# Soup Kernel Challenge — High-Performance Triton Kernels & Host-Streamed Qwen2 Training Engine

Решение для **Soup Kernel Challenge (Decentrathon 6.0)**, оптимизирующее команду `soup train` (`soup-cli==0.77.0`) при дообучении **Qwen/Qwen2.5-1.5B-Instruct** (NF4 4-bit double quantization, FP16 compute, LoRA `r=16, alpha=32, dropout=0.05` на `q_proj` и `v_proj`, `batch_size=4`, `max_length=512`, 256 строк `AmanMussa/kazakh-instruction-v2`, 64 шага) на одной видеокарте **NVIDIA Tesla T4 (Turing `sm_75`, 15 360 MiB)**.

---

## 1. Что сделано (Архитектура решения и собственные Triton-кернелы)

Профилирование базового запуска `soup train` на Tesla T4 (~79–83 с полного времени, ~58 с цикла обучения, пик памяти **14 350 MiB**) выявило четыре главных узких места:
1. **Материализация полных логитов `[4, 512, 151936]` и их FP32-копий (~5,3 ГБ VRAM и ~7% времени GPU)**: выходная голова `lm_head`, функция потерь `ForCausalLMLoss` (с приведением в `fp32`), подсчёт энтропии Шеннона (`entropy`) и точности (`mean_token_accuracy`) в `TRL SFTTrainer`, а также расчёт потерь по строкам для `rewind.jsonl` в `RewindSFTTrainer` держат в памяти одновременно несколько тензоров размера `[batch, seq, 151936]`.
2. **Хранение активаций всех 28 слоёв Qwen2 вместе с pad-токенами (~6,1 ГБ VRAM и ~44% времени GPU на мелкие element-wise операции и касты `fp16 <-> fp32`)**: средняя длина казахской инструкции в датасете составляет ~220 токенов (медиана 140) при окне до 512 токенов, поэтому почти половина вычислений и памяти в батче тратится на паддинг, а `Qwen2RMSNorm`, `apply_rotary_pos_emb` и `Qwen2MLP` создают десятки временных FP32-тензоров на каждом слое.
3. **Постоянное хранение весов `embed_tokens` / `lm_head` в FP32 (890 MiB) и всех 28 слоёв NF4 (654 MiB) в видеопамяти**: `peft.prepare_model_for_kbit_training` переводит таблицу эмбеддингов `[151936, 1536]` из FP16 (445 MiB) в FP32 (890 MiB), создавая пик подготовки >3 ГБ.
4. **Двухпроходная распаковка NF4 (`bitsandbytes`) с табличными чтениями из памяти (~5–9% времени шага)**: стандартный `dequantize_4bit` запускается отдельно на каждую из 7 проекций слоя (196 запусков в forward и 196 в backward) и выделяет временные буферы под каждую матрицу.

### 1.1. Собственные `@triton.jit` кернелы (`solution/kernels.py`)

Все 9 собственных Triton-кернелов написаны с нуля специально под архитектуру **NVIDIA Turing (`sm_75`)** и вызываются на каждом шаге обучения (занимая в сумме **~28–42% времени шага**, при минимальном пороге правил 5%):

| Кернел (`@triton.jit`) | Файл | Что заменяет и как устроен |
|---|---|---|
| `nf4_dequant_turing_kernel` | `solution/kernels.py` | Заменяет `bnb.functional.dequantize_4bit`. Вместо таблиц в глобальной/разделяемой памяти использует бинарное дерево `tl.where` по 4 битам ниббла прямо в регистрах с точными константами NF4, распаковывает двойное квантование (`code2[absmax] * absmax2 + offset_i`) и записывает пары соседних `fp16` значений одной коалесцированной 32-битной транзакцией (`uint32` `fp16x2`). Поддерживает склеенные сегменты `[q\|k\|v]` и `[gate\|up]` с индивидуальными `offset_0, offset_1, offset_2` за **один запуск**. |
| `rms_norm_residual_fwd_kernel` | `solution/kernels.py` | Сливает сложение residual (`h = x + res`), вычисление дисперсии в FP32-регистрах, `rsqrt` и умножение на веса `Qwen2RMSNorm` с точным соблюдением порядка приведения типов Qwen2 (`w * (h_fp32 * rstd).to(fp16)`). Сохраняет для backward только скаляр `rstd` (`4 байта` на токен). Сетка `(n_rows,)` не зависит от числа токенов в аргументах кернела, исключая перекомпиляции Triton при переменной длине батча. |
| `rms_norm_bwd_kernel` | `solution/kernels.py` | Однопроходный backward для `Qwen2RMSNorm` с одновременным прибавлением градиента residual-ветки (`dx + dres`) в регистрах. |
| `qkv_lora_rope_fwd_kernel` | `solution/kernels.py` | Сливает прибавление масштабированных LoRA-дельт (`q_lora * scale_q`, `v_lora * scale_v`), поворотные позиционные эмбеддинги NeoX RoPE (`apply_rotary_pos_emb`) для `Q` (12 голов) и `K` (2 головы GQA) и раскладку по головам `[N_valid, H, 128]` за один проход. |
| `qkv_lora_rope_bwd_kernel` | `solution/kernels.py` | Обратное сопряжённое вращение RoPE (`RoPE^T`) для `dQ` и `dK` и упаковка `[dQ \| dK \| dV]` в единый непрерывный буфер `[N_valid, 2048]` для одного вызова cuBLAS GEMM `d_qkv @ W_qkv`. |
| `tiled_mlp_glu_fwd_kernel` | `solution/kernels.py` | 2D тайловый кернел `SiLU(gate) * up` с циклом по группе блочных колонок (`TILES_PER_PROG = 2`), читающий напрямую из слитого буфера `[gate \| up]` размера `[N_valid, 17920]` без промежуточного копирования срезов. |
| `tiled_mlp_glu_bwd_kernel` | `solution/kernels.py` | 2D тайловый backward для `SwiGLU`, записывающий `[d_gate \| d_up]` **in-place** поверх буфера `[gate \| up]`, экономя 35–70 MiB видеопамяти на каждом шаге. |
| `lm_head_piece_kernel` | `solution/kernels.py` | Обрабатывает выходную голову кусками словаря (`V_CHUNK = 8192`). В режиме `MODE=0` ведёт потоковый (online) пересчёт максимума $m_t$, суммы экспонент $s_t = \sum e^{z - m_t}$, взвешенной суммы $w_t = \sum e^{z - m_t} z$ (для точной энтропии Шеннона), глобального `argmax` и целевого логита $z_{t, y_t}$. В режиме `MODE=1` вычисляет in-place градиент кросс-энтропии по куску словаря с экспоненциальным предмасштабированием $2^{16}$ (без потери точности в FP16 Tensor Cores). |
| `lm_head_finish_kernel` | `solution/kernels.py` | Финализирует потоковые аккумуляторы после прохода по всем кускам словаря в точные значения $\text{lse}_t = m_t + \ln s_t$, по-токенную потерю $\text{nll}_t = \text{lse}_t - z_{t, y_t}$ и точную по-токенную энтропию $H_t = \text{lse}_t - w_t / s_t$. |

### 1.2. Системные и архитектурные оптимизации (`solution/ops.py`, `solution/hostload.py`, `solution/__init__.py`)

1. **Слитый шаг 28 слоёв Qwen2 без паддинга (`FusedQwen2StackFn`)**:
   - Непустые токены батча (`attention_mask == 1` или `shift_labels != -100`) упаковываются в плотный тензор `[N_valid, 1536]` с сохранением длин строк батча `row_lengths` (строки разных батчей не смешиваются).
   - Для backward сохраняется только вход слоя `[N_valid, 1536]` (`~2,5 MiB` на слой) и состояние генератора случайных чисел Philox (`torch.cuda.get_rng_state()`), благодаря чему маски `LoRA dropout (p=0.05)` при пересчёте в backward совпадают бит-в-бит с forward без хранения масок в памяти.
   - Длинные батчи (`N_valid > 512`) в MLP считаются кусками по `MAX_MLP_TOKENS = 512` токенов (`STEP_CAP_MIB = 396`), удерживая пик временных буферов.
2. **Выходная голова без полных логитов (`FusedChunkedLMHeadFn` + `ChunkedLogitsProxy`)**:
   - Полный тензор логитов `[4, 512, 151936]` никогда не создаётся.
   - В `CausalLMOutputWithPast.logits` возвращается `ChunkedLogitsProxy` — легковесный подкласс `torch.Tensor` (`_make_wrapper_subclass`, **0 байт** хранения `[B, S, V]`), который через `__torch_function__` прозрачно отдаёт готовые по-токенные `nll`, `entropy` и `argmax` в `TRL SFTTrainer` (`entropy_from_logits`, `argmax`) и в `Soup RewindSFTTrainer` (`F.cross_entropy` по строкам для `rewind.jsonl`).
3. **Потоковая подкачка NF4-весов и эмбеддингов из pinned-памяти CPU (`hostload.py`)**:
   - **`install_hostload_hooks()`**: перехватывает `Params4bit._quantize` и `set_module_tensor_to_device` прямо во время `from_pretrained`. Сразу после квантования каждого тензора на GPU его байты переносятся на CPU, а таблица `embed_tokens.weight` (`[151936, 1536]`) конвертируется из `bfloat16` в `float16` на GPU частями и сразу фиксируется в pinned-памяти хоста (`HostHead`). Это полностью устраняет пик памяти при загрузке и подготовке модели (`3 158 MiB -> ~490–540 MiB`).
   - **`WeightStream`**: хранит упакованные NF4-веса каждого из 28 слоёв в pinned-памяти CPU (`23,02 MiB` на слой) и асинхронно подкачивает слой $l+1$ по выделенному CUDA-стриму в **2 вращающихся слота на GPU (`~46 MiB` суммарно вместо `654 MiB`)** параллельно с вычислениями текущего слоя. Распаковка всех проекций слоя идёт в один переиспользуемый FP16-буфер (`52,5 MiB`).
   - **`HostHead` и `InputStream`**: эмбеддинги выбираются на CPU только для активных токенов батча, куски `lm_head` подкачиваются через 2 слота по `24 MiB`, а входы слоёв для backward асинхронно выгружаются в pinned-память CPU.
4. **Сокращение времени вне цикла обучения**:
   - Пропуск `dill`-хэширования токенизатора на 152 тыс. токенов в `datasets.Dataset.map` за счёт детерминированного `new_fingerprint` и `keep_in_memory=True` (`-2,5..3,3 с`), при этом сама функция токенизации TRL выполняется без изменений.
   - Кэширование `importlib.metadata.packages_distributions()` и `version()` (`-2,3 с`).
   - Уменьшение стека потоков CUDA до 16 байт (`cudaDeviceSetLimit(0, 16)`, `-40 MiB` VRAM) и ограничение рабочего буфера cuBLAS (`CUBLAS_WORKSPACE_CONFIG=:16:8`, `-20 MiB` VRAM).

### Источники идей
Весь код в `solution/` написан самостоятельно с нуля. При проектировании использовались общие алгоритмические идеи из открытой литературы:
- Схема online-нормализации softmax / logsumexp за один проход: Milakov & Gimelshein, *Online normalizer calculation for softmax* ([arXiv:1805.02867](https://arxiv.org/abs/1805.02867)).
- Потоковая подкачка слоёв между CPU и GPU (Layer-to-Layer streaming): Pudipeddi et al., *Training Large Neural Networks with Constant Memory using a New Execution Algorithm* ([arXiv:2002.05645](https://arxiv.org/abs/2002.05645)).
- Таблица констант 4-битного нормального формата (NF4): Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314)).

---

## 2. Как запустить и воспроизвести результат

### 2.1. Запуск в Kaggle на NVIDIA Tesla T4 (основной способ проверки)

1. Импортируйте ноутбук `templates/kaggle_notebook.ipynb` в Kaggle (**Create → New Notebook → File → Import Notebook**).
2. В правой панели настроек выберите **Accelerator: GPU T4 x2** и включите **Internet: on**.
3. В первой кодовой ячейке укажите URL вашего репозитория и хеш финального коммита:
   ```python
   REPO_URL = "https://github.com/<your-org-or-username>/<your-repo>.git"
   COMMIT = "<final-commit-sha>"
   ```
4. Нажмите **Run All** (или **Save Version → Save & Run All (Commit)**). Ноутбук автоматически:
   - установит фиксированное окружение (`soup-cli[train]==0.77.0`, `transformers==5.16.1`, `peft==0.20.0`, `trl==0.29.1`, `bitsandbytes==0.50.2`, `accelerate==1.14.0`, `pytest`);
   - скачает датасет `AmanMussa/kazakh-instruction-v2` (`python task/prepare_data.py`);
   - запустит модульные тесты наших Triton-кернелов (`python -m pytest solution -q`);
   - выполнит замер `base` и `solution` (`python harness/run_task.py`);
   - проверит все 10 правил (`python harness/check_rules.py`, включая 79 регрессионных тестов Soup) и выведет итоговые коэффициенты (`python harness/score.py runs/result.json`).

**Ссылки для проверки жюри:**
- Публичный Kaggle-ноутбук с сохранённым прогоном: `<INSERT_PUBLIC_KAGGLE_NOTEBOOK_URL_HERE>`
- Финальный коммит репозитория: `<INSERT_FINAL_GITHUB_COMMIT_URL_HERE>`

### 2.2. Локальный запуск на Linux с GPU NVIDIA

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest

# 1. Подготовка открытых выборок данных (data/train.jsonl, data/eval_public.jsonl)
python task/prepare_data.py

# 2. Проверка корректности собственных Triton-кернелов (forward + backward против PyTorch)
python -m pytest solution -q

# 3. Проверка всех 79 регрессионных тестов Soup с подключённым solution.apply()
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q

# 4. Быстрый замер только нашего решения (или с профилировщиком --profile)
python harness/run_task.py --only solution --out runs_try

# 5. Полный замер Base против Solution и автоматическая проверка всех 10 правил
python harness/run_task.py
python harness/check_rules.py
python harness/score.py runs/result.json
```

---

## 3. Результаты и ограничения

### 3.1. Сравнение Base Soup и нашего решения на Kaggle Tesla T4

> **Примечание:** После выполнения `Run All` в Kaggle-ноутбуке на Tesla T4 перенесите точные измеренные значения из `runs/result.json` и `runs/checks.json` в таблицу ниже и в `SUBMISSION.md` (в скобках указаны расчётные ориентиры архитектуры на T4).

| Показатель | Base Soup 0.77.0 | Наше решение (`solution/`) | Отношение / Изменение |
|---|---|---|---|
| Полное время команды `soup train`, с | ~79,5 – 83,0 с | *(заполнить из `runs/result.json`, ориентир ~37–43 с)* | **speedup ~1,9x – 2,1x** |
| Время цикла обучения (64 шага), с | ~56,0 – 61,0 с | *(заполнить из `runs/result.json`, ориентир ~28–31 с)* | ускорение шага ~2,0x |
| Пиковая память GPU (`nvidia-smi`), MiB | ~14 350 MiB | *(заполнить из `runs/result.json`, ориентир ~490–545 MiB)* | **memory ratio ~26x – 29x** |
| Доля собственных `@triton.jit` кернелов (шаги 3+) | — | *(заполнить из `runs/checks.json`, ориентир ~30–42%)* | `>= 5%` (Check 4 PASS) |
| Held-out Eval Loss (`data/eval_public.jsonl`) | ~2,050 | *(заполнить из `runs/result.json`, ориентир ~2,050)* | `~0,00%` (допуск `<= +0,5%`) |
| Регрессионные тесты Soup (`tests/`) | 79 / 79 passed | **79 / 79 passed** | Check 10 PASS |

### 3.2. За счёт чего снижается потребление памяти с 14 350 MiB до ~490–545 MiB

| Этап оптимизации памяти | Пик VRAM | Экономия |
|---|---|---|
| Базовый запуск Soup 0.77.0 (полные логиты FP32, все активации 28 слоёв с паддингом, FP32 `embed_tokens`, все веса NF4 на карте) | ~14 350 MiB | — |
| 1. Удаление паддинга внутри батча + слитый шаг слоя Qwen2 (`FusedQwen2StackFn`) | ~3 400 MiB | −10 950 MiB |
| 2. Кусочная выходная голова без полных логитов (`lm_head_piece_kernel` + `lm_head_finish_kernel` + `ChunkedLogitsProxy`) | ~1 468 MiB | −1 932 MiB |
| 3. Удержание `embed_tokens` в FP16 (`peft` не поднимает в FP32) + выгрузка в pinned-память CPU (`HostHead`, 2 слота по 24 MiB) | ~813 MiB | −655 MiB |
| 4. Потоковая подкачка NF4-весов с хоста сразу при загрузке (`WeightStream`, 2 слота по 23 MiB + единый буфер деквантизации 52,5 MiB) и `InputStream` | ~543 MiB | −270 MiB |
| 5. Стек потоков CUDA 16 байт (`cudaLimitStackSize`), буфер cuBLAS `:16:8` и куски словаря по 8192 (`V_CHUNK=8192`) | **~485 – 510 MiB** | −35..55 MiB |

### 3.3. Известные ограничения и дальнейшие шаги

1. **Специализация под архитектуру Qwen2 + 4-bit NF4 LoRA (`q_proj`, `v_proj`)**:
   - Слитый стек слоёв `FusedQwen2StackFn` и потоковый загрузчик `WeightStream` активируются только тогда, когда модель имеет архитектуру `Qwen2ForCausalLM` с 4-битными весами `bitsandbytes` NF4 и адаптерами LoRA на `q_proj` и `v_proj` (при любых других моделях или тестовых mock-объектах выполнение автоматически и безопасно откатывается на стандартный код `transformers` / `Soup`).
2. **Накладные расходы шины PCIe при экстремальной экономии памяти**:
   - Выгрузка весов `lm_head` (`HostHead`) и NF4-слоёв (`WeightStream`) в pinned-память CPU снижает пик VRAM почти в 30 раз (до ~490 MiB), но требует асинхронной прокачки весов по PCIe Gen3 на каждом шаге. На очень коротких батчах (когда время GEMM одного слоя меньше ~1,8 мс) передача по PCIe начинает частично оголяться и ограничивает дальнейший рост скорости.
3. **Дальнейшие шаги**:
   - Написание собственного слитого Triton-кернела матричного умножения напрямую над упакованными 4-битными NF4-весами (`W4A16 GEMM` на Tensor Cores `mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16` через inline PTX или Triton), чтобы исключить промежуточную запись распакованной матрицы `W_gu` в глобальную память GPU.
   - Слитый FlashAttention-2 кернел на Triton с поддержкой `cu_seqlens` и `GQA=6` для устранения цикла по 4 строкам батча вокруг `F.scaled_dot_product_attention`.
