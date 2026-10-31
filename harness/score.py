"""Points from result.json files, by the formula of the task statement.

    python harness/score.py runs/result.json                        # your ratios
    python harness/score.py runs/*/result.json                      # jury: every team, one folder each

Speed s = base time / solution time, memory m = base peak / solution peak, both measured in the
same session. A gain under 5% is a tie and counts as 1.0. Points: speed 35 * ln(s) / ln(best s),
memory 30 * ln(m) / ln(best m), never below 0, where "best" is the best admitted team. A team is
admitted when every check of check_rules.py passed on this very result.json (checks.json next to
it); only an admitted team gets the 20 points for "works and correct" and any speed or memory
points. README (15 points) is scored by the jury by hand. On equal points the team with the
better memory ratio ranks higher.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_rules import CHECK_COUNT  # noqa: E402
from common import file_sha256, read_json  # noqa: E402

TIE = 0.05
SPEED_POINTS, MEMORY_POINTS, CORRECT_POINTS = 35, 30, 20


def number(value):
    """A finite number, or None for anything else (strings, inf, nan, booleans)."""
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    return value if ok else None


def effective(ratio):
    ratio = number(ratio)
    return ratio if ratio is not None and ratio >= 1 + TIE else 1.0


def points(value, best, weight):
    if best <= 1.0 or value <= 1.0:
        return 0.0
    return weight * math.log(value) / math.log(best)


def admitted(result_path):
    """Every check of check_rules.py passed, on exactly this result.json."""
    checks = read_json(Path(result_path).parent / "checks.json")
    if not isinstance(checks, dict):
        return False
    rows, sha = checks.get("checks"), checks.get("result_sha256")
    return (isinstance(sha, str) and sha == file_sha256(result_path)
            and isinstance(rows, list) and len(rows) == CHECK_COUNT
            and all(isinstance(row, dict) and row.get("ok") is True for row in rows))


def shown(value, fmt):
    value = number(value)
    return format(value, fmt) if value is not None else "-"


def ratios_of(path):
    result = read_json(path)
    summary = result.get("summary") if isinstance(result, dict) else None
    ratios = summary.get("ratios") if isinstance(summary, dict) else None
    return ratios if isinstance(ratios, dict) else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+")
    args = parser.parse_args()

    teams = []
    for path in args.results:
        ratios = ratios_of(path)
        teams.append({"name": Path(path).parent.name if len(args.results) > 1 else path,
                      "speedup": ratios.get("speedup"), "memory": ratios.get("memory_ratio"),
                      "quality": ratios.get("quality_delta_pct"), "admitted": admitted(path)})

    if len(teams) == 1:
        team = teams[0]
        print(f"speedup {shown(team['speedup'], '.3f')}x (counts as "
              f"{effective(team['speedup']):.3f}), memory base/solution "
              f"{shown(team['memory'], '.3f')}x (counts as {effective(team['memory']):.3f}), "
              f"eval loss vs base {shown(team['quality'], '+.3f')} %, "
              f"all checks passed on this result: {team['admitted']}")
        print("Speed and memory points depend on the best admitted team: "
              f"{SPEED_POINTS} * ln(s) / ln(best s) and {MEMORY_POINTS} * ln(m) / ln(best m).")
        return

    good = [t for t in teams if t["admitted"]]
    best_s = max((effective(t["speedup"]) for t in good), default=1.0)
    best_m = max((effective(t["memory"]) for t in good), default=1.0)
    for team in teams:
        team["points"] = (round(CORRECT_POINTS
                                + points(effective(team["speedup"]), best_s, SPEED_POINTS)
                                + points(effective(team["memory"]), best_m, MEMORY_POINTS), 2)
                          if team["admitted"] else None)
    teams.sort(key=lambda t: (-(t["points"] if t["points"] is not None else -1),
                              -(number(t["memory"]) or 0)))
    print(f"{'team':30s} {'speedup':>8s} {'memory':>8s} {'quality%':>9s} {'points':>12s}"
          "  (+ README up to 15, by hand)")
    for team in teams:
        points_text = "not admitted" if team["points"] is None else f"{team['points']:12.2f}"
        print(f"{team['name'][:30]:30s} {shown(team['speedup'], '8.3f')} "
              f"{shown(team['memory'], '8.3f')} {shown(team['quality'], '9.3f')} {points_text}")


if __name__ == "__main__":
    main()
