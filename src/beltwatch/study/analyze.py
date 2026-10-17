"""Analyse usability-study sessions exactly as declared in docs/usability_study.md.

Usage::

    uv run python -m beltwatch.study.analyze data/study/<name> --results results-P*.json \\
        --out reports/<name>

Writes ``study_report.json``. Only complete sessions are analysed; excluded
sessions are listed with the reason. No trial is dropped. Intervals are a
percentile bootstrap over participants (whole participants are resampled, so
each person's two conditions stay paired).
"""

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from beltwatch.study.prepare import StudyError, load_study
from beltwatch.study.submit import AUDITS_FILE, PILOT

log = logging.getLogger(__name__)

RESULTS_VERSION = 1
CONDITIONS = ("unaided", "assisted")
MATERIALS = ("cardboard", "soft_plastic", "rigid_plastic", "metal")
TIME_REDUCTION_TARGET = 0.30
RECALL_DIFFERENCE_TARGET = -0.05
MIN_GROUPS = 5
N_RESAMPLES = 2000


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def load_session(path: Path, study: dict[str, Any], audits: dict[str, Any]) -> dict[str, Any]:
    """Validate one results file; raises StudyError with the exclusion reason."""
    data = json.loads(path.read_text())
    if data.get("results_version") != RESULTS_VERSION:
        raise StudyError("unsupported results version")
    if data.get("study") != study["name"]:
        raise StudyError(f"results are for study {data.get('study')!r}")
    if data.get("participant") == PILOT:
        raise StudyError("pilot session (never analysed)")
    if not data.get("completed"):
        raise StudyError("session not completed")
    lookup = {
        image_id: (set_name, info)
        for set_name, entry in audits["audits"].items()
        for image_id, info in entry["images"].items()
    }
    truth = {i["study_id"]: i for i in study["images"]}
    trials = []
    seen: dict[str, set[str]] = {c: set() for c in CONDITIONS}
    sets: dict[str, set[str]] = {c: set() for c in CONDITIONS}
    for t in data["trials"]:
        if t["practice"]:
            continue
        if t["image_id"] not in lookup or t["condition"] not in CONDITIONS:
            raise StudyError(f"unknown image {t['image_id']} or condition {t['condition']}")
        set_name, info = lookup[t["image_id"]]
        if t["image_id"] in seen[t["condition"]]:
            raise StudyError(f"image {t['image_id']} answered twice")
        seen[t["condition"]].add(t["image_id"])
        sets[t["condition"]].add(set_name)
        gt = truth[info["study_id"]]
        trials.append(
            {
                **t,
                "study_id": info["study_id"],
                "set": set_name,
                "truth_positive": gt["audit_positive"],
                "truth_materials": gt["materials_present"],
                "suggestion": info["suggestion"] if t["condition"] == "assisted" else None,
            }
        )
    for condition in CONDITIONS:
        if len(sets[condition]) != 1:
            raise StudyError(f"{condition} trials do not come from exactly one set")
        (set_name,) = sets[condition]
        expected = set(audits["audits"][set_name]["images"])
        if seen[condition] != expected:
            raise StudyError(f"{condition}: {len(expected - seen[condition])} images unanswered")
    if sets["unaided"] == sets["assisted"]:
        raise StudyError("both conditions used the same set")
    return {"participant": data["participant"], "file": path.name, "trials": trials}


def _condition_stats(trials: list[dict[str, Any]]) -> dict[str, Any]:
    pos = [t for t in trials if t["truth_positive"]]
    neg = [t for t in trials if not t["truth_positive"]]
    materials = {}
    for m in MATERIALS:
        tp = sum(m in t["materials"] and m in t["truth_materials"] for t in trials)
        fn = sum(m not in t["materials"] and m in t["truth_materials"] for t in trials)
        fp = sum(m in t["materials"] and m not in t["truth_materials"] for t in trials)
        materials[m] = {
            "recall": _ratio(tp, tp + fn),
            "precision": _ratio(tp, tp + fp),
            "n_present": tp + fn,
        }
    return {
        "n_trials": len(trials),
        "median_seconds": float(np.median([t["seconds"] for t in trials])) if trials else None,
        "recall": _ratio(sum(t["follow_up"] for t in pos), len(pos)),
        "false_alarm_rate": _ratio(sum(t["follow_up"] for t in neg), len(neg)),
        "n_positive": len(pos),
        "n_negative": len(neg),
        "materials": materials,
    }


def _assisted_extras(trials: list[dict[str, Any]]) -> dict[str, Any]:
    changed, n_changes, wrong, kept_wrong = 0, [], 0, 0
    for t in trials:
        s = t["suggestion"]
        diff = int(t["follow_up"] != s["follow_up"]) + len(
            set(t["materials"]) ^ set(s["materials"])
        )
        n_changes.append(diff)
        changed += diff > 0
        if s["follow_up"] != t["truth_positive"]:
            wrong += 1
            kept_wrong += t["follow_up"] == s["follow_up"]
    return {
        "changed_suggestion_share": _ratio(changed, len(trials)),
        "mean_changed_answers": float(np.mean(n_changes)) if n_changes else None,
        "wrong_suggestions": wrong,
        "followed_wrong_suggestion_share": _ratio(kept_wrong, wrong),
    }


def _pooled(sessions: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    return [t for s in sessions for t in s["trials"] if t["condition"] == condition]


def _time_reduction(session: dict[str, Any]) -> float:
    med = {
        c: float(np.median([t["seconds"] for t in session["trials"] if t["condition"] == c]))
        for c in CONDITIONS
    }
    return 1.0 - med["assisted"] / med["unaided"]


def _recall_difference(sessions: list[dict[str, Any]]) -> float:
    r = {c: _condition_stats(_pooled(sessions, c))["recall"] for c in CONDITIONS}
    return float(r["assisted"] - r["unaided"])


def _bootstrap(
    sessions: list[dict[str, Any]], stat: Callable[[list[dict[str, Any]]], float]
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    n = len(sessions)
    values = [stat([sessions[i] for i in rng.integers(0, n, n)]) for _ in range(N_RESAMPLES)]
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {
        "low": float(lo),
        "high": float(hi),
        "n_groups": n,
        "too_few_groups": n < MIN_GROUPS,
        "resamples": N_RESAMPLES,
        "unit": "participant",
    }


def analyze(study_dir: Path, result_files: Sequence[Path]) -> dict[str, Any]:
    study = load_study(study_dir)
    audits_path = study_dir / AUDITS_FILE
    if not audits_path.is_file():
        raise StudyError(f"no {AUDITS_FILE}; run beltwatch.study.submit first")
    audits = json.loads(audits_path.read_text())
    sessions, excluded = [], []
    for path in sorted(result_files):
        try:
            sessions.append(load_session(path, study, audits))
        except (StudyError, KeyError, json.JSONDecodeError) as exc:
            excluded.append({"file": path.name, "reason": str(exc)})
    ids = [s["participant"] for s in sessions]
    if len(set(ids)) != len(ids):
        raise StudyError(f"duplicate participant codes: {sorted(ids)}")
    if not sessions:
        raise StudyError("no complete sessions to analyse")

    def median_reduction(group: list[dict[str, Any]]) -> float:
        return float(np.median([_time_reduction(s) for s in group]))

    reduction = median_reduction(sessions)
    recall_diff = _recall_difference(sessions)
    assisted = _pooled(sessions, "assisted")
    by_condition = {c: _condition_stats(_pooled(sessions, c)) for c in CONDITIONS}
    by_condition["assisted"] |= _assisted_extras(assisted)
    reference = {}
    for set_name in ("A", "B"):
        entry = audits["audits"][set_name]
        truth = {i["study_id"]: i["audit_positive"] for i in study["images"]}
        flags = [
            (info["suggestion"]["follow_up"], truth[info["study_id"]])
            for info in entry["images"].values()
        ]
        pos = [f for f, t in flags if t]
        neg = [f for f, t in flags if not t]
        reference[set_name] = {
            "recall": _ratio(sum(pos), len(pos)),
            "false_alarm_rate": _ratio(sum(neg), len(neg)),
        }
    few = len(sessions) < MIN_GROUPS
    return {
        "study": study["name"],
        "label": "usability study with peer participants (not a facility pilot)",
        "model_versions": sorted({e["model_version"] for e in audits["audits"].values()}),
        "n_sessions": len(sessions),
        "excluded_sessions": excluded,
        "prevalence": {k: v["prevalence"] for k, v in study["sets"].items() if k in ("A", "B")},
        "primary": {
            "median_time_reduction": reduction,
            "median_time_reduction_interval": _bootstrap(sessions, median_reduction),
            "time_reduction_target": TIME_REDUCTION_TARGET,
            "time_reduction_met": reduction >= TIME_REDUCTION_TARGET,
            "recall_difference": recall_diff,
            "recall_difference_interval": _bootstrap(sessions, _recall_difference),
            "recall_difference_target": RECALL_DIFFERENCE_TARGET,
            "recall_met": recall_diff >= RECALL_DIFFERENCE_TARGET,
            "too_few_groups": few,
        },
        "by_condition": by_condition,
        "per_participant": [
            {
                "participant": s["participant"],
                "time_reduction": _time_reduction(s),
                **{
                    f"{c}_median_seconds": _condition_stats(
                        [t for t in s["trials"] if t["condition"] == c]
                    )["median_seconds"]
                    for c in CONDITIONS
                },
            }
            for s in sessions
        ],
        "model_suggestion_reference": reference,
    }


def pilot_summary(path: Path) -> dict[str, Any]:
    """Session length and decision times of a pilot. No accuracy, by design."""
    data = json.loads(path.read_text())
    if data.get("participant") != PILOT:
        raise StudyError(f"{path.name} is not a pilot session")
    trials = data.get("trials", [])
    started, finished = data.get("started_at"), data.get("finished_at")
    minutes = None
    if started and finished:
        delta = datetime.fromisoformat(finished) - datetime.fromisoformat(started)
        minutes = delta.total_seconds() / 60
    by_condition = {}
    for condition in CONDITIONS:
        seconds = [
            t["seconds"] for t in trials if t["condition"] == condition and not t["practice"]
        ]
        by_condition[condition] = {
            "n_trials": len(seconds),
            "median_seconds": float(np.median(seconds)) if seconds else None,
            "max_seconds": max(seconds) if seconds else None,
        }
    return {
        "label": "pilot session: protocol check only, not a usability-study result",
        "completed": bool(data.get("completed")),
        "n_trials": len(trials),
        "session_minutes": minutes,
        "by_condition": by_condition,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--pilot", action="store_true", help="summarise a PILOT session (timing only)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.pilot:
        try:
            for path in args.results:
                print(json.dumps(pilot_summary(path), indent=2))
        except StudyError as exc:
            log.error("%s", exc)
            return 1
        return 0
    if args.out is None:
        parser.error("--out is required unless --pilot is given")
    try:
        report = analyze(args.study_dir, args.results)
    except StudyError as exc:
        log.error("%s", exc)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "study_report.json").write_text(json.dumps(report, indent=2) + "\n")
    p = report["primary"]
    log.info(
        "%d sessions (%d excluded): median time reduction %.1f%% (target 30%%: %s), "
        "recall difference %+.1f pp (target >= -5 pp: %s)%s",
        report["n_sessions"],
        len(report["excluded_sessions"]),
        100 * p["median_time_reduction"],
        "met" if p["time_reduction_met"] else "missed",
        100 * p["recall_difference"],
        "met" if p["recall_met"] else "missed",
        " [too few participants for intervals]" if p["too_few_groups"] else "",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
