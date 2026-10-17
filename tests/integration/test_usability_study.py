import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from beltwatch.jobs.worker import Worker
from beltwatch.study.analyze import analyze
from beltwatch.study.prepare import StudyError, load_study, prepare_study
from beltwatch.study.submit import ARMS, submit_study
from tests.integration.conftest import png_bytes

PX = ("background", "cardboard", "soft_plastic", "rigid_plastic", "metal")
TOTAL_PX = 10_000


def make_dataset(root: Path) -> dict[str, Path]:
    """Synthetic manifests: 3 val sequences (frames 100 apart) plus test images."""
    raw = root / "raw"
    rows, splits = [], []
    n = 0
    for seq in ("01", "02", "03"):
        for k in range(40):
            n += 1
            # Every 4th image negative (2% coverage); positives from 6% to 45%.
            total = 0.02 if k % 4 == 0 else 0.06 + 0.01 * k
            cardboard = int(total * TOTAL_PX * 0.7)
            soft = int(total * TOTAL_PX) - cardboard
            px = [TOTAL_PX - cardboard - soft, cardboard, soft, 0, 0]
            image_id = f"val/{seq}_frame_{k * 100:06d}"
            path = f"val/data/{seq}_frame_{k * 100:06d}.png"
            (raw / path).parent.mkdir(parents=True, exist_ok=True)
            (raw / path).write_bytes(png_bytes(n))
            rows.append(
                {
                    "image_id": image_id,
                    "sequence_id": seq,
                    "frame_index": k * 100,
                    "image_path": path,
                    "image_sha256": hashlib.sha256(png_bytes(n)).hexdigest(),
                    **{f"px_{c}": v for c, v in zip(PX, px, strict=True)},
                }
            )
            splits.append({"image_id": image_id, "split": "val"})
    for k in range(10):  # locked test images: must never be selected
        image_id = f"test/09_frame_{k * 100:06d}"
        rows.append(
            {
                "image_id": image_id,
                "sequence_id": "09",
                "frame_index": k * 100,
                "image_path": "test/missing.png",
                "image_sha256": "0" * 64,
                **{f"px_{c}": v for c, v in zip(PX, [5000, 5000, 0, 0, 0], strict=True)},
            }
        )
        splits.append({"image_id": image_id, "split": "test"})
    files = {"splits_csv": root / "splits.csv", "images_csv": root / "images.csv", "raw_dir": raw}
    for path, data in ((files["splits_csv"], splits), (files["images_csv"], rows)):
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    return files


@pytest.fixture
def study_dir(tmp_path: Path) -> Path:
    out = tmp_path / "study-test"
    prepare_study(out, **make_dataset(tmp_path / "ds"), seed=0)  # type: ignore[arg-type]
    return out


def test_prepare_selects_matched_val_sets(study_dir: Path, tmp_path: Path) -> None:
    study = load_study(study_dir)
    images = study["images"]
    by_set = {s: [i for i in images if i["set"] == s] for s in ("A", "B", "practice")}
    assert [len(by_set[s]) for s in ("A", "B", "practice")] == [20, 20, 2]
    assert all(i["source_image_id"].startswith("val/") for i in images)
    for s in ("A", "B"):
        assert sum(i["audit_positive"] for i in by_set[s]) == 14
    assert sorted(i["audit_positive"] for i in by_set["practice"]) == [False, True]
    assert len({i["source_image_id"] for i in images}) == 42
    # No two selected frames of a sequence are closer than the declared gap.
    for a in images:
        for b in images:
            if a is not b and a["sequence_id"] == b["sequence_id"]:
                assert abs(a["frame_index"] - b["frame_index"]) >= 100
    # Coverage-paired assignment keeps the sets matched.
    mean = {s: study["sets"][s]["mean_total_coverage"] for s in ("A", "B")}
    assert abs(mean["A"] - mean["B"]) < 0.02
    first = next(i for i in images if i["audit_positive"])
    assert first["materials_present"] == ["cardboard", "soft_plastic"]
    assert (study_dir / first["file"]).is_file()
    with pytest.raises(StudyError, match="not empty"):
        prepare_study(study_dir, **make_dataset(tmp_path / "ds2"))  # type: ignore[arg-type]


def test_prepare_reports_missing_frames(tmp_path: Path) -> None:
    files = make_dataset(tmp_path / "ds")
    for png in (files["raw_dir"] / "val" / "data").glob("01_*.png"):
        png.unlink()
    with pytest.raises(StudyError, match="not under"):
        prepare_study(tmp_path / "out", **files)  # type: ignore[arg-type]


def test_prepare_refuses_frames_that_differ_from_the_manifest(tmp_path: Path) -> None:
    files = make_dataset(tmp_path / "ds")
    for png in (files["raw_dir"] / "val" / "data").glob("*.png"):
        png.write_bytes(png_bytes(999))
    with pytest.raises(StudyError, match="SHA-256"):
        prepare_study(tmp_path / "out", **files)  # type: ignore[arg-type]


def client_transport(client: TestClient, worker: Worker):  # type: ignore[no-untyped-def]
    def call(method: str, path: str, body: bytes | None, headers: dict[str, str]) -> Any:
        if method == "GET":
            worker.run_once()
        response = client.request(method, path, content=body, headers=headers)
        assert response.status_code < 300, response.text
        return response.json()

    return call


def test_submit_creates_audits_and_counterbalanced_plans(
    study_dir: Path, client: TestClient, worker: Worker
) -> None:
    transport = client_transport(client, worker)
    record = submit_study(study_dir, transport, participants=4, poll_seconds=0)
    audits = record["audits"]
    assert set(audits) == {"practice", "A", "B"}
    assert {a["model_version"] for a in audits.values()} == {"test-1"}
    assert len(audits["A"]["images"]) == 20 and len(audits["practice"]["images"]) == 2
    # all-background never suggests follow-up or a material.
    assert all(not i["suggestion"]["follow_up"] for i in audits["A"]["images"].values())
    # Submitting again replays the same audits (idempotency keys).
    again = submit_study(study_dir, transport, participants=4, poll_seconds=0)
    assert {k: v["audit_id"] for k, v in again["audits"].items()} == {
        k: v["audit_id"] for k, v in audits.items()
    }
    for p in range(1, 5):
        text = (study_dir / "plans" / f"P{p:02d}.json").read_text()
        plan = json.loads(text)
        arm = ARMS[p - 1]
        assert [(b["set"], b["condition"]) for b in plan["blocks"]] == list(arm)
        for block in plan["blocks"]:
            assert sorted(block["image_ids"]) == sorted(audits[block["set"]]["images"])
        # Plans carry no ground truth.
        for leak in (
            "audit_positive",
            "coverage",
            "total_coverage",
            "source_image_id",
            "materials_present",
            "study_id",
        ):
            assert f'"{leak}"' not in text
        assert not re.search(r"\bval/", text)


def fake_session(
    study_dir: Path,
    plan_file: Path,
    unaided_s: float,
    assisted_s: float,
    answer: str,
    completed: bool = True,
) -> dict[str, Any]:
    """Answer every trial: by ground truth ("truth") or by accepting the suggestion."""
    plan = json.loads(plan_file.read_text())
    audits = json.loads((study_dir / "audits.json").read_text())["audits"]
    truth = {i["study_id"]: i for i in load_study(study_dir)["images"]}
    trials = []
    for b, block in enumerate(plan["blocks"]):
        for practice, audit_key, ids in (
            (True, "practice", block["practice_image_ids"]),
            (False, block["set"], block["image_ids"]),
        ):
            for pos, image_id in enumerate(ids):
                info = audits[audit_key]["images"][image_id]
                gt = truth[info["study_id"]]
                assisted = block["condition"] == "assisted"
                suggestion = info["suggestion"] if assisted else None
                follow, materials = (
                    (suggestion["follow_up"], suggestion["materials"])
                    if assisted and answer == "suggestion"
                    else (gt["audit_positive"], gt["materials_present"])
                )
                trials.append(
                    {
                        "block": b,
                        "condition": block["condition"],
                        "set": audit_key,
                        "practice": practice,
                        "audit_id": audits[audit_key]["audit_id"],
                        "image_id": image_id,
                        "position": pos,
                        "seconds": (assisted_s if assisted else unaided_s) + 0.01 * pos,
                        "follow_up": follow,
                        "materials": materials,
                        "suggestion": suggestion,
                    }
                )
    return {
        "results_version": 1,
        "study": plan["study"],
        "participant": plan["participant"],
        "arm": plan["arm"],
        "completed": completed,
        "trials": trials,
    }


def test_analysis_follows_the_declared_rules(
    study_dir: Path, client: TestClient, worker: Worker, tmp_path: Path
) -> None:
    submit_study(study_dir, client_transport(client, worker), participants=6, poll_seconds=0)
    files = []
    for p in range(1, 7):
        session = fake_session(
            study_dir,
            study_dir / "plans" / f"P{p:02d}.json",
            unaided_s=10.0,
            assisted_s=6.0,
            answer="suggestion",
            completed=p != 6,
        )
        path = tmp_path / f"results-P{p:02d}.json"
        path.write_text(json.dumps(session))
        files.append(path)
    report = analyze(study_dir, files)
    assert report["n_sessions"] == 5
    assert report["excluded_sessions"] == [
        {"file": "results-P06.json", "reason": "session not completed"}
    ]
    primary = report["primary"]
    assert primary["median_time_reduction"] == pytest.approx(0.4, abs=0.01)
    assert primary["time_reduction_met"] and not primary["too_few_groups"]
    # Accepting an all-background model's "no follow-up" loses every positive.
    unaided, assisted = report["by_condition"]["unaided"], report["by_condition"]["assisted"]
    assert unaided["recall"] == 1.0 and assisted["recall"] == 0.0
    assert primary["recall_difference"] == pytest.approx(-1.0) and not primary["recall_met"]
    assert assisted["followed_wrong_suggestion_share"] == 1.0
    assert assisted["wrong_suggestions"] == 5 * 14
    assert assisted["changed_suggestion_share"] == 0.0
    assert report["prevalence"] == {"A": 0.7, "B": 0.7}
    assert report["model_suggestion_reference"]["A"]["recall"] == 0.0


def test_analysis_rejects_inconsistent_sessions(
    study_dir: Path, client: TestClient, worker: Worker, tmp_path: Path
) -> None:
    submit_study(study_dir, client_transport(client, worker), participants=1, poll_seconds=0)
    session = fake_session(study_dir, study_dir / "plans" / "P01.json", 10.0, 6.0, "truth")
    main_trials = [t for t in session["trials"] if not t["practice"]]
    session["trials"].append(main_trials[0])  # the same image answered twice
    bad = tmp_path / "results-P01.json"
    bad.write_text(json.dumps(session))
    with pytest.raises(StudyError, match="no complete sessions"):
        analyze(study_dir, [bad])
    session["trials"] = session["trials"][:-2]  # duplicate removed, one image now unanswered
    bad.write_text(json.dumps(session))
    with pytest.raises(StudyError, match="no complete sessions"):
        analyze(study_dir, [bad])
    good = fake_session(study_dir, study_dir / "plans" / "P01.json", 10.0, 6.0, "truth")
    bad.write_text(json.dumps(good))
    report = analyze(study_dir, [bad])
    assert report["primary"]["recall_difference"] == 0.0
    assert report["primary"]["too_few_groups"]
