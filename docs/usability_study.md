# Usability study protocol

This protocol was written, and its analysis declared, **before any session
was run**. It tests the product targets in [design.md §27](design.md#27-success-metrics):
whether BeltWatch-assisted review is faster than unaided review without
losing detection recall.

With peer participants this is a **usability study**, not a facility pilot.
It says nothing about savings in an operating facility.

## Question

On the same kind of images, does a reviewer using BeltWatch (overlay,
estimated visible coverage, suggested answers, priority order) decide faster
than a reviewer looking at the photo alone, and do they miss more
follow-up-worthy images?

## Design

- **Within-participant, two conditions.** Each participant reviews one image
  set unaided and the other assisted.
- **Two matched sets, A and B**, of 20 val images each. Every participant sees
  both sets, one per condition, so neither set appears twice for anyone.
- **Counterbalancing.** Participants cycle through four arms: (A unaided →
  B assisted), (B unaided → A assisted), (A assisted → B unaided),
  (B assisted → A unaided). This balances set difficulty and order effects.
- **Practice.** Two practice images (one positive, one negative) open each
  block in that block's condition. Practice trials are never analysed.

### Conditions

| | Unaided | Assisted |
|---|---|---|
| Image | Original photo | Overlay, with original and uncertainty views |
| Order | Random (seeded per participant) | BeltWatch review-queue priority |
| Answers | Start blank | Pre-filled with BeltWatch's suggestion, which the participant confirms or changes |
| Extra information | None | Estimated visible coverage per material, review reasons |

The assisted suggestion follows the project conventions: **follow-up** when
estimated visible target coverage exceeds 5% of the inspection region, and a
material is suggested as present when its estimated coverage is at least 1%.
The page states that a "no follow-up" suggestion does not certify the image
as clean.

### Task (both conditions)

For each image the participant answers:

1. **Does this image need follow-up?** Yes means unwanted material (cardboard,
   soft plastic, rigid plastic, metal) appears to cover more than about 5% of
   the image.
2. **Which materials are present?** (any of the four).

The inspection region is the full frame, so participants do not have to
judge region boundaries.

## Images and ground truth

- **Pool:** the `val` split only. Never `test` (locked) or `calibration`.
  The model was selected on val, so assisted performance may be slightly
  optimistic; this is a stated limitation.
- **Ground truth** comes from the ZeroWaste-f masks, through the per-class
  pixel counts in the committed image manifest:
  - an image is **audit-positive** when ground-truth target coverage is
    above 5% (the project convention, not an industry standard);
  - a material is **present** when its ground-truth coverage is at least 1%.
- **Stratification.** Val is 89% audit-positive and has only 39 negatives.
  Each set gets 14 positives and 6 negatives (70% prevalence), which keeps
  enough negatives to measure false alarms. The positives are spread over
  the coverage tertiles of the positive val images. The study prevalence is
  reported with every result.
- **No near-repeats.** Within a sequence, any two selected frames (sets and
  practice together) are at least 100 frames apart, so a participant cannot
  recognise a scene from the other set.
- Selection is seeded (`--seed`, default 0) and recorded in `study.json`.
  Each copied frame is checked against the SHA-256 in the image manifest.
  The page never receives ground truth.

## Procedure

1. The facilitator prepares and submits the study once (commands below). The
   images go through the **real audit path**; the assisted view shows the
   live results of the active release, not precomputed output.
2. Per participant: read the consent script, open `/study.html`, load that
   participant's plan file, and let them work. The facilitator does not help
   during blocks.
3. At the end, the page downloads `results-<participant>.json`. The page also
   keeps a copy in the browser until it is downloaded.

### Consent script (read aloud)

> This is a usability study of a software prototype, not a test of you. You
> will look at about 44 photos of a recycling conveyor and answer two
> questions for each. It takes about 20 minutes. I record only an anonymous
> participant code, your answers, and how long each answer took. No name, no
> video, no audio. You can stop at any time, and your answers will then be
> discarded. Do you agree to take part?

Only an anonymous code (P01, P02, …) is stored. A participant who stops, or
who withdraws later, is excluded and their file deleted.

## Pilot session

Declared on 2026-10-17, before the pilot was run.

The developer runs one session on `plans/PILOT.json` (arm 0: set A unaided,
then set B assisted) to check the protocol, not BeltWatch:

- how long a session takes (target about 20 minutes);
- whether the instructions and the "about 5%" question are clear;
- whether anything on the page gets in the way.

The developer built the model and has seen its error analysis, so their
answers and times would measure that knowledge. The pilot therefore:

- is **never analysed** as a study session: `analyze` excludes any `PILOT`
  results file;
- is summarised for timing only (`analyze --pilot`), with no recall or
  false-alarm figures;
- may change the session length or instructions before the first
  participant. Any change is recorded here with the date, and the analysis
  above stays as declared;
- rules the developer out as a study participant.

## Declared analysis

Run by `python -m beltwatch.study.analyze`. Fixed now and not changed after
seeing results.

- **Unit:** completed sessions only. Incomplete sessions are excluded and
  counted.
- **No trial exclusions.** Long or short decision times are kept; the
  analysis uses medians, which are robust to them.
- **Decision time** runs from the image finishing loading to the answer
  being submitted.

| Measure | Definition |
|---|---|
| Median decision time | Per participant and condition, the median over the 20 analysed trials |
| **Time reduction** (primary) | Per participant, 1 − assisted median ÷ unaided median; reported as the median over participants |
| **Recall** (primary) | Audit-positive images answered "follow-up", pooled over participants, per condition |
| Recall difference | Assisted − unaided, in percentage points |
| False-alarm rate | Audit-negative images answered "follow-up", per condition |
| Material recall and precision | Pooled over images and participants, per material, per condition |
| Correction effort | Assisted only: share of trials where the participant changed the suggestion, and the number of changed answers per trial |
| Followed a wrong suggestion | Assisted only: on images where BeltWatch's follow-up suggestion is wrong, the share where the participant kept it (automation bias) |

**Intervals:** a percentile bootstrap over participants (2,000 resamples,
seed 0), resampling whole participants, so each person's paired conditions
stay together. Below 5 participants, intervals are flagged `too_few_groups`,
as everywhere else in the project.

**Targets (proposed, design.md §27):**

- median time reduction of **at least 30%**;
- recall difference of **−5 pp or better** (assisted at most 5 points below
  unaided).

A missed target is reported as missed. No other metric replaces it.

## Limitations to state with any result

- Peer participants, not trained facility operators; a short session.
- Val images, which also selected the model; one facility and one camera.
- Stratified prevalence (70%), not the 89% of val or any facility's rate.
- The 5% threshold is judged by eye in both conditions.
- Pre-filled answers can induce automation bias; the "followed a wrong
  suggestion" measure reports it rather than hiding it.

## Commands

```bash
# 1. Select the two sets and practice images from val, and copy the frames.
uv run python -m beltwatch.study.prepare --out data/study/study-1

# 2. Start the service with a long retention (sessions can span days), then
#    submit the images as audits and write one plan per participant.
BELTWATCH_RETENTION_HOURS=720 uv run uvicorn beltwatch.api.app:create_app --factory --port 8000
BELTWATCH_RETENTION_HOURS=720 uv run python -m beltwatch.jobs.worker
uv run python -m beltwatch.study.submit data/study/study-1 --base-url http://localhost:8000 --participants 8

# 3. Each session: open http://localhost:8000/study.html and load plans/P0n.json.

#    The developer's protocol check uses plans/PILOT.json; summarise it with
uv run python -m beltwatch.study.analyze data/study/study-1 --results results-PILOT.json --pilot

# 4. Analyse the downloaded results.
uv run python -m beltwatch.study.analyze data/study/study-1 --results results-P*.json \
    --out reports/study-1
```

`data/study/` holds dataset frames and is never committed. On a machine
without the extracted archive, fetch only the selected val frames (their
paths are in the error that `prepare` prints) and pass `--raw-dir`.

## Status

The kit was dry-run on 2026-10-17 against the `beltwatch-0.1.0` release:
selection from the committed manifest, the 42 frames through the real audit
path, two complete scripted sessions in headless Chromium (no console
errors), and the analysis. That checks the mechanics only. **No participant
session has been run, so there is no result yet.** The developer pilot
(above) is pending.
