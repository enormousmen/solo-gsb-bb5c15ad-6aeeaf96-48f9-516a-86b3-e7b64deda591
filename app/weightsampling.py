"""Batch weight double-sampling acceptance audit over parsed GS1 scans.

The request carries one sampling *plan* per (GTIN, batch): an inclusive
weight interval ``min_weight`` / ``max_weight`` (expressed in
``target_unit``) plus the classic double-sampling parameters
``n1, a1, r1, n2, a2, r2``.  Every *sample* names its ``stage`` (1 or 2)
and a GS1 scan fed through :func:`app.gs1.parse`; the scan must then
contain:

* exactly one AI 01 (GTIN),
* exactly one AI 10 (batch/lot),
* exactly one AI 21 (serial number),
* exactly one weight element of the 310n (kg) / 320n (lb) family.

AI 17 (expiry date) is optional (at most one) and ignored.  A sample
joins the unique plan matching its GTIN + AI 10 batch; a sample without
a matching plan fails the whole request with
:class:`SampleValidationError` (``NO_MATCHING_PLAN``) naming the
``sample_index``.  Physical items are identified by (GTIN, serial); the
same pair may appear only once across *all* samples
(``DUPLICATE_ITEM``, carrying both the current and the first index).

Plans are validated before any sample is parsed:

* ``n1 >= 1`` and ``n2 > n1``,
* ``0 <= a1 < r1 <= n1`` and ``0 <= a2 < r2 <= n2``,
* ``min_weight <= max_weight`` (the interval is closed),
* the (GTIN, batch) key is unique.

Any violation fails with :class:`PlanValidationError` naming the
``plan_index``.

A sample whose converted weight lies outside the closed interval is a
defect.  Decisions follow the double-sampling state machine in request
order:

* Stage 1 accepts once ``n1`` stage-1 samples are collected and the
  defect count is ``<= a1``; it rejects when the count is ``>= r1``.
* Otherwise the plan continues into stage 2; stage-2 samples are only
  accepted while the plan is in that continuation, and the cumulative
  sample count may never exceed ``n2``.
* Once the cumulative count reaches ``n2`` the total defect count
  decides: ``<= a2`` accepts, ``>= r2`` rejects, anything in between
  stays pending.
* Before ``n1``/``n2`` is reached the plan is pending.

A stage-1 decision is final -- any further sample for a plan already
accepted or rejected at stage 1, any stage-2 sample arriving before
stage 1 is complete, any stage-1 sample arriving after stage 1 is
complete, and any sample past the cumulative ``n2`` limit fails the
whole request as ``STAGE_SAMPLE_OVERFLOW`` naming the ``sample_index``
(and ``plan_index``).

All weights are rebuilt with :class:`decimal.Decimal` from the original
six digits, converted with the fixed factors of :mod:`app.summary` and
compared against the bounds in :class:`Decimal`; rounding to six places
happens only when the response is rendered.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

from .gs1 import GS1ParseError, Element, decode_weight, parse
from .summary import ARITHMETIC_PRECISION, ScanValidationError, _to_target

REQUIRED_AI_TEXT = (
    "exactly one AI 01 (GTIN), exactly one AI 10 (batch/lot), "
    "exactly one AI 21 (serial number), and exactly one weight AI 310n/320n"
)


class WeightSamplingError(Exception):
    """Base class for double-sampling audit failures carrying a 422 detail."""

    def __init__(self, detail: dict):
        super().__init__(detail["error"]["message"])
        self.detail = detail


class PlanValidationError(WeightSamplingError):
    """A sampling plan is invalid (thresholds, interval or duplicate key).

    ``detail`` is the complete 422 payload: ``plan_index`` (plus
    ``first_index`` for a duplicate plan) and the error under ``error``.
    """


class SampleValidationError(WeightSamplingError):
    """A sample failed parsing, matching, dedup or the stage quota.

    ``detail`` is the complete 422 payload: ``sample_index`` (plus
    ``first_index`` for a duplicate item or ``plan_index`` for a stage
    overflow) and the error under ``error``.
    """


def _plan_error(index: int, code: str, message: str, expected, found) -> None:
    raise PlanValidationError(
        {
            "plan_index": index,
            "error": {
                "error": code,
                "message": message,
                "expected": expected,
                "found": found,
            },
        }
    )


def _validate_plans(plans: list[tuple]) -> list[dict]:
    """Validate thresholds and intervals; index plans by (GTIN, batch)."""
    normalized = []
    index_of: dict[tuple[str, str], int] = {}
    for plan_index, (
        gtin,
        batch,
        min_weight,
        max_weight,
        n1,
        a1,
        r1,
        n2,
        a2,
        r2,
    ) in enumerate(plans):
        first_index = index_of.get((gtin, batch))
        if first_index is not None:
            raise PlanValidationError(
                {
                    "plan_index": plan_index,
                    "first_index": first_index,
                    "error": {
                        "error": "DUPLICATE_PLAN",
                        "message": (
                            f"plan {plan_index}: GTIN '{gtin}' batch "
                            f"'{batch}' is already defined by plan "
                            f"{first_index}; each GTIN+batch may have only "
                            "one plan"
                        ),
                        "expected": "a unique GTIN+batch per plan",
                        "found": f"GTIN '{gtin}' batch '{batch}'",
                    },
                }
            )

        if n1 < 1:
            _plan_error(
                plan_index,
                "INVALID_THRESHOLD",
                f"plan {plan_index}: n1 must be a positive sample size, got {n1}",
                expected="n1 >= 1",
                found=str(n1),
            )
        if n2 <= n1:
            _plan_error(
                plan_index,
                "INVALID_THRESHOLD",
                f"plan {plan_index}: n2 ({n2}) must be greater than n1 ({n1})",
                expected=f"n2 > n1 ({n1})",
                found=f"n2 = {n2}",
            )

        for stage, n, a, r in ((1, n1, a1, r1), (2, n2, a2, r2)):
            if a < 0:
                _plan_error(
                    plan_index,
                    "INVALID_THRESHOLD",
                    (
                        f"plan {plan_index}: a{stage} must be non-negative, "
                        f"got {a}"
                    ),
                    expected=f"0 <= a{stage} < r{stage} <= n{stage}",
                    found=f"a{stage} = {a}",
                )
            if r <= a:
                _plan_error(
                    plan_index,
                    "INVALID_THRESHOLD",
                    (
                        f"plan {plan_index}: r{stage} ({r}) must be greater "
                        f"than a{stage} ({a})"
                    ),
                    expected=f"a{stage} < r{stage}",
                    found=f"a{stage} = {a}, r{stage} = {r}",
                )
            if r > n:
                _plan_error(
                    plan_index,
                    "INVALID_THRESHOLD",
                    (
                        f"plan {plan_index}: r{stage} ({r}) must not exceed "
                        f"the stage sample size n{stage} ({n})"
                    ),
                    expected=f"r{stage} <= n{stage} ({n})",
                    found=f"r{stage} = {r}",
                )

        if min_weight > max_weight:
            raise PlanValidationError(
                {
                    "plan_index": plan_index,
                    "error": {
                        "error": "WEIGHT_RANGE_INVERTED",
                        "message": (
                            f"plan {plan_index}: min_weight {min_weight} is "
                            f"greater than max_weight {max_weight}; the "
                            "accepted interval is closed and needs "
                            "min_weight <= max_weight"
                        ),
                        "expected": "min_weight <= max_weight",
                        "found": f"{min_weight} > {max_weight}",
                    },
                }
            )

        index_of[(gtin, batch)] = plan_index
        normalized.append(
            {
                "plan_index": plan_index,
                "gtin": gtin,
                "batch": batch,
                "min_weight": min_weight,
                "max_weight": max_weight,
                "n1": n1,
                "a1": a1,
                "r1": r1,
                "n2": n2,
                "a2": a2,
                "r2": r2,
                # Runtime state (see _StageRuntime below / audit loop):
                # phase is one of "stage1", "stage2", "done2", "locked1".
                "phase": "stage1",
                "decision": None,
                "s1": 0,
                "s2": 0,
                "d1": 0,
                "d2": 0,
                "oor1": [],
                "oor2": [],
            }
        )
    return normalized


def _extract(
    index: int, elements: list[Element]
) -> tuple[str, str, str, Decimal, str]:
    """Pull (GTIN, batch, serial, exact weight, weight unit) from one scan."""
    gtins = [e for e in elements if e.ai == "01"]
    batches = [e for e in elements if e.ai == "10"]
    serials = [e for e in elements if e.ai == "21"]
    expiries = [e for e in elements if e.ai == "17"]
    weights = [e for e in elements if e.unit is not None]

    if len(gtins) != 1:
        if not gtins:
            raise ScanValidationError(
                "MISSING_AI",
                f"sample {index}: missing required AI 01 (GTIN)",
                expected=REQUIRED_AI_TEXT,
            )
        raise ScanValidationError(
            "DUPLICATE_AI",
            f"sample {index}: AI 01 (GTIN) appears {len(gtins)} times; exactly one is required",
            expected="exactly one AI 01 (GTIN)",
            found=f"{len(gtins)} AI 01 elements",
            offset=gtins[1].offset,
        )

    if len(batches) != 1:
        if not batches:
            raise ScanValidationError(
                "MISSING_AI",
                f"sample {index}: missing required AI 10 (batch/lot)",
                expected=REQUIRED_AI_TEXT,
            )
        raise ScanValidationError(
            "DUPLICATE_AI",
            f"sample {index}: AI 10 (batch/lot) appears {len(batches)} times; exactly one is required",
            expected="exactly one AI 10 (batch/lot)",
            found=f"{len(batches)} AI 10 elements",
            offset=batches[1].offset,
        )

    if len(serials) != 1:
        if not serials:
            raise ScanValidationError(
                "MISSING_AI",
                f"sample {index}: missing required AI 21 (serial number)",
                expected=REQUIRED_AI_TEXT,
            )
        raise ScanValidationError(
            "DUPLICATE_AI",
            f"sample {index}: AI 21 (serial number) appears {len(serials)} times; exactly one is required",
            expected="exactly one AI 21 (serial number)",
            found=f"{len(serials)} AI 21 elements",
            offset=serials[1].offset,
        )

    if len(expiries) > 1:
        raise ScanValidationError(
            "DUPLICATE_AI",
            f"sample {index}: AI 17 (expiry date) appears {len(expiries)} times; at most one is allowed",
            expected="at most one AI 17 (expiry date)",
            found=f"{len(expiries)} AI 17 elements",
            offset=expiries[1].offset,
        )

    if len(weights) != 1:
        if not weights:
            raise ScanValidationError(
                "MISSING_AI",
                f"sample {index}: missing required weight AI 310n/320n",
                expected=REQUIRED_AI_TEXT,
            )
        ais = ", ".join(e.ai for e in weights)
        raise ScanValidationError(
            "DUPLICATE_AI",
            f"sample {index}: {len(weights)} weight elements ({ais}); "
            "exactly one 310n/320n is required",
            expected="exactly one weight AI 310n (kg) or 320n (lb)",
            found=ais,
            offset=weights[1].offset,
        )

    weight = weights[0]
    value = decode_weight(weight.value, int(weight.ai[3]))
    return gtins[0].value, batches[0].value, serials[0].value, value, weight.unit


def _overflow(sample_index: int, plan: dict, message: str, expected, found) -> None:
    raise SampleValidationError(
        {
            "sample_index": sample_index,
            "plan_index": plan["plan_index"],
            "error": {
                "error": "STAGE_SAMPLE_OVERFLOW",
                "message": message,
                "expected": expected,
                "found": found,
            },
        }
    )


def _check_stage(sample_index: int, stage: int, plan: dict) -> None:
    """Reject a sample that violates the plan's double-sampling sequence."""
    phase = plan["phase"]
    p = plan["plan_index"]

    if phase == "locked1":
        _overflow(
            sample_index,
            plan,
            (
                f"sample {sample_index}: plan {p} already reached a stage-1 "
                f"decision ('{plan['decision']}'); that decision is final "
                "and no further samples are accepted for this plan"
            ),
            expected="no further samples after a stage-1 accept/reject",
            found=f"stage {stage} sample",
        )

    if phase == "done2":
        _overflow(
            sample_index,
            plan,
            (
                f"sample {sample_index}: plan {p} already holds the maximum "
                f"cumulative sample size n2 = {plan['n2']}; no further "
                "samples are accepted for this plan"
            ),
            expected=f"at most {plan['n2']} cumulative samples",
            found=f"stage {stage} sample after n2 was reached",
        )

    if stage == 2 and phase == "stage1":
        remaining = plan["n1"] - plan["s1"]
        _overflow(
            sample_index,
            plan,
            (
                f"sample {sample_index}: plan {p} has collected "
                f"{plan['s1']} of n1 = {plan['n1']} stage-1 samples; a "
                "stage-2 sample is accepted only after stage 1 is full "
                "and inconclusive"
            ),
            expected=f"{remaining} more stage-1 sample(s) to reach n1 = {plan['n1']}",
            found="stage 2 sample",
        )

    if stage == 1 and phase == "stage2":
        _overflow(
            sample_index,
            plan,
            (
                f"sample {sample_index}: stage 1 of plan {p} is already "
                f"complete with n1 = {plan['n1']} samples and was "
                "inconclusive; only stage-2 samples may follow"
            ),
            expected=f"a stage-2 sample (up to {plan['n2'] - plan['n1']})",
            found="stage 1 sample",
        )

    if stage == 2 and phase == "stage2" and plan["s1"] + plan["s2"] >= plan["n2"]:
        _overflow(
            sample_index,
            plan,
            (
                f"sample {sample_index}: plan {p} already holds n2 = "
                f"{plan['n2']} cumulative samples; at most "
                f"{plan['n2'] - plan['n1']} stage-2 samples may follow "
                f"the {plan['n1']} stage-1 samples"
            ),
            expected=f"at most {plan['n2'] - plan['n1']} stage-2 samples",
            found=f"stage-2 sample beyond cumulative n2 = {plan['n2']}",
        )


def audit(
    target_unit: str,
    plans: list[tuple],
    samples: list[tuple[int, str]],
) -> dict:
    """Run the double-sampling acceptance audit.

    ``plans`` is a list of ``(gtin, batch, min_weight, max_weight, n1,
    a1, r1, n2, a2, r2)`` tuples in request order; ``samples`` is a list
    of ``(stage, scan)`` tuples in sampling order.  Returns
    ``{"target_unit": ..., "plans": [...]}`` with one result per plan in
    the original order; the echoed bounds are unrounded
    :class:`Decimal` values in the target unit.
    """
    runtime = _validate_plans(plans)
    plan_of: dict[tuple[str, str], dict] = {
        (plan["gtin"], plan["batch"]): plan for plan in runtime
    }
    first_seen: dict[tuple[str, str], int] = {}

    with localcontext() as ctx:
        ctx.prec = ARITHMETIC_PRECISION

        for sample_index, (stage, scan_text) in enumerate(samples):
            try:
                elements = parse(scan_text)
            except GS1ParseError as exc:
                raise SampleValidationError(
                    {"sample_index": sample_index, "error": exc.to_dict()}
                ) from exc
            try:
                gtin, batch, serial, weight, unit = _extract(
                    sample_index, elements
                )
            except ScanValidationError as exc:
                raise SampleValidationError(
                    {"sample_index": sample_index, "error": exc.to_dict()}
                ) from exc

            plan = plan_of.get((gtin, batch))
            if plan is None:
                raise SampleValidationError(
                    {
                        "sample_index": sample_index,
                        "error": {
                            "error": "NO_MATCHING_PLAN",
                            "message": (
                                f"sample {sample_index}: no sampling plan "
                                f"matches GTIN '{gtin}' batch '{batch}'"
                            ),
                            "expected": (
                                "the sample's GTIN (AI 01) and batch "
                                "(AI 10) to name a plan in this request"
                            ),
                            "found": f"GTIN '{gtin}' batch '{batch}'",
                        },
                    }
                )

            key = (gtin, serial)
            first_index = first_seen.get(key)
            if first_index is not None:
                raise SampleValidationError(
                    {
                        "sample_index": sample_index,
                        "first_index": first_index,
                        "error": {
                            "error": "DUPLICATE_ITEM",
                            "message": (
                                f"sample {sample_index}: item GTIN "
                                f"'{gtin}' serial '{key[1]}' was already "
                                f"sampled at index {first_index}; the same "
                                "GTIN+serial may be sampled only once"
                            ),
                            "expected": (
                                f"the item first seen at sample "
                                f"{first_index} to be the only occurrence"
                            ),
                            "found": (
                                f"GTIN '{gtin}' serial '{key[1]}' at "
                                f"sample {sample_index}"
                            ),
                        },
                    }
                )

            _check_stage(sample_index, stage, plan)
            first_seen[key] = sample_index

            converted = _to_target(weight, unit, target_unit)
            out_of_range = (
                converted < plan["min_weight"] or converted > plan["max_weight"]
            )

            if stage == 1:
                plan["s1"] += 1
                if out_of_range:
                    plan["d1"] += 1
                    plan["oor1"].append(sample_index)
                if plan["s1"] == plan["n1"]:
                    if plan["d1"] <= plan["a1"]:
                        plan["phase"] = "locked1"
                        plan["decision"] = "accept"
                    elif plan["d1"] >= plan["r1"]:
                        plan["phase"] = "locked1"
                        plan["decision"] = "reject"
                    else:
                        plan["phase"] = "stage2"
            else:
                plan["s2"] += 1
                if out_of_range:
                    plan["d2"] += 1
                    plan["oor2"].append(sample_index)
                if plan["s1"] + plan["s2"] == plan["n2"]:
                    total_defects = plan["d1"] + plan["d2"]
                    if total_defects <= plan["a2"]:
                        plan["decision"] = "accept"
                    elif total_defects >= plan["r2"]:
                        plan["decision"] = "reject"
                    else:
                        plan["decision"] = "pending"
                    plan["phase"] = "done2"

    results = []
    for plan in runtime:
        if plan["phase"] == "locked1" or plan["phase"] == "done2":
            decision = plan["decision"]
        else:
            # "stage1": fewer than n1 samples; "stage2": cumulative < n2.
            decision = "pending"
        results.append(
            {
                "plan_index": plan["plan_index"],
                "gtin": plan["gtin"],
                "batch": plan["batch"],
                "min_weight": plan["min_weight"],
                "max_weight": plan["max_weight"],
                "unit": target_unit,
                "n1": plan["n1"],
                "a1": plan["a1"],
                "r1": plan["r1"],
                "n2": plan["n2"],
                "a2": plan["a2"],
                "r2": plan["r2"],
                "stages": [
                    {
                        "stage": 1,
                        "sample_count": plan["s1"],
                        "defect_count": plan["d1"],
                        "out_of_range_indices": list(plan["oor1"]),
                    },
                    {
                        "stage": 2,
                        "sample_count": plan["s2"],
                        "defect_count": plan["d2"],
                        "out_of_range_indices": list(plan["oor2"]),
                    },
                ],
                "decision": decision,
            }
        )

    return {"target_unit": target_unit, "plans": results}
