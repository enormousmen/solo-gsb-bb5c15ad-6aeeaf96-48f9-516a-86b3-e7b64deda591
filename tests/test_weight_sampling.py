"""End-to-end tests for the /weight-sampling-audit endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

GS = "\x1d"
GTIN_RAW = "09501101530003"  # valid Mod-10 check digit
GTIN_PRODUCT = "00000000000017"  # second valid GTIN


def audit(plans, samples, target_unit="kg"):
    return client.post(
        "/weight-sampling-audit",
        json={"target_unit": target_unit, "plans": plans, "samples": samples},
    )


def plan(
    gtin=GTIN_RAW,
    batch="B1",
    min_weight=1.9,
    max_weight=2.1,
    n1=5,
    a1=0,
    r1=2,
    n2=10,
    a2=1,
    r2=2,
):
    return {
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
    }


def sample(serial, stage=1, weight="3103002000", batch="B1", gtin=GTIN_RAW):
    """Element-string sample scan with 01/10/21 and one weight AI (default 2 kg)."""
    return {"stage": stage, "scan": f"01{gtin}10{batch}{GS}21{serial}{GS}{weight}"}


def stage1_all_good(prefix="S", count=5, **kw):
    return [sample(f"{prefix}{i}", **kw) for i in range(count)]


# ---------------------------------------------------------------- happy path


def test_empty_audit_returns_empty_result():
    resp = audit([], [])
    assert resp.status_code == 200
    assert resp.json() == {"target_unit": "kg", "plans": []}


def test_plan_without_samples_is_pending_with_zeroed_stages():
    resp = audit([plan()], [])
    assert resp.status_code == 200
    entry = resp.json()["plans"][0]
    assert entry["plan_index"] == 0
    assert entry["gtin"] == GTIN_RAW
    assert entry["batch"] == "B1"
    assert entry["min_weight"] == 1.9
    assert entry["max_weight"] == 2.1
    assert entry["unit"] == "kg"
    assert (entry["n1"], entry["a1"], entry["r1"]) == (5, 0, 2)
    assert (entry["n2"], entry["a2"], entry["r2"]) == (10, 1, 2)
    assert entry["decision"] == "pending"
    assert entry["stages"] == [
        {"stage": 1, "sample_count": 0, "defect_count": 0, "out_of_range_indices": []},
        {"stage": 2, "sample_count": 0, "defect_count": 0, "out_of_range_indices": []},
    ]


def test_stage1_accepts_when_defects_at_or_below_a1():
    samples = stage1_all_good()
    resp = audit([plan(a1=0, r1=2)], samples)
    assert resp.status_code == 200
    entry = resp.json()["plans"][0]
    assert entry["decision"] == "accept"
    assert entry["stages"][0]["sample_count"] == 5
    assert entry["stages"][0]["defect_count"] == 0
    assert entry["stages"][0]["out_of_range_indices"] == []
    assert entry["stages"][1]["sample_count"] == 0


def test_stage1_rejects_when_defects_at_or_above_r1():
    # Two 3 kg samples are outside [1.9, 2.1] -> 2 defects = r1.
    samples = [
        sample("S0", weight="3103003000"),
        sample("S1", weight="3103003000"),
        sample("S2"),
        sample("S3"),
        sample("S4"),
    ]
    resp = audit([plan(a1=0, r1=2)], samples)
    assert resp.status_code == 200
    entry = resp.json()["plans"][0]
    assert entry["decision"] == "reject"
    assert entry["stages"][0]["defect_count"] == 2
    assert entry["stages"][0]["out_of_range_indices"] == [0, 1]


def test_decision_is_not_made_before_stage1_sample_size_is_reached():
    # Even two early defects must wait until the n1-th sample.
    samples = [
        sample("S0", weight="3103003000"),
        sample("S1", weight="3103003000"),
        sample("S2"),
    ]
    resp = audit([plan()], samples)
    assert resp.status_code == 200
    entry = resp.json()["plans"][0]
    assert entry["decision"] == "pending"
    assert entry["stages"][0]["sample_count"] == 3
    assert entry["stages"][0]["defect_count"] == 2
    assert entry["stages"][0]["out_of_range_indices"] == [0, 1]


def test_inconclusive_stage1_without_stage2_stays_pending():
    # a1=0 < 1 defect < r1=2.
    samples = [sample("S0"), sample("S1"), sample("S2"), sample("S3"), sample("S4", weight="3103003000")]
    resp = audit([plan()], samples)
    assert resp.status_code == 200
    assert resp.json()["plans"][0]["decision"] == "pending"


def test_stage2_accepts_on_cumulative_a2():
    samples = [sample(f"S{i}") for i in range(4)] + [sample("S4", weight="3103003000")]
    samples += [sample(f"S{i+5}", stage=2) for i in range(5)]
    resp = audit([plan(a1=0, r1=2, a2=1, r2=2)], samples)
    assert resp.status_code == 200
    entry = resp.json()["plans"][0]
    assert entry["decision"] == "accept"
    assert entry["stages"][0] == {
        "stage": 1,
        "sample_count": 5,
        "defect_count": 1,
        "out_of_range_indices": [4],
    }
    assert entry["stages"][1]["sample_count"] == 5
    assert entry["stages"][1]["defect_count"] == 0


def test_stage2_rejects_on_cumulative_r2():
    samples = [sample(f"S{i}") for i in range(4)] + [sample("S4", weight="3103003000")]
    samples += [sample(f"S{i+5}", stage=2) for i in range(4)]
    samples += [sample("S9", stage=2, weight="3103003000")]
    resp = audit([plan(a1=0, r1=2, a2=1, r2=2)], samples)
    assert resp.status_code == 200
    entry = resp.json()["plans"][0]
    assert entry["decision"] == "reject"
    assert entry["stages"][1]["defect_count"] == 1
    assert entry["stages"][1]["out_of_range_indices"] == [9]


def test_between_a2_and_r2_at_n2_stays_pending():
    # r2 = 3: cumulative 2 defects with a2 = 1 stays pending.
    samples = [sample(f"S{i}") for i in range(4)] + [sample("S4", weight="3103003000")]
    samples += [sample(f"S{i+5}", stage=2) for i in range(4)]
    samples += [sample("S9", stage=2, weight="3103003000")]
    resp = audit([plan(a1=0, r1=2, a2=1, r2=3)], samples)
    assert resp.status_code == 200
    assert resp.json()["plans"][0]["decision"] == "pending"


def test_incomplete_stage2_stays_pending():
    samples = [sample(f"S{i}") for i in range(4)] + [sample("S4", weight="3103003000")]
    samples += [sample("S5", stage=2, weight="3103003000")]
    resp = audit([plan(a1=0, r1=2, a2=1, r2=2)], samples)
    assert resp.status_code == 200
    entry = resp.json()["plans"][0]
    assert entry["decision"] == "pending"
    assert entry["stages"][1]["sample_count"] == 1
    assert entry["stages"][1]["out_of_range_indices"] == [5]


def test_interval_bounds_are_inclusive():
    # Exactly 1.9 and 2.1 kg must pass; 2 kg too.
    samples = [
        sample("S0", weight="3104019000"),  # 1.9 kg
        sample("S1", weight="3104021000"),  # 2.1 kg
        sample("S2"),
        sample("S3"),
        sample("S4"),
    ]
    resp = audit([plan(min_weight=1.9, max_weight=2.1, a1=0, r1=1)], samples)
    assert resp.status_code == 200
    entry = resp.json()["plans"][0]
    assert entry["stages"][0]["defect_count"] == 0
    assert entry["decision"] == "accept"


def test_lb_scans_convert_before_interval_comparison():
    # 25.13 lb = 11.3987... kg, far outside [1.9, 2.1] kg.
    samples = stage1_all_good(prefix="F", count=4)
    samples.append(sample("F4", weight="3202002513"))
    resp = audit([plan(a1=1, r1=2)], samples)
    assert resp.status_code == 200
    entry = resp.json()["plans"][0]
    assert entry["stages"][0]["defect_count"] == 1
    assert entry["decision"] == "accept"  # 1 <= a1


def test_target_unit_lb_compares_converted_kilogram_scans():
    # 2 kg ~= 4.409 lb lies inside [4, 5] lb.
    resp = audit(
        [plan(min_weight=4, max_weight=5, n1=2, a1=0, r1=1, n2=4, a2=1, r2=2)],
        stage1_all_good(count=2),
        target_unit="lb",
    )
    assert resp.status_code == 200
    entry = resp.json()["plans"][0]
    assert entry["unit"] == "lb"
    assert entry["min_weight"] == 4.0
    assert entry["max_weight"] == 5.0
    assert entry["decision"] == "accept"


def test_decimal_bound_strings_are_exact():
    resp = audit(
        [
            plan(
                min_weight="1.900000",
                max_weight="2.100000",
                a1=0,
                r1=1,
            )
        ],
        stage1_all_good(),
    )
    assert resp.status_code == 200
    entry = resp.json()["plans"][0]
    assert entry["min_weight"] == 1.9
    assert entry["max_weight"] == 2.1


def test_weights_render_with_exactly_six_decimal_places():
    resp = client.post(
        "/weight-sampling-audit",
        json={
            "target_unit": "kg",
            "plans": [
                {
                    "gtin": GTIN_RAW,
                    "batch": "B1",
                    "min_weight": "1.90",
                    "max_weight": "2.10",
                    "n1": 5,
                    "a1": 0,
                    "r1": 2,
                    "n2": 10,
                    "a2": 1,
                    "r2": 2,
                }
            ],
            "samples": [],
        },
    )
    assert resp.status_code == 200
    text = resp.text
    assert '"min_weight":1.900000' in text
    assert '"max_weight":2.100000' in text


def test_digital_link_scans_are_accepted():
    samples = stage1_all_good()
    samples[0] = {
        "stage": 1,
        "scan": (
            "https://id.gs1.org/01/09501101530003/10/B1/21/S0?3103=002000"
        ),
    }
    resp = audit([plan(a1=0, r1=1)], samples)
    assert resp.status_code == 200
    assert resp.json()["plans"][0]["decision"] == "accept"


def test_several_plans_run_independently_with_global_sample_indices():
    plans = [
        plan(gtin=GTIN_RAW, batch="A", n1=2, a1=0, r1=2, n2=4, a2=1, r2=2),
        plan(gtin=GTIN_PRODUCT, batch="B", min_weight=2.5, max_weight=3.5, n1=2, a1=0, r1=2, n2=4, a2=1, r2=2),
    ]
    samples = [
        sample("A0", batch="A"),
        sample("B0", weight="3103003000", batch="B", gtin=GTIN_PRODUCT),
        sample("A1", weight="3103003000", batch="A"),
        sample("B1", weight="3103002000", batch="B", gtin=GTIN_PRODUCT),
    ]
    resp = audit(plans, samples)
    assert resp.status_code == 200
    pa, pb = resp.json()["plans"]
    assert pa["stages"][0]["out_of_range_indices"] == [2]
    assert pb["stages"][0]["out_of_range_indices"] == [3]


def test_interleaved_two_stage_plans_decide_independently():
    plans = [
        plan(gtin=GTIN_RAW, batch="A", n1=2, a1=0, r1=2, n2=4, a2=1, r2=2),
        plan(gtin=GTIN_PRODUCT, batch="B", n1=2, a1=0, r1=2, n2=4, a2=1, r2=2),
    ]
    samples = [
        sample("J0", batch="A"),
        sample("K0", weight="3103003000", batch="B", gtin=GTIN_PRODUCT),
        sample("J1", weight="3103003000", batch="A"),
        sample("K1", batch="B", gtin=GTIN_PRODUCT),
        sample("J2", stage=2, batch="A"),
        sample("K2", stage=2, weight="3103003000", batch="B", gtin=GTIN_PRODUCT),
        sample("J3", stage=2, batch="A"),
        sample("K3", stage=2, batch="B", gtin=GTIN_PRODUCT),
    ]
    resp = audit(plans, samples)
    assert resp.status_code == 200
    pa, pb = resp.json()["plans"]
    assert pa["decision"] == "accept"  # cumulative 1 defect <= a2
    assert pb["decision"] == "reject"  # cumulative 2 defects >= r2


# ------------------------------------------------------------- error: plans


def test_n2_not_greater_than_n1_is_rejected_with_plan_index():
    resp = audit([plan(n1=5, n2=5)], [])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["plan_index"] == 0
    assert detail["error"]["error"] == "INVALID_THRESHOLD"


def test_n1_zero_is_rejected():
    resp = audit([plan(n1=0, n2=5)], [])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["plan_index"] == 0
    assert detail["error"]["error"] == "INVALID_THRESHOLD"


def test_a1_negative_is_rejected():
    resp = audit([plan(a1=-1)], [])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["plan_index"] == 0
    assert detail["error"]["error"] == "INVALID_THRESHOLD"


def test_r1_not_greater_than_a1_is_rejected():
    resp = audit([plan(a1=2, r1=2)], [])
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"]["error"] == "INVALID_THRESHOLD"


def test_r1_above_n1_is_rejected():
    resp = audit([plan(n1=5, a1=0, r1=6)], [])
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"]["error"] == "INVALID_THRESHOLD"


def test_r2_above_n2_is_rejected():
    resp = audit([plan(n2=10, a2=1, r2=11)], [])
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"]["error"] == "INVALID_THRESHOLD"


def test_inverted_weight_interval_is_rejected():
    resp = audit([plan(min_weight=2.1, max_weight=1.9)], [])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["plan_index"] == 0
    assert detail["error"]["error"] == "WEIGHT_RANGE_INVERTED"


def test_duplicate_plan_key_is_rejected_with_first_index():
    resp = audit([plan(batch="B1"), plan(batch="B1")], [])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["plan_index"] == 1
    assert detail["first_index"] == 0
    assert detail["error"]["error"] == "DUPLICATE_PLAN"


def test_same_gtin_with_different_batches_are_separate_plans():
    resp = audit(
        [plan(batch="B1"), plan(batch="B2", n1=2, a1=0, r1=1, n2=4, a2=1, r2=2)],
        [],
    )
    assert resp.status_code == 200


def test_plan_validation_runs_before_sample_parsing():
    bad_scan = "not-a-scan"
    resp = audit([plan(n1=5, n2=5)], [{"stage": 1, "scan": bad_scan}])
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"]["error"] == "INVALID_THRESHOLD"


# ------------------------------------------------------------ error: samples


def test_sample_without_matching_plan_is_rejected():
    resp = audit([plan(batch="B1")], [sample("S0", batch="ZZ")])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["sample_index"] == 0
    assert detail["error"]["error"] == "NO_MATCHING_PLAN"


def test_duplicate_gtin_serial_is_rejected_with_first_index():
    resp = audit([plan()], [sample("DUP", weight="3103002000"), sample("DUP")])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["sample_index"] == 1
    assert detail["first_index"] == 0
    assert detail["error"]["error"] == "DUPLICATE_ITEM"


def test_duplicate_serial_under_different_batches_is_still_rejected():
    # Identity is GTIN + serial, independent of the batch.
    plans = [
        plan(batch="A"),
        plan(batch="B"),
    ]
    resp = audit(plans, [sample("X1", batch="A"), sample("X1", batch="B")])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["sample_index"] == 1
    assert detail["first_index"] == 0
    assert detail["error"]["error"] == "DUPLICATE_ITEM"


def test_extra_stage1_sample_after_n1_is_overflow():
    resp = audit([plan()], stage1_all_good(count=6))
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["sample_index"] == 5
    assert detail["plan_index"] == 0
    assert detail["error"]["error"] == "STAGE_SAMPLE_OVERFLOW"


def test_sample_after_stage1_decision_is_overflow():
    # Two defects at the first two positions only reject once n1 is full;
    # the sixth sample then overruns the locked decision.
    samples = [
        sample("S0", weight="3103003000"),
        sample("S1", weight="3103003000"),
        sample("S2"),
        sample("S3"),
        sample("S4"),
        sample("S5"),
    ]
    resp = audit([plan()], samples)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["sample_index"] == 5
    assert detail["plan_index"] == 0
    assert detail["error"]["error"] == "STAGE_SAMPLE_OVERFLOW"


def test_stage2_before_stage1_complete_is_overflow():
    resp = audit([plan()], [sample("S0"), sample("S1", stage=2)])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["sample_index"] == 1
    assert detail["error"]["error"] == "STAGE_SAMPLE_OVERFLOW"


def test_stage1_sample_after_inconclusive_stage1_is_overflow():
    samples = [sample(f"S{i}") for i in range(4)] + [sample("S4", weight="3103003000")]
    samples.append(sample("S5"))
    resp = audit([plan()], samples)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["sample_index"] == 5
    assert detail["error"]["error"] == "STAGE_SAMPLE_OVERFLOW"


def test_more_than_n2_minus_n1_stage2_samples_is_overflow():
    samples = [sample(f"S{i}") for i in range(4)] + [sample("S4", weight="3103003000")]
    samples += [sample(f"S{i+5}", stage=2) for i in range(6)]
    resp = audit([plan(n1=5, n2=10)], samples)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["sample_index"] == 10
    assert detail["error"]["error"] == "STAGE_SAMPLE_OVERFLOW"


def test_sample_after_completed_n2_is_overflow():
    # Stage 2 rejects at n2 (two cumulative defects); an extra sample fails.
    samples = [sample(f"S{i}") for i in range(4)] + [sample("S4", weight="3103003000")]
    samples += [sample(f"S{i+5}", stage=2) for i in range(4)]
    samples += [sample("S9", stage=2, weight="3103003000"), sample("S10", stage=2)]
    resp = audit([plan()], samples)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["sample_index"] == 10
    assert detail["error"]["error"] == "STAGE_SAMPLE_OVERFLOW"


def test_invalid_stage_value_is_pydantic_422():
    resp = audit([plan()], [{"stage": 3, "scan": f"01{GTIN_RAW}10B1{GS}21S0{GS}3103002000"}])
    assert resp.status_code == 422


def test_missing_ai21_is_rejected_with_sample_index():
    resp = audit([plan()], [{"stage": 1, "scan": f"01{GTIN_RAW}10B1{GS}3103002000"}])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["sample_index"] == 0
    assert detail["error"]["error"] == "MISSING_AI"


def test_missing_ai10_is_rejected():
    resp = audit([plan()], [{"stage": 1, "scan": f"01{GTIN_RAW}21S0{GS}3103002000"}])
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"]["error"] == "MISSING_AI"


def test_missing_weight_ai_is_rejected():
    resp = audit([plan()], [{"stage": 1, "scan": f"01{GTIN_RAW}10B1{GS}21S0"}])
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"]["error"] == "MISSING_AI"


def test_duplicate_weight_ai_is_rejected():
    scan = f"01{GTIN_RAW}10B1{GS}21S0{GS}31030020003202002513"
    resp = audit([plan()], [{"stage": 1, "scan": scan}])
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"]["error"] == "DUPLICATE_AI"


def test_parser_error_is_wrapped_with_sample_index():
    resp = audit([plan()], [{"stage": 1, "scan": "0109501101530004"}])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["sample_index"] == 0
    assert detail["error"]["error"] == "INVALID_CHECK_DIGIT"


def test_ai17_is_allowed_but_ignored():
    scan = f"01{GTIN_RAW}1725013110B1{GS}21S0{GS}3103002000"
    resp = audit([plan(a1=0, r1=1)], [{"stage": 1, "scan": scan}])
    # n1 = 5 but only one sample; still parses and stays pending.
    assert resp.status_code == 200
    assert resp.json()["plans"][0]["decision"] == "pending"


def test_two_ai17_elements_are_rejected():
    scan = f"01{GTIN_RAW}1725013110B1{GS}1725022821S0{GS}3103002000"
    resp = audit([plan()], [{"stage": 1, "scan": scan}])
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"]["error"] == "DUPLICATE_AI"


def test_default_target_unit_is_kg():
    resp = client.post(
        "/weight-sampling-audit",
        json={
            "plans": [
                {
                    "gtin": GTIN_RAW,
                    "batch": "B1",
                    "min_weight": 1.9,
                    "max_weight": 2.1,
                    "n1": 1,
                    "a1": 0,
                    "r1": 1,
                    "n2": 2,
                    "a2": 0,
                    "r2": 1,
                }
            ],
            "samples": [{"stage": 1, "scan": f"01{GTIN_RAW}10B1{GS}21S0{GS}3103002000"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["target_unit"] == "kg"
    assert body["plans"][0]["decision"] == "accept"


def test_single_sample_plan_with_a1_zero_r1_one_decides_at_one():
    # n1 = 1, a1 = 0, r1 = 1: a good sample accepts, a bad one rejects.
    good = audit(
        [plan(n1=1, a1=0, r1=1, n2=2, a2=0, r2=1)],
        [sample("S0")],
    )
    assert good.status_code == 200
    assert good.json()["plans"][0]["decision"] == "accept"

    bad = audit(
        [plan(n1=1, a1=0, r1=1, n2=2, a2=0, r2=1)],
        [sample("S0", weight="3103003000")],
    )
    assert bad.status_code == 200
    assert bad.json()["plans"][0]["decision"] == "reject"
