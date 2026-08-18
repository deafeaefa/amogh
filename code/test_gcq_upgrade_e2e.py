"""CPU-only schema smoke for proxy -> shortlist -> beam -> controls."""
import json

from allocate_gcq_beam import BeamRun, load_candidates
from build_gcq_controls import POLICIES, build_controls_manifest
from gcq_profile_metrics import aggregate_coordinate_candidate, build_shortlist
from score_gcq_plan import score_plan


CONTEXT = "ca" * 32
MANIFEST = "da" * 32


def _proxy_rows(repair):
    rows = []
    for task in ("rec", "coco_grounding"):
        for q in range(1, 5):
            rows.append({
                "uid": f"{task}:q{q}",
                "task": task,
                "area_quartile": q,
                "w4_coordinate_token_kl": [[1.0], [1.0], [1.0], [1.0]],
                "w8_coordinate_token_kl": [
                    [1.0 - repair], [1.0 - repair], [1.0 - repair], [1.0 - repair]
                ],
            })
    return rows


def _decoded(state_id, score):
    rows = []
    for task in ("rec", "coco_grounding"):
        for q in range(1, 5):
            rows.append({
                "uid": f"{task}:q{q}",
                "image_id": len(rows),
                "task": task,
                "area_quartile": q,
                "giou": score,
                "precise_iou": score,
                "box1000": [1, 1, 2, 2],
                "manifest_sha256": MANIFEST,
                "allocation_state_id": state_id,
            })
    return rows


def test_upgrade_artifacts_interoperate_end_to_end(tmp_path):
    repairs = {"a": 0.04, "b": 0.03, "c": 0.02, "d": 0.01}
    summaries = []
    for name, repair in repairs.items():
        summary = aggregate_coordinate_candidate(_proxy_rows(repair))
        summaries.append({
            "module_name": name,
            "delta_bytes": 1,
            "repair_macro": summary["repair_macro"],
        })
    shortlist = build_shortlist(summaries, top_k=4)
    shortlist_path = tmp_path / "shortlist.json"
    shortlist_path.write_text(json.dumps(shortlist))
    candidates, _ = load_candidates(shortlist_path)
    run = BeamRun.initialize(
        tmp_path / "beam", candidates, 2, beam_width=4, context_sha256=CONTEXT
    )

    objective = {
        frozenset(): 0.20,
        frozenset({"a"}): 0.25,
        frozenset({"b"}): 0.24,
        frozenset({"c"}): 0.23,
        frozenset({"d"}): 0.22,
        frozenset({"a", "b"}): 0.26,
        frozenset({"a", "c"}): 0.255,
        frozenset({"a", "d"}): 0.251,
        frozenset({"b", "c"}): 0.50,
        frozenset({"b", "d"}): 0.245,
        frozenset({"c", "d"}): 0.235,
    }
    while not run.is_complete:
        plan = run.plan()
        if plan is None:
            assert run.is_complete
            break
        result_rows = {
            row["state_id"]: _decoded(
                row["state_id"], objective[frozenset(row["members"])]
            )
            for row in plan["states"]
        }
        scored = score_plan(
            plan,
            result_rows,
            manifest_sha256=MANIFEST,
            context_sha256=CONTEXT,
        )
        run.record(scored["scores"])

    final = run.status_summary()["final"]
    assert final["members"] == ["b", "c"]
    assert final["cost_bytes"] == 2
    score_maps = {
        policy: {candidate.name: repairs[candidate.name] for candidate in candidates}
        for policy in POLICIES
    }
    controls = build_controls_manifest(
        candidates,
        final["cost_bytes"],
        score_maps,
        [1, 2, 3],
        CONTEXT,
    )
    assert controls["validation"]["all_score_driven_costs_equal_target"]
    assert controls["validation"]["all_random_costs_equal_target"]
