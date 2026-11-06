"""Combine the memdiag.py outputs into MEMDIAG.json (DIRECTIVE-VLM-VAD-PACKFIX-001-002). Diagnosis only.

Usage: python memdiag_summary.py <output_root> <MEMDIAG.json>
Exit 1 when workerdict_init.json or an agentloop_<pid>.json is missing.
"""

import glob
import json
import os
import sys


def main(out, dest):
    wd_path = os.path.join(out, "workerdict_init.json")
    al_paths = sorted(glob.glob(os.path.join(out, "agentloop_*.json")))
    if not os.path.exists(wd_path) or not al_paths:
        print(f"missing measurement files: workerdict {os.path.exists(wd_path)} agentloop {len(al_paths)}")
        return 1
    wd = json.load(open(wd_path))
    workers = [json.load(open(p)) for p in al_paths]
    keys = sorted({k for w in workers for k in w["output_totals_gib"]})
    totals = {k: sum(w["output_totals_gib"].get(k, 0.0) for w in workers) for k in keys}
    summary = {
        "workerdict": wd,
        "agent_loop_workers": len(workers),
        "agent_loop_rows": sum(w["rows"] for w in workers),
        "agent_loop_output_totals_gib_all_workers": totals,
        "pixel_values_gib_unshared_all_workers": sum(w["pixel_values_gib_unshared"] for w in workers),
        "pixel_values_gib_if_shared_all_workers": sum(w["pixel_values_gib_if_shared_per_group"] for w in workers),
        "image_bytes_sent_to_vllm_gib_all_workers": sum(w["image_bytes_sent_to_vllm_gib"] for w in workers),
        "groups": sum(w["groups"] for w in workers),
        "groups_pixel_values_equal": sum(w["groups_pixel_values_equal"] for w in workers),
        "groups_image_grid_thw_equal": sum(w["groups_image_grid_thw_equal"] for w in workers),
        "pickle_tests": [t for w in workers for t in w["pickle_tests"]],
        "per_worker": workers,
    }
    with open(dest, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in summary if k not in ("workerdict", "per_worker")}, indent=1))
    print("workerdict parts:", json.dumps(wd["parts"]), "rss", wd["rss_gib"],
          "predicted adam state gib", wd["predicted_adam_state_gib_after_first_update"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
