"""Peak of the container memory charge in a memwatch.log (DIRECTIVE-VLM-VAD-PACKFIX-001-004 action 6).

python mempeak.py <memwatch.log> <MEMPEAK.json>
MEMPEAK.json: the sample with the largest ctr_cur (all its fields), the maximum of each field, the first and last
ctr_oom_kill, and the sample count. Values are GiB, as memwatch.sh writes them.
"""

import json
import re
import sys

SAMPLE = re.compile(r"^(\d\d:\d\d:\d\d) (.*)$")
FIELD = re.compile(r"([a-z_]+) (-?[0-9.]+)")


def parse(path):
    samples = []
    for line in open(path):
        m = SAMPLE.match(line)
        if not m:
            continue
        fields = {k: float(v) for k, v in FIELD.findall(m.group(2))}
        if "ctr_cur" in fields:
            samples.append({"time": m.group(1), **fields})
    return samples


def summary(samples):
    if not samples:
        return {"samples": 0}
    peak = max(samples, key=lambda s: s["ctr_cur"])
    keys = sorted({k for s in samples for k in s if k != "time"})
    return {
        "samples": len(samples),
        "first_time": samples[0]["time"],
        "last_time": samples[-1]["time"],
        "peak_ctr_cur_sample": peak,
        "max": {k: max(s[k] for s in samples if k in s) for k in keys},
        "oom_kill_first": samples[0].get("ctr_oom_kill"),
        "oom_kill_last": samples[-1].get("ctr_oom_kill"),
    }


def main(src, dest):
    result = summary(parse(src))
    with open(dest, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: result.get(k) for k in ("samples", "peak_ctr_cur_sample", "oom_kill_first", "oom_kill_last")}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
