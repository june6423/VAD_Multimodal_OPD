#!/bin/bash
# VLM-VAD-ROLLOUT-001 device-level GPU memory sampler (DT answer to VERDICT-VLM-VAD-ROLLOUT-001-002 D1).
# Usage: [GPUS=0,1,2,3,4,5] [INTERVAL=15] [DURATION=10800] scripts/vad_rollout/gpu_sample.sh <output_root>
# Runs nvidia-smi every INTERVAL seconds for DURATION seconds on the listed GPUs, without a GPU allocation.
# Records: GPU_SAMPLES.csv (time, index, memory.used MiB, memory.total MiB, utilization %) and GPU_PEAK.txt
# (per GPU the sample count, the peak memory.used with its time, and the mean utilization).
set -uo pipefail

out="$1"
gpus="${GPUS:-0,1,2,3,4,5}"
interval="${INTERVAL:-15}"
duration="${DURATION:-10800}"
mkdir -p "$out"
samples="$out/GPU_SAMPLES.csv"
echo "time,index,memory_used_mib,memory_total_mib,utilization_pct" > "$samples"

end=$(( $(date +%s) + duration ))
while [[ $(date +%s) -lt $end ]]; do
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  nvidia-smi -i "$gpus" --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
    | sed "s/, /,/g; s/^/$now,/" >> "$samples"
  sleep "$interval"
done

echo "gpu samples peak_memory_used_mib memory_total_mib peak_time mean_utilization_pct" > "$out/GPU_PEAK.txt"
awk -F, 'NR > 1 {
  n[$2]++; util[$2] += $5
  if ($3 + 0 > peak[$2] + 0 || !($2 in at)) { peak[$2] = $3; at[$2] = $1; total[$2] = $4 }
} END {
  for (g in n) printf "%s %d %s %s %s %.1f\n", g, n[g], peak[g], total[g], at[g], util[g] / n[g]
}' "$samples" | sort -n >> "$out/GPU_PEAK.txt"
