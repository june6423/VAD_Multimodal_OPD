#!/bin/bash
# VLM-VAD-ROLLOUT-001 W&B check (HANDOFF-VLM-VAD-ROLLOUT-001-R4, Part A item 3, gate G3). No GPU work.
# Usage: WANDB_ENV_FILE=<file that sets WANDB_API_KEY> ENV_ROOT=<environment root> \
#   scripts/vad_rollout/wandb_check.sh <output_root>
# The wandb_check.sh of 1f12b81d (scripts/vad_repro/) with the secrets path from WANDB_ENV_FILE and the
# wandb of the VAD environment. Runs wandb login --verify with the key in WANDB_API_KEY. The key value is
# never printed: the record shows each secret-named variable as <set> or <unset>, and the wandb output has
# the key masked. The wandb config, cache, data and netrc paths point inside <output_root>. If wandb writes
# a netrc file, the script reports it and removes it. The record is <output_root>/WANDB_CHECK.txt.
set -uo pipefail
set +x

out="$1"
secrets="${WANDB_ENV_FILE:?WANDB_ENV_FILE must name the W&B key file}"
env_root="${ENV_ROOT:-$HOME/vad-rollout/envs}"
export PATH="$env_root/vad/bin:$PATH"
mkdir -p "$out/wandb-config" "$out/wandb-cache" "$out/wandb-data"
export WANDB_CONFIG_DIR="$out/wandb-config" WANDB_CACHE_DIR="$out/wandb-cache" \
  WANDB_DATA_DIR="$out/wandb-data" WANDB_DIR="$out" NETRC="$out/netrc" WANDB_SILENT=false
log="$out/WANDB_CHECK.txt"

{
  echo "== date"; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== secrets file"; ls -l "$secrets" | awk '{print $1, $3, $4, $NF}'
  echo "variable names in the file (values not shown):"
  grep -oE '^(export +)?[A-Za-z_][A-Za-z0-9_]*=' "$secrets" | sed -E 's/^export +//; s/=$//' | sort -u
  # Error text of a malformed line could hold the key, so stderr of source is dropped.
  set -a
  # shellcheck source=/dev/null
  source "$secrets" 2>/dev/null
  echo "source exit $?"
  set +a
  for name in $(compgen -e | grep -E 'KEY|TOKEN|SECRET|PASSWORD' | sort); do
    if [[ -n "${!name}" ]]; then echo "$name=<set>"; else echo "$name=<unset>"; fi
  done
  echo "== wandb"; command -v wandb; wandb --version
  echo "== wandb login --verify (key from WANDB_API_KEY, masked)"
  text=$(wandb login --verify 2>&1)
  status=$?
  if [[ -n "${WANDB_API_KEY:-}" ]]; then text="${text//"$WANDB_API_KEY"/<masked>}"; fi
  printf '%s\n' "$text"
  echo "wandb login --verify exit $status"
  if [[ -e "$NETRC" ]]; then
    echo "netrc written by wandb: yes (removed)"
    rm -f "$NETRC"
  else
    echo "netrc written by wandb: no"
  fi
  echo "== files under the output root"
  (cd "$out" && find . -type f ! -name WANDB_CHECK.txt -printf '%s %p\n' | sort -k2)
} > "$log" 2>&1

grep -q "^wandb login --verify exit 0$" "$log"
