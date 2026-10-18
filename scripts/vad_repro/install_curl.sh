#!/bin/bash
# VLM-VAD-REPRO-001 R3 curl install (DIRECTIVE-VLM-VAD-REPRO-001-002), container VAD_A100_001. No GPU work.
# Usage: scripts/vad_repro/install_curl.sh <output_root>
# Runs apt-get update && apt-get install -y --no-install-recommends curl, with no other package named.
# This script is on branch exp/VLM-VAD-REPRO-001-tools, so the run branch stays at the commit of
# VERDICT-VLM-VAD-REPRO-001-003. The record is <output_root>/INSTALL_CURL.txt.
set -uo pipefail

out="$1"
mkdir -p "$out"
export DEBIAN_FRONTEND=noninteractive
{
  echo "== date"; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== curl before: $(command -v curl || echo missing)"
  echo "== dpkg list before (count)"; dpkg-query -W -f='${Package}\n' | wc -l
  dpkg-query -W -f='${Package} ${Version}\n' | sort > "$out/dpkg-before.txt"
  echo "== apt-get update && apt-get install -y --no-install-recommends curl"
  apt-get update && apt-get install -y --no-install-recommends curl
  echo "apt exit $?"
  dpkg-query -W -f='${Package} ${Version}\n' | sort > "$out/dpkg-after.txt"
  echo "== dpkg diff (before -> after)"; diff "$out/dpkg-before.txt" "$out/dpkg-after.txt"
  echo "== curl after: $(command -v curl || echo missing)"
  curl --version
} > "$out/INSTALL_CURL.txt" 2>&1

grep -q "^apt exit 0$" "$out/INSTALL_CURL.txt" && command -v curl > /dev/null
