#!/bin/bash
# DIRECTIVE-VLM-VAD-ROLLOUT-001-015: inventory of the workspace and SHA-256 manifest of the experiment outputs before
# the data move. No GPU, no package install. It reads files and writes only under <output root>.
# Usage: inventory.sh <output root> [workspace root, default /workspace]
# Classes (by path):
#   A       harness/outputs/<experiment>/<task> (task output roots) and untracked files of the repo checkouts
#   B       harness/data/* (downloads and derived data), harness/envs/* (Python environments)
#   REPO    a repo checkout as a whole (tracked code is B by its git commit; untracked entries: REPO_UNTRACKED.tsv)
#   SECRET  harness/secrets and files named netrc, .netrc or *.env: path only, never hashed or copied
#   CHECK   any other path: hashed in a separate manifest, so that nothing is lost before a decision
#   SELF    the output root of this task (not hashed)
# Manifests: "<sha256>  <path relative to the workspace root>", sorted by path, in 16 (A) and 4 (CHECK) parts.
# Check at the destination: cd <destination workspace root> && cat MANIFEST_A.part* | sha256sum -c --quiet
set -uo pipefail
out="$1"
ws="$(realpath "${2:-/workspace}")"
mkdir -p "$out"
out="$(realpath "$out")"
export CUDA_VISIBLE_DEVICES=""
jobs=$(nproc)
(( jobs > 32 )) && jobs=32
env_log="$out/INVENTORY_ENV.txt"
inv="$out/INVENTORY.tsv"
secrets="$out/SECRETS_PATHS.txt"
untracked="$out/REPO_UNTRACKED.tsv"
sources="$out/B_SOURCES.txt"
: > "$secrets"
: > "$out/HASH_A.list"
: > "$out/HASH_CHECK.list"
printf 'class\tbytes\tfiles\tsymlinks\tpath\n' > "$inv"
printf 'class\tbytes\tfiles\tpath\n' > "$untracked"
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "== workspace $ws"
  echo "== output root $out"
  echo "== nproc $(nproc), hash jobs $jobs"
} > "$env_log"

is_secret_name() { case "$(basename "$1")" in netrc|.netrc|*.env) return 0 ;; esac; return 1; }

item() {  # item <class> <path>: one inventory row; class A and CHECK items go to the hash lists
  local class="$1" path="$2" bytes files links
  bytes=$(du -sb "$path" 2>/dev/null | cut -f1)
  files=$(find "$path" -type f 2>/dev/null | wc -l)
  links=$(find "$path" -type l 2>/dev/null | wc -l)
  printf '%s\t%s\t%s\t%s\t%s\n' "$class" "${bytes:-0}" "$files" "$links" "${path#"$ws"/}" >> "$inv"
  [[ "$class" == A ]] && printf '%s\0' "$path" >> "$out/HASH_A.list"
  [[ "$class" == CHECK ]] && printf '%s\0' "$path" >> "$out/HASH_CHECK.list"
  return 0
}

repo_item() {  # repo_item <repo dir>: the whole checkout is one REPO row (tracked code is B by git); untracked and
  # ignored entries are listed in REPO_UNTRACKED.tsv and hashed as A, except Python build files (B)
  local repo="$1" line status path class bytes files
  item REPO "$repo"
  echo "repo ${repo#"$ws"/} HEAD $(git -C "$repo" rev-parse HEAD 2>/dev/null) branch $(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null)" >> "$sources"
  while IFS= read -r -d '' line; do
    status="${line:0:2}"
    [[ "$status" == "??" || "$status" == "!!" ]] || continue
    path="$repo/${line:3}"
    path="${path%/}"
    case "$path" in
      */__pycache__|*/__pycache__/*|*.pyc|*.egg-info|*.egg-info/*) class=B ;;
      *) class=A ;;
    esac
    if is_secret_name "$path"; then echo "${path#"$ws"/}" >> "$secrets"; continue; fi
    bytes=$(du -sb "$path" 2>/dev/null | cut -f1)
    files=$(find "$path" -type f 2>/dev/null | wc -l)
    printf '%s\t%s\t%s\t%s\n' "$class" "${bytes:-0}" "$files" "${path#"$ws"/}" >> "$untracked"
    [[ "$class" == A ]] && printf '%s\0' "$path" >> "$out/HASH_A.list"
  done < <(git -C "$repo" status --porcelain=v1 -z --ignored --untracked-files=normal 2>/dev/null)
}

hf_revision() {  # hf_revision <dir>: commit hash from the hf download metadata, if any
  local meta
  meta=$(find "$1" -path '*/.cache/huggingface/download/*.metadata' -type f 2>/dev/null | head -n 1)
  [[ -n "$meta" ]] && head -n 1 "$meta" || echo "no hf metadata"
}

# Top level of the workspace.
for path in "$ws"/* "$ws"/.[!.]*; do
  [[ -e "$path" ]] || continue
  [[ "$path" == "$ws/harness" ]] && continue
  item CHECK "$path"
done
# harness/
for path in "$ws"/harness/* "$ws"/harness/.[!.]*; do
  [[ -e "$path" ]] || continue
  name="${path#"$ws"/harness/}"
  case "$name" in
    outputs)
      for exp in "$path"/*; do
        [[ -e "$exp" ]] || continue
        if [[ -d "$exp" ]]; then
          for task in "$exp"/*; do
            [[ -e "$task" ]] || continue
            if [[ "$(realpath "$task")" == "$out" ]]; then item SELF "$task"; else item A "$task"; fi
          done
        else
          item A "$exp"
        fi
      done ;;
    data)
      for sub in "$path"/*; do
        [[ -e "$sub" ]] || continue
        case "${sub##*/}" in
          models|hub) for d in "$sub"/*; do [[ -e "$d" ]] && item B "$d" && echo "${d#"$ws"/} $(hf_revision "$d")" >> "$sources"; done ;;
          *) item B "$sub" ;;
        esac
      done ;;
    envs) for d in "$path"/*; do [[ -e "$d" ]] && item B "$d"; done ;;
    secrets) find "$path" -mindepth 1 2>/dev/null | sed "s#^$ws/##" >> "$secrets" ;;
    *) if [[ -d "$path/.git" ]]; then repo_item "$path"; else item CHECK "$path"; fi ;;
  esac
done
for d in "$ws"/harness/data/vad/benchmarks/*_data "$ws"/harness/data/vad/benchmarks/yifanzhang114_*; do
  [[ -d "$d" ]] && echo "${d#"$ws"/} $(hf_revision "$d")" >> "$sources"
done

# Information only (outside the workspace).
{
  echo "== df -B1 $ws"; df -B1 "$ws"
  echo "== df -h $ws"; df -h "$ws"
  echo "== outside the workspace (information only)"
  for p in /tmp/ray /root/.cache; do [[ -e "$p" ]] && echo "$(du -sb "$p" 2>/dev/null | cut -f1) $p"; done
} > "$out/DF.txt" 2>&1

# Manifests. Secret-named files are never hashed; their paths go to SECRETS_PATHS.txt.
manifest() {  # manifest <list> <name> <parts>
  local list="$1" name="$2" parts="$3" start
  start=$(date +%s)
  if [[ -s "$list" ]]; then
    xargs -0 -I{} find {} -type f -print0 < "$list" \
      | while IFS= read -r -d '' f; do
          if is_secret_name "$f"; then echo "${f#"$ws"/}" >> "$secrets"; else printf '%s\0' "$f"; fi
        done \
      | xargs -0 -P "$jobs" -n 64 sha256sum \
      | sed "s#  $ws/#  #" | sort -k2 > "$out/$name.all"
  else
    : > "$out/$name.all"
  fi
  split -n "l/$parts" -d -a 2 "$out/$name.all" "$out/$name.part"
  echo "$name lines $(wc -l < "$out/$name.all") bytes $(wc -c < "$out/$name.all") sha256 $(sha256sum "$out/$name.all" | cut -d' ' -f1) parts $parts seconds $(( $(date +%s) - start ))" >> "$out/MANIFEST_INFO.txt"
}
: > "$out/MANIFEST_INFO.txt"
manifest "$out/HASH_A.list" MANIFEST_A 16
manifest "$out/HASH_CHECK.list" MANIFEST_CHECK 4
sort -u -o "$secrets" "$secrets"

# Totals by class (inventory rows plus untracked repo entries).
{
  printf 'class\tbytes\tGB\tfiles\n'
  tail -n +2 "$inv" | cut -f1,2,3 > "$out/.rows"
  tail -n +2 "$untracked" | cut -f1,2,3 | sed 's/^\(A\|B\)\t/\1(repo)\t/' >> "$out/.rows"
  sort "$out/.rows" | awk -F'\t' '{b[$1]+=$2; f[$1]+=$3} END {for (c in b) printf "%s\t%d\t%.1f\t%d\n", c, b[c], b[c]/1e9, f[c]}' | sort
  rm -f "$out/.rows"
} > "$out/INVENTORY_SUMMARY.txt"
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$env_log"
