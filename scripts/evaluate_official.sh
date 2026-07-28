#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  MODEL_API_BASE=http://127.0.0.1:8000/v1 \
  MODEL_ID=VCD \
  MODEL_TAG=vcd_checkpoint \
  bash scripts/evaluate_official.sh

Optional answer reuse:
  REUSE_ANSWER_ROOT=/path/to/model_answer \
  REUSE_MODEL_TAG=old_model_tag

The reuse root must contain:
  <benchmark>/<REUSE_MODEL_TAG>_seed42_answer.jsonl
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -gt 0 ]]; then
  echo "This entrypoint is configured through environment variables." >&2
  usage >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_CODE="${PROJECT_ROOT}/evaluation/vision_opd_official"
BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-${PROJECT_ROOT}/data/benchmarks}"

MODEL_API_BASE="${MODEL_API_BASE:-${API_BASE:-}}"
MODEL_API_KEY="${MODEL_API_KEY:-${OPENAI_API_KEY:-EMPTY}}"
MODEL_ID="${MODEL_ID:-${OPENAI_MODEL_ID:-}}"
MODEL_TAG="${MODEL_TAG:-VCD}"
MODEL_PATH="${MODEL_PATH:-}"

JUDGE_API_BASE="${JUDGE_API_BASE:-http://127.0.0.1:8900/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-2048}"

SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
ENABLE_THINKING="${ENABLE_THINKING:-False}"
MAX_RETRIES="${MAX_RETRIES:-3}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-256}"
FORMAL_BENCHMARKS="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn"
BENCHMARKS="${BENCHMARKS:-${FORMAL_BENCHMARKS}}"

REUSE_ANSWER_ROOT="${REUSE_ANSWER_ROOT:-}"
REUSE_MODEL_TAG="${REUSE_MODEL_TAG:-${MODEL_TAG}}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${MODEL_TAG}_seed${SEED}_${RUN_STAMP}}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/outputs/official_eval/${RUN_NAME}}"

if [[ ! "${MODEL_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "MODEL_TAG may contain only letters, numbers, dots, underscores, and hyphens: ${MODEL_TAG}" >&2
  exit 2
fi
if [[ "${SEED}" != "42" || "${MAX_TOKENS}" != "8192" || "${ENABLE_THINKING}" != "False" ]]; then
  echo "Formal reproduction requires SEED=42, MAX_TOKENS=8192, and ENABLE_THINKING=False." >&2
  exit 2
fi
if [[ "${JUDGE_MODEL}" != "openai/gpt-oss-120b" ]]; then
  echo "Formal main-table evaluation requires JUDGE_MODEL=openai/gpt-oss-120b." >&2
  exit 2
fi
if [[ "${JUDGE_MAX_TOKENS}" != "2048" ]]; then
  echo "Formal main-table evaluation requires JUDGE_MAX_TOKENS=2048." >&2
  exit 2
fi
if [[ "${BENCHMARKS}" != "${FORMAL_BENCHMARKS}" ]]; then
  echo "Formal main-table evaluation requires all six benchmarks in the documented order." >&2
  exit 2
fi
if [[ -z "${MODEL_ID}" && -z "${REUSE_ANSWER_ROOT}" ]]; then
  echo "MODEL_ID is required when model answers are not being reused." >&2
  exit 2
fi
if [[ -z "${MODEL_API_BASE}" && -z "${REUSE_ANSWER_ROOT}" ]]; then
  echo "MODEL_API_BASE is required when model answers are not being reused." >&2
  exit 2
fi

benchmark_json_name() {
  case "$1" in
    vstar) echo "vstar.json" ;;
    zoombench) echo "zoombench.json" ;;
    hrbench-4k) echo "hr_bench_4k.json" ;;
    hrbench-8k) echo "hr_bench_8k.json" ;;
    mme-realworld) echo "MME_RealWorld.json" ;;
    mme-realworld-cn) echo "MME_RealWorld_CN.json" ;;
    *)
      echo "Unsupported formal benchmark: $1" >&2
      return 1
      ;;
  esac
}

check_endpoint() {
  local api_base="$1"
  local api_key="$2"
  local expected_model="$3"
  local response
  response="$(
    curl -fsS --max-time 20 \
      -H "Authorization: Bearer ${api_key}" \
      "${api_base%/}/models"
  )"
  if [[ "${response}" != *"\"${expected_model}\""* ]]; then
    echo "Endpoint ${api_base} does not advertise ${expected_model}." >&2
    exit 1
  fi
}

mkdir -p "${RUN_DIR}/model_answer" "${RUN_DIR}/judge" "${RUN_DIR}/logs" "${RUN_DIR}/audits"
check_endpoint "${JUDGE_API_BASE}" "${JUDGE_API_KEY}" "${JUDGE_MODEL}"
if [[ -z "${REUSE_ANSWER_ROOT}" ]]; then
  check_endpoint "${MODEL_API_BASE}" "${MODEL_API_KEY}" "${MODEL_ID}"
fi

IFS=',' read -r -a BENCHMARK_ARRAY <<<"${BENCHMARKS}"
for raw_benchmark in "${BENCHMARK_ARRAY[@]}"; do
  benchmark="$(echo "${raw_benchmark}" | xargs)"
  [[ -z "${benchmark}" ]] && continue
  benchmark_json="${BENCHMARK_DATA_DIR}/$(benchmark_json_name "${benchmark}")"
  if [[ ! -f "${benchmark_json}" ]]; then
    python3 "${EVAL_CODE}/prepare_data.py" \
      --benchmark "${benchmark}" \
      --data_dir "${BENCHMARK_DATA_DIR}"
  fi

  answer_name="${MODEL_TAG}_seed${SEED}_answer.jsonl"
  answer_json="${RUN_DIR}/model_answer/${benchmark}/${answer_name}"
  answer_source="generated"
  answer_source_path="${answer_json}"
  mkdir -p "$(dirname "${answer_json}")" "${RUN_DIR}/judge/${benchmark}"

  if [[ -n "${REUSE_ANSWER_ROOT}" ]]; then
    source_answer="${REUSE_ANSWER_ROOT}/${benchmark}/${REUSE_MODEL_TAG}_seed${SEED}_answer.jsonl"
    if [[ ! -f "${source_answer}" ]]; then
      echo "Reusable answer is missing: ${source_answer}" >&2
      exit 1
    fi
    ln -sfn "$(realpath "${source_answer}")" "${answer_json}"
    answer_source="reused"
    answer_source_path="$(realpath "${source_answer}")"
  elif [[ -f "${answer_json}" ]]; then
    answer_source="resumed"
  fi

  if [[ "${answer_source}" != "reused" ]]; then
    python3 "${EVAL_CODE}/infer.py" \
      --benchmark "${benchmark}" \
      --benchmark_json "${benchmark_json}" \
      --out_dir "${RUN_DIR}/model_answer" \
      --model_name "${MODEL_TAG}_seed${SEED}" \
      --seed "${SEED}" \
      --api_base "${MODEL_API_BASE}" \
      --api_key "${MODEL_API_KEY}" \
      --model_id "${MODEL_ID}" \
      --max_tokens "${MAX_TOKENS}" \
      --max_retries "${MAX_RETRIES}" \
      --parallel_workers "${PARALLEL_WORKERS}" \
      --enable_thinking "${ENABLE_THINKING}" \
      2>&1 | tee "${RUN_DIR}/logs/${benchmark}.infer.log"
  fi

  python3 "${EVAL_CODE}/audit_results.py" \
    --benchmark "${benchmark}" \
    --benchmark-json "${benchmark_json}" \
    --answer-json "${answer_json}" \
    --answer-source "${answer_source}" \
    --answer-source-path "${answer_source_path}" \
    >"${RUN_DIR}/logs/${benchmark}.answer_audit.log"

  (
    cd "${RUN_DIR}"
    python3 "${EVAL_CODE}/judge_qwenlm.py" \
      --benchmark "${benchmark}" \
      --model "${MODEL_TAG}_seed${SEED}" \
      --api_base "${JUDGE_API_BASE}" \
      --api_key "${JUDGE_API_KEY}" \
      --judge_model "${JUDGE_MODEL}" \
      --judge_max_tokens "${JUDGE_MAX_TOKENS}"
  ) 2>&1 | tee "${RUN_DIR}/logs/${benchmark}.judge.log"

  judge_json="${RUN_DIR}/judge/${benchmark}/${answer_name}"
  python3 "${EVAL_CODE}/repair_judges.py" \
    --judge-json "${judge_json}" \
    --api-base "${JUDGE_API_BASE}" \
    --api-key "${JUDGE_API_KEY}" \
    --judge-model "${JUDGE_MODEL}" \
    --judge-max-tokens "${JUDGE_MAX_TOKENS}" \
    2>&1 | tee -a "${RUN_DIR}/logs/${benchmark}.judge.log"

  python3 "${EVAL_CODE}/audit_results.py" \
    --benchmark "${benchmark}" \
    --benchmark-json "${benchmark_json}" \
    --answer-json "${answer_json}" \
    --judge-json "${judge_json}" \
    --answer-source "${answer_source}" \
    --answer-source-path "${answer_source_path}" \
    --output "${RUN_DIR}/audits/${benchmark}.json" \
    >"${RUN_DIR}/logs/${benchmark}.audit.log"

  python3 "${EVAL_CODE}/cal_acc.py" \
    --benchmark "${benchmark}" \
    --judge_json "${judge_json}" \
    --benchmark_json "${benchmark_json}" \
    2>&1 | tee "${RUN_DIR}/logs/${benchmark}.score.log"
done

python3 "${EVAL_CODE}/summarize_results.py" \
  --run-dir "${RUN_DIR}" \
  --benchmarks "${BENCHMARKS}" \
  --model-tag "${MODEL_TAG}" \
  --model-id "${MODEL_ID:-reused-model-answer}" \
  --model-path "${MODEL_PATH}" \
  --model-api-base "${MODEL_API_BASE}" \
  --judge-api-base "${JUDGE_API_BASE}" \
  --judge-model "${JUDGE_MODEL}" \
  --judge-max-tokens "${JUDGE_MAX_TOKENS}" \
  --seed "${SEED}" \
  --max-tokens "${MAX_TOKENS}" \
  --enable-thinking "${ENABLE_THINKING}" \
  --evaluator-commit c2e345f

echo "Official evaluation complete: ${RUN_DIR}"
