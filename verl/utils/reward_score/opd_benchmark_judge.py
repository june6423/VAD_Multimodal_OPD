"""OPD benchmark scorer shared by in-training validation and eval/judge_qwenlm.py."""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

MCQ_BENCHMARKS = [
    "hrbench-4k",
    "hrbench-8k",
    "vstar",
    "mme-realworld",
    "mme-realworld-cn",
    "ai2d",
    "mmmu",
    "mmstar",
    "zwz_rl_vqa_bbox_teacher",
]

YESNO_BENCHMARKS = [
    "hallusionbench",
]

PROMPT_TEMPLATE = (
    "Your task is to judge whether the response expresses the same meaning "
    "as the answer of a question.\n"
    "The question is: {question}\n"
    "The answer is: {gt}\n"
    "The response is: {response}\n"
    "Please check and compare them and then judge. "
    "If the response is correct, your output should be Yes. "
    "Otherwise, your output should be No. Directly give me your output."
)


def _response_for_judge_prompt(response: str) -> str:
    max_chars = int(os.getenv("OPD_BENCH_JUDGE_MAX_RESPONSE_CHARS", "12000"))
    if max_chars <= 0 or len(response) <= max_chars:
        return response
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return (
        response[:head_chars]
        + "\n\n[... response truncated for judge context length ...]\n\n"
        + response[-tail_chars:]
    )


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return (
        "e429" in text
        or "exceed quota" in text
        or "rate limit" in text
        or "too many requests" in text
    )


def _judge_api_retry_config() -> tuple[float, int]:
    backoff_sec = float(
        os.getenv("OPD_BENCH_JUDGE_API_ERROR_BACKOFF_SEC")
        or os.getenv("BENCHMARK_EVAL_JUDGE_API_ERROR_BACKOFF_SEC")
        or os.getenv("JUDGE_API_ERROR_BACKOFF_SEC")
        or 180
    )
    max_retries = int(
        os.getenv("OPD_BENCH_JUDGE_API_ERROR_MAX_RETRIES")
        or os.getenv("BENCHMARK_EVAL_JUDGE_API_ERROR_MAX_RETRIES")
        or os.getenv("JUDGE_API_ERROR_MAX_RETRIES")
        or 3
    )
    return backoff_sec, max_retries


def extract_first_option(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.strip()
    patterns = [
        r"(?:final\s+answer|correct\s+answer|answer|答案)\s*[:：]\s*[*_`]*\s*[\(\[]?\s*([A-F])\b",
        r"(?:final\s+answer|correct\s+answer|answer|答案)\s+is\s+[*_`]*\s*[\(\[]?\s*([A-F])\b",
        r"(?:option|choice)\s*[*_`]*\s*[\(\[]?\s*([A-F])\b",
        r"(?:choose|select|selected|pick|picked)\s+(?:option|choice)?\s*[*_`]*\s*[\(\[]?\s*([A-F])\b",
        r"^\s*[*_`]*\s*[\(\[]?\s*([A-F])\s*[\)\]]?\s*[*_`]*\s*$",
        r"^\s*[*_`]*\s*[\(\[]?\s*([A-F])\s*[\)\].:：]",
        r"^\s*[*_`]*\s*[\(\[]?\s*([A-F])\s*[\)\]]?\s+(?:because|is|should|seems|appears)\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return ""


def extract_mcq_option(answer):
    if not isinstance(answer, str) or not answer:
        return ""
    text = answer.strip()
    pattern = r"^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))"
    match = re.match(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-F])\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


def first_letter_match(gt, answer):
    gt_val = extract_mcq_option(gt)
    pred_val = extract_first_option(answer)
    return bool(gt_val and pred_val and gt_val == pred_val)


def extract_answer(model_answer_raw):
    if not isinstance(model_answer_raw, str):
        return ""
    if "<answer>" in model_answer_raw:
        start = model_answer_raw.find("<answer>")
        end = model_answer_raw.find("</answer>")
        if start != -1 and end != -1:
            return model_answer_raw[start + len("<answer>") : end].strip()
    if "Answer:" in model_answer_raw:
        return model_answer_raw[model_answer_raw.rfind("Answer:") :].strip()
    return model_answer_raw.strip()


def normalize_judge_label(judge_text: Any) -> str:
    value = str(judge_text or "").strip()
    match = re.match(r"^\s*(yes|no)\b", value, flags=re.IGNORECASE)
    if not match:
        return value
    return "Yes" if match.group(1).lower() == "yes" else "No"


def _is_yes(judge_text: Any) -> bool:
    return normalize_judge_label(judge_text) == "Yes"


def _mathruler_grade(gt: Any, extracted_answer: str) -> bool:
    try:
        from mathruler.grader import grade_answer
    except ImportError:
        return False
    try:
        return bool(grade_answer(gt, extracted_answer))
    except Exception:
        return False


def _normalize_text(text: Any) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"^(?:final\s+answer|answer|答案)\s*[:：]\s*", "", value)
    value = value.strip(" \t\r\n\"'`.,;:()[]{}")
    value = re.sub(r"\s+", " ", value)
    return value


def _normalize_text_compact(text: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "", _normalize_text(text))


def _iter_gt_candidates(gt: Any):
    if isinstance(gt, (list, tuple, set)):
        for item in gt:
            yield item
        return
    yield gt


def exact_or_list_match(gt: Any, extracted_answer: str) -> bool:
    pred_norm = _normalize_text(extracted_answer)
    pred_compact = _normalize_text_compact(extracted_answer)
    if not pred_norm:
        return False
    for candidate in _iter_gt_candidates(gt):
        cand_norm = _normalize_text(candidate)
        if cand_norm and pred_norm == cand_norm:
            return True
        cand_compact = _normalize_text_compact(candidate)
        if cand_compact and pred_compact == cand_compact:
            return True
    return False


def _extract_yes_no(text: Any) -> str:
    value = _normalize_text(text)
    if not value:
        return ""
    matches = re.findall(r"\b(yes|no|true|false)\b", value)
    if matches:
        token = matches[-1]
        return "yes" if token in {"yes", "true"} else "no"
    if value in {"1", "y"}:
        return "yes"
    if value in {"0", "n"}:
        return "no"
    return ""


def yes_no_match(gt: Any, extracted_answer: str) -> bool:
    gt_val = _extract_yes_no(gt)
    pred_val = _extract_yes_no(extracted_answer)
    return bool(gt_val and pred_val and gt_val == pred_val)


def _llm_fallback_result(result: dict[str, Any], gt: Any, extracted_answer: str, question: str) -> dict[str, Any]:
    judge_response = _response_for_judge_prompt(extracted_answer)
    result.update(
        {
            "judge": "",
            "judge_source": "llm",
            "needs_llm": True,
            "judge_prompt": PROMPT_TEMPLATE.format(gt=gt, response=judge_response, question=question),
        }
    )
    return result


def rule_judge_one(benchmark: str, question: str, gt: Any, model_answer_raw: str) -> dict[str, Any]:
    extracted_answer = extract_answer(model_answer_raw)

    is_correct = exact_or_list_match(gt, extracted_answer) or _mathruler_grade(gt, extracted_answer)

    result = {
        "extracted_answer": extracted_answer,
        "gt": str(gt),
        "question": question,
    }
    if is_correct:
        result.update({"judge": "Yes", "judge_source": "rule", "needs_llm": False})
    elif benchmark in MCQ_BENCHMARKS:
        try:
            gt_val = extract_mcq_option(gt)
            pred_val = extract_first_option(extracted_answer)
        except Exception:
            gt_val = ""
            pred_val = ""
        if gt_val and pred_val and gt_val == pred_val:
            result.update({"judge": "Yes", "judge_source": "first letter", "needs_llm": False})
        else:
            _llm_fallback_result(result, gt, extracted_answer, question)
    elif benchmark in YESNO_BENCHMARKS:
        try:
            gt_val = _extract_yes_no(gt)
            pred_val = _extract_yes_no(extracted_answer)
        except Exception:
            gt_val = ""
            pred_val = ""
        if gt_val and pred_val and gt_val == pred_val:
            result.update({"judge": "Yes", "judge_source": "yes/no", "needs_llm": False})
        else:
            _llm_fallback_result(result, gt, extracted_answer, question)
    else:
        _llm_fallback_result(result, gt, extracted_answer, question)
    return result


def judge_via_api(
    prompts,
    api_base,
    api_key,
    judge_model,
    judge_max_tokens,
    enable_thinking=None,
    parallel_workers=32,
    show_progress=True,
    request_timeout=600,
    min_request_interval=0.0,
    thinking_extra_body_style="chat_template_kwargs",
):
    from openai import OpenAI

    thread_local = threading.local()
    api_error_backoff_sec, api_error_max_retries = _judge_api_retry_config()
    api_bases = [base.strip() for base in str(api_base).split(",") if base.strip()]
    if not api_bases:
        raise ValueError("judge_via_api requires at least one API base URL")

    def get_client(base_url: str):
        clients = getattr(thread_local, "clients", None)
        if clients is None:
            clients = {}
            thread_local.clients = clients
        c = clients.get(base_url)
        if c is None:
            c = OpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout)
            clients[base_url] = c
        return c

    results = [""] * len(prompts)
    rate_lock = threading.Lock()
    next_request_time = [0.0]

    def wait_for_rate_slot():
        if min_request_interval <= 0:
            return
        with rate_lock:
            now = time.monotonic()
            wait_s = next_request_time[0] - now
            if wait_s > 0:
                time.sleep(wait_s)
                now = time.monotonic()
            next_request_time[0] = max(next_request_time[0], now) + min_request_interval

    def call_one(idx, prompt):
        base_url = api_bases[idx % len(api_bases)]
        client = get_client(base_url)
        last_error = None
        quota_retry_count = 0
        general_retry_count = 0
        while True:
            try:
                wait_for_rate_slot()
                extra_kwargs = {}
                if enable_thinking is not None:
                    enabled = enable_thinking == "True"
                    if thinking_extra_body_style == "direct":
                        extra_kwargs["extra_body"] = {"enable_thinking": enabled}
                    elif thinking_extra_body_style == "chat_template_kwargs":
                        extra_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enabled}}
                resp = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=judge_max_tokens,
                    **extra_kwargs,
                )
                return idx, (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                last_error = exc
                if _is_quota_error(exc):
                    quota_retry_count += 1
                    if api_error_max_retries > 0 and quota_retry_count >= api_error_max_retries:
                        break
                    print(
                        f"[judge_api_retry] idx={idx} endpoint={base_url} quota error; sleeping {api_error_backoff_sec:.1f}s before retry {quota_retry_count}",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(api_error_backoff_sec)
                    continue
                general_retry_count += 1
                if general_retry_count < 3:
                    time.sleep(1.0)
                    continue
        print(f"[judge_api_error] idx={idx} endpoint={base_url} error={last_error}", file=sys.stderr, flush=True)
        return idx, f"[JUDGE_API_ERROR] {last_error}"

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [executor.submit(call_one, i, p) for i, p in enumerate(prompts)]
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="LLM Judge")
        for future in iterator:
            idx, text = future.result()
            results[idx] = text

    return results


def judge_via_api_stream(
    prompts,
    api_base,
    api_key,
    judge_model,
    judge_max_tokens,
    enable_thinking=None,
    parallel_workers=32,
    request_timeout=600,
    min_request_interval=0.0,
    thinking_extra_body_style="chat_template_kwargs",
    show_progress=True,
):
    from openai import OpenAI

    thread_local = threading.local()
    api_error_backoff_sec, api_error_max_retries = _judge_api_retry_config()
    api_bases = [base.strip() for base in str(api_base).split(",") if base.strip()]
    if not api_bases:
        raise ValueError("judge_via_api_stream requires at least one API base URL")

    def get_client(base_url: str):
        clients = getattr(thread_local, "clients", None)
        if clients is None:
            clients = {}
            thread_local.clients = clients
        c = clients.get(base_url)
        if c is None:
            c = OpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout)
            clients[base_url] = c
        return c

    rate_lock = threading.Lock()
    next_request_time = [0.0]

    def wait_for_rate_slot():
        if min_request_interval <= 0:
            return
        with rate_lock:
            now = time.monotonic()
            wait_s = next_request_time[0] - now
            if wait_s > 0:
                time.sleep(wait_s)
                now = time.monotonic()
            next_request_time[0] = max(next_request_time[0], now) + min_request_interval

    def call_one(idx, prompt):
        base_url = api_bases[idx % len(api_bases)]
        client = get_client(base_url)
        last_error = None
        quota_retry_count = 0
        general_retry_count = 0
        while True:
            try:
                wait_for_rate_slot()
                extra_kwargs = {}
                if enable_thinking is not None:
                    enabled = enable_thinking == "True"
                    if thinking_extra_body_style == "direct":
                        extra_kwargs["extra_body"] = {"enable_thinking": enabled}
                    elif thinking_extra_body_style == "chat_template_kwargs":
                        extra_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enabled}}
                resp = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=judge_max_tokens,
                    **extra_kwargs,
                )
                return idx, (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                last_error = exc
                if _is_quota_error(exc):
                    quota_retry_count += 1
                    if api_error_max_retries > 0 and quota_retry_count >= api_error_max_retries:
                        break
                    print(
                        f"[judge_api_retry] idx={idx} endpoint={base_url} quota error; sleeping {api_error_backoff_sec:.1f}s before retry {quota_retry_count}",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(api_error_backoff_sec)
                    continue
                general_retry_count += 1
                if general_retry_count < 3:
                    time.sleep(1.0)
                    continue
        text = f"[JUDGE_API_ERROR] {last_error}"
        print(f"[judge_api_error] idx={idx} endpoint={base_url} error={last_error}", file=sys.stderr, flush=True)
        return idx, text

    if parallel_workers <= 1:
        iterator = range(len(prompts))
        if show_progress:
            iterator = tqdm(iterator, total=len(prompts), desc="LLM Judge")
        for idx in iterator:
            yield call_one(idx, prompts[idx])
        return

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [executor.submit(call_one, i, p) for i, p in enumerate(prompts)]
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="LLM Judge")
        for future in iterator:
            yield future.result()


def judge_via_vllm(prompts, judge_model_path, judge_max_tokens):
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(max_tokens=judge_max_tokens, temperature=0)
    llm = LLM(model=judge_model_path, tensor_parallel_size=1, gpu_memory_utilization=0.9)
    tokenizer = AutoTokenizer.from_pretrained(judge_model_path)

    chat_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_prompts.append(text)

    outputs = llm.generate(chat_prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]


def _first_nonempty(*values):
    for value in values:
        if value is not None and str(value) != "":
            return value
    return None


def judge_batched(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos=None,
    judge_api_base=None,
    judge_api_key=None,
    judge_model=None,
    judge_model_path=None,
    judge_max_tokens=2048,
    enable_thinking=None,
    parallel_workers=32,
    request_timeout=600,
    min_request_interval=0.0,
    thinking_extra_body_style="direct",
    show_progress=True,
):
    extra_infos = extra_infos if extra_infos is not None else [{} for _ in solution_strs]
    results: list[dict[str, Any]] = []
    llm_indices = []
    llm_prompts = []

    for i, (data_source, solution, gt) in enumerate(zip(data_sources, solution_strs, ground_truths, strict=True)):
        extra_info = extra_infos[i] or {}
        benchmark = str(extra_info.get("benchmark") or data_source)
        question = str(extra_info.get("query") or extra_info.get("question") or "").replace("<image>", "")
        judged = rule_judge_one(benchmark, question, gt, str(solution or ""))
        if judged.pop("needs_llm"):
            llm_indices.append(i)
            llm_prompts.append(judged.pop("judge_prompt"))
        results.append(judged)

    if llm_prompts:
        judge_api_base = _first_nonempty(
            judge_api_base,
            os.getenv("OPD_BENCH_JUDGE_API_BASE"),
            os.getenv("BENCHMARK_EVAL_JUDGE_API_BASE"),
            os.getenv("JUDGE_API_BASE"),
            "http://127.0.0.1:8900/v1",
        )
        judge_model_path = _first_nonempty(
            judge_model_path,
            os.getenv("OPD_BENCH_JUDGE_MODEL_PATH"),
            os.getenv("BENCHMARK_EVAL_JUDGE_MODEL_PATH"),
            os.getenv("JUDGE_MODEL_PATH"),
        )
        judge_api_key = _first_nonempty(
            judge_api_key,
            os.getenv("OPD_BENCH_JUDGE_API_KEY"),
            os.getenv("BENCHMARK_EVAL_JUDGE_API_KEY"),
            os.getenv("JUDGE_API_KEY"),
            "EMPTY",
        )
        judge_model = _first_nonempty(
            judge_model,
            os.getenv("OPD_BENCH_JUDGE_MODEL"),
            os.getenv("BENCHMARK_EVAL_JUDGE_MODEL"),
            os.getenv("JUDGE_MODEL"),
            "openai/gpt-oss-120b",
        )
        judge_max_tokens = int(
            _first_nonempty(
                os.getenv("OPD_BENCH_JUDGE_MAX_TOKENS"),
                os.getenv("BENCHMARK_EVAL_JUDGE_MAX_TOKENS"),
                os.getenv("JUDGE_MAX_TOKENS"),
                judge_max_tokens,
                2048,
            )
        )
        parallel_workers = int(
            _first_nonempty(
                os.getenv("OPD_BENCH_JUDGE_PARALLEL_WORKERS"),
                os.getenv("BENCHMARK_EVAL_JUDGE_PARALLEL_WORKERS"),
                parallel_workers,
                32,
            )
        )
        request_timeout = float(
            _first_nonempty(
                os.getenv("OPD_BENCH_JUDGE_REQUEST_TIMEOUT"),
                os.getenv("BENCHMARK_EVAL_JUDGE_REQUEST_TIMEOUT"),
                os.getenv("JUDGE_REQUEST_TIMEOUT"),
                request_timeout,
                600,
            )
        )
        min_request_interval = float(
            _first_nonempty(
                os.getenv("OPD_BENCH_JUDGE_MIN_REQUEST_INTERVAL"),
                os.getenv("BENCHMARK_EVAL_JUDGE_MIN_REQUEST_INTERVAL"),
                os.getenv("JUDGE_MIN_REQUEST_INTERVAL"),
                min_request_interval,
                0.0,
            )
        )
        enable_thinking = _first_nonempty(
            enable_thinking,
            os.getenv("OPD_BENCH_JUDGE_ENABLE_THINKING"),
            os.getenv("BENCHMARK_EVAL_JUDGE_ENABLE_THINKING"),
            os.getenv("JUDGE_ENABLE_THINKING"),
        )
        thinking_extra_body_style = _first_nonempty(
            os.getenv("OPD_BENCH_JUDGE_THINKING_EXTRA_BODY_STYLE"),
            os.getenv("BENCHMARK_EVAL_JUDGE_THINKING_EXTRA_BODY_STYLE"),
            os.getenv("JUDGE_THINKING_EXTRA_BODY_STYLE"),
            thinking_extra_body_style,
            "direct",
        )

        if judge_api_base:
            llm_outputs = judge_via_api(
                llm_prompts,
                judge_api_base,
                judge_api_key,
                judge_model,
                judge_max_tokens,
                enable_thinking,
                parallel_workers,
                request_timeout=request_timeout,
                min_request_interval=min_request_interval,
                thinking_extra_body_style=thinking_extra_body_style,
                show_progress=show_progress,
            )
        elif judge_model_path:
            llm_outputs = judge_via_vllm(llm_prompts, judge_model_path, judge_max_tokens)
        else:
            raise ValueError(
                "OPD benchmark judge requires judge_api_base/JUDGE_API_BASE or "
                "judge_model_path/JUDGE_MODEL_PATH for LLM fallback cases."
            )

        for original_idx, response_text in zip(llm_indices, llm_outputs, strict=True):
            results[original_idx]["judge"] = normalize_judge_label(response_text)
            results[original_idx]["judge_source"] = "llm"

    for result in results:
        score = float(_is_yes(result.get("judge")))
        result["score"] = score
        result["acc"] = score
    return results


def compute_score_batched(data_sources, solution_strs, ground_truths, extra_infos=None, **kwargs):
    return judge_batched(data_sources, solution_strs, ground_truths, extra_infos, **kwargs)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return judge_batched([data_source], [solution_str], [ground_truth], [extra_info or {}], **kwargs)[0]
