import csv
import json
import os
import sys
import glob
import fcntl
from pathlib import Path
from typing import Dict, List
import argparse

import dotenv
import pandas as pd
import requests
from tqdm import tqdm
from sklearn.metrics import f1_score
from src.veracity.llm_nli import (
    predict_stance_ollama,
    predict_stance_openai,
    predict_stance_openrouter,
    predict_stance_openai_batch,
    predict_stance_openrouter_batch,
    predict_stance_ollama_batch,
)
from src.veracity.stance_inference import (
    BERTStancePredictor, STANCE_ID2LABEL, STANCE_LABEL2ID,
)
from src.llm_utils.openai_utils import OpenAIUtils
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.utils import get_access_token, load_lang_codes, load_json
import random


# Load secrets (Azure OpenAI / Anthropic / Factiverse keys) from the ml-models
# repo root .env so it works regardless of the process working directory
# (e.g. when launched by systemd). Falls back to default discovery.
dotenv.load_dotenv(Path(__file__).resolve().parents[3] / ".env")
dotenv.load_dotenv()



def factiverse_verify(query, lang, access_token):
    api_link = os.getenv("SERVER_ENDPOINT")
    api_endpoint = f"{api_link}/stance_detection"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    payload = {
        "claim": query,  # Your search query "lang": "en", # Language code
        "lang": lang
    }
    # print(payload)
    response = requests.post(api_endpoint, headers=headers, json=payload)
    return response


# ── Local Factiverse stance model (fine-tuned mmBERT) ─────────────────────────
# Repo root is three levels up from this file (factcheck-editor/src/veracity/…).
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_checkpoint(path: str, fallback_glob: str) -> str:
    """Return an existing checkpoint path, falling back to the newest match."""
    p = Path(path)
    if p.exists():
        return str(p)
    candidates = sorted(_REPO_ROOT.glob(fallback_glob))
    if candidates:
        chosen = str(candidates[-1])
        print(f"[warn] checkpoint not found: {path}; using {chosen}")
        return chosen
    return path


FACTIVERSE_LOCAL_STANCE_PATH = os.getenv(
    "FACTIVERSE_STANCE_MODEL_PATH",
    str(_REPO_ROOT / "results/mmbert_balanced/best_model/"),
)
FACTIVERSE_XLMR_LOCAL_STANCE_PATH = os.getenv(
    "FACTIVERSE_XLMR_STANCE_MODEL_PATH",
    str(_REPO_ROOT / "results/xlmr_stance_balanced/checkpoint-9672"),
)
FACTIVERSE_XLMR_LOCAL_STANCE_PATH = _resolve_checkpoint(
    FACTIVERSE_XLMR_LOCAL_STANCE_PATH,
    "results/xlmr_stance_balanced/checkpoint-*",
)

# Lazy singleton — the mmBERT stance model loads on GPU only when
# `factiverse_local` is actually requested.
_stance_model: BERTStancePredictor | None = None
_xlmr_stance_model: BERTStancePredictor | None = None


def get_stance_model() -> BERTStancePredictor:
    """Return the local fine-tuned mmBERT stance model, loading on first call."""
    global _stance_model
    if _stance_model is None:
        print(f"[factiverse_local] Loading stance model from {FACTIVERSE_LOCAL_STANCE_PATH} …")
        _stance_model = BERTStancePredictor(FACTIVERSE_LOCAL_STANCE_PATH)
    return _stance_model


def get_xlmr_stance_model() -> BERTStancePredictor:
    """Return the local fine-tuned XLM-R stance model, loading on first call."""
    global _xlmr_stance_model
    if _xlmr_stance_model is None:
        print(f"[factiverse_xlmr_local] Loading stance model from {FACTIVERSE_XLMR_LOCAL_STANCE_PATH} …")
        _xlmr_stance_model = BERTStancePredictor(FACTIVERSE_XLMR_LOCAL_STANCE_PATH)
    return _xlmr_stance_model


def verify_factiverse_local_batch(claims: list, evidences: list) -> list:
    """Predict stance for (claim, evidence) pairs with the local mmBERT model.

    Returns a list of
    ``{"int_label": 0|1|2, "label": "REFUTES|SUPPORTS|MIXED",
       "score": <P(top class)>, "scores": [p0, p1, p2]}``.
    """
    pairs = tuple((c or "", e or "") for c, e in zip(claims, evidences))
    preds, scores = get_stance_model().predict(pairs)
    out = []
    for p, s in zip(preds, scores):
        out.append({
            "int_label": int(p),
            "label": STANCE_ID2LABEL.get(int(p), str(p)),
            "score": float(max(s)),
            "scores": [float(x) for x in s],
        })
    return out


def verify_factiverse_xlmr_local_batch(claims: list, evidences: list) -> list:
    """Predict stance for (claim, evidence) pairs with the local XLM-R model.

    Returns a list of
    ``{"int_label": 0|1|2, "label": "REFUTES|SUPPORTS|MIXED",
       "score": <P(top class)>, "scores": [p0, p1, p2]}``.
    """
    pairs = tuple((c or "", e or "") for c, e in zip(claims, evidences))
    preds, scores = get_xlmr_stance_model().predict(pairs)
    out = []
    for p, s in zip(preds, scores):
        out.append({
            "int_label": int(p),
            "label": STANCE_ID2LABEL.get(int(p), str(p)),
            "score": float(max(s)),
            "scores": [float(x) for x in s],
        })
    return out


# Models that can run on local claim+evidence (no remote stance API).
LOCAL_MODELS = ["factiverse_local", "factiverse_xlmr_local", "gpt52", "claude", "openrouter", "qwen3-8b"]

# model key → (label column, int-label column) written into the merged preds.
_STANCE_PRED_KEYS = {
    "factiverse_local": ("facti_local_label", "facti_local_int_label"),
    "factiverse_xlmr_local": ("facti_xlmr_local_label", "facti_xlmr_local_int_label"),
    "gpt52":            ("gpt5_label", "gpt5_int_label"),
    "claude":           ("claude-opus-4-6_label", "claude-opus-4-6_int_label"),
    "openrouter":       ("openrouter_label", "openrouter_int_label"),
    "qwen3-8b":          ("ollama_label", "ollama_int_label"),
}
_STANCE_DISPLAY = {
    "factiverse_local": "Factiverse (local mmBERT)",
    "factiverse_xlmr_local": "Factiverse (local XLM-R)",
    "gpt52":            "OpenAI GPT-5.2",
    "claude":           "Claude Opus 4-6",
    "openrouter":       "OpenRouter",
    "qwen3-8b":          "Ollama (qwen3-8b)",
}


def _stance_label_to_id(label) -> int:
    """Map an LLM string label to a class id; unknown/NEI → -1 (counts wrong)."""
    return STANCE_LABEL2ID.get(str(label).strip().upper(), -1)


def _report_stance_f1(name: str, lang: str, gt: list, pred: list) -> None:
    if not gt or len(gt) != len(pred):
        print(f"{name} [{lang}] - no predictions to evaluate")
        return
    macro = f1_score(gt, pred, average="macro", zero_division=0)
    micro = f1_score(gt, pred, average="micro", zero_division=0)
    per = f1_score(gt, pred, average=None, labels=[0, 1, 2], zero_division=0)
    acc = sum(int(g == p) for g, p in zip(gt, pred)) / max(len(gt), 1)
    print(f"{name} [{lang}] n={len(gt)}  Acc={acc:.4f}  "
          f"Macro-F1={macro:.4f}  Micro-F1={micro:.4f}  "
          f"F1(REFUTES)={per[0]:.4f} F1(SUPPORTS)={per[1]:.4f} F1(MIXED)={per[2]:.4f}")


def run_local_eval(
    models, split: str, batch_size: int = 8,
    openrouter_model: str = "google/gemma-4-31b-it:free",
    ollama_model: str = "qwen3-8b",
    lang: str = None,
    data_dir: str = "data/veracity_prediction/",
    use_pred_cache: bool = True,
    lang_workers: int = 1,
) -> None:
    """Evaluate any subset of LOCAL_MODELS on ``{lang}_{split}.json`` files.

    Reads claim AND evidence straight from the local JSON (e.g.
    ``data/veracity_prediction_gcloud_gcloud/en_test.json``) — no remote ``stance_detection``
    API call. Each requested model predicts on the same pairs; predictions are
    merged per row into ``{lang}_{split}_local_pred.json`` and macro/micro/
    per-class F1 is printed per model against ground-truth ``labels``
    (0=REFUTES, 1=SUPPORTS, 2=MIXED).
    """
    models = [m for m in LOCAL_MODELS if m in set(models)]
    if not models:
        return
    if lang:
        files = [os.path.join(data_dir, f"{lang}_{split}.json")]
        files = [f for f in files if os.path.exists(f)]
        if not files:
            print(f"[local-eval] No file {data_dir}/{lang}_{split}.json")
            return
    else:
        files = sorted(glob.glob(f"{data_dir}/*_{split}.json"))
    if not files:
        print(f"[local-eval] No files matching *_{split}.json in {data_dir}")
        return

    langs = load_lang_codes()

    def _run_single_file(data_file: str) -> None:
        lang_code = os.path.basename(data_file)[: -len(f"_{split}.json")]
        lang_name = langs.get(lang_code, {}).get("name", lang_code)
        data = load_json(data_file)
        if not data:
            print(f"[local-eval] {data_file} is empty — skipping")
            return
        print(f"\n[local-eval] {lang_code} ({len(data)} rows) — models: {', '.join(models)}")

        # Keep one OpenAI client per language-task to avoid sharing mutable
        # client state across worker threads.
        open_ai_utils = OpenAIUtils() if ({"gpt52", "claude"} & set(models)) else None

        claims = [d.get("claim", "") for d in data]
        evidences = [d.get("evidence", "") for d in data]
        # Ground truth is optional per row; rows without it are still predicted
        # but excluded from F1.
        def _gt(d):
            v = d.get("labels")
            return int(v) if v is not None else None
        gts = [_gt(d) for d in data]

        # Merge with any existing pred file so predictions from earlier runs
        # (other models' columns) are preserved instead of overwritten.
        out_file = f"{data_dir}/{lang_code}_{split}_local_pred.json"
        existing_by_claim = {}
        if use_pred_cache and os.path.exists(out_file):
            try:
                with open(out_file, encoding="utf-8") as f:
                    for r in json.load(f):
                        existing_by_claim[r.get("claim", "")] = r
                print(f"  merging with {len(existing_by_claim)} existing rows in {out_file}")
            except Exception as e:
                print(f"  [warn] could not read {out_file}: {e} — starting fresh")
        elif not use_pred_cache and os.path.exists(out_file):
            print(f"  cache disabled: recomputing all rows and overwriting {out_file}")

        rows = []
        for k, d in enumerate(data):
            row = dict(existing_by_claim.get(d.get("claim", ""), {}))
            row.update({
                "claim": d.get("claim", ""),
                "evidence": d.get("evidence", ""),
                "lang": d.get("lang", lang_code),
                "dataset_name": d.get("dataset_name"),
                "labels": gts[k],
                "gt_label": STANCE_ID2LABEL.get(gts[k]) if gts[k] is not None else None,
            })
            rows.append(row)

        def _pred_keys(m):
            """Prediction column names. For ollama/openrouter the actual model
            is embedded (e.g. 'ollama/qwen3:8b', 'openrouter/google/gemma-…')
            so different models don't overwrite each other in the pred file."""
            if m == "qwen3-8b":
                return (f"ollama/{ollama_model}_label", f"ollama/{ollama_model}_int_label")
            if m == "openrouter":
                return (f"openrouter/{openrouter_model}_label", f"openrouter/{openrouter_model}_int_label")
            return _STANCE_PRED_KEYS[m]

        def _dispatch(name, bc, be):
            """Run one model on a batch; return list of string labels (LLMs)
            or list of local-model result dicts (factiverse_local / factiverse_xlmr_local)."""
            if name == "factiverse_local":
                return verify_factiverse_local_batch(bc, be)
            if name == "factiverse_xlmr_local":
                return verify_factiverse_xlmr_local_batch(bc, be)
            if name == "gpt52":
                return predict_stance_openai_batch(claims=bc, evidences=be, lang=lang_name,
                                                   open_ai_utils=open_ai_utils, model="gpt-5.2")
            if name == "claude":
                return predict_stance_openai_batch(claims=bc, evidences=be, lang=lang_name,
                                                   open_ai_utils=open_ai_utils, model="claude-opus-4-6")
            if name == "openrouter":
                return predict_stance_openrouter_batch(claims=bc, evidences=be, lang=lang_name,
                                                       model=openrouter_model)
            if name == "qwen3-8b":
                return predict_stance_ollama_batch(claims=bc, evidences=be, lang=lang_name,
                                                   model=ollama_model)
            return [None] * len(bc)

        def _done(row, m):
            """A row is already predicted for model m if it has a non-ERROR label."""
            lkey, _ = _pred_keys(m)
            v = row.get(lkey)
            return v is not None and v != "ERROR"

        def _assign(row, m, r):
            lkey, ikey = _pred_keys(m)
            if m in {"factiverse_local", "factiverse_xlmr_local"}:
                row[lkey] = r["label"]
                row[ikey] = r["int_label"]
                if m == "factiverse_local":
                    row["facti_local_score"] = r.get("score")
                    row["facti_local_scores"] = r.get("scores")
                else:
                    row["facti_xlmr_local_score"] = r.get("score")
                    row["facti_xlmr_local_scores"] = r.get("scores")
            else:
                row[lkey] = r
                row[ikey] = _stance_label_to_id(r)

        def _save():
            # Merge-on-write under an inter-process lock so concurrent runs
            # for different models don't clobber each other's columns.
            lock_path = out_file + ".lock"
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(lock_path, "w", encoding="utf-8") as lock_f:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
                latest_by_claim = {}
                if os.path.exists(out_file):
                    try:
                        with open(out_file, encoding="utf-8") as f:
                            for r in json.load(f):
                                latest_by_claim[r.get("claim", "")] = r
                    except Exception:
                        latest_by_claim = {}

                # Preserve existing columns from the latest on disk and update
                # only fields present in this worker's rows.
                for r in rows:
                    k = r.get("claim", "")
                    merged = dict(latest_by_claim.get(k, {}))
                    merged.update(r)
                    latest_by_claim[k] = merged

                merged_rows = []
                seen = set()
                for r in rows:
                    k = r.get("claim", "")
                    merged_rows.append(latest_by_claim.get(k, r))
                    seen.add(k)
                for k, v in latest_by_claim.items():
                    if k not in seen:
                        merged_rows.append(v)

                tmp = out_file + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(merged_rows, f, indent=2, ensure_ascii=False)
                os.replace(tmp, out_file)
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

        for i in tqdm(
            range(0, len(data), batch_size),
            desc=f"local-eval {lang_code}",
            disable=(lang_workers > 1),
        ):
            batch_rows = rows[i:i + batch_size]
            bc, be = claims[i:i + batch_size], evidences[i:i + batch_size]
            # Resume: only call each model for rows it hasn't predicted yet.
            need = {m: [j for j in range(len(batch_rows)) if not _done(batch_rows[j], m)]
                    for m in models}
            if not any(need.values()):
                continue  # whole batch already done (e.g. after a reboot)

            with ThreadPoolExecutor(max_workers=max(len(models), 1)) as ex:
                futures = {m: ex.submit(_dispatch, m, [bc[j] for j in need[m]],
                                        [be[j] for j in need[m]])
                           for m in models if need[m]}
                for m, fut in futures.items():
                    try:
                        out = fut.result()
                    except Exception as e:
                        print(f"  [warn] {m} batch failed: {type(e).__name__}: {str(e)[:120]}")
                        out = ([{"int_label": -1, "label": "ERROR", "score": 0.0, "scores": []}] * len(need[m])
                               if m in {"factiverse_local", "factiverse_xlmr_local"} else ["ERROR"] * len(need[m]))
                    for local_j, r in zip(need[m], out):
                        _assign(batch_rows[local_j], m, r)

            _save()  # checkpoint after every batch for reboot-resume

        _save()

        print(f"\n{'─'*60}")
        for m in models:
            display = (f"Ollama ({ollama_model})" if m == "qwen3-8b"
                       else _STANCE_DISPLAY[m])
            _, ikey = _pred_keys(m)
            labeled = [(gts[k], rows[k].get(ikey))
                       for k in range(len(rows))
                       if gts[k] is not None and rows[k].get(ikey) is not None]
            if not labeled:
                print(f"{display} [{lang_code}] - no ground-truth labels, skipping F1")
                continue
            g, p = map(list, zip(*labeled))
            _report_stance_f1(display, lang_code, g, p)
        print(f"  saved → {out_file}\n{'─'*60}")

    if lang_workers <= 1 or len(files) <= 1:
        for data_file in files:
            _run_single_file(data_file)
        return

    print(f"[local-eval] Parallelizing across languages with {lang_workers} workers")
    with ThreadPoolExecutor(max_workers=lang_workers) as ex:
        futures = {ex.submit(_run_single_file, data_file): data_file for data_file in files}
        for fut in as_completed(futures):
            data_file = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[local-eval] {data_file} failed: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run veracity prediction with specific models")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["qwen3-8b", "gpt52", "claude", "factiverse", "factiverse_local", "factiverse_xlmr_local", "openrouter", "all"],
        default=["all"],
        help="Specify which models to run. Options: qwen3-8b, gpt52, claude, factiverse, factiverse_local, factiverse_xlmr_local, openrouter, all. Default: all"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for local stance models (factiverse_local / factiverse_xlmr_local). Default: 8"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Restrict the local eval to a single language file ({lang}_{split}.json), e.g. --lang en"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test_jan_2026",
        help="Data split to process. Default: test_jan_2026"
    )
    parser.add_argument(
        "--retrieval-model",
        type=str,
        default="paraphrase-multilingual",
        choices=["paraphrase-multilingual", "bge-m3"],
        help="Retrieval model to use. Default: paraphrase-multilingual"
    )
    parser.add_argument(
        "--low-resource-only",
        action="store_true",
        help="Process only low-resource languages"
    )
    parser.add_argument(
        "--include-no-evidence",
        action="store_true",
        help="Include claims with no evidence in output"
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="qwen3:8b",
        help="Ollama model to use for qwen3-8b predictions. Default: qwen3:8b"
    )
    parser.add_argument(
        "--openrouter-model",
        type=str,
        default="google/gemma-4-31b-it:free",
        help="OpenRouter model to use. Default: google/gemma-4-31b-it:free"
    )
    parser.add_argument(
        "--no-pred-cache",
        action="store_true",
        help="Disable *_local_pred.json resume cache and recompute all local predictions from scratch"
    )
    parser.add_argument(
        "--lang-workers",
        type=int,
        default=1,
        help="Number of parallel language workers for local eval. Default: 1"
    )
    args = parser.parse_args()
    
    # Determine which models to run
    models_to_run = set(args.models)
    if "all" in models_to_run:
        models_to_run = {"qwen3-8b", "gpt52", "claude", "factiverse", "openrouter"}

    print(f"\n{'='*60}")
    print(f"Running ONLY these models: {', '.join(sorted(models_to_run))}")
    print(f"{'='*60}\n")

    # ── Local evaluation path ────────────────────────────────────────────────
    # Self-contained: reads claim+evidence from {lang}_{split}.json directly
    # (no remote stance_detection API). Handles factiverse_local (mmBERT) plus
    # gpt52 / claude / openrouter / qwen3-8b on the same local pairs. Runs before
    # the API-based flow so a local-only run needs no access token.
    local_requested = [m for m in LOCAL_MODELS if m in models_to_run]
    if local_requested:
        lang_workers = max(int(args.lang_workers), 1)
        local_gpu_models = {"factiverse_local", "factiverse_xlmr_local"}
        if lang_workers > 1 and any(m in local_gpu_models for m in local_requested):
            print("[warn] --lang-workers>1 with local GPU models can overcommit VRAM; forcing --lang-workers=1")
            lang_workers = 1
        run_local_eval(
            local_requested,
            split=args.split,
            batch_size=args.batch_size,
            openrouter_model=args.openrouter_model,
            ollama_model=args.ollama_model,
            lang=args.lang,
            use_pred_cache=not args.no_pred_cache,
            lang_workers=lang_workers,
        )
        models_to_run.difference_update(local_requested)
        if not models_to_run:
            sys.exit(0)

    input = "data/veracity_prediction_gcloud_gcloud/test_dec_2025.jsonl"
    access_token = get_access_token()
    count = 0
    missing_evidence = 0
    ISO639_FILE = {}
    langs = load_lang_codes()

    # Low resource languages - South Indian languages and other low resource languages
    low_resource_langs = {
        "ta",  # Tamil
        "te",  # Telugu
        "kn",  # Kannada
        "ml",  # Malayalam
        "bn",  # Bengali
        "gu",  # Gujarati
        "pa",  # Punjabi
        "or",  # Odia
        "hi",  # Hindi
        "ur",  # Urdu
        "am",  # Amharic
        "ha",  # Hausa
        "sw",  # Swahili
        "my",  # Burmese
        "th",  # Thai
        "vi",  # Vietnamese
        "fil", # Filipino
        "id",  # Indonesian
        "jv",  # Javanese
    }

    split = args.split
    retrieval_model = args.retrieval_model

    fact_checked_data = []
    cur_timestamp = pd.Timestamp.now().strftime("%Y%m%d%H")

    # Find all language files matching the pattern {lang}_{split}.json
    data_dir = "data/veracity_prediction_gcloud/"
    language_files = sorted(glob.glob(f"{data_dir}/*_{split}.json"))

    if not language_files:
        print(f"No files found matching pattern: *_{split}.json in {data_dir}")
        print("Available splits: test, train, dev, etc.")
        exit(1)

    print(f"Found {len(language_files)} language files to process")
    print(f"Running models: {', '.join(sorted(models_to_run))}")
    if "qwen3-8b" in models_to_run:
        print(f"Ollama model: {args.ollama_model}")
    print(f"Retrieval model: {retrieval_model}\n")
    
    # Process each language file
    for lang in langs.keys():
        # Extract language code from filename
        filename = f"{lang}_{split}.json"
        data_file = os.path.join(data_dir, filename)
        if not os.path.exists(data_file):
            print(f"File not found for language {lang}: {data_file}")
            continue
        lang_code = lang
        
        print(f"\n{'='*60}")
        print(f"Processing language: {lang_code} ({filename})")
        print(f"{'='*60}")
        
        data = load_json(data_file)
        print(f"Total items loaded for {lang_code}: {len(data)}")
        
        # Filter to include only low resource languages if specified
        if args.low_resource_only:
            data = [item for item in data if item.get("lang") in low_resource_langs]
        
        if len(data) == 0:
            print(f"No data found for language {lang_code}")
            continue
        
        # lang_distribution = Counter([item.get("lang") for item in data])
        # print("Language distribution:")
        # for lang, count_val in sorted(lang_distribution.items(), key=lambda x: x[1], reverse=True):
        #     print(f"  {lang}: {count_val}")
        
        run_llm_verification = any(model in models_to_run for model in ["qwen3-8b", "gpt52", "claude"])
        random.Random(42).shuffle(data)
        # sort data by language code
        data = sorted(data, key=lambda x: x.get("lang", ""))
        
        # Create output files for this language
        output_results_file = f"data/veracity_prediction_gcloud_gcloud/{lang_code}_{split}_results_debug_{retrieval_model}_{cur_timestamp}.jsonl"
        output_pred_file = f"data/veracity_prediction_gcloud_gcloud/{lang_code}_{split}_nli_pred_debug_{retrieval_model}_{cur_timestamp}.json"
        
        lang_fact_checked_data = []
        lang_count = 0
        lang_missing_evidence = 0
        
        with open(
            output_results_file,
            mode="w",
            encoding="utf-8",
        ) as f:
            # Process each claim one at a time
            for item in tqdm(data, desc=f"Processing {lang_code}"):
                lang_name = langs[lang_code]["name"]
                response = factiverse_verify(item["claim"], lang_code, access_token)
                
                if response.status_code == 200:
                    response_data = response.json()
                    # Check if we have evidence or if we should include anyway
                    if len(response_data["evidence"]) > 0 or args.include_no_evidence:
                        if len(response_data["evidence"]) == 0:
                            lang_missing_evidence += 1
                        
                        # Collect evidence
                        full_evidence = ""
                        for evidence in response_data["evidence"]:
                            if evidence.get("snippet", ""):
                                if evidence.get("rewritten_query", ""):
                                    full_evidence += evidence["rewritten_query"] + " "
                                full_evidence += evidence["snippet"] + " "
                        
                        # Run LLM predictions if needed
                        verified_data = {}
                        verified_data["claim"] = response_data["claim"]
                        verified_data["label"] = item["label"]
                        verified_data["lang"] = lang_code
                        
                        if run_llm_verification:
                            try:
                                # Run models sequentially on single claim
                                if "qwen3-8b" in models_to_run:
                                    try:
                                        ollama_result = predict_stance_ollama(
                                            claim=response_data["claim"],
                                            evidence=full_evidence,
                                            lang=lang_name,
                                            model=args.ollama_model
                                        )
                                        verified_data["ollama_label"] = ollama_result
                                    except Exception as e:
                                        error_msg = str(e)
                                        if "404" in error_msg or "not found" in error_msg.lower():
                                            print(f"\n⚠️  Ollama model '{args.ollama_model}' not found. Skipping qwen3-8b predictions.")
                                            print(f"   Error: {error_msg[:100]}")
                                        else:
                                            print(f"\n❌ qwen3-8b error: {error_msg[:100]}")
                                
                                if "gpt52" in models_to_run:
                                    try:
                                        gpt52_result = predict_stance_openai(
                                            claim=response_data["claim"],
                                            lang=lang_name,
                                            evidence=full_evidence,
                                            model="gpt-5.2"
                                        )
                                        verified_data["gpt5_label"] = gpt52_result
                                    except Exception as e:
                                        error_msg = str(e)
                                        if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                                            print(f"\n⚠️  GPT-5.2 model not available. Skipping GPT-5.2 predictions.")
                                        else:
                                            print(f"\n❌ GPT-5.2 error: {error_msg[:100]}")
                                
                                if "claude" in models_to_run:
                                    try:
                                        claude_result = predict_stance_openai(
                                            claim=response_data["claim"],
                                            lang=lang_name,
                                            evidence=full_evidence,
                                            model="claude-opus-4-6"
                                        )
                                        verified_data["claude-opus-4-6_label"] = claude_result
                                    except Exception as e:
                                        error_msg = str(e)
                                        if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                                            print(f"\n⚠️  Claude model not available. Skipping Claude predictions.")
                                        else:
                                            print(f"\n❌ Claude error: {error_msg[:100]}")

                                if "openrouter" in models_to_run:
                                    try:
                                        openrouter_result = predict_stance_openrouter(
                                            claim=response_data["claim"],
                                            evidence=full_evidence,
                                            lang=lang_name,
                                            model=args.openrouter_model
                                        )
                                        verified_data["openrouter_label"] = openrouter_result
                                    except Exception as e:
                                        error_msg = str(e)
                                        if "not found" in error_msg.lower() or "credential" in error_msg.lower():
                                            print(f"\n⚠️  OpenRouter API key not found. Skipping OpenRouter predictions.")
                                        else:
                                            print(f"\n❌ OpenRouter error: {error_msg[:100]}")

                            except Exception as e:
                                print(f"Exception processing claim '{response_data['claim'][:50]}...': {str(e)}")
                        
                        # Add Factiverse data
                        if "factiverse" in models_to_run:
                            verified_data["factiverse_response"] = response_data
                            verified_data["factiverse_score"] = response_data["finalScore"]
                            verified_data["factiverse_int_label"] = response_data["finalPrediction"]
                            verified_data["factiverse_label"] = response_data["finalLabelDescription"]
                        
                        lang_count += 1
                        f.write(json.dumps(verified_data) + "\n")
                        f.flush()
                        lang_fact_checked_data.append(verified_data)
                    else:
                        lang_missing_evidence += 1
                else:
                    print(f"API Error for claim: Status {response.status_code}")
                    print(f"Response: {response.text[:200]}")
        
        # Save language-specific predictions
        with open(
            output_pred_file,
            mode="w",
            encoding="utf-8",
        ) as f:
            json.dump(lang_fact_checked_data, f, indent=4)
        
        print(f"\n[{lang_code}] Processed: {lang_count}, Missing evidence: {lang_missing_evidence}")
        print(f"Results saved to:")
        print(f"  - {output_results_file}")
        print(f"  - {output_pred_file}")
        
        # Add to overall data
        fact_checked_data.extend(lang_fact_checked_data)
    
    # Final summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total languages processed: {len(language_files)}")
