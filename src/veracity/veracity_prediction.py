import csv
import json
import os
import glob
from typing import Dict, List
import argparse

import dotenv
import pandas as pd
import requests
from tqdm import tqdm
from src.veracity.llm_nli import (
    predict_stance_ollama,
    predict_stance_openai,
    predict_stance_openrouter,
)
from src.llm_utils.openai_utils import OpenAIUtils
from collections import Counter
from src.utils.utils import get_access_token, load_lang_codes, load_json
import random


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run veracity prediction with specific models")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["mistral", "gpt52", "claude", "factiverse", "openrouter", "all"],
        default=["all"],
        help="Specify which models to run. Options: mistral, gpt52, claude, factiverse, openrouter, all. Default: all"
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
        default="mistral",
        help="Ollama model to use for mistral predictions. Default: mistral"
    )
    parser.add_argument(
        "--openrouter-model",
        type=str,
        default="google/gemma-4-31b-it:free",
        help="OpenRouter model to use. Default: google/gemma-4-31b-it:free"
    )
    args = parser.parse_args()
    
    # Determine which models to run
    models_to_run = set(args.models)
    if "all" in models_to_run:
        models_to_run = {"mistral", "gpt52", "claude", "factiverse", "openrouter"}

    print(f"\n{'='*60}")
    print(f"Running ONLY these models: {', '.join(sorted(models_to_run))}")
    print(f"{'='*60}\n")

    input = "data/veracity_prediction/test_dec_2025.jsonl"
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
    data_dir = "data/veracity_prediction"
    language_files = sorted(glob.glob(f"{data_dir}/*_{split}.json"))

    if not language_files:
        print(f"No files found matching pattern: *_{split}.json in {data_dir}")
        print("Available splits: test, train, dev, etc.")
        exit(1)

    print(f"Found {len(language_files)} language files to process")
    print(f"Running models: {', '.join(sorted(models_to_run))}")
    if "mistral" in models_to_run:
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
        
        run_llm_verification = any(model in models_to_run for model in ["mistral", "gpt52", "claude"])
        random.Random(42).shuffle(data)
        # sort data by language code
        data = sorted(data, key=lambda x: x.get("lang", ""))
        
        # Create output files for this language
        output_results_file = f"data/veracity_prediction/{lang_code}_{split}_results_debug_{retrieval_model}_{cur_timestamp}.jsonl"
        output_pred_file = f"data/veracity_prediction/{lang_code}_{split}_nli_pred_debug_{retrieval_model}_{cur_timestamp}.json"
        
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
                                if "mistral" in models_to_run:
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
                                            print(f"\n⚠️  Ollama model '{args.ollama_model}' not found. Skipping Mistral predictions.")
                                            print(f"   Error: {error_msg[:100]}")
                                        else:
                                            print(f"\n❌ Mistral error: {error_msg[:100]}")
                                
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
