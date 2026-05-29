import csv
import json
import os
from typing import Dict, List
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import dotenv
import pandas as pd
import requests
from tqdm import tqdm
from src.veracity.llm_nli import (
    predict_stance_ollama,
    predict_stance_openai,
    predict_stance_ollama_batch,
    predict_stance_openai_batch,
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
        choices=["mistral", "gpt52", "claude", "factiverse", "all"],
        default=["all"],
        help="Specify which models to run. Options: mistral, gpt52, claude, factiverse, all. Default: all"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of claims to process per batch. Default: 10"
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
    args = parser.parse_args()
    
    # Determine which models to run
    models_to_run = set(args.models)
    if "all" in models_to_run:
        models_to_run = {"mistral", "gpt52", "claude", "factiverse"}
    
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
    batch_size = args.batch_size
    retrieval_model = args.retrieval_model
    
    fact_checked_data = []
    cur_timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    data_file = f"data/veracity_prediction/{lang}_{split}.json"
    data = load_json(data_file)
    
    # Filter to include only low resource languages if specified
    if args.low_resource_only:
        data = [item for item in data if item.get("lang") in low_resource_langs]
    
    lang_distribution = Counter([item.get("lang") for item in data])
    print("Language distribution:")
    for lang, count_val in sorted(lang_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lang}: {count_val}")
    
    print(f"\nRunning models: {', '.join(sorted(models_to_run))}")
    print(f"Batch size: {batch_size}")
    print(f"Retrieval model: {retrieval_model}\n")
    
    run_llm_verification = any(model in models_to_run for model in ["mistral", "gpt52", "claude"])
    random.Random(42).shuffle(data)
    # sort data by language code
    data = sorted(data, key=lambda x: x.get("lang", ""))

        with open(
            output_results_file,
            mode="w",
            encoding="utf-8",
        ) as f:
            # Process data in batches
            for batch_idx in tqdm(range(0, len(data), batch_size), desc=f"Processing {lang_code}"):
                batch = data[batch_idx:batch_idx + batch_size]
                batch_verified_data = []
                batch_claims = []
                batch_evidences = []
                batch_langs = []
                batch_items = []
                
                # First, get Factiverse responses for all items in the batch
                for item in batch:
                    lang = item["lang"]
                    lang_name = langs[lang]["name"]
                    response = factiverse_verify(item["claim"], lang, access_token)
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        if len(response_data["evidence"]) > 0:
                            # Collect evidence
                            full_evidence = ""
                            for evidence in response_data["evidence"]:
                                if evidence.get("snippet", ""):
                                    if evidence.get("rewritten_query", ""):
                                        full_evidence += evidence["rewritten_query"] + " "
                                    full_evidence += evidence["snippet"] + " "
                            
                            batch_claims.append(response_data["claim"])
                            batch_evidences.append(full_evidence)
                            batch_langs.append(lang_name)
                            batch_items.append({
                                "item": item,
                                "response_data": response_data,
                                "lang": lang,
                                "full_evidence": full_evidence
                            })
                        else:
                            lang_missing_evidence += 1
                    else:
                        pass
                
                # Now run LLM predictions in parallel for the batch
                if run_llm_verification and len(batch_claims) > 0:
                    try:
                        # Group by language for batch processing
                        lang_groups = {}
                        for idx, lang_name in enumerate(batch_langs):
                            if lang_name not in lang_groups:
                                lang_groups[lang_name] = {"claims": [], "evidences": [], "indices": []}
                            lang_groups[lang_name]["claims"].append(batch_claims[idx])
                            lang_groups[lang_name]["evidences"].append(batch_evidences[idx])
                            lang_groups[lang_name]["indices"].append(idx)
                        
                        # Initialize results storage
                        all_results = {
                            "mistral": [None] * len(batch_claims),
                            "gpt52": [None] * len(batch_claims),
                            "claude": [None] * len(batch_claims),
                        }
                        
                        # Process each language group
                        for lang_name, lang_data in lang_groups.items():
                            futures = {}
                            with ThreadPoolExecutor(max_workers=3) as executor:
                                if "mistral" in models_to_run:
                                    futures["mistral"] = executor.submit(
                                        predict_stance_ollama_batch,
                                        claims=lang_data["claims"],
                                        evidences=lang_data["evidences"],
                                        lang=lang_name
                                    )
                                if "gpt52" in models_to_run:
                                    futures["gpt52"] = executor.submit(
                                        predict_stance_openai_batch,
                                        claims=lang_data["claims"],
                                        evidences=lang_data["evidences"],
                                        lang=lang_name,
                                        open_ai_utils=OpenAIUtils(),
                                        model="gpt-5.2"
                                    )
                                if "claude" in models_to_run:
                                    futures["claude"] = executor.submit(
                                        predict_stance_openai_batch,
                                        claims=lang_data["claims"],
                                        evidences=lang_data["evidences"],
                                        lang=lang_name,
                                        open_ai_utils=OpenAIUtils(),
                                        model="claude-opus-4-6"
                                    )
                                
                                # Wait for all to complete and store results
                                for model_name, future in futures.items():
                                    model_results = future.result()
                                    for local_idx, global_idx in enumerate(lang_data["indices"]):
                                        all_results[model_name][global_idx] = model_results[local_idx]
                        
                        # Now process the results for each item
                        for idx, batch_item in enumerate(batch_items):
                            verified_data = {}
                            item = batch_item["item"]
                            response_data = batch_item["response_data"]
                            lang = batch_item["lang"]
                            
                            # Add model predictions
                            if "mistral" in models_to_run:
                                verified_data["ollama_label"] = all_results["mistral"][idx]
                            
                            if "gpt52" in models_to_run:
                                verified_data["gpt5_label"] = all_results["gpt52"][idx]
                            
                            if "claude" in models_to_run:
                                verified_data["claude-opus-4-6_label"] = all_results["claude"][idx]
                            
                            # Add common data
                            verified_data["claim"] = response_data["claim"]
                            if "factiverse" in models_to_run:
                                verified_data["factiverse_response"] = response_data
                                verified_data["factiverse_score"] = response_data["finalScore"]
                                verified_data["factiverse_int_label"] = response_data["finalPrediction"]
                                verified_data["factiverse_label"] = response_data["finalLabelDescription"]
                            
                            verified_data["label"] = item["label"]
                            verified_data["lang"] = lang
                            
                            lang_count += 1
                            
                            f.write(json.dumps(verified_data) + "\n")
                            f.flush()
                            lang_fact_checked_data.append(verified_data)
                    
                    except Exception as e:
                        print(f"Exception occurred while processing batch: {str(e)}")
                        import traceback
                        traceback.print_exc()
                else:
                    # No LLM verification, just save Factiverse results
                    for batch_item in batch_items:
                        verified_data = {}
                        item = batch_item["item"]
                        response_data = batch_item["response_data"]
                        lang = batch_item["lang"]
                        
                        verified_data["claim"] = response_data["claim"]
                        if "factiverse" in models_to_run:
                            verified_data["factiverse_response"] = response_data
                            verified_data["factiverse_score"] = response_data["finalScore"]
                            verified_data["factiverse_int_label"] = response_data["finalPrediction"]
                            verified_data["factiverse_label"] = response_data["finalLabelDescription"]
                        
                        verified_data["label"] = item["label"]
                        verified_data["lang"] = lang
                        
                        lang_count += 1
                        
                        f.write(json.dumps(verified_data) + "\n")
                        f.flush()
                        lang_fact_checked_data.append(verified_data)
        
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
