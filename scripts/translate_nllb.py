"""Translate a JSONL claim-detection file to multiple languages using NLLB-200.

Uses facebook/nllb-200-distilled-600M for fast GPU inference.
Outputs one JSONL file per target language in the same directory.

Usage:
    uv run python scripts/translate_nllb.py \
        --input data/claim_detection/en_clean_test.jsonl \
        --langs fr de es ar hi zh ja ko ru \
        --batch-size 64

    # All XLM-R languages (intersection with NLLB):
    uv run python scripts/translate_nllb.py \
        --input data/claim_detection/en_clean_test.jsonl \
        --all-xlmr \
        --batch-size 64
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── ISO 639-1 → NLLB BCP-47 mapping ─────────────────────────────────────────
# NLLB uses flores-200 codes like "fra_Latn", "deu_Latn", etc.
# This maps the ISO 639-1 codes used in lang_codes.json to NLLB codes.
ISO_TO_NLLB = {
    "af": "afr_Latn", "am": "amh_Ethi", "ar": "arb_Arab", "as": "asm_Beng",
    "az": "azj_Latn", "be": "bel_Cyrl", "bg": "bul_Cyrl", "bn": "ben_Beng",
    "bs": "bos_Latn", "ca": "cat_Latn", "cs": "ces_Latn", "cy": "cym_Latn",
    "da": "dan_Latn", "de": "deu_Latn", "el": "ell_Grek", "en": "eng_Latn",
    "eo": "epo_Latn", "es": "spa_Latn", "et": "est_Latn", "eu": "eus_Latn",
    "fa": "pes_Arab", "fi": "fin_Latn", "fr": "fra_Latn", "ga": "gle_Latn",
    "gd": "gla_Latn", "gl": "glg_Latn", "gu": "guj_Gujr", "ha": "hau_Latn",
    "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn", "ht": "hat_Latn",
    "hu": "hun_Latn", "hy": "hye_Armn", "id": "ind_Latn", "ig": "ibo_Latn",
    "is": "isl_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "jv": "jav_Latn",
    "ka": "kat_Geor", "kk": "kaz_Cyrl", "km": "khm_Khmr", "kn": "kan_Knda",
    "ko": "kor_Hang", "ky": "kir_Cyrl", "lo": "lao_Laoo", "lt": "lit_Latn",
    "lv": "lvs_Latn", "mg": "plt_Latn", "mi": "mri_Latn", "mk": "mkd_Cyrl",
    "ml": "mal_Mlym", "mn": "khk_Cyrl", "mr": "mar_Deva", "ms": "zsm_Latn",
    "mt": "mlt_Latn", "my": "mya_Mymr", "ne": "npi_Deva", "nl": "nld_Latn",
    "no": "nob_Latn", "nb": "nob_Latn", "nn": "nno_Latn", "ny": "nya_Latn",
    "or": "ory_Orya", "pa": "pan_Guru", "pl": "pol_Latn", "ps": "pbt_Arab",
    "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "rw": "kin_Latn",
    "sd": "snd_Arab", "si": "sin_Sinh", "sk": "slk_Latn", "sl": "slv_Latn",
    "sn": "sna_Latn", "so": "som_Latn", "sq": "als_Latn", "sr": "srp_Cyrl",
    "su": "sun_Latn", "sv": "swe_Latn", "sw": "swh_Latn", "ta": "tam_Taml",
    "te": "tel_Telu", "tg": "tgk_Cyrl", "th": "tha_Thai", "tl": "tgl_Latn",
    "tr": "tur_Latn", "uk": "ukr_Cyrl", "ur": "urd_Arab", "uz": "uzn_Latn",
    "vi": "vie_Latn", "xh": "xho_Latn", "yi": "ydd_Hebr", "yo": "yor_Latn",
    "zh": "zho_Hans", "zu": "zul_Latn", "fil": "tgl_Latn", "iw": "heb_Hebr",
    "ln": "lin_Latn", "lg": "lug_Latn", "om": "gaz_Latn", "ti": "tir_Ethi",
    "tk": "tuk_Latn", "tt": "tat_Cyrl", "ug": "uig_Arab", "wo": "wol_Latn",
    "ak": "aka_Latn", "ay": "ayr_Latn", "bm": "bam_Latn", "dv": "div_Thaa",
    "ee": "ewe_Latn", "gn": "grn_Latn", "lb": "ltz_Latn", "sm": "smo_Latn",
    "st": "sot_Latn", "ts": "tso_Latn",
    # Fallbacks for langs not natively in NLLB — use closest relative:
    "br": "fra_Latn",   # Breton → French
    "fy": "nld_Latn",   # West Frisian → Dutch
    "ku": "ckb_Arab",   # Kurdish → Central Kurdish (Sorani)
    "la": "ita_Latn",   # Latin → Italian
    "sa": "hin_Deva",   # Sanskrit → Hindi
    "co": "ita_Latn",   # Corsican → Italian
    "qu": "spa_Latn",   # Quechua → Spanish (closest high-resource)
}

# All target languages for multilingual evaluation.
TARGET_LANGS = {
    "af", "ak", "am", "ar", "as", "ay", "az", "be", "bg", "bm", "bn", "bs",
    "ca", "co", "cs", "cy", "da", "de", "dv", "ee", "el", "en", "eo", "es",
    "et", "eu", "fa", "fi", "fr", "fy", "ga", "gd", "gl", "gn", "gu", "ha",
    "hi", "hr", "ht", "hu", "hy", "id", "ig", "is", "it", "iw", "ja", "ka",
    "kk", "km", "kn", "ko", "ku", "ky", "la", "lb", "lg", "ln", "lo", "lt",
    "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl",
    "no", "ny", "om", "or", "pa", "pl", "ps", "pt", "qu", "ro", "ru", "rw",
    "sa", "sd", "si", "sk", "sl", "sm", "sn", "so", "sq", "sr", "st", "su",
    "sv", "sw", "ta", "te", "tg", "th", "ti", "tk", "tl", "tr", "ts", "tt",
    "ug", "uk", "ur", "uz", "vi", "xh", "yi", "yo", "zu",
}


def get_target_langs(args):
    """Return list of ISO codes to translate to."""
    if args.all_xlmr:
        # All target langs, minus English
        return sorted(lang for lang in TARGET_LANGS if lang != "en")
    return [l for l in args.langs if l != "en"]


def load_data(path: str):
    """Load JSONL (one JSON per line) or JSON array."""
    with open(path, "r") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]


def get_text_key(row: dict) -> str:
    for k in ("text", "claim", "sentence"):
        if k in row:
            return k
    raise KeyError(f"No text field found in row: {list(row.keys())}")


def translate_batch(texts, tokenizer, model, tgt_nllb_code, max_length=256):
    """Translate a batch of texts to the target language."""
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    ).to(model.device)
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_nllb_code)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tgt_id,
            max_new_tokens=max_length,
            max_length=None,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Translate JSONL with NLLB-200")
    parser.add_argument("--input", required=True, help="Input JSONL/JSON file")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--langs", nargs="+", default=[],
                        help="Target ISO 639-1 language codes")
    parser.add_argument("--all-xlmr", action="store_true",
                        help="Translate to all XLM-R supported languages")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M",
                        help="NLLB model to use")
    args = parser.parse_args()

    target_langs = get_target_langs(args)
    if not target_langs:
        print("No target languages specified. Use --langs or --all-xlmr")
        return

    # Determine output directory
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_data(args.input)
    text_key = get_text_key(data[0])
    texts = [row[text_key] for row in data]
    print(f"Loaded {len(texts)} rows from {args.input} (text_key={text_key!r})")

    # Load model
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"Model on {device}")

    # Filter to supported languages
    unsupported = [l for l in target_langs if l not in ISO_TO_NLLB]
    if unsupported:
        print(f"Skipping {len(unsupported)} langs not in NLLB: {unsupported}")
        target_langs = [l for l in target_langs if l in ISO_TO_NLLB]

    print(f"Translating to {len(target_langs)} languages: {target_langs}")

    # Translate per language
    for lang in target_langs:
        nllb_code = ISO_TO_NLLB[lang]
        # Derive output filename: en_clean_test.jsonl → fr_clean_test.jsonl
        stem = input_path.stem
        if stem.startswith("en_"):
            out_stem = f"{lang}_{stem[3:]}"
        else:
            out_stem = f"{lang}_{stem}"
        out_path = output_dir / f"{out_stem}{input_path.suffix}"

        # Skip if already translated
        if out_path.exists():
            print(f"  [{lang}] Already exists: {out_path} — skipping")
            continue

        print(f"  [{lang}] Translating → {out_path} ...")
        translated_texts = []
        for i in tqdm(range(0, len(texts), args.batch_size), desc=f"  {lang}", leave=False):
            batch = texts[i:i + args.batch_size]
            translated = translate_batch(batch, tokenizer, model, nllb_code)
            translated_texts.extend(translated)

        # Write output
        with open(out_path, "w") as f:
            for row, trans in zip(data, translated_texts):
                out_row = dict(row)
                out_row[text_key] = trans
                out_row["lang"] = lang
                f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print("\nDone!")


if __name__ == "__main__":
    main()
