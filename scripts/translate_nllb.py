"""Translate claim/veracity data files to multiple languages using NLLB-200.

Uses facebook/nllb-200-distilled-600M for fast GPU inference.
Supports both JSONL and JSON-array input files.
Outputs one translated file per target language in the same directory.

Usage:
    # Auto-resolve input from task/split:
    uv run python scripts/translate_nllb.py \
        --task claim_detection \
        --split test \
        --langs fr de es ar hi zh ja ko ru

    uv run python scripts/translate_nllb.py \
        --task veracity \
        --split test \
        --langs fr de es ar hi zh ja ko ru

    uv run python scripts/translate_nllb.py \
        --input data/claim_detection/en_clean_test.jsonl \
        --langs fr de es ar hi zh ja ko ru \
        --batch-size 64

    # All XLM-R languages (intersection with NLLB):
    uv run python scripts/translate_nllb.py \
        --input data/claim_detection/en_clean_test.jsonl \
        --all-lang-codes \
        --batch-size 64

    # Veracity prediction test split (translates claim + evidence):
    uv run python scripts/translate_nllb.py \
        --input data/veracity_prediction/en_test.json \
        --langs fr de es ar hi zh ja ko ru \
        --batch-size 32
"""

import argparse
import json
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


def load_lang_codes(path: Path):
    """Load ISO language codes from lang_codes.json and drop English."""
    with open(path, "r") as f:
        data = json.load(f)
    return sorted(lang for lang in data.keys() if lang != "en")


def get_target_langs(args):
    """Return list of ISO codes to translate to."""
    if args.all_lang_codes or args.all_xlmr:
        # Backward compatible: --all-xlmr now behaves like --all-lang-codes.
        return load_lang_codes(Path(args.lang_codes_file))
    return [l for l in args.langs if l != "en"]


def load_data(path: str):
    """Load JSONL (one JSON per line) or JSON array.

    Returns:
        tuple[list[dict], str]: Data rows and input format ("jsonl" or "json").
    """
    with open(path, "r") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f), "json"
        else:
            return [json.loads(line) for line in f if line.strip()], "jsonl"


def resolve_input_path(task: str, split: str):
    """Resolve input file from task/split using known naming conventions."""
    task_dir_map = {
        "claim_detection": "data/claim_detection",
        "veracity": "data/veracity_prediction",
    }
    task_dir = Path(task_dir_map[task])

    if task == "claim_detection":
        candidates = [
            task_dir / f"en_clean_{split}.jsonl",
            task_dir / f"en_{split}.jsonl",
            task_dir / f"en_{split}.json",
        ]
    else:
        candidates = [
            task_dir / f"en_{split}.json",
            task_dir / f"en_{split}.jsonl",
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not resolve input file for task/split. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


def get_text_keys(row: dict, requested_text_keys):
    """Return text keys to translate for a row schema.

    If --text-keys is provided, only those keys are used (and validated).
    Otherwise, all known text fields present in the row are translated.
    """
    candidate_keys = ("text", "claim", "sentence", "evidence")
    if requested_text_keys:
        missing = [k for k in requested_text_keys if k not in row]
        if missing:
            raise KeyError(
                f"Requested text key(s) not found: {missing}; available keys: {list(row.keys())}"
            )
        return requested_text_keys

    detected = [k for k in candidate_keys if k in row]
    if not detected:
        raise KeyError(f"No text field found in row: {list(row.keys())}")
    return detected


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
    parser.add_argument(
        "--task",
        choices=["claim_detection", "veracity"],
        default="claim_detection",
        help="Task to translate (used to auto-resolve input when --input is omitted)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Data split name used with --task when --input is omitted",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional explicit input JSONL/JSON file; overrides --task/--split resolution",
    )
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--langs", nargs="+", default=[],
                        help="Target ISO 639-1 language codes")
    parser.add_argument(
        "--all-lang-codes",
        action="store_true",
        help="Translate to all languages from src/utils/lang_codes.json",
    )
    parser.add_argument("--all-xlmr", action="store_true",
                        help="Deprecated alias for --all-lang-codes")
    parser.add_argument(
        "--lang-codes-file",
        default=str(Path(__file__).resolve().parent.parent / "src" / "utils" / "lang_codes.json"),
        help="Path to lang_codes.json used by --all-lang-codes",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M",
                        help="NLLB model to use")
    parser.add_argument(
        "--text-keys",
        nargs="+",
        default=None,
        help="Optional text fields to translate (e.g. --text-keys claim evidence). "
        "Default: auto-detect from first row.",
    )
    args = parser.parse_args()

    target_langs = get_target_langs(args)
    if not target_langs:
        print("No target languages specified. Use --langs or --all-lang-codes")
        return

    # Determine input/output paths
    input_path = Path(args.input) if args.input else resolve_input_path(args.task, args.split)
    print(f"Using input: {input_path}")
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data, input_format = load_data(str(input_path))
    text_keys = get_text_keys(data[0], args.text_keys)
    print(f"Loaded {len(data)} rows from {input_path} (text_keys={text_keys!r}, format={input_format})")

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
        out_rows = [dict(row) for row in data]
        for text_key in text_keys:
            texts = [str(row.get(text_key, "")) for row in data]
            translated_texts = []
            for i in tqdm(range(0, len(texts), args.batch_size), desc=f"  {lang}:{text_key}", leave=False):
                batch = texts[i:i + args.batch_size]
                translated = translate_batch(batch, tokenizer, model, nllb_code)
                translated_texts.extend(translated)

            for out_row, trans in zip(out_rows, translated_texts):
                out_row[text_key] = trans

        for out_row in out_rows:
            out_row["lang"] = lang

        # Preserve the same file format as input.
        with open(out_path, "w") as f:
            if input_format == "json":
                json.dump(out_rows, f, ensure_ascii=False, indent=2)
            else:
                for row in out_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nDone!")


if __name__ == "__main__":
    main()
