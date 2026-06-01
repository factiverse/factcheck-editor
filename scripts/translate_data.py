import json
import copy
from deep_translator import GoogleTranslator
from src.utils.utils import load_json



import argparse
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(args: argparse.Namespace) -> None:
    """Main function.

    Args:
        args: Arguments from command-line call.
    """
    logger.info("Starting main function.")
    logger.debug(f"Arguments: {args}")
    
    
def parse_args() -> argparse.Namespace:
    """Parses arguments from command-line call.

    Returns:
        Arguments from command-line call.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        choices=["claim_detection", "veracity_prediction"],
        help="Which task to translate the data for.",
        default="claim_detection",
    )
    parser.add_argument(
        "-s",
        "--split",
        dest="split",
        choices=["train", "dev", "test", "fv_claim_test"],
        help="Which split to translate.",
        default="test",
    )
    return parser.parse_args()

def google_translate_text(text: str, target_language: str) -> str:
    """Translate text into target_language.

    Args:
        text: Source text.
        target_language: Target language.

    Returns:
        Translated text.
    """
    translator = GoogleTranslator(target=target_language)
    return translator.translate(text)

if __name__ == "__main__":
    args = parse_args()
    task = args.task
    split = args.split
    source_lang = "en"

    # Support both .json and .jsonl input files
    json_path = f"data/{task}/{source_lang}_{split}.json"
    jsonl_path = f"data/{task}/{source_lang}_{split}.jsonl"
    import os
    if os.path.exists(jsonl_path):
        input_path = jsonl_path
        use_jsonl = True
    else:
        input_path = json_path
        use_jsonl = False

    if use_jsonl:
        with open(input_path, "r") as f:
            data = [json.loads(line) for line in f if line.strip()]
    else:
        data = load_json(input_path)

    # All target languages for Google Translate
    TARGET_LANGS = sorted({
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
    })

    for lang in TARGET_LANGS:
        if lang == source_lang:  # Skip the source language.
            continue

        # Skip if output already exists
        ext = ".jsonl" if use_jsonl else ".json"
        out_path = f"data/{task}/{lang}_{split}{ext}"
        if os.path.exists(out_path):
            print(f"Already exists: {out_path} — skipping {lang}")
            continue

        if not GoogleTranslator().is_language_supported(language=lang):
            print(f"Language {lang} not supported by Google Translate — skipping")
            continue
        lang_data = []
        print(f"Translating data from English to {lang}")
        for row in data:
            # Detect which key holds the text to translate
            if "claim" in row:
                text_key = "claim"
            elif "sentence" in row:
                text_key = "sentence"
            elif "text" in row:
                text_key = "text"
            else:
                logger.warning(f"Row has no 'claim', 'sentence', or 'text' key: {row}")
                continue
            try:
                google_translation = google_translate_text(row[text_key], lang)
            except Exception as e:
                logger.error(e)
                logger.warning(f"Failed to translate {row[text_key]} to {lang}")
                continue
            new_row = copy.deepcopy(row)
            lang_data.append(new_row)
            new_row[text_key] = google_translation
            # if len(lang_data) == 10:
            #     break

        if use_jsonl:
            with open(out_path, "w") as f:
                for row in lang_data:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            with open(out_path, "w") as json_file:
                json.dump(lang_data, json_file, indent=4)
        # break