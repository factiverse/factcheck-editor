import json
import copy
from deep_translator import GoogleTranslator, DeeplTranslator
from code.utils.utils import load_json, load_lang_codes



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
        choices=["train", "dev", "test"],
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
    data = load_json(f"data/{task}/{source_lang}_{split}.json")
    lang_codes = load_lang_codes()

    for lang in lang_codes.keys():
        if (
            lang == source_lang # Skip the source language.
            or not GoogleTranslator().is_language_supported(language=lang)
        ):
            continue
        lang_data = []
        print(f"Translating data from English to {lang}")
        for row in data:
            try:
                google_translation = google_translate_text(row["claim"], lang)
            except Exception as e:
                logger.error(e)
                logger.warning(f"Failed to translate {row['claim']} to {lang}")
                continue
            new_row = copy.deepcopy(row)
            lang_data.append(new_row)
            new_row["claim"] = google_translation
        with open(f"data/{task}/{lang}_{split}.json", "w") as json_file:
            json.dump(lang_data, json_file, indent=4)