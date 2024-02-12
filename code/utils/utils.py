import requests
import json
import os

from typing import Dict


def get_access_token() -> str:
    """Get access token from Factiverse API.

    Exception: If token request fails.

    Returns:
        Access token.
    """
    client_id = os.getenv("AUTH0_CLIENT_ID")
    client_secret = os.getenv("AUTH0_SECRET")
    token_url = os.getenv("AUTH0_TOKEN_URL")
    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "audience": os.getenv("AUTH0_AUDIENCE"),
    }
    response = requests.post(token_url, data=payload)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(
            f"Failed to obtain token: {response.status_code} {response.text}"
        )


def load_json(file_name: str):
    """Loads the claim data from the claim detection dataset."""
    with open(file_name, "r") as json_file:
        return json.load(json_file)

def load_lang_codes() -> Dict[str, Dict[str, str]]:
    lang_codes = {}
    with open("code/utils/lang_codes.json", "r") as iso639_file:
        lang_codes = json.load(iso639_file)
    return lang_codes