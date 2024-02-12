# FactCheck Editor

## Translate data
* To translate data use `translate_data.py`. 
    - Usage: 
        - Claim detection test split: `python -m scripts.translate_data --task claim_detection --split test`
        - Veracity prediction test split: `python -m scripts.translate_data --task veracity_prediction --split test`

## Setup
* Make a copy of the .env.example file in `.env``
* Contact Factiverse for obtaining the values for `SERVER_ENDPOINT`, `AUTH0_AUDIENCE`, `AUTH0_TOKEN_URL`, `AUTH0_CLIENT_ID`, `AUTH0_SECRET`
* Setup ollama following guide at ollama.ai and specify the endpoint for `OLLAMA_HOST`
* Specify OpenAI credentials deployed in Azure for `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_MODELID`

## Evaluate claim detection
* First run claim detection predictions using `python -m code.claim_detection.claim_detection`
* Then run the plotting script `python -m scripts.claim_detection_plots`

## Evaluate veracity prediction
* First run veracity prediction: `python -m code.veracity.veracity_prediction`
* Then run veracity eval: `python -m code.veracity.nli_eval`
* Then run the plotting script: `python -m scripts.veracity_plots`