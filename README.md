# FactCheck Editor

## Translate data
* To translate data use `translate_data.py`. 
    - Usage: 
        - Claim detection test split: `python -m scripts.translate_data --task claim_detection --split test`
        - Veracity prediction test split: `python -m scripts.translate_data --task veracity_prediction --split test`

## Setup
* Make a copy of the .env.example file
* Contact Factiverse for obtaining the values for 

## Evaluate claim detection
    - First run claim detection predictions using `python -m code.claim_detection.claim_detection`
    - Then run the plotting script `python -m scripts.claim_detection_plots`

## Evaluate veracity prediction
* To plot veracity prediction plots run 
    - First run the 