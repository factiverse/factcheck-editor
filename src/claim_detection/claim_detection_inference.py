"""Detects sentences with claims"""

from functools import lru_cache
from typing import List, Tuple

from pathlib import Path
import logging
import json
import torch
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification
from transformers.trainer_utils import set_seed
from transformers.tokenization_utils_base import BatchEncoding
import numpy as np
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class BERTClaimPredictor:
    def __init__(
        self,
        model_path: str,
        model_type: str,
        cache_dir: str,
    ) -> None:
        """The claim predictor.

        Class to use Huggingface BERT model via simpletransformers library
        to detect claims.

        Args:
            model_path: Path to the trained Huggingface BERT model.
            model_type: Type of the model. For example, roberta or bert.
            cache_dir: cache directory.
        """
        set_seed(0)
        self.model_path = model_path
        self.model_type = model_type
        self.softmax = torch.nn.Softmax(dim=1)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(
            "FacebookAI/xlm-roberta-large",
            cache_dir=cache_dir,
            use_fast=True,
        )
        self._model = XLMRobertaForSequenceClassification.from_pretrained(
            self.model_path, ignore_mismatched_sizes=True
        ).to(self._device)

    @staticmethod
    def sigmoid(x):
        """Compute sigmoid function.

        Args:
            x: input value.

        Returns:
            sigmoid value of x.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def logit(p):
        """Compute logit function.

        Args:
            p: input value.

        Returns:
            logit value of p.
        """
        # Ensure p is not exactly 0 or 1 for stability
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    @staticmethod
    def scale_probabilities_logit(probs, scale_factor=10):
        """Scale probabilities using logit function.

        Args:
            probs: input probabilities.
            scale_factor: scale factor.

        Returns:
            scaled probabilities.
        """
        logits = BERTClaimPredictor.logit(probs)
        scaled_logits = scale_factor * logits
        return BERTClaimPredictor.sigmoid(scaled_logits)

    def _encode(self, claims: Tuple[str]) -> BatchEncoding:
        """Encodes the input into a Batch.

        Tokenize and collate a number of single inputs, adding special tokens
        and padding.

        Args:
            input: Input to encode.

        Returns:
            Batch: Input IDs, attention masks, token type IDs
        """
        inputs = self._tokenizer.batch_encode_plus(
            [claim for claim in claims],
            add_special_tokens=True,
            return_token_type_ids=True,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        return inputs

    @lru_cache(maxsize=10000)
    def predict(self, claims: Tuple[str]) -> Tuple[List[int], List[List[float]]]:
        """Predicts if given sentences are claims.

        staticmethods moved to enable lru_cache.

        Args:
            claims: List of string claims.

        Returns:
            List of predictions and softmax scores for the predictions
            corresponding to claims.
        """
        claims_list = list(claims)
        inputs = self._encode(claims_list).to(self._device)
        with torch.no_grad():
            logits = self._model(**inputs).logits
            softmax_score = self.softmax(logits).tolist()
            predictions = (np.argmax(softmax_score, axis=1).flatten()).tolist()
            return (
                predictions,
                BERTClaimPredictor.scale_probabilities_logit(
                    np.array(softmax_score)
                ).tolist(),
            )


def init(model_path):
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    
    model = BERTClaimPredictor(
        model_path,
        "unquantized",
        model_path,
    )
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    """
    logging.info("claim_detection_model: request received")
    data = json.loads(raw_data)["data"]
    sentences: list[str] = data

    preds, scores = model.predict(tuple(sentences))

    logging.info("Request processed")
    logging.info(scores)

    return preds, scores

if __name__ == "__main__":
    init("claim_detection_unquantized")
    sentences = ["Eating after 7 PM will make you gain weight.", "What and how much you eat will determine weight gain or loss."]
    preds, scores = model.predict(tuple(sentences))
    print("Predictions:", preds)
    print("Scores:", scores)
    