"""Detects sentences with claims"""

from functools import lru_cache
from typing import List, Tuple

from pathlib import Path
import logging
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
        inference_batch_size: int = 8,
    ) -> None:
        """The claim predictor.

        Args:
            model_path: Path to the trained Huggingface BERT model.
            model_type: Type of the model. For example, roberta or bert.
            cache_dir: cache directory.
            inference_batch_size: Forward-pass sub-batch size used by
                ``predict()``. Independent of training's
                ``per_device_eval_batch_size`` (which lives in model_args.json
                and is consumed by HF Trainer at training time, not here).
                Lower this if you OOM, raise it for throughput. For XLM-R-XL
                on H100 80GB, 8 is conservative; 16 usually fits.
        """
        set_seed(0)
        self.model_path = model_path
        self.model_type = model_type
        self.inference_batch_size = inference_batch_size
        self.softmax = torch.nn.Softmax(dim=1)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Tokeniser: use whatever the saved checkpoint shipped with (so XL,
        # XXL, large all just work). Falls back to xlm-roberta-large if the
        # checkpoint dir doesn't include a tokenizer.
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, cache_dir=cache_dir, use_fast=True,
            )
        except (OSError, ValueError):
            self._tokenizer = AutoTokenizer.from_pretrained(
                "FacebookAI/xlm-roberta-large", cache_dir=cache_dir, use_fast=True,
            )
        # AutoModelForSequenceClassification dispatches to the right HF class
        # based on the checkpoint's config.json — critical for XLM-R-XL, which
        # needs XLMRobertaXLForSequenceClassification (pre-norm), not the
        # XLMRobertaForSequenceClassification used for base/large (post-norm).
        # Load weights directly in bf16 to halve memory: XLM-R-XL is 3.5B
        # params, so fp32 weights alone are ~14GB; bf16 brings that to 7GB
        # and shrinks all forward-pass activations by 2x.
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
        ).to(self._device)
        self._model.eval()

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
        inputs = self._tokenizer(
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
        """Predicts if given sentences are claims (chunked to bound GPU mem)."""
        claims_list = list(claims)
        all_softmax_scores: List[List[float]] = []
        all_predictions: List[int] = []
        with torch.no_grad():
            for start in range(0, len(claims_list), self.inference_batch_size):
                chunk = claims_list[start:start + self.inference_batch_size]
                inputs = self._encode(chunk).to(self._device)
                logits = self._model(**inputs).logits.float()  # cast back to fp32 for stable softmax
                softmax_score = self.softmax(logits).tolist()
                all_softmax_scores.extend(softmax_score)
                all_predictions.extend(
                    np.argmax(softmax_score, axis=1).flatten().tolist()
                )
        return (
            all_predictions,
            BERTClaimPredictor.scale_probabilities_logit(
                np.array(all_softmax_scores)
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
    