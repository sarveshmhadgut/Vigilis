import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score

from utils.logger import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class BertProcessor:
    """
    Handles log classification using a pre-trained BERT model.

    Responsibilities:
        - Load a pre-trained SentenceTransformer model and a classifier (pickled).
        - Generate embeddings for log messages.
        - Predict the category using the classifier and map it to a human-readable label.
        - Handle low-confidence predictions by returning 'Unclassified'.
    """

    def __init__(self) -> None:
        """
        Initializes the BertProcessor by loading the pre-trained model and classifier.
        """
        self.LABEL_MAP: Dict[int, str] = {
            0: "Critical Error",
            1: "Error",
            2: "HTTP Status",
            3: "Resource Usage",
            4: "Security Alert",
        }
        self.transformer: Any = None
        self.clf: Any = None
        self._load_models()

    def _load_models(self) -> None:
        """
        Loads the SentenceTransformer and the pickled classifier.

        Raises:
            FileNotFoundError: If the model pickle file does not exist.
            Exception: If loading the models fails.
        """
        try:
            models_dir: Path = Path(__file__).parent.parent / "models"
            model_path: Path = models_dir / "model.pkl"

            if not model_path.exists():
                logging.error(f"Model file not found at {model_path}")
                raise FileNotFoundError(f"Model file not found at {model_path}")

            self.transformer = SentenceTransformer("all-MiniLM-L6-v2")

            with open(model_path, "rb") as f:
                self.clf = pickle.load(f)

        except Exception as e:
            logging.error(f"Failed to load BERT models: {e}", exc_info=True)
            raise

    def classify(self, message: str) -> str:
        """
        Analyzes a log message using a BERT-based classifier to determine its category.

        Args:
            message (str): The log message string to analyze.

        Returns:
            str: The category label if a match is found with high confidence, otherwise "Unclassified".

        Raises:
            Exception: Captures and logs any unexpected errors during processing.
        """
        try:
            if not self.transformer or not self.clf:
                return "Unclassified"

            embeddings: ndarray = self.transformer.encode(
                [message], show_progress_bar=False
            )

            proba: Any = self.clf.predict_proba(embeddings)
            max_proba: Any = proba.max()

            if max_proba < 0.5:
                return "Unclassified"

            label_index: int = int(self.clf.predict(embeddings)[0])

            return self.LABEL_MAP.get(label_index, "Unclassified")

        except Exception as e:
            logging.info(f"Error in BertProcessor: {e}", exc_info=True)
            return "Unclassified"


def main() -> None:
    """
    Main function to test the BertProcessor class with sample logs.
    """
    processor: BertProcessor = BertProcessor()

    df = pd.read_csv("dataset/syn_logs_5k.csv")
    bert_df = df[df["complexity"] == "bert"]

    y_hat = bert_df["log_message"].apply(processor.classify)
    y_test = bert_df["target_label"]

    unclassified = (y_hat == "Unclassified").sum()
    print(f"BERT Accuracy: {accuracy_score(y_test, y_hat) * 100:.2f}%")


if __name__ == "__main__":
    main()
