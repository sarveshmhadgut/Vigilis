import pickle
from pathlib import Path
from typing import Any, Dict, List
from numpy import ndarray
import sys
import os

# Add project root to sys.path to allow running this script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer


from utils.logger import logging


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
        self.transformer = None
        self.clf = None
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

            self.transformer: SentenceTransformer = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )

            with open(model_path, "rb") as f:
                self.clf: Any = pickle.load(f)

        except Exception as e:
            logging.error(f"Failed to load BERT models: {str(e)}", exc_info=True)
            raise e

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

            # Encode the message to get embeddings
            embeddings: ndarray = self.transformer.encode(
                [message], show_progress_bar=False
            )

            # Predict probabilities
            proba: Any = self.clf.predict_proba(embeddings)
            max_proba: Any = proba.max()

            # Threshold for classification confidence
            if max_proba < 0.5:
                return "Unclassified"

            # Get the predicted label index
            label_index: str = self.clf.predict(embeddings)[0]
            # Map index to label
            return self.LABEL_MAP.get(label_index, "Unclassified")

        except Exception as e:
            logging.info(f"Error in BertProcessor: {str(e)}", exc_info=True)
            return "Unclassified"


def main() -> None:
    """
    Main function to test the BertProcessor class with sample logs.
    """
    processor: BertProcessor = BertProcessor()

    test_logs: List[str] = [
        # HTTP Status
        "nova.osapi_compute.wsgi.server returned HTTP 404 for GET /v2/servers/detail",
        "Request to compute API failed with HTTP 500 response code",
        "Upstream service responded with HTTP 503 during metadata fetch",
        # Critical Error
        "Email delivery subsystem failed causing outage in notification service",
        "Critical failure occurred while initializing billing workflow",
        "System encountered fatal exception during startup sequence",
        # Security Alert
        "Unauthorized access to protected customer data was detected",
        "Multiple authentication failures recorded for user account",
        "Suspicious activity detected while validating access tokens",
        # Error
        "Failed to process input record due to malformed payload",
        "Application encountered an error while reading configuration values",
        "Error occurred during transformation of analytics dataset",
        # Resource Usage
        "High memory consumption observed during batch analytics job",
        "CPU utilization spiked above threshold while processing reports",
        "Disk space running low on storage node during backup operation",
        # Low-signal / ambiguous
        "Service initialized and awaiting requests",
        "Background task executed successfully",
        # Workflow Error
        "ERROR: Job execution failed due to unhandled exception",
        "Fatal error: pipeline aborted after task preprocess_data failed",
        # Configuration Error
        "ERROR: Environment variable DATABASE_URL is not set",
        "Invalid configuration: expected integer for batch_size, got string",
        # Dependency / Environment Issue
        "ModuleNotFoundError: No module named 'ipykernel'",
        "ImportError: numpy>=2.0 required, but version 1.26.4 is installed",
        # Deprecation Warning
        "DeprecationWarning: keras.backend.set_session is deprecated and will be removed in a future release",
        "WARNING: The --use_legacy_api flag is deprecated and will be removed in v3.0",
        # Performance Warning
        "WARNING: Inference latency exceeded threshold (1200ms > 500ms)",
        "PerformanceWarning: DataLoader is using a single worker and may be slow",
        # Resource Exhaustion
        "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "ERROR: No space left on device while writing checkpoint",
        # Security / Permission Issue
        "PermissionError: [Errno 13] Permission denied: '/var/log/app.log'",
        "403 Forbidden: user does not have access to this resource",
        # Data / Input Error
        "ValueError: could not convert string to float: 'abc'",
        "JSONDecodeError: Expecting ',' delimiter at line 4",
        # Informational / Status
        "INFO: Training job started with run_id=42",
        "Server listening on port 8080",
        # Unclassified
        "Hello from worker process",
        "Debug mode enabled",
    ]

    for msg in test_logs:
        label: str = processor.classify(msg)
        print(f"{msg} -> {label}")


if __name__ == "__main__":
    main()
