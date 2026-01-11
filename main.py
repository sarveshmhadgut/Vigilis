import pandas as pd
from typing import List, Dict
from processors.regex_processing import RegexProcessor
from processors.bert_processing import BertProcessor
from processors.llm_processing import LlmProcessor


class LogClassifier:
    """
    Orchestrates the log classification process using a combination of methods.

    Responsibilities:
        - Initialize specific processors (Regex, BERT, LLM).
        - Implement a tiered classification strategy:
            1. Use LLM for specific sources (e.g., LegacyCRM).
            2. Try Regex matching for known patterns.
            3. Fallback to BERT classification for semantic understanding.
        - Process batches of logs from CSV files.
        - Generate and save labelled datasets.
    """

    def __init__(self) -> None:
        """
        Initializes the LogClassifier by instantiating the specific processors.
        """
        self.regex_processor: RegexProcessor = RegexProcessor()
        self.bert_processor: BertProcessor = BertProcessor()
        self.llm_processor: LlmProcessor = LlmProcessor()

    def classify_message(self, src: str, message: str) -> str:
        """
        Classifies a single log message based on its source and content.

        Args:
            src (str): The source system of the log.
            message (str): The log message content.

        Returns:
            str: The determined classification label.
        """
        label = self.regex_processor.classify(message=message)
        if label:
            return label

        label = self.bert_processor.classify(message=message)
        if label != "Unclassified":
            return label

        return self.llm_processor.classify(message=message)

    def batch_classify(self, logs: List[Dict[str, str]]) -> List[str]:
        """
        Classifies a batch of log messages.

        Args:
            logs (List[Dict[str, str]]): A list of dictionaries containing 'source' and 'log_message'.

        Returns:
            List[str]: A list of classification labels corresponding to the input logs.
        """
        import time

        results = []
        for log in logs:
            results.append(self.classify_message(log["source"], log["log_message"]))
            time.sleep(0.2)  # Slight delay to respect Rate Limits
        return results

    def generate_labelled_logs(
        self, logs_dirpath: str, output_path: str = "./artifacts/labelled_logs.csv"
    ) -> None:
        """
        Reads logs from a CSV, classifies them, and saves the results.

        Args:
            logs_dirpath (str): Path to the input CSV file.
            output_path (str): Path to save the labelled CSV file.

        Raises:
            FileNotFoundError: If the input CSV file is not found.
            Exception: If an error occurs during processing or saving.
        """
        df: pd.DataFrame = pd.read_csv(logs_dirpath)

        logs: List[Dict[str, str]] = df[["source", "log_message"]].to_dict("records")
        df["label"] = self.batch_classify(logs)

        df.to_csv(output_path, index=False)


def main() -> None:
    """
    Entry point for the log classification application.

    Instantiates the LogClassifier and triggers the generation of labelled logs
    from the test dataset.
    """
    classifier: LogClassifier = LogClassifier()
    classifier.generate_labelled_logs("./artifacts/test.csv")


if __name__ == "__main__":
    main()
