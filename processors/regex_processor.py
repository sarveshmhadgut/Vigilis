import os
import re
import sys
from typing import List, Optional, Pattern, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score

from utils.logger import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class RegexProcessor:
    """
    Handles log classification using regular expression matching.

    Responsibilities:
        - Maintain a prioritized list of regex patterns mapped to classification labels.
        - Iterate through rules to find the first matching pattern for a given log message.
        - Return 'USER_ACTION', 'SYSTEM_NOTIFICATION', or 'SECURITY_ALERT' based on matches.
    """

    def __init__(self):
        """
        Initializes the RegexProcessor with a list of compiled regex rules.
        """
        self.REGEX_RULES: List[Tuple[Pattern, str]] = [
            (
                re.compile(r"^User User\d+ logged (in|out)\.$"),
                "USER_ACTION",
            ),
            (
                re.compile(r"^Account with ID \d+ created by User\d+\.$"),
                "USER_ACTION",
            ),
            (
                re.compile(
                    r"^Backup started at \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.$"
                ),
                "SYSTEM_NOTIFICATION",
            ),
            (
                re.compile(r"^Backup ended at \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.$"),
                "SYSTEM_NOTIFICATION",
            ),
            (
                re.compile(r"^Backup completed successfully\.$"),
                "SYSTEM_NOTIFICATION",
            ),
            (
                re.compile(r"^System updated to version \d+\.\d+\.\d+\.$"),
                "SYSTEM_NOTIFICATION",
            ),
            (
                re.compile(r"^File .+ uploaded successfully by user User\d+\.$"),
                "SYSTEM_NOTIFICATION",
            ),
            (
                re.compile(r"^Disk cleanup completed successfully\.$"),
                "SYSTEM_NOTIFICATION",
            ),
            (
                re.compile(r"^System reboot initiated by user User\d+\.$"),
                "SYSTEM_NOTIFICATION",
            ),
            (
                re.compile(r"^User login successful\.$", re.IGNORECASE),
                "USER_ACTION",
            ),
            (
                re.compile(
                    r"\bunauthorized\b|\bfailed login\b|\bblocked\b|\bsuspicious\b",
                    re.IGNORECASE,
                ),
                "SECURITY_ALERT",
            ),
        ]

    def classify(self, message: str) -> Optional[str]:
        """
        Analyzes a log message using regular expressions to determine its category.

        Args:
            message (str): The log message string to analyze.

        Returns:
            Optional[str]: The category label if a match is found, otherwise None.

        Raises:
            Exception: Captures and logs any unexpected errors during processing.
        """
        try:
            for pattern, label in self.REGEX_RULES:
                if pattern.search(message):
                    return label

            return None

        except Exception as e:
            logging.info(f"Error in RegexProcessor: {e}", exc_info=True)
            return None


def main() -> None:
    """
    Main function to test the RegexProcessor class with sample logs.
    """
    processor: RegexProcessor = RegexProcessor()
    label_map = {
        "USER_ACTION": "User Action",
        "SYSTEM_NOTIFICATION": "System Notification",
        "SECURITY_ALERT": "Security Alert",
    }

    df = pd.read_csv("dataset/syn_logs_5k.csv")
    regex_df = df[df["complexity"] == "regex"]

    y_hat = regex_df["log_message"].apply(processor.classify).map(label_map)
    y_test = regex_df["target_label"]

    print(f"Regex Accuracy: {accuracy_score(y_test, y_hat) * 100:.2f}%")


if __name__ == "__main__":
    main()
