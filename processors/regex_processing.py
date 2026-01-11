import re
import sys
import os

# Add project root to sys.path to allow running this script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import logging
from typing import Optional, List, Tuple, Pattern


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
                re.compile(r"^User login successful\.$", re.I),
                "USER_ACTION",
            ),
            (
                re.compile(
                    r"\bunauthorized\b|\bfailed login\b|\bblocked\b|\bsuspicious\b",
                    re.I,
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
            # Iterate through rules and return the first match
            for pattern, label in self.REGEX_RULES:
                if pattern.search(message):
                    return label

            return None

        except Exception as e:
            logging.info(f"Error in RegexProcessor: {str(e)}", exc_info=True)
            return None


def main() -> None:
    """
    Main function to test the RegexProcessor class with sample logs.
    """
    processor: RegexProcessor = RegexProcessor()

    test_logs: List[str] = [
        "User User123 logged out.",
        "User User123 logged in.",
        "Backup started at 2022-01-01 12:00:00.",
        "Backup ended at 2022-01-01 13:00:00.",
        "System updated to version 1.0.0.",
        "File report.pdf uploaded successfully by user User456.",
        "Disk cleanup completed successfully.",
        "System reboot initiated by user User789.",
        "Account with ID 12345 created by User999.",
        "Backup completed successfully.",
        "Unauthorized access attempt detected.",
        "Multiple failed login attempts from IP 192.168.1.1.",
        "Account blocked due to suspicious activity.",
        "User login successful.",
        "Non-matching log message.",
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
        result: str = processor.classify(msg)
        print(f"'{msg}' -> {result}")


if __name__ == "__main__":
    main()
