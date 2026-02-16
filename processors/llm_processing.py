import os
import sys
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()


class LlmSchema(BaseModel):
    label: str = Field(description="Classification label of the log message.")


class LlmProcessor:
    """
    Handles log classification using Large Language Models provided by Google Gemini via LangChain.

    Responsibilities:
        - Initialize the Google Generative AI client.
        - Construct a prompt with classification instructions and categories.
        - Send the prompt to the LLM and parse the response to extract the label.
        - Handle API errors and fall back to 'Unclassified'.
    """

    def __init__(self) -> None:
        """
        Initializes the LlmProcessor by setting up the Google Gemini client.
        """
        try:
            self.parser: PydanticOutputParser[LlmSchema] = PydanticOutputParser(
                pydantic_object=LlmSchema
            )

            self.prompt = PromptTemplate(
                template="""
                You are an expert system log analyzer.
                Classify the following log message into one of these categories:
                - User Action
                - System Notification
                - HTTP Status
                - Critical Error
                - Security Alert
                - Error
                - Resource Usage
                - Workflow Error
                - Configuration Error
                - Dependency / Environment Issue
                - Deprecation Warning
                - Performance Warning
                - Resource Exhaustion
                - Security / Permission Issue
                - Data / Input Error
                - Informational / Status
                - Miscellaneous

                If the log does not fit well into any specific category, use "Miscellaneous".

                Log Message:
                {message}

                {parser_instructions}
                """,
                input_variables=["message"],
                partial_variables={
                    "parser_instructions": self.parser.get_format_instructions()
                },
            )

            self.model = ChatGoogleGenerativeAI(
                model="gemini-3-pro-preview", temperature=0
            )

            self.chain = self.prompt | self.model | self.parser

        except Exception as e:
            print(f"Error initializing LlmProcessor: {e}")
            self.model = None

    def classify(self, message: str) -> str:
        """
        Analyzes a log message using Google Gemini to determine its category.

        Args:
            message (str): The log message string to analyze.

        Returns:
            str: The extracted category label if successful, otherwise "Miscellaneous" or "Unclassified".

        Raises:
            Exception: Captures and logs any unexpected errors during processing.
        """
        if not self.model:
            return "Unclassified"

        try:
            response = self.chain.invoke({"message": message})
            return response.label

        except Exception as e:
            print(f"Error classifying log: {e}")
            return "Unclassified"


def main() -> None:
    """
    Main function to test the LlmProcessor function with sample logs.
    """
    processor: LlmProcessor = LlmProcessor()

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
