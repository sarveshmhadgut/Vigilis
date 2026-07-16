import os
import sys
from typing import List

import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score
from tqdm import tqdm

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

            self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

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
    Evaluate the LLM tier over its dataset rows using concurrent batched calls.

    Uses chain.batch with a concurrency cap (fast, rate-limit friendly) and a
    tqdm progress bar that advances per chunk.
    """

    processor = LlmProcessor()

    df = pd.read_csv("dataset/syn_logs_5k.csv")
    llm_df = df[df["complexity"] == "llm"]
    messages = llm_df["log_message"].tolist()

    CHUNK = 20
    MAX_CONCURRENCY = 10

    labels: List[str] = []
    for start in tqdm(range(0, len(messages), CHUNK), desc="LLM classifying"):
        chunk = messages[start : start + CHUNK]
        inputs = [{"message": m} for m in chunk]
        try:
            results = processor.chain.batch(
                inputs, config={"max_concurrency": MAX_CONCURRENCY}
            )
            labels.extend(r.label for r in results)
        except Exception as e:
            print(f"Chunk starting at {start} failed: {e}")
            labels.extend(["Unclassified"] * len(chunk))

    y_hat = pd.Series(labels, index=llm_df.index)
    y_test = llm_df["target_label"]

    failed = int((y_hat == "Unclassified").sum())
    print(f"LLM Accuracy: {accuracy_score(y_test, y_hat) * 100:.2f}%")


if __name__ == "__main__":
    main()
