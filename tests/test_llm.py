import unittest
from io import StringIO
from unittest.mock import MagicMock, patch
from processors.llm_processing import LlmProcessor


class TestLlmProcessor(unittest.TestCase):
    def setUp(self):
        # Patch the dependencies
        self.patcher_chat = patch("processors.llm_processing.ChatGoogleGenerativeAI")
        self.patcher_prompt = patch("processors.llm_processing.PromptTemplate")
        self.patcher_parser = patch("processors.llm_processing.PydanticOutputParser")

        self.MockChat = self.patcher_chat.start()
        self.MockPrompt = self.patcher_prompt.start()
        self.MockParser = self.patcher_parser.start()

    def tearDown(self):
        self.patcher_chat.stop()
        self.patcher_prompt.stop()
        self.patcher_parser.stop()

    def test_classify_success(self):
        # Setup the chain mock
        processor = LlmProcessor()
        processor.chain = MagicMock()

        # Mock successful response
        mock_response = MagicMock()
        mock_response.label = "Security Alert"
        processor.chain.invoke.return_value = mock_response

        label = processor.classify("Unauthorized access attempt")
        self.assertEqual(label, "Security Alert")

    @patch("sys.stdout", new_callable=StringIO)
    def test_classify_api_error(self, mock_stdout):
        processor = LlmProcessor()
        processor.chain = MagicMock()

        # Mock exception
        processor.chain.invoke.side_effect = Exception("API Error")

        label = processor.classify("Some message")
        self.assertEqual(label, "Unclassified")

    @patch("sys.stdout", new_callable=StringIO)
    def test_init_failure(self, mock_stdout):
        # Simulate initialization failure
        with patch(
            "processors.llm_processing.ChatGoogleGenerativeAI",
            side_effect=Exception("Init Error"),
        ):
            processor = LlmProcessor()
            self.assertIsNone(processor.model)
            self.assertEqual(processor.classify("msg"), "Unclassified")


if __name__ == "__main__":
    unittest.main()
