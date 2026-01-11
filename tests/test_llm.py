import unittest
from unittest.mock import MagicMock, patch
from processors.llm_processing import LlmProcessor


class TestLlmProcessor(unittest.TestCase):
    def setUp(self):
        self.groq_patcher = patch("processors.llm_processing.Groq")
        self.mock_groq_class = self.groq_patcher.start()

        self.mock_groq_instance = MagicMock()
        self.mock_groq_class.return_value = self.mock_groq_instance

        self.processor = LlmProcessor()

    def tearDown(self):
        self.groq_patcher.stop()

    def test_classify_success(self):
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "<label>Security Alert</label>"

        self.mock_groq_instance.chat.completions.create.return_value = mock_completion

        label = self.processor.classify("Unauthorized access attempt")
        self.assertEqual(label, "Security Alert")

    def test_classify_no_label_match(self):
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "This looks like a security issue."

        self.mock_groq_instance.chat.completions.create.return_value = mock_completion

        label = self.processor.classify("Some message")
        self.assertEqual(label, "Miscellaneous")

    def test_classify_api_error(self):
        self.mock_groq_instance.chat.completions.create.side_effect = Exception(
            "API Error"
        )

        label = self.processor.classify("Some message")
        self.assertEqual(label, "Unclassified")

    def test_init_failure(self):
        self.groq_patcher.stop()

        with patch(
            "processors.llm_processing.Groq", side_effect=Exception("Init Error")
        ):
            processor = LlmProcessor()
            self.assertIsNone(processor.groq)
            self.assertEqual(processor.classify("msg"), "Unclassified")

        self.groq_patcher.start()


if __name__ == "__main__":
    unittest.main()
