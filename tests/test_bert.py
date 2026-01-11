import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from processors.bert_processing import BertProcessor


class TestBertProcessor(unittest.TestCase):
    @patch("processors.bert_processing.SentenceTransformer")
    @patch("processors.bert_processing.pickle.load")
    @patch("processors.bert_processing.open")
    @patch("processors.bert_processing.Path.exists")
    def setUp(self, mock_exists, mock_open, mock_pickle_load, mock_transformer_cls):
        mock_exists.return_value = True

        self.mock_clf = MagicMock()
        mock_pickle_load.return_value = self.mock_clf

        self.mock_transformer = MagicMock()
        mock_transformer_cls.return_value = self.mock_transformer

        self.processor = BertProcessor()

    def test_classify_high_confidence(self):
        self.mock_transformer.encode.return_value = np.array([[0.1, 0.2]])

        self.mock_clf.predict_proba.return_value = np.array([[0.1, 0.9]])

        self.mock_clf.predict.return_value = [4]

        label = self.processor.classify("System security breach")

        self.assertEqual(label, "Security Alert")
        self.mock_transformer.encode.assert_called()
        self.mock_clf.predict.assert_called()

    def test_classify_low_confidence(self):
        self.mock_transformer.encode.return_value = np.array([[0.1, 0.2]])

        self.mock_clf.predict_proba.return_value = np.array([[0.6, 0.4]])
        proba_mock = MagicMock()
        proba_mock.max.return_value = 0.4
        self.mock_clf.predict_proba.return_value = proba_mock

        label = self.processor.classify("Ambiguous message")
        self.assertEqual(label, "Unclassified")

    def test_classify_not_initialized(self):
        self.processor.transformer = None
        self.processor.clf = None

        label = self.processor.classify("msg")
        self.assertEqual(label, "Unclassified")

    @patch("processors.bert_processing.logging")
    def test_classify_error(self, mock_logging):
        self.mock_transformer.encode.side_effect = Exception("Encode Error")

        label = self.processor.classify("msg")
        self.assertEqual(label, "Unclassified")


if __name__ == "__main__":
    unittest.main()
