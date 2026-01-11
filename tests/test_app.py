import unittest
from unittest.mock import MagicMock
import io
from fastapi.testclient import TestClient
from app.app import app, classifier


class TestApp(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        classifier.classify_message = MagicMock(return_value="Mocked_Label")
        classifier.batch_classify = MagicMock(return_value=["Mocked_Label"])

    def tearDown(self):
        classifier.classify_message = self.original_classify_message
        classifier.batch_classify = self.original_batch_classify

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Vigilis", response.text)

    def test_classify_api(self):
        payload = {"source": "TestSrc", "log_message": "Test Message"}
        response = self.client.post("/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["label"], "Mocked_Label")
        self.assertEqual(data["source"], "TestSrc")

    def test_classify_batch_api(self):
        payload = {
            "logs": [
                {"source": "S1", "log_message": "M1"},
                {"source": "S2", "log_message": "M2"},
            ]
        }
        classifier.batch_classify.return_value = ["L1", "L2"]

        response = self.client.post("/classify/batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["results"]), 2)
        self.assertEqual(data["results"][0]["label"], "L1")

    def test_predict_upload_success(self):
        # Create a dummy CSV file in memory
        csv_content = (
            "source,log_message\nCRM,Error connecting to DB\nFirewall,Block IP"
        )
        files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}

        classifier.batch_classify.return_value = ["Critical Error", "Security Alert"]

        response = self.client.post("/predict", files=files)
        self.assertEqual(response.status_code, 200)
        self.assertIn("test.csv", response.text)
        self.assertIn("Critical Error", response.text)

    def test_predict_upload_missing_columns(self):
        csv_content = "wrong_col,data\n1,2"
        files = {"file": ("bad.csv", io.BytesIO(csv_content.encode()), "text/csv")}

        response = self.client.post("/predict", files=files)
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "CSV must contain &#39;source&#39; and &#39;log_message&#39; columns",
            response.text,
        )


if __name__ == "__main__":
    unittest.main()
