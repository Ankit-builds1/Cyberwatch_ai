import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import download_models


EXPECTED_METRICS = {
    "network_xgboost_test_acc": 97.14,
    "network_lstm_test_acc": 94.06,
    "phishing_ensemble_test_acc": 93.78,
    "malware_cnn_test_acc": 97.13,
    "social_bert_test_acc": 91.55,
    "anomaly_combined_detection": 80.4,
}


class DownloadModelsTests(unittest.TestCase):
    def test_models_complete_requires_every_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            with patch.object(download_models, "MODELS_DIR", model_dir):
                self.assertFalse(download_models.models_complete())
                for relative in download_models.REQUIRED:
                    path = model_dir / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                self.assertTrue(download_models.models_complete())

    def test_install_corrected_metrics_overwrites_release_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            metrics_path = model_dir / "metrics.json"
            metrics_path.write_text(
                '{"network_xgboost_test_acc": 95.13}', encoding="utf-8"
            )
            with patch.object(download_models, "MODELS_DIR", model_dir):
                download_models.install_corrected_metrics()
            self.assertEqual(
                json.loads(metrics_path.read_text(encoding="utf-8")),
                EXPECTED_METRICS,
            )


if __name__ == "__main__":
    unittest.main()
