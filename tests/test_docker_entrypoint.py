import sys
import unittest
from unittest.mock import patch

import docker_entrypoint


class DockerEntrypointTests(unittest.TestCase):
    @patch("docker_entrypoint.os.execv")
    @patch("docker_entrypoint.download_models.main")
    def test_initializes_models_then_forwards_arguments(self, init_models, execv):
        docker_entrypoint.main(["url", "http://example.com"])
        init_models.assert_called_once_with()
        execv.assert_called_once_with(
            sys.executable,
            [sys.executable, "predict.py", "url", "http://example.com"],
        )

    @patch("docker_entrypoint.os.execv")
    @patch("docker_entrypoint.download_models.main")
    def test_defaults_to_info(self, init_models, execv):
        docker_entrypoint.main([])
        init_models.assert_called_once_with()
        execv.assert_called_once_with(
            sys.executable,
            [sys.executable, "predict.py", "info"],
        )

    @patch("docker_entrypoint.os.execv")
    @patch("docker_entrypoint.download_models.main")
    def test_live_fails_before_model_initialization(self, init_models, execv):
        with self.assertRaisesRegex(SystemExit, "not supported inside Docker"):
            docker_entrypoint.main(["live"])
        init_models.assert_not_called()
        execv.assert_not_called()


if __name__ == "__main__":
    unittest.main()
