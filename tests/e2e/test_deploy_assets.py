from __future__ import annotations

import pathlib
import re
import unittest


class TestDeployAssets(unittest.TestCase):
    def test_kubernetes_job_uses_immutable_image_digest_and_mounts_inputs(self) -> None:
        path = pathlib.Path("deploy/k8s/single-node-runner-job.yaml")
        content = path.read_text(encoding="utf-8")

        self.assertRegex(content, r"image:\s+.+@sha256:[a-f0-9]{64}")
        self.assertIn("RUNNER_POD_MANIFEST_PATH", content)
        self.assertIn("RUNNER_POD_LOCKFILE_PATH", content)
        self.assertIn("RUNNER_POD_RUNTIME_CLOSURE_PATH", content)
        self.assertIn("RUNNER_ENABLE_HOST_PROBE", content)
        self.assertIn("mountPath: /run-inputs", content)
        self.assertIn("manifest.json", content)
        self.assertIn("lockfile.json", content)
        self.assertIn("runtime_closure_digest.txt", content)

    def test_helm_chart_pins_runner_image_by_digest(self) -> None:
        values_path = pathlib.Path("deploy/helm/deterministic-serving/values.yaml")
        template_path = pathlib.Path("deploy/helm/deterministic-serving/templates/runner-job.yaml")
        values = values_path.read_text(encoding="utf-8")
        template = template_path.read_text(encoding="utf-8")

        self.assertRegex(values, r"digest:\s+sha256:[a-f0-9]{64}")
        self.assertIn("@{{ .Values.runnerImage.digest }}", template)
        self.assertIn("RUNNER_POD_RUNTIME_CLOSURE_PATH", template)

    def test_nix_assets_exist_for_runtime_closure_and_image(self) -> None:
        expected = [
            pathlib.Path("nix/packages/runtime-closure.nix"),
            pathlib.Path("nix/images/runtime-image.nix"),
        ]
        for path in expected:
            with self.subTest(path=str(path)):
                self.assertTrue(path.is_file(), f"Missing Nix asset: {path}")


if __name__ == "__main__":
    unittest.main()
