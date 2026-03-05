import pathlib
import unittest


class TestRepoLayout(unittest.TestCase):
    def test_required_directories_exist(self) -> None:
        required = [
            "cmd/resolver",
            "cmd/builder",
            "cmd/runner",
            "cmd/verifier",
            "pkg/manifest",
            "pkg/lockfile",
            "pkg/provenance",
            "pkg/hardware",
            "pkg/networkdet",
            "pkg/batchtrace",
            "nix/packages",
            "nix/images",
            "deploy/k8s",
            "deploy/helm",
            "tests/unit",
            "tests/integration",
            "tests/e2e",
            "tests/determinism",
            "tests/chaos",
            "tests/fixtures",
            "docs/adr",
            "docs/conformance",
            "schemas",
        ]
        for rel in required:
            with self.subTest(path=rel):
                self.assertTrue(pathlib.Path(rel).is_dir(), f"Missing directory: {rel}")


if __name__ == "__main__":
    unittest.main()
