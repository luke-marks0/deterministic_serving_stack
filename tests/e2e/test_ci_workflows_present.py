import pathlib
import unittest


class TestCIWorkflows(unittest.TestCase):
    def test_workflows_exist(self) -> None:
        expected = [
            ".github/workflows/pr-gate.yml",
            ".github/workflows/main-gate.yml",
            ".github/workflows/nightly.yml",
            ".github/workflows/release-gate.yml",
        ]
        for rel in expected:
            with self.subTest(path=rel):
                self.assertTrue(pathlib.Path(rel).is_file(), f"Missing workflow: {rel}")


if __name__ == "__main__":
    unittest.main()
