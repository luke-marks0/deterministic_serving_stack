import pathlib
import unittest


class TestChaosScaffold(unittest.TestCase):
    def test_chaos_runner_script_exists(self) -> None:
        self.assertTrue(pathlib.Path("scripts/ci/test_nightly.sh").is_file())


if __name__ == "__main__":
    unittest.main()
