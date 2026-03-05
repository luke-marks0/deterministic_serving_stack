from __future__ import annotations

import subprocess
import unittest


class TestFixtureContracts(unittest.TestCase):
    def test_positive_and_negative_fixtures(self) -> None:
        subprocess.run(["python3", "scripts/ci/fixture_validate.py"], check=True)


if __name__ == "__main__":
    unittest.main()
