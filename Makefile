.PHONY: lint schema test-fast test-full test-nightly test-release ci-pr ci-main ci-nightly ci-release build-libnetdet

lint:
	bash scripts/ci/lint.sh

schema:
	bash scripts/ci/schema_gate.sh

test-fast:
	bash scripts/ci/test_fast.sh

test-full:
	bash scripts/ci/test_full.sh

test-nightly:
	bash scripts/ci/test_nightly.sh

test-release:
	bash scripts/ci/test_release.sh

ci-pr: lint schema test-fast

ci-main: lint schema test-full

ci-nightly: lint schema test-nightly

ci-release: lint schema test-release

build-libnetdet:
	cd native/libnetdet && make
