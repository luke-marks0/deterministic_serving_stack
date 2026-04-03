import os
from pose.prover import Prover
from pose.verifier import Verifier
from pose.memory.dram import DramRegion
from pose.noise import generate_block

BLOCK_SIZE = 4096
NUM_BLOCKS = 100  # Small for testing


def test_fill_and_respond():
    """Prover fills memory from a block stream and can return any block."""
    seed = os.urandom(32)
    dram = DramRegion(size_bytes=NUM_BLOCKS * BLOCK_SIZE, block_size=BLOCK_SIZE)

    # Verifier produces the stream; prover stores it without seeing the seed
    verifier = Verifier(total_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE, seed=seed)
    prover = Prover(regions={"dram": dram}, block_size=BLOCK_SIZE)
    prover.fill(verifier.noise_stream())

    for idx in [0, 49, 99]:
        response = prover.respond(idx)
        assert verifier.verify(idx, response)

    dram.close()


def test_fill_multiple_regions():
    """Prover fills across multiple regions from a single stream."""
    seed = os.urandom(32)
    dram = DramRegion(size_bytes=50 * BLOCK_SIZE, block_size=BLOCK_SIZE)
    dram2 = DramRegion(size_bytes=50 * BLOCK_SIZE, block_size=BLOCK_SIZE)

    verifier = Verifier(total_blocks=100, block_size=BLOCK_SIZE, seed=seed)
    prover = Prover(
        regions={"dram": dram, "hbm": dram2},
        block_size=BLOCK_SIZE,
    )
    prover.fill(verifier.noise_stream())

    # Block 0 in dram, block 50 in "hbm"
    assert verifier.verify(0, prover.respond(0))
    assert verifier.verify(50, prover.respond(50))

    dram.close()
    dram2.close()


def test_prover_does_not_receive_seed():
    """Prover has no attribute or reference to the seed."""
    dram = DramRegion(size_bytes=10 * BLOCK_SIZE, block_size=BLOCK_SIZE)
    prover = Prover(regions={"dram": dram}, block_size=BLOCK_SIZE)
    assert not hasattr(prover, "seed")
    dram.close()
