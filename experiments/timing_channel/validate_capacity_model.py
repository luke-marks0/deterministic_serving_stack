#!/usr/bin/env python3
"""Validation suite for the capacity model in capacity_visualizer.html.

Mirrors the JS math in Python so we can sweep the parameter space and
catch modeling bugs that don't surface from one slider position. Three
sections:

  1. Hard tests — physical sanity, limits, monotonicity. Must all PASS.
  2. CF vs MC cross-check — closed form is the optimal-decoder upper
     bound; MC is the naive-decoder estimate. CF >= MC must hold.
  3. Harness validation — does the model reproduce bucket_experiment.json?
  4. Vibe check — print realistic threat-model sweeps for eyeballing.

Run: python3 experiments/timing_channel/validate_capacity_model.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Model (mirror of the JS in capacity_visualizer.html)
# ---------------------------------------------------------------------------

def Q(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2))


def Hb(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


# 16-point Gauss-Hermite quadrature for ∫ f(t) e^{-t²} dt
HG_NODES = (
    -4.688738939305818, -3.869447904860123, -3.176999161979956, -2.546202157847481,
    -1.951787990916254, -1.380258539198881, -0.822951449144656, -0.273481046138152,
     0.273481046138152,  0.822951449144656,  1.380258539198881,  1.951787990916254,
     2.546202157847481,  3.176999161979956,  3.869447904860123,  4.688738939305818,
)
HG_WEIGHTS = (
    2.65480747401119e-10, 2.32098084485497e-07, 2.71186007426073e-05, 9.32284008624180e-04,
    1.28803115863748e-02, 8.38100413749661e-02, 2.80647458528532e-01, 5.07929479016613e-01,
    5.07929479016613e-01, 2.80647458528532e-01, 8.38100413749661e-02, 1.28803115863748e-02,
    9.32284008624180e-04, 2.71186007426073e-05, 2.32098084485497e-07, 2.65480747401119e-10,
)
_SQRT_PI = math.sqrt(math.pi)
_SQRT_2 = math.sqrt(2)


def m_pam_awgn_capacity(M: int, d: float, sigma: float) -> float:
    """I(X;Y) bits/symbol for uniform-input M-PAM with AWGN noise σ, separation d.

    The optimal-decoder capacity. Replaces the old BSC approximation
    log2(M)·(1−H_b(SER)), which is a loose lower bound and was failing
    the CF >= MC ordering test under moderate noise.
    """
    if M <= 1:
        return 0.0
    if sigma <= 0:
        return math.log2(M)
    cond = 0.0
    for x in range(M):
        inner = 0.0
        for w_i, t_i in zip(HG_WEIGHTS, HG_NODES):
            z = sigma * _SQRT_2 * t_i
            s = 0.0
            for xp in range(M):
                diff = (x - xp) * d
                s += math.exp(-(diff * diff + 2 * diff * z) / (2 * sigma * sigma))
            inner += w_i * math.log2(s)
        cond += inner / _SQRT_PI
    return max(0.0, math.log2(M) - cond / M)


def bucket_channel_capacity(M, base, step, B, sigma):
    """Optimal-decoder capacity through bucket pacer + AWGN.

    Inter-arrival Y given symbol s: Y = q_s·B with prob 1−p_s, (q_s+1)·B
    with prob p_s, where q_s = ⌊Δ_s/B⌋, p_s = (Δ_s mod B)/B, Δ_s = base+step·s.
    Then Gaussian noise σ. Replaces the M_eff = ⌊range/B⌋+1 cliff that
    artificially zeroed capacity at large B.
    """
    if M <= 1: return 0.0
    if B <= 0: return m_pam_awgn_capacity(M, step, sigma)

    sd = []  # (Y_low, Y_high, p_high) per symbol
    for s in range(M):
        Delta = base + step * s
        q = math.floor(Delta / B)
        rho = Delta - q * B
        sd.append((q * B, (q + 1) * B, rho / B))

    if sigma <= 1e-9:
        # Discrete MI
        joint = {}
        for s, (yl, yh, p) in enumerate(sd):
            if 1 - p > 1e-12: joint[(s, yl)] = joint.get((s, yl), 0.0) + (1 - p) / M
            if p > 1e-12:     joint[(s, yh)] = joint.get((s, yh), 0.0) + p / M
        py = {}
        for (s, y), v in joint.items(): py[y] = py.get(y, 0.0) + v
        if len(py) <= 1: return 0.0
        mi = 0.0
        for (s, y), pst in joint.items():
            if pst > 0 and py[y] > 0:
                mi += pst * math.log2(pst / ((1.0 / M) * py[y]))
        return max(0.0, mi)

    # Continuous: 16-pt Gauss-Hermite around each component peak
    sqrt2pi_sig = sigma * math.sqrt(2 * math.pi)
    def pYS(y, s):
        yl, yh, p = sd[s]
        return ((1 - p) * math.exp(-0.5 * ((y - yl) / sigma) ** 2)
                + p     * math.exp(-0.5 * ((y - yh) / sigma) ** 2)) / sqrt2pi_sig
    mi = 0.0
    for s in range(M):
        yl, yh, p_h = sd[s]
        for mu, w_comp in ((yl, 1 - p_h), (yh, p_h)):
            if w_comp <= 0: continue
            integral = 0.0
            for w_i, t_i in zip(HG_WEIGHTS, HG_NODES):
                y = mu + sigma * _SQRT_2 * t_i
                ps = pYS(y, s)
                py = sum(pYS(y, s2) for s2 in range(M)) / M
                if ps > 0 and py > 0:
                    integral += w_i * math.log2(ps / py)
            integral /= _SQRT_PI
            mi += w_comp * integral
    mi /= M
    return max(0.0, mi)


def capacity(M, base, step, sigma, B, rate, multistream=False):
    """Closed-form optimal-decoder capacity using the bucket channel model."""
    rng = step * (M - 1)
    if B <= 0 or B <= step:
        M_eff, d_eff = M, step
    else:
        M_eff = max(1, min(M, int(rng // B) + 1))
        d_eff = B

    bpp = bucket_channel_capacity(M, base, step, B, sigma)
    # Diagnostic SER (informational)
    if M_eff <= 1:
        ser = 0.5
    elif sigma <= 0:
        ser = 0.0
    else:
        ser = min(0.5, 2 * (M_eff - 1) / M_eff * Q(d_eff / (2 * sigma)))

    mean_delay = base + step * (M - 1) / 2
    max_rate = (1000 / mean_delay) if mean_delay > 0 else float("inf")
    eff_rate = rate if multistream else min(rate, max_rate)

    return {
        "M_eff": M_eff,
        "d_eff": d_eff,
        "ser": ser,
        "bpp": bpp,
        "max_rate": max_rate,
        "eff_rate": eff_rate,
        "bps": eff_rate * bpp,
        "raw_bps": eff_rate * math.log2(M),
    }


def simulate_mc(M, base, step, sigma, B, rate, N=10000, multistream=False, seed=1):
    """Naive decoder; bits/packet computed as actual MI of (s, ŝ) joint."""
    random.seed(seed)
    last_emit = 0.0
    last_bucketed = None
    counted = 0
    errors = 0
    joint = [[0] * M for _ in range(M)]
    phase = random.random() * B if B > 0 else 0.0

    mean_delay = base + step * (M - 1) / 2
    max_rate = (1000 / mean_delay) if mean_delay > 0 else float("inf")
    eff_rate = rate if multistream else min(rate, max_rate)

    for _ in range(N):
        s = random.randrange(M)
        intended = base + step * s
        emit = last_emit + intended
        t_b = (math.ceil((emit + phase) / B) * B - phase) if B > 0 else emit
        if last_bucketed is not None:
            delta = (t_b - last_bucketed) + random.gauss(0, sigma)
            s_hat = max(0, min(M - 1, round((delta - base) / step)))
            joint[s][s_hat] += 1
            if s_hat != s:
                errors += 1
            counted += 1
        last_emit = emit
        last_bucketed = t_b

    if counted == 0:
        return {"ser": 0, "bpp": 0, "bps": 0, "eff_rate": eff_rate}

    ser = errors / counted
    ps = [sum(joint[s]) for s in range(M)]
    pt = [sum(joint[s][t] for s in range(M)) for t in range(M)]
    mi = 0.0
    for s in range(M):
        for t in range(M):
            c = joint[s][t]
            if c > 0:
                pst = c / counted
                psm = ps[s] / counted
                ptm = pt[t] / counted
                mi += pst * math.log2(pst / (psm * ptm))
    mi = max(0.0, mi)
    return {"ser": ser, "bpp": mi, "bps": eff_rate * mi, "eff_rate": eff_rate}


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

class T:
    def __init__(self):
        self.failures = []
        self.checks = 0

    def check(self, name, condition, detail=""):
        self.checks += 1
        if not condition:
            self.failures.append(f"  ✗ {name}  {detail}")
            print(f"  ✗ {name}  {detail}")
        # silent on pass — keep output focused on failures + summary

    def close(self, name, actual, expected, tol):
        self.check(
            name,
            abs(actual - expected) <= tol,
            f"got {actual:.4g}, expected {expected:.4g} ± {tol:.4g}",
        )


# ---------------------------------------------------------------------------
# 1. Hard tests
# ---------------------------------------------------------------------------

def test_physical_sanity(t: T):
    print("\n[1a] Physical sanity (must hold for all parameter combos)")
    rng = random.Random(42)
    for _ in range(200):
        M = rng.choice([2, 4, 8, 16, 32, 64])
        base = rng.uniform(0, 100)
        step = 10 ** rng.uniform(-1, 2)
        sigma = 10 ** rng.uniform(-2, 1.5)
        B = rng.choice([0, 0, 0, rng.uniform(1, 200)])  # bias toward unbucketed
        rate = 10 ** rng.uniform(0, 5)
        c = capacity(M, base, step, sigma, B, rate)
        t.check("bps >= 0", c["bps"] >= 0, f"M={M} σ={sigma:.2g} B={B:.2g} rate={rate:.2g} → {c['bps']:.2g}")
        t.check("bpp <= log2(M)", c["bpp"] <= math.log2(M) + 1e-9, f"bpp={c['bpp']:.4f} log2M={math.log2(M):.4f}")
        t.check("bps <= raw_bps", c["bps"] <= c["raw_bps"] + 1e-9)
        t.check("0 <= ser <= 0.5", 0 <= c["ser"] <= 0.5 + 1e-9)
        t.check("M_eff <= M", c["M_eff"] <= M)


def test_closed_form_limits(t: T):
    print("\n[1b] Closed-form limit cases")
    # No bucket, no noise: bpp = log2(M), bps = rate * log2(M)
    c = capacity(M=16, base=5, step=2, sigma=0, B=0, rate=50)
    t.close("no-bucket no-noise: bpp", c["bpp"], 4.0, 1e-9)
    t.close("no-bucket no-noise: bps", c["bps"], 200.0, 1e-9)

    # Huge sigma: bpp -> 0
    c = capacity(M=16, base=5, step=2, sigma=1000, B=0, rate=50)
    t.check("σ→∞: bpp→0", c["bpp"] < 0.01, f"got {c['bpp']:.4g}")

    # Bucket >> mod range: M_eff=1, bpp very small (residual leakage, not 0)
    c = capacity(M=16, base=5, step=2, sigma=0.5, B=10000, rate=50)
    t.check("B→∞: M_eff=1", c["M_eff"] == 1)
    t.check("B→∞: bpp small (residual)", 0 <= c["bpp"] < 0.01, f"got {c['bpp']:.5f}")

    # B <= step: should equal no-bucket case
    c0 = capacity(M=16, base=5, step=2, sigma=0.5, B=0, rate=50)
    c1 = capacity(M=16, base=5, step=2, sigma=0.5, B=1, rate=50)  # B < step
    t.close("B<step ≡ no-bucket: bpp", c1["bpp"], c0["bpp"], 1e-9)

    # M=1: bps = 0
    c = capacity(M=1, base=5, step=2, sigma=0.5, B=0, rate=50)
    t.check("M=1: bps=0", c["bps"] == 0)

    # rate=0: bps=0
    c = capacity(M=16, base=5, step=2, sigma=0.5, B=0, rate=0)
    t.check("rate=0: bps=0", c["bps"] == 0)


def test_monotonicity(t: T):
    print("\n[1c] Monotonicity")
    # bps decreasing in sigma (fixed everything else)
    base_p = dict(M=16, base=5, step=2, B=0, rate=50)
    sigmas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    bpss = [capacity(sigma=s, **base_p)["bps"] for s in sigmas]
    monotone = all(bpss[i] >= bpss[i + 1] - 1e-9 for i in range(len(bpss) - 1))
    t.check("bps decreasing in σ", monotone, f"sweep: {[f'{b:.2f}' for b in bpss]}")

    # bps decreasing in B (once B > step)
    base_p = dict(M=16, base=5, step=2, sigma=0.5, rate=50)
    Bs = [3, 5, 10, 20, 50, 100, 200]
    bpss = [capacity(B=B, **base_p)["bps"] for B in Bs]
    monotone = all(bpss[i] >= bpss[i + 1] - 1e-9 for i in range(len(bpss) - 1))
    t.check("bps decreasing in B (B>step)", monotone, f"sweep: {[f'{b:.2f}' for b in bpss]}")

    # bps increasing in rate (until cap)
    base_p = dict(M=16, base=5, step=2, sigma=0.5, B=0)
    rates = [1, 5, 10, 25, 49]  # all below max_rate=50
    bpss = [capacity(rate=r, **base_p)["bps"] for r in rates]
    monotone = all(bpss[i] <= bpss[i + 1] + 1e-9 for i in range(len(bpss) - 1))
    t.check("bps increasing in rate (below cap)", monotone, f"sweep: {[f'{b:.2f}' for b in bpss]}")


def test_rate_cap(t: T):
    print("\n[1d] Rate cap behavior")
    # max_rate = 1000 / (5 + 2*7.5) = 50
    # rate above max_rate gets capped (single-stream)
    c100 = capacity(M=16, base=5, step=2, sigma=0.5, B=0, rate=100, multistream=False)
    c50 = capacity(M=16, base=5, step=2, sigma=0.5, B=0, rate=50, multistream=False)
    t.close("single-stream: rate=100 capped to 50", c100["bps"], c50["bps"], 1e-9)

    # multistream: rate=100 actually doubles bps
    c100ms = capacity(M=16, base=5, step=2, sigma=0.5, B=0, rate=100, multistream=True)
    t.close("multi-stream: rate=100 → 2x bps", c100ms["bps"], 2 * c50["bps"], 1e-9)


# ---------------------------------------------------------------------------
# 2. CF vs MC cross-check
# ---------------------------------------------------------------------------

def test_cf_vs_mc(t: T):
    print("\n[2] CF vs MC")
    # No-bucket, low noise: both should be near log2(M)
    c = capacity(M=16, base=5, step=2, sigma=0.3, B=0, rate=50)
    m = simulate_mc(M=16, base=5, step=2, sigma=0.3, B=0, rate=50, N=20000, seed=1)
    t.close("no-bucket low-σ: MC bpp ≈ CF bpp", m["bpp"], c["bpp"], 0.15)

    # Huge bucket: both small (CF now captures the residual instead of zeroing)
    c = capacity(M=16, base=5, step=2, sigma=0.5, B=500, rate=50)
    m = simulate_mc(M=16, base=5, step=2, sigma=0.5, B=500, rate=50, N=10000, seed=1)
    t.check("B→∞: CF bpp small", c["bpp"] < 0.05, f"got {c['bpp']:.5f}")
    t.check("B→∞: MC bpp ≈ 0", m["bpp"] < 0.05, f"got {m['bpp']:.4f}")

    # CF >= MC (CF is optimal-decoder upper bound)
    cases = [
        (16, 5, 2, 0.5, 7),
        (16, 5, 2, 0.5, 15),
        (16, 5, 2, 1.0, 0),
        (8, 10, 5, 0.5, 0),
        (32, 1, 0.5, 0.2, 0),
    ]
    for (M, base, step, sigma, B) in cases:
        c = capacity(M, base, step, sigma, B, rate=50)
        m = simulate_mc(M, base, step, sigma, B, rate=50, N=10000, seed=1)
        # Allow small MC noise tolerance
        t.check(
            f"CF >= MC at M={M},B={B},σ={sigma}",
            c["bpp"] >= m["bpp"] - 0.05,
            f"CF={c['bpp']:.4f}, MC={m['bpp']:.4f}",
        )


# ---------------------------------------------------------------------------
# 3. Harness validation
# ---------------------------------------------------------------------------

def test_harness_json(t: T):
    print("\n[3] Validation against bucket_experiment_results.json")
    json_path = Path(__file__).parent / "bucket_experiment_results.json"
    if not json_path.exists():
        print("  (skip — file not found)")
        return
    data = json.loads(json_path.read_text())
    cfg = data["config"]
    M, base, step = cfg["levels"], cfg["base_delay_s"] * 1000, cfg["step_s"] * 1000
    R, sigma = cfg["token_rate_approx"], 0.55  # calibrated against undefended SER

    print(f"  Params: M={M}, base={base}ms, step={step}ms, σ={sigma}ms, R={R}/s")
    print(f"  {'B(ms)':>6}  {'CF SER':>7}  {'MC SER':>7}  {'obs SER':>7}  {'CF bps':>7}  {'MC bps':>7}  {'obs bps':>8}")
    for r in data["results"]:
        B = r["bucket_ms"] or 0
        c = capacity(M, base, step, sigma, B, R)
        m = simulate_mc(M, base, step, sigma, B, R, N=10000, seed=1)
        Bs = "none" if B == 0 else f"{B}"
        print(
            f"  {Bs:>6}  {c['ser']:>7.3f}  {m['ser']:>7.3f}  {r['ser']:>7.3f}"
            f"  {c['bps']:>7.1f}  {m['bps']:>7.1f}  {r['effective_bps']:>8.1f}"
        )

    # Check no-bucket calibration is tight
    c0 = capacity(M, base, step, sigma, 0, R)
    obs0 = next(r for r in data["results"] if r["bucket_ms"] is None)
    t.close("undefended SER matches harness", c0["ser"], obs0["ser"], 0.02)


# ---------------------------------------------------------------------------
# 4. Vibe check
# ---------------------------------------------------------------------------

def vibe_check():
    print("\n[4] Vibe check — capacity across realistic threat models")
    threats = [
        ("Single LLM stream, no defense", dict(M=16, base=5, step=2, sigma=0.55, B=0, rate=50)),
        ("Single LLM stream, LAN, no defense", dict(M=16, base=5, step=2, sigma=0.2, B=0, rate=50)),
        ("Single LLM stream, WiFi", dict(M=16, base=5, step=2, sigma=8, B=0, rate=50)),
        ("Single LLM stream, B=20ms bucket", dict(M=16, base=5, step=2, sigma=0.55, B=20, rate=50)),
        ("Single LLM stream, B=100ms bucket", dict(M=16, base=5, step=2, sigma=0.55, B=100, rate=50)),
        ("Adaptive vs B=100ms", dict(M=16, base=100, step=150, sigma=0.55, B=100, rate=1000)),
        ("Datacenter egress, 1M pps, no defense", dict(M=16, base=5, step=2, sigma=0.55, B=0, rate=1_000_000, multistream=True)),
        ("Datacenter egress, 1M pps, B=20ms", dict(M=16, base=5, step=2, sigma=0.55, B=20, rate=1_000_000, multistream=True)),
        ("Datacenter egress, 1M pps, B=100ms", dict(M=16, base=5, step=2, sigma=0.55, B=100, rate=1_000_000, multistream=True)),
        ("Boundary: 2-PAM, 50 pps, σ=8 (WiFi)", dict(M=2, base=20, step=20, sigma=8, B=0, rate=25)),
    ]
    print(f"  {'scenario':<48} {'CF bps':>10}  {'MC bps':>10}  {'M_eff':>5}  {'eff rate':>10}")
    for name, p in threats:
        c = capacity(**p)
        m = simulate_mc(**p, N=10000, seed=1)
        print(
            f"  {name:<48} {fmt(c['bps']):>10}  {fmt(m['bps']):>10}  "
            f"{c['M_eff']:>5}  {fmt(c['eff_rate'])+' pps':>10}"
        )


def fmt(x):
    if x < 1:
        return f"{x:.3f}"
    if x < 1000:
        return f"{x:.1f}"
    if x < 1e6:
        return f"{x/1e3:.1f}k"
    if x < 1e9:
        return f"{x/1e6:.1f}M"
    return f"{x/1e9:.1f}G"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t = T()
    test_physical_sanity(t)
    test_closed_form_limits(t)
    test_monotonicity(t)
    test_rate_cap(t)
    test_cf_vs_mc(t)
    test_harness_json(t)
    vibe_check()

    print(f"\n{'=' * 60}")
    if t.failures:
        print(f"FAIL: {len(t.failures)} of {t.checks} checks failed")
        for f in t.failures:
            print(f)
        sys.exit(1)
    else:
        print(f"PASS: all {t.checks} checks")


if __name__ == "__main__":
    main()
