"""
W0 — Numerical verification of the LASER-VFL reweighting theorem.

Two claims are checked here, both by brute force against closed forms.

CLAIM 1 (counting identity)
    For a sample whose observed block set has size m, the total weight it
    contributes to the LASER-VFL objective is

        W(m) = sum over k in K_o, sum over I with k in I subset K_o, of 1/|I|
             = 2^m - 1

CLAIM 2 (expected weight)
    If each of K blocks is observed independently with probability pi,
    so that m ~ Binomial(K, pi), then

        E[W] = (1 + pi)^K - 1

Everything downstream (the bias ratio R, the prevalence shift) rests on
these two identities. Run this before writing anything.

Usage:  python verify_theorem.py
Dependencies: standard library only.
"""

import itertools
import math
import random
from typing import Callable


# ----------------------------------------------------------------------
# CLAIM 1 — the counting identity
# ----------------------------------------------------------------------

def weight_brute_force(m: int) -> float:
    """
    Compute W(m) exactly as the paper's definition states it: iterate over
    every (k, I) pair with k in I subset of {1..m}, accumulating 1/|I|.

    No shortcuts. This is deliberately the naive double loop so that it
    tests the closed form rather than reusing its logic.
    """
    blocks = list(range(m))
    total = 0.0
    for k in blocks:                                   # outer: sum over k in K_o
        for size in range(1, m + 1):                   # inner: sum over I containing k
            for subset in itertools.combinations(blocks, size):
                if k in subset:
                    total += 1.0 / len(subset)
    return total


def weight_closed_form(m: int) -> float:
    """The claimed closed form."""
    return 2.0 ** m - 1.0


def verify_counting_identity(max_m: int = 12) -> bool:
    """Check W(m) == 2^m - 1 for m = 0 .. max_m."""
    print("=" * 70)
    print("CLAIM 1 — counting identity:  W(m) = 2^m - 1")
    print("=" * 70)
    print(f"{'m':>3}  {'brute force':>14}  {'2^m - 1':>14}  {'match':>7}")
    print("-" * 70)

    all_ok = True
    for m in range(0, max_m + 1):
        brute = weight_brute_force(m)
        closed = weight_closed_form(m)
        ok = math.isclose(brute, closed, rel_tol=1e-9, abs_tol=1e-9)
        all_ok &= ok
        print(f"{m:>3}  {brute:>14.6f}  {closed:>14.6f}  {'OK' if ok else 'FAIL':>7}")

    print()
    print(f"RESULT: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


def show_the_swap(m: int = 3) -> None:
    """
    Display the summation swap that makes the identity work, so you can see
    the mechanism rather than just the answer.

    The proof reorders the same set of (k, I) pairs: instead of grouping by
    k, group by I. Once I is fixed, 1/|I| is added exactly |I| times, which
    contributes exactly 1 per subset. Hence W(m) = number of non-empty
    subsets = 2^m - 1.
    """
    print("=" * 70)
    print(f"THE SWAP, shown explicitly for m = {m}")
    print("=" * 70)
    print("Group the (k, I) pairs by subset I instead of by client k.")
    print("Each subset then contributes |I| copies of 1/|I|, i.e. exactly 1.\n")
    print(f"{'subset I':>16}  {'|I|':>4}  {'k in I':>18}  {'contribution':>13}")
    print("-" * 70)

    blocks = list(range(m))
    running = 0.0
    for size in range(1, m + 1):
        for subset in itertools.combinations(blocks, size):
            contribution = len(subset) * (1.0 / len(subset))   # = 1, always
            running += contribution
            members = ",".join(str(k) for k in subset)
            print(f"{'{' + members + '}':>16}  {len(subset):>4}  "
                  f"{members:>18}  {contribution:>13.4f}")

    print("-" * 70)
    print(f"{'TOTAL':>16}  {'':>4}  {'':>18}  {running:>13.4f}")
    print(f"Non-empty subsets of a {m}-element set: 2^{m} - 1 = {2**m - 1}")
    print()


# ----------------------------------------------------------------------
# CLAIM 2 — the expected weight
# ----------------------------------------------------------------------

def expected_weight_exact(K: int, pi: float) -> float:
    """
    E[2^m - 1] computed by explicitly summing the binomial pmf.
    Independent of the closed form, so it is a genuine check.
    """
    total = 0.0
    for j in range(K + 1):
        p_j = math.comb(K, j) * (pi ** j) * ((1.0 - pi) ** (K - j))
        total += p_j * (2.0 ** j - 1.0)
    return total


def expected_weight_closed_form(K: int, pi: float) -> float:
    """The claimed closed form: (1 + pi)^K - 1."""
    return (1.0 + pi) ** K - 1.0


def expected_weight_simulated(K: int, pi: float, n_trials: int = 200_000,
                              seed: int = 0) -> float:
    """Monte Carlo estimate of E[2^m - 1], m ~ Bin(K, pi)."""
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n_trials):
        m = sum(1 for _ in range(K) if rng.random() < pi)
        total += 2.0 ** m - 1.0
    return total / n_trials


def verify_expected_weight() -> bool:
    """Check exact == closed form, and that simulation agrees."""
    print("=" * 70)
    print("CLAIM 2 — expected weight:  E[W] = (1 + pi)^K - 1")
    print("=" * 70)
    print(f"{'K':>3}  {'pi':>5}  {'exact sum':>12}  {'(1+pi)^K-1':>12}  "
          f"{'simulated':>12}  {'match':>7}")
    print("-" * 70)

    all_ok = True
    for K in (2, 4, 8):
        for pi in (0.0, 0.25, 0.5, 0.6, 0.75, 1.0):
            exact = expected_weight_exact(K, pi)
            closed = expected_weight_closed_form(K, pi)
            sim = expected_weight_simulated(K, pi, n_trials=50_000)

            ok = math.isclose(exact, closed, rel_tol=1e-9, abs_tol=1e-9)
            all_ok &= ok
            print(f"{K:>3}  {pi:>5.2f}  {exact:>12.6f}  {closed:>12.6f}  "
                  f"{sim:>12.6f}  {'OK' if ok else 'FAIL':>7}")
        print()

    print(f"RESULT: {'PASS' if all_ok else 'FAIL'}")
    print("(Simulated column should be close to the others but will not match")
    print(" exactly — it is a finite sample. Large K with pi near 1 is the")
    print(" slowest to converge, since 2^m has heavy right tail.)")
    print()
    return all_ok


# ----------------------------------------------------------------------
# SANITY CHECKS from the plan
# ----------------------------------------------------------------------

def run_sanity_checks() -> bool:
    """The four checks specified in W0, stated as assertions."""
    print("=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    checks = []

    # Small-m values, computed by brute force
    for m, expected in [(1, 1), (2, 3), (3, 7), (4, 15)]:
        got = weight_brute_force(m)
        ok = math.isclose(got, expected, abs_tol=1e-9)
        checks.append(ok)
        print(f"  m = {m}  ->  W = {got:.4f}   (expect {expected})   "
              f"{'OK' if ok else 'FAIL'}")

    # pi = 0: nothing is ever observed, so nothing is ever counted
    K = 4
    got = expected_weight_closed_form(K, 0.0)
    ok = math.isclose(got, 0.0, abs_tol=1e-12)
    checks.append(ok)
    print(f"  pi = 0, K = {K}  ->  E[W] = {got:.6f}   (expect 0)          "
          f"{'OK' if ok else 'FAIL'}")

    # pi = 1: everything observed, so weight is the full 2^K - 1
    got = expected_weight_closed_form(K, 1.0)
    ok = math.isclose(got, 2.0 ** K - 1.0, abs_tol=1e-9)
    checks.append(ok)
    print(f"  pi = 1, K = {K}  ->  E[W] = {got:.6f}   (expect {2**K - 1})        "
          f"{'OK' if ok else 'FAIL'}")

    # pi_1 == pi_0: no reweighting, R must be exactly 1
    for pi in (0.3, 0.5, 0.7):
        r = bias_ratio_laser(K, pi, pi)
        ok = math.isclose(r, 1.0, rel_tol=1e-12)
        checks.append(ok)
        print(f"  pi_1 = pi_0 = {pi}, K = {K}  ->  R = {r:.12f}   (expect 1)  "
              f"{'OK' if ok else 'FAIL'}")

    all_ok = all(checks)
    print()
    print(f"RESULT: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


# ----------------------------------------------------------------------
# CONSEQUENCES — bias ratio and prevalence shift
# ----------------------------------------------------------------------

def bias_ratio_laser(K: int, pi0: float, pi1: float) -> float:
    """Relative class weighting under LASER-VFL (and Combinatorial)."""
    num = (1.0 + pi1) ** K - 1.0
    den = (1.0 + pi0) ** K - 1.0
    return num / den


def bias_ratio_standard(K: int, pi0: float, pi1: float) -> float:
    """
    Relative class weighting under standard VFL, which trains only on
    fully-observed samples.

    NOTE: assumes blocks are observed INDEPENDENTLY given the label. If
    blocks are correlated in your data (likely in clinical settings), this
    overstates how much data is dropped. Measure the correlation in W4.4
    and recompute with the empirical joint distribution if it is large.
    """
    if pi0 <= 0.0:
        raise ValueError(
            "pi0 must be > 0: if negatives never have all blocks observed, "
            "standard VFL trains on no negatives at all and the ratio is "
            "undefined."
        )
    return (pi1 / pi0) ** K


def bias_ratio_local(K: int, pi0: float, pi1: float) -> float:
    """Local / Ensemble methods: a User is used iff its own block is present."""
    return pi1 / pi0


def print_bias_table(pi0: float = 0.5) -> None:
    """Reproduce the R table from the plan."""
    print("=" * 70)
    print(f"BIAS RATIO R,  pi_0 = {pi0}")
    print("=" * 70)
    print("R = 1 means no reweighting. R = 1.5 means the positive class")
    print("counts 50% more than it should.\n")

    header = (f"{'pi_1':>6}  {'delta':>6}  "
              f"{'std K=4':>9}  {'std K=8':>9}  "
              f"{'LASER K=4':>10}  {'LASER K=8':>10}  {'local':>7}")
    print(header)
    print("-" * 70)

    for pi1 in (0.50, 0.52, 0.55, 0.60, 0.65, 0.70):
        delta = abs(pi1 - pi0)
        print(f"{pi1:>6.2f}  {delta:>6.2f}  "
              f"{bias_ratio_standard(4, pi0, pi1):>9.3f}  "
              f"{bias_ratio_standard(8, pi0, pi1):>9.3f}  "
              f"{bias_ratio_laser(4, pi0, pi1):>10.3f}  "
              f"{bias_ratio_laser(8, pi0, pi1):>10.3f}  "
              f"{bias_ratio_local(4, pi0, pi1):>7.3f}")
    print()


def delta_required_for(target_R: float, K: int, pi0: float,
                       ratio_fn: Callable[[int, float, float], float],
                       tol: float = 1e-10) -> float:
    """
    Invert the bias ratio: find the smallest gap delta = pi_1 - pi_0 that
    produces a reweighting of at least target_R.

    This is what calibrates the W4 gate. The question is not "is delta big
    enough to infer labels" but "is delta big enough to bias training",
    and those have very different thresholds.

    Solved by bisection on pi_1 in (pi0, 1].
    """
    lo, hi = pi0, 1.0
    if ratio_fn(K, pi0, hi) < target_R:
        return float("nan")          # unreachable even at pi_1 = 1

    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if ratio_fn(K, pi0, mid) < target_R:
            lo = mid
        else:
            hi = mid
    return hi - pi0


def print_gate_calibration(pi0: float = 0.5, targets=(1.05, 1.10, 1.20, 1.50)) -> None:
    """
    For each method and each K, report the delta needed to reach a given
    level of bias. Use this to set the W4 go/no-go threshold before you
    look at the data.
    """
    print("=" * 70)
    print(f"GATE CALIBRATION — delta needed to reach a given R  (pi_0 = {pi0})")
    print("=" * 70)
    print("Read this BEFORE measuring, so the threshold is not chosen")
    print("after seeing the answer.\n")

    methods = [
        ("standard VFL", bias_ratio_standard),
        ("LASER-VFL", bias_ratio_laser),
        ("local", bias_ratio_local),
    ]

    print(f"{'method':>14}  {'K':>3}  " +
          "  ".join(f"{'R>=' + str(t):>9}" for t in targets))
    print("-" * 70)
    for name, fn in methods:
        for K in (4, 8):
            cells = []
            for t in targets:
                d = delta_required_for(t, K, pi0, fn)
                cells.append("  n/a" if math.isnan(d) else f"{d:>9.4f}")
            print(f"{name:>14}  {K:>3}  " + "  ".join(f"{c:>9}" for c in cells))
    print()
    print("Interpretation: any measured delta above the R>=1.20 column")
    print("produces a reweighting large enough to matter. Note how small")
    print("that threshold is for standard VFL at K = 8.")
    print()


def prevalence_shift(p_true: float, K: int, pi0: float, pi1: float,
                     weight_fn: Callable[[int, float], float]) -> float:
    """
    Effective positive rate in the training objective.

    weight_fn(K, pi) returns the expected weight of a sample whose class has
    availability pi. Reweighting by class turns the true prevalence p_true
    into the effective prevalence returned here.
    """
    w1 = weight_fn(K, pi1)
    w0 = weight_fn(K, pi0)
    pos = p_true * w1
    neg = (1.0 - p_true) * w0
    return pos / (pos + neg)


def print_prevalence_table(p_true: float = 0.10, pi0: float = 0.5,
                           pi1: float = 0.6) -> None:
    """
    Show what the reweighting does to class balance. This is the most
    quotable number in the paper: it converts an abstract ratio into
    "your training set thinks the disease is twice as common as it is."
    """
    print("=" * 70)
    print(f"PREVALENCE SHIFT   (true rate = {p_true:.1%}, "
          f"pi_0 = {pi0}, pi_1 = {pi1})")
    print("=" * 70)
    print(f"{'K':>3}  {'method':>12}  {'effective prevalence':>21}  "
          f"{'inflation':>10}")
    print("-" * 70)

    methods = [
        ("standard VFL", lambda K, pi: pi ** K),
        ("LASER-VFL", lambda K, pi: (1.0 + pi) ** K - 1.0),
        ("local", lambda K, pi: pi),
        ("zero-fill", lambda K, pi: 1.0),
    ]

    for K in (4, 8):
        for name, fn in methods:
            eff = prevalence_shift(p_true, K, pi0, pi1, fn)
            print(f"{K:>3}  {name:>12}  {eff:>20.1%}   {eff / p_true:>9.2f}x")
        print()


# ----------------------------------------------------------------------

def main() -> None:
    print()
    print("#" * 70)
    print("# W0 — VERIFICATION OF THE REWEIGHTING THEOREM")
    print("#" * 70)
    print()

    show_the_swap(m=3)
    ok1 = verify_counting_identity(max_m=12)
    ok2 = verify_expected_weight()
    ok3 = run_sanity_checks()

    print_bias_table(pi0=0.5)
    print_gate_calibration(pi0=0.5)
    print_prevalence_table(p_true=0.10, pi0=0.5, pi1=0.6)

    print("=" * 70)
    if ok1 and ok2 and ok3:
        print("ALL CHECKS PASSED — the theorem holds numerically.")
        print()
        print("Reminder of what is NOT verified here:")
        print("  * That blocks are observed independently. The standard-VFL")
        print("    ratio (pi_1/pi_0)^K assumes it. Measure this in W4.4.")
        print("  * That pi_1 != pi_0 in real data. That is W3 and W4.")
        print("  * That the reweighting changes trained models. That is Stage 2.")
    else:
        print("SOMETHING FAILED — do not proceed until this is resolved.")
        print("Recheck the summation swap in the derivation first.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()