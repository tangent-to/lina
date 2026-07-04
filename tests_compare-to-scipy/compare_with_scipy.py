#!/usr/bin/env python3
"""
Compare @tangent.to/lina against numpy/scipy.linalg on seeded random
matrices: solve, det, inv, cholesky, qr, svd, eigSym (eigh), lstsq,
pinv, rank, cond.

Decomposition factors are compared invariantly (reconstruction, spectra,
orthogonality) rather than entry-by-entry, since factor signs/orders are
conventions. Run from the package root:

    uv run --with scipy python3 tests_compare-to-scipy/compare_with_scipy.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NODE_SCRIPT = ROOT / "tests_compare-to-scipy" / "compare_lina.mjs"

FAILURES = []


def check(label, err, tol):
    ok = err < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  (err={err:.2e}, tol={tol:.0e})")
    if not ok:
        FAILURES.append(label)


def run_node(spec):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(spec, fh)
        path = fh.name
    r = subprocess.run(["node", str(NODE_SCRIPT), path], check=True,
                       capture_output=True, text=True, cwd=ROOT)
    return json.loads(r.stdout)


def main():
    rng = np.random.default_rng(2026)
    print("numpy/scipy comparison for @tangent.to/lina")

    # --- solve / det / inv on well-conditioned square systems ---
    for n in (3, 8, 25):
        A = rng.standard_normal((n, n)) + n * np.eye(n)
        b = rng.standard_normal(n)
        js = run_node({"op": "solve", "A": A.tolist(), "b": b.tolist()})
        check(f"solve n={n}", float(np.max(np.abs(js["x"] - np.linalg.solve(A, b)))), 1e-10)
        js = run_node({"op": "det", "A": A.tolist()})
        check(f"det n={n}", abs(js["det"] - np.linalg.det(A)) / max(1, abs(np.linalg.det(A))), 1e-10)
        js = run_node({"op": "inv", "A": A.tolist()})
        check(f"inv n={n}", float(np.max(np.abs(np.asarray(js["inv"]) - np.linalg.inv(A)))), 1e-10)

    # --- cholesky ---
    for n in (4, 12):
        B = rng.standard_normal((n, n))
        A = B @ B.T + n * np.eye(n)
        js = run_node({"op": "cholesky", "A": A.tolist()})
        L = np.asarray(js["L"])
        check(f"cholesky n={n} (reconstruction)", float(np.max(np.abs(L @ L.T - A))), 1e-9)
        check(f"cholesky n={n} (vs numpy)", float(np.max(np.abs(L - np.linalg.cholesky(A)))), 1e-9)

    # --- qr (invariants) ---
    for shape in ((6, 6), (10, 4)):
        A = rng.standard_normal(shape)
        js = run_node({"op": "qr", "A": A.tolist()})
        Q, R = np.asarray(js["Q"]), np.asarray(js["R"])
        check(f"qr {shape} A=QR", float(np.max(np.abs(Q @ R - A))), 1e-11)
        check(f"qr {shape} Q'Q=I", float(np.max(np.abs(Q.T @ Q - np.eye(Q.shape[1])))), 1e-11)

    # --- svd: singular values vs numpy, reconstruction ---
    for shape in ((6, 6), (12, 5), (5, 12)):
        A = rng.standard_normal(shape)
        js = run_node({"op": "svd", "A": A.tolist()})
        s_np = np.linalg.svd(A, compute_uv=False)
        check(f"svd {shape} singular values", float(np.max(np.abs(np.asarray(js["s"]) - s_np))), 1e-10)
        U, s, V = np.asarray(js["U"]), np.asarray(js["s"]), np.asarray(js["V"])
        check(f"svd {shape} reconstruction", float(np.max(np.abs(U @ np.diag(s) @ V.T - A))), 1e-10)

    # --- eigSym vs numpy.linalg.eigh ---
    for n in (5, 20):
        B = rng.standard_normal((n, n))
        A = (B + B.T) / 2
        js = run_node({"op": "eigSym", "A": A.tolist()})
        w_np = np.linalg.eigh(A)[0][::-1]  # eigh ascending -> descending
        check(f"eigSym n={n} eigenvalues", float(np.max(np.abs(np.asarray(js["values"]) - w_np))), 1e-10)
        Vv = np.asarray(js["vectors"])
        D = np.diag(js["values"])
        check(f"eigSym n={n} reconstruction", float(np.max(np.abs(Vv @ D @ Vv.T - A))), 1e-9)

    # --- lstsq vs numpy ---
    A = rng.standard_normal((30, 4))
    x_true = np.array([1.5, -2.0, 0.5, 3.0])
    b = A @ x_true + 0.01 * rng.standard_normal(30)
    js = run_node({"op": "lstsq", "A": A.tolist(), "b": b.tolist()})
    x_np = np.linalg.lstsq(A, b, rcond=None)[0]
    check("lstsq 30x4", float(np.max(np.abs(np.asarray(js["x"]) - x_np))), 1e-10)

    # --- pinv (incl. rank-deficient) / rank / cond ---
    A = rng.standard_normal((7, 4))
    js = run_node({"op": "pinv", "A": A.tolist()})
    check("pinv 7x4", float(np.max(np.abs(np.asarray(js["pinv"]) - np.linalg.pinv(A)))), 1e-9)
    B = np.outer(rng.standard_normal(6), rng.standard_normal(4))  # rank 1
    js = run_node({"op": "pinv", "A": B.tolist()})
    check("pinv rank-1 6x4", float(np.max(np.abs(np.asarray(js["pinv"]) - np.linalg.pinv(B)))), 1e-9)
    js = run_node({"op": "rank_cond", "A": B.tolist()})
    check("rank rank-1", abs(js["rank"] - 1), 0.5)
    A = rng.standard_normal((8, 8)) + 8 * np.eye(8)
    js = run_node({"op": "rank_cond", "A": A.tolist()})
    check("cond 8x8", abs(js["cond"] - np.linalg.cond(A)) / np.linalg.cond(A), 1e-9)

    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nAll comparisons passed.")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
