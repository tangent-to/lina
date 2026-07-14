/**
 * LU factorization with scaled partial pivoting, and LU-based solve/det/inv.
 *
 * Public APIs accept and return nested row-major matrices; computation runs
 * on flat Float64Array storage (see _mat.js).
 */

import { fromNested, toNested, vecFrom } from './_mat.js';

/**
 * Factor a flat square matrix in place-of-copy with scaled partial pivoting.
 *
 * Returns the combined LU storage (unit-diagonal L strictly below the
 * diagonal, U on and above), the row permutation `perm` (row k of the
 * factored matrix is row perm[k] of the input), the permutation sign, and a
 * singularity flag (any pivot with |pivot| <= 1e-13 * maxAbs of the input).
 *
 * @param {Float64Array} src - Flat row-major n*n values (not modified)
 * @param {number} n - Dimension
 * @returns {{a: Float64Array, perm: Int32Array, sign: number, singular: boolean}}
 */
function luDecompose(src, n) {
  const a = Float64Array.from(src);
  const perm = new Int32Array(n);
  const scale = new Float64Array(n);
  let maxAbs = 0;
  for (let i = 0; i < a.length; i++) maxAbs = Math.max(maxAbs, Math.abs(a[i]));
  const tol = 1e-13 * maxAbs;
  let sign = 1;
  let singular = maxAbs === 0;

  for (let i = 0; i < n; i++) {
    perm[i] = i;
    let s = 0;
    for (let j = 0; j < n; j++) s = Math.max(s, Math.abs(a[i * n + j]));
    scale[i] = s;
  }

  for (let k = 0; k < n; k++) {
    // Pick the row maximizing |a[i][k]| / scale[i] among rows k..n-1.
    let p = k;
    let best = -1;
    for (let i = k; i < n; i++) {
      const v = scale[i] > 0 ? Math.abs(a[i * n + k]) / scale[i] : 0;
      if (v > best) {
        best = v;
        p = i;
      }
    }
    if (p !== k) {
      for (let j = 0; j < n; j++) {
        const t = a[k * n + j];
        a[k * n + j] = a[p * n + j];
        a[p * n + j] = t;
      }
      const ts = scale[k];
      scale[k] = scale[p];
      scale[p] = ts;
      const tp = perm[k];
      perm[k] = perm[p];
      perm[p] = tp;
      sign = -sign;
    }
    const pivot = a[k * n + k];
    if (Math.abs(pivot) <= tol) {
      singular = true;
      continue;
    }
    for (let i = k + 1; i < n; i++) {
      const f = a[i * n + k] / pivot;
      a[i * n + k] = f;
      for (let j = k + 1; j < n; j++) {
        a[i * n + j] -= f * a[k * n + j];
      }
    }
  }

  return { a, perm, sign, singular };
}

/**
 * Validate a square nested matrix and return its flat form.
 *
 * @param {Array<Array<number>>} A - Nested matrix
 * @param {string} caller - Function name for error messages
 * @returns {{data: Float64Array, m: number, n: number}}
 */
function squareFromNested(A, caller) {
  const M = fromNested(A, 'A');
  if (M.m !== M.n) {
    throw new Error(`${caller}: matrix must be square (got ${M.m}x${M.n})`);
  }
  return M;
}

/**
 * LU factorization with partial pivoting: P A = L U.
 *
 * @param {Array<Array<number>>} A - Square nested matrix
 * @returns {{L: Array<Array<number>>, U: Array<Array<number>>, P: Array<Array<number>>}}
 *   L unit lower triangular, U upper triangular, P a permutation matrix.
 */
export function lu(A) {
  const { data, n } = squareFromNested(A, 'lu');
  const { a, perm } = luDecompose(data, n);
  const L = new Float64Array(n * n);
  const U = new Float64Array(n * n);
  const P = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    L[i * n + i] = 1;
    P[i * n + perm[i]] = 1;
    for (let j = 0; j < i; j++) L[i * n + j] = a[i * n + j];
    for (let j = i; j < n; j++) U[i * n + j] = a[i * n + j];
  }
  return { L: toNested(L, n, n), U: toNested(U, n, n), P: toNested(P, n, n) };
}

/**
 * LU factorization in packed flat storage, for callers that back-substitute
 * many right-hand sides against one factorization on a hot path.
 *
 * Unlike lu(), this skips building the nested L, U and dense permutation
 * matrix P: it returns the combined LU array directly (unit-diagonal L
 * strictly below the diagonal, U on and above, in row-major n*n storage)
 * together with the permutation vector, so no nested round-trip or
 * permutation-matrix scan is needed. Pair it with luFactorSolve().
 *
 * @param {Array<Array<number>>} A - Square nested matrix
 * @returns {{lu: Float64Array, perm: Int32Array, n: number, sign: number, singular: boolean}}
 *   `lu` is the combined LU storage (row-major n*n); `perm` maps factored row k
 *   to input row perm[k]; `sign` is the permutation sign; `singular` flags a
 *   pivot at or below 1e-13 * maxAbs(A).
 */
export function luFactor(A) {
  const { data, n } = squareFromNested(A, 'luFactor');
  const { a, perm, sign, singular } = luDecompose(data, n);
  return { lu: a, perm, n, sign, singular };
}

/**
 * Solve A x = b from a packed factorization returned by luFactor().
 *
 * @param {{lu: Float64Array, perm: Int32Array, n: number}} fac - luFactor() result
 * @param {Array<number>|Float64Array} b - Right-hand side (length n, not modified)
 * @returns {Float64Array} Solution vector x
 */
export function luFactorSolve(fac, b) {
  const { lu, perm, n } = fac;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = b[perm[i]];
  luSolveInPlace(lu, perm, n, x);
  return x;
}

/**
 * Solve L U x = P b for one right-hand side, in place in `x`.
 *
 * @param {Float64Array} a - Combined LU storage
 * @param {Int32Array} perm - Row permutation
 * @param {number} n - Dimension
 * @param {Float64Array} x - Permuted right-hand side; overwritten with the solution
 */
function luSolveInPlace(a, perm, n, x) {
  // Forward substitution with unit lower triangular L.
  for (let i = 1; i < n; i++) {
    let s = x[i];
    for (let j = 0; j < i; j++) s -= a[i * n + j] * x[j];
    x[i] = s;
  }
  // Back substitution with U.
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let j = i + 1; j < n; j++) s -= a[i * n + j] * x[j];
    x[i] = s / a[i * n + i];
  }
}

/**
 * Solve A x = b via LU factorization with partial pivoting.
 *
 * @param {Array<Array<number>>} A - Square nested matrix
 * @param {Array<number>|Array<Array<number>>} b - Right-hand side vector (length n)
 *   or nested matrix of right-hand sides (n x k, solved column by column)
 * @returns {Array<number>|Array<Array<number>>} Solution vector, or nested n x k matrix
 */
export function solve(A, b) {
  const { data, n } = squareFromNested(A, 'solve');
  const { a, perm, singular } = luDecompose(data, n);
  if (singular) {
    throw new Error('solve: matrix is singular (pivot below tolerance)');
  }

  const matrixRhs = Array.isArray(b) && Array.isArray(b[0]);
  if (!matrixRhs) {
    const bv = vecFrom(b, n, 'b');
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) x[i] = bv[perm[i]];
    luSolveInPlace(a, perm, n, x);
    return Array.from(x);
  }

  const B = fromNested(b, 'b');
  if (B.m !== n) {
    throw new Error(`solve: b must have ${n} rows (got ${B.m})`);
  }
  const k = B.n;
  const X = new Float64Array(n * k);
  const x = new Float64Array(n);
  for (let c = 0; c < k; c++) {
    for (let i = 0; i < n; i++) x[i] = B.data[perm[i] * k + c];
    luSolveInPlace(a, perm, n, x);
    for (let i = 0; i < n; i++) X[i * k + c] = x[i];
  }
  return toNested(X, n, k);
}

/**
 * Determinant via LU factorization (permutation sign times product of pivots).
 * Returns 0 for singular matrices instead of throwing.
 *
 * @param {Array<Array<number>>} A - Square nested matrix
 * @returns {number}
 */
export function det(A) {
  const { data, n } = squareFromNested(A, 'det');
  const { a, sign, singular } = luDecompose(data, n);
  if (singular) return 0;
  let d = sign;
  for (let i = 0; i < n; i++) d *= a[i * n + i];
  return d;
}

/**
 * Matrix inverse via solve(A, I).
 *
 * @param {Array<Array<number>>} A - Square nested matrix
 * @returns {Array<Array<number>>}
 */
export function inv(A) {
  const M = squareFromNested(A, 'inv');
  const n = M.n;
  const I = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(n).fill(0);
    row[i] = 1;
    I[i] = row;
  }
  return solve(A, I);
}
