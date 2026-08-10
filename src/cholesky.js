/**
 * Cholesky factorization A = L L^T for symmetric positive definite matrices.
 *
 * Public APIs accept and return nested row-major matrices; computation runs
 * on flat Float64Array storage (see _mat.js).
 */

import { assertSymmetric, fromNested, toNested, vecFrom } from './_mat.js';

/**
 * Cholesky factorization of a symmetric positive definite matrix.
 *
 * @param {Array<Array<number>>} A - Symmetric positive definite nested matrix
 * @returns {Array<Array<number>>} Lower triangular L with A = L L^T
 * @throws {Error} When A is not symmetric, or a diagonal pivot is <= 0
 *   (not positive definite)
 */
export function cholesky(A) {
  const M = fromNested(A, 'A');
  assertSymmetric(M, 'cholesky');
  return toNested(choleskyFlat(M, 'cholesky'), M.n, M.n);
}

/**
 * Cholesky factorization on flat storage, for callers that already hold a
 * validated square matrix and want to avoid the nested round trip.
 *
 * @param {{data: Float64Array, n: number}} M - Square matrix from fromNested()
 * @param {string} caller - Function name for error messages
 * @returns {Float64Array} Row-major n*n lower triangular L with A = L L^T
 * @throws {Error} When a diagonal pivot is <= 0 (not positive definite)
 */
export function choleskyFlat(M, caller = 'cholesky') {
  const { data, n } = M;
  const L = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    let d = data[j * n + j];
    for (let k = 0; k < j; k++) d -= L[j * n + k] * L[j * n + k];
    if (d <= 0) {
      throw new Error(`${caller}: matrix is not positive definite (pivot ${j} is ${d})`);
    }
    const ljj = Math.sqrt(d);
    L[j * n + j] = ljj;
    for (let i = j + 1; i < n; i++) {
      let s = data[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      L[i * n + j] = s / ljj;
    }
  }
  return L;
}

/**
 * Forward substitution L X = B for lower triangular L, on flat storage.
 *
 * @param {Float64Array} L - Row-major n*n lower triangular factor
 * @param {number} n - Dimension of L
 * @param {Float64Array} B - Row-major n*k right-hand side (not modified)
 * @param {number} k - Number of right-hand side columns
 * @returns {Float64Array} Row-major n*k solution X
 */
export function solveLowerFlat(L, n, B, k) {
  const X = new Float64Array(n * k);
  for (let i = 0; i < n; i++) {
    const lii = L[i * n + i];
    for (let c = 0; c < k; c++) {
      let s = B[i * k + c];
      for (let j = 0; j < i; j++) s -= L[i * n + j] * X[j * k + c];
      X[i * k + c] = s / lii;
    }
  }
  return X;
}

/**
 * Back substitution L^T X = B for lower triangular L, on flat storage.
 *
 * @param {Float64Array} L - Row-major n*n lower triangular factor
 * @param {number} n - Dimension of L
 * @param {Float64Array} B - Row-major n*k right-hand side (not modified)
 * @param {number} k - Number of right-hand side columns
 * @returns {Float64Array} Row-major n*k solution X
 */
export function solveLowerTransposeFlat(L, n, B, k) {
  const X = new Float64Array(n * k);
  for (let i = n - 1; i >= 0; i--) {
    const lii = L[i * n + i];
    for (let c = 0; c < k; c++) {
      let s = B[i * k + c];
      for (let j = i + 1; j < n; j++) s -= L[j * n + i] * X[j * k + c];
      X[i * k + c] = s / lii;
    }
  }
  return X;
}

/**
 * Solve A x = b given the Cholesky factor L of A (A = L L^T), by forward
 * substitution (L y = b) then back substitution (L^T x = y).
 *
 * @param {Array<Array<number>>} L - Lower triangular factor from cholesky()
 * @param {Array<number>} b - Right-hand side vector
 * @returns {Array<number>}
 */
export function choleskySolve(L, b) {
  const M = fromNested(L, 'L');
  if (M.m !== M.n) {
    throw new Error(`choleskySolve: L must be square (got ${M.m}x${M.n})`);
  }
  const { data, n } = M;
  const x = vecFrom(b, n, 'b');
  // Forward substitution: L y = b.
  for (let i = 0; i < n; i++) {
    let s = x[i];
    for (let j = 0; j < i; j++) s -= data[i * n + j] * x[j];
    x[i] = s / data[i * n + i];
  }
  // Back substitution: L^T x = y.
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let j = i + 1; j < n; j++) s -= data[j * n + i] * x[j];
    x[i] = s / data[i * n + i];
  }
  return Array.from(x);
}

/**
 * Test positive definiteness by attempting a Cholesky factorization.
 * Never throws; non-symmetric or malformed input returns false.
 *
 * @param {Array<Array<number>>} A - Nested matrix
 * @returns {boolean}
 */
export function isPositiveDefinite(A) {
  try {
    cholesky(A);
    return true;
  } catch {
    return false;
  }
}
