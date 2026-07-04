/**
 * QR decomposition via Householder reflections, and least squares on top of it.
 *
 * Public APIs take and return nested row-major matrices (Array<Array<number>>)
 * and plain vectors (Array<number>); computation runs on flat Float64Array
 * storage via the helpers in _mat.js.
 */

import { fromNested, toNested, vecFrom } from './_mat.js';

/**
 * Factor a flat matrix in place with Householder reflectors.
 *
 * After the call, `data` holds R in its upper triangle (subdiagonal entries of
 * the processed columns are set to exact zeros). The reflectors H_j =
 * I - beta_j v_j v_j^T (with v_j supported on rows j..m-1) satisfy
 * H_{k-1} ... H_0 A = R, i.e. A = H_0 ... H_{k-1} R.
 *
 * Uses the standard stable sign choice v = x + sign(x0) ||x|| e1.
 *
 * @param {Float64Array} data - Row-major m*n storage, overwritten with R
 * @param {number} m - Rows
 * @param {number} n - Columns
 * @returns {{vs: Array<Float64Array>, betas: Float64Array, k: number}}
 */
function householderFactor(data, m, n) {
  const k = Math.min(m, n);
  const vs = new Array(k);
  const betas = new Float64Array(k);
  for (let j = 0; j < k; j++) {
    let norm = 0;
    for (let i = j; i < m; i++) {
      const t = data[i * n + j];
      norm += t * t;
    }
    norm = Math.sqrt(norm);
    const v = new Float64Array(m - j);
    vs[j] = v;
    if (norm === 0) {
      betas[j] = 0;
      continue;
    }
    const x0 = data[j * n + j];
    const sign = x0 >= 0 ? 1 : -1;
    v[0] = x0 + sign * norm;
    for (let i = j + 1; i < m; i++) v[i - j] = data[i * n + j];
    let vv = 0;
    for (let t = 0; t < v.length; t++) vv += v[t] * v[t];
    const beta = 2 / vv;
    betas[j] = beta;
    // Apply H_j to the trailing columns.
    for (let c = j + 1; c < n; c++) {
      let s = 0;
      for (let i = j; i < m; i++) s += v[i - j] * data[i * n + c];
      s *= beta;
      for (let i = j; i < m; i++) data[i * n + c] -= s * v[i - j];
    }
    // Column j maps exactly to -sign * norm * e1.
    data[j * n + j] = -sign * norm;
    for (let i = j + 1; i < m; i++) data[i * n + j] = 0;
  }
  return { vs, betas, k };
}

/**
 * Accumulate Q = H_0 H_1 ... H_{k-1} applied to the first qCols columns of I.
 *
 * @param {Array<Float64Array>} vs - Reflector vectors
 * @param {Float64Array} betas - Reflector scalings
 * @param {number} m - Rows of Q
 * @param {number} qCols - Columns of Q (min(m, n) for reduced, m for full)
 * @returns {Float64Array} Row-major m*qCols storage of Q
 */
function accumulateQ(vs, betas, m, qCols) {
  const Q = new Float64Array(m * qCols);
  for (let i = 0; i < qCols; i++) Q[i * qCols + i] = 1;
  for (let j = vs.length - 1; j >= 0; j--) {
    const v = vs[j];
    const beta = betas[j];
    if (beta === 0) continue;
    for (let c = 0; c < qCols; c++) {
      let s = 0;
      for (let i = j; i < m; i++) s += v[i - j] * Q[i * qCols + c];
      s *= beta;
      for (let i = j; i < m; i++) Q[i * qCols + c] -= s * v[i - j];
    }
  }
  return Q;
}

/**
 * QR decomposition A = Q R via Householder reflections.
 *
 * Works for any shape: m >= n and m < n alike.
 *
 * @param {Array<Array<number>>} A - m x n matrix
 * @param {Object} [options]
 * @param {string} [options.mode='reduced'] - 'reduced' (Q is m x min(m,n),
 *   R is min(m,n) x n) or 'full' (Q is m x m, R is m x n)
 * @returns {{Q: Array<Array<number>>, R: Array<Array<number>>}}
 */
export function qr(A, options = {}) {
  const { data, m, n } = fromNested(A, 'A');
  const mode = options.mode !== undefined ? options.mode : 'reduced';
  if (mode !== 'reduced' && mode !== 'full') {
    throw new Error(`qr: options.mode must be 'reduced' or 'full' (got '${mode}')`);
  }
  const { vs, betas, k } = householderFactor(data, m, n);
  const qCols = mode === 'full' ? m : k;
  const Qflat = accumulateQ(vs, betas, m, qCols);
  const rRows = mode === 'full' ? m : k;
  return {
    Q: toNested(Qflat, m, qCols),
    R: toNested(data.subarray(0, rRows * n), rRows, n),
  };
}

/**
 * Least-squares solution of A x ≈ b via reduced QR.
 *
 * Requires m >= n and full column rank: solves R x = Q^T b by back
 * substitution. Throws for rank-deficient R.
 *
 * @param {Array<Array<number>>} A - m x n matrix with m >= n
 * @param {Array<number>} b - Right-hand side of length m
 * @returns {{x: Array<number>, residualNorm: number}} residualNorm = ||A x - b||_2
 */
export function lstsq(A, b) {
  const { data, m, n } = fromNested(A, 'A');
  if (m < n) {
    throw new Error(`lstsq: A must have at least as many rows as columns (got ${m}x${n})`);
  }
  const bv = vecFrom(b, m, 'b');
  const original = data.slice();
  const { vs, betas, k } = householderFactor(data, m, n);

  // Rank check on the diagonal of R against its largest entry.
  let maxR = 0;
  for (let i = 0; i < k; i++) {
    for (let j = i; j < n; j++) maxR = Math.max(maxR, Math.abs(data[i * n + j]));
  }
  for (let i = 0; i < n; i++) {
    if (Math.abs(data[i * n + i]) < 1e-12 * maxR || maxR === 0) {
      throw new Error(
        `lstsq: A is rank deficient (R[${i}][${i}] is negligible); ` +
        'use pinv for a minimum-norm solution'
      );
    }
  }

  // Compute Q^T b = H_{k-1} ... H_0 b in place.
  for (let j = 0; j < k; j++) {
    const v = vs[j];
    const beta = betas[j];
    if (beta === 0) continue;
    let s = 0;
    for (let i = j; i < m; i++) s += v[i - j] * bv[i];
    s *= beta;
    for (let i = j; i < m; i++) bv[i] -= s * v[i - j];
  }

  // Back substitution: R x = (Q^T b)[0..n-1].
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = bv[i];
    for (let j = i + 1; j < n; j++) s -= data[i * n + j] * x[j];
    x[i] = s / data[i * n + i];
  }

  // Residual norm from the original A.
  let rr = 0;
  for (let i = 0; i < m; i++) {
    let s = -b[i];
    for (let j = 0; j < n; j++) s += original[i * n + j] * x[j];
    rr += s * s;
  }
  return { x, residualNorm: Math.sqrt(rr) };
}
