/**
 * Symmetric eigendecomposition by the cyclic Jacobi rotation method.
 *
 * Slower than tridiagonal QL for large n but simple, unconditionally
 * stable, and accurate to machine precision — the right trade-off for
 * the suite's target sizes (covariance/correlation matrices, n up to a
 * few hundred). Eigenvalues are returned in descending order with
 * orthonormal eigenvectors as matrix columns.
 */

import { assertSymmetric, fromNested, toNested } from './_mat.js';

/**
 * Eigendecomposition of a symmetric matrix: A = V diag(values) V^T.
 *
 * @param {Array<Array<number>>} A - Symmetric matrix (validated to 1e-10)
 * @param {Object} [options]
 * @param {number} [options.maxSweeps=60] - Maximum Jacobi sweeps
 * @param {number} [options.tol=1e-14] - Off-diagonal convergence tolerance,
 *   relative to the Frobenius norm of the diagonal
 * @returns {{values: Array<number>, vectors: Array<Array<number>>}}
 *   values[i] descending; vectors' column i is the eigenvector for values[i]
 */
export function eigSym(A, options = {}) {
  const M = fromNested(A);
  assertSymmetric(M, 'eigSym');
  const n = M.n;
  const maxSweeps = options.maxSweeps || 60;
  const tol = options.tol || 1e-14;

  const a = Float64Array.from(M.data);
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;

  let scale = 0;
  for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(a[i * n + i]));
  for (let i = 0; i < a.length; i++) scale = Math.max(scale, Math.abs(a[i]));
  const thresholdBase = tol * Math.max(1, scale);

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    // Sum of off-diagonal magnitudes — the convergence measure
    let off = 0;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) off += Math.abs(a[p * n + q]);
    }
    if (off <= thresholdBase * n) break;

    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (Math.abs(apq) <= thresholdBase * 1e-2) continue;

        const app = a[p * n + p];
        const aqq = a[q * n + q];
        const theta = (aqq - app) / (2 * apq);
        // Stable tangent of the rotation angle
        const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1)) ||
          1 / (theta + Math.sign(theta || 1) * Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        const tau = s / (1 + c);

        // Update the p and q rows/columns of A (symmetric Jacobi update)
        a[p * n + p] = app - t * apq;
        a[q * n + q] = aqq + t * apq;
        a[p * n + q] = 0;
        a[q * n + p] = 0;

        for (let i = 0; i < n; i++) {
          if (i !== p && i !== q) {
            const aip = a[i * n + p];
            const aiq = a[i * n + q];
            a[i * n + p] = aip - s * (aiq + tau * aip);
            a[i * n + q] = aiq + s * (aip - tau * aiq);
            a[p * n + i] = a[i * n + p];
            a[q * n + i] = a[i * n + q];
          }
          // Accumulate the rotation into the eigenvector matrix
          const vip = v[i * n + p];
          const viq = v[i * n + q];
          v[i * n + p] = vip - s * (viq + tau * vip);
          v[i * n + q] = viq + s * (vip - tau * viq);
        }
      }
    }
  }

  // Extract eigenvalues, sort descending, reorder eigenvector columns
  const order = Array.from({ length: n }, (_, i) => i)
    .sort((i, j) => a[j * n + j] - a[i * n + i]);
  const values = order.map((i) => a[i * n + i]);
  const vectors = new Float64Array(n * n);
  for (let col = 0; col < n; col++) {
    const src = order[col];
    // Sign convention: largest-magnitude component positive (deterministic)
    let maxAbs = 0;
    let sign = 1;
    for (let i = 0; i < n; i++) {
      const val = v[i * n + src];
      if (Math.abs(val) > maxAbs) {
        maxAbs = Math.abs(val);
        sign = val >= 0 ? 1 : -1;
      }
    }
    for (let i = 0; i < n; i++) {
      vectors[i * n + col] = sign * v[i * n + src];
    }
  }

  return { values, vectors: toNested(vectors, n, n) };
}
