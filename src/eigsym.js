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
import { choleskyFlat, solveLowerFlat, solveLowerTransposeFlat } from './cholesky.js';

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

/** Transpose an n x n matrix held in flat row-major storage. */
function transposeFlat(X, n) {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) out[j * n + i] = X[i * n + j];
  }
  return out;
}

/** Average an n x n flat matrix with its transpose, in place. */
function symmetrizeFlat(X, n) {
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const avg = (X[i * n + j] + X[j * n + i]) / 2;
      X[i * n + j] = avg;
      X[j * n + i] = avg;
    }
  }
  return X;
}

/** Product of two n x n matrices in flat row-major storage. */
function matmulFlat(A, B, n) {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < n; k++) {
      const a = A[i * n + k];
      if (a === 0) continue;
      for (let j = 0; j < n; j++) out[i * n + j] += a * B[k * n + j];
    }
  }
  return out;
}

/**
 * Optionally scale each column to unit euclidean length, then fix the sign so
 * the largest-magnitude component is positive (eigSym's convention).
 *
 * A column that lies in the discarded null space comes out of the reduction at
 * roundoff magnitude rather than exactly zero; scaling it would amplify pure
 * noise into a unit vector, so columns far below the largest one are zeroed
 * instead. The 1e-12 relative bound sits well below any genuinely retained
 * column (bounded by the reduction's own cutoff, ~1e-8 relative at the default
 * rcond) and well above roundoff, ~1e-16 relative.
 */
function normalizeColumnsFlat(X, n, unitLength) {
  if (unitLength) {
    const norms = new Float64Array(n);
    let maxNorm = 0;
    for (let c = 0; c < n; c++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += X[i * n + c] * X[i * n + c];
      norms[c] = Math.sqrt(sum);
      if (norms[c] > maxNorm) maxNorm = norms[c];
    }
    const floor = 1e-12 * maxNorm;
    for (let c = 0; c < n; c++) {
      if (norms[c] > floor) {
        for (let i = 0; i < n; i++) X[i * n + c] /= norms[c];
      } else {
        for (let i = 0; i < n; i++) X[i * n + c] = 0;
      }
    }
  }
  for (let c = 0; c < n; c++) {
    let maxAbs = 0;
    let sign = 1;
    for (let i = 0; i < n; i++) {
      const val = X[i * n + c];
      if (Math.abs(val) > maxAbs) {
        maxAbs = Math.abs(val);
        sign = val >= 0 ? 1 : -1;
      }
    }
    if (sign < 0) {
      for (let i = 0; i < n; i++) X[i * n + c] = -X[i * n + c];
    }
  }
  return X;
}

/**
 * Inverse square root of a symmetric positive semidefinite matrix on flat
 * storage: V diag(w) V^T with w_i = 1/sqrt(lambda_i) above the cutoff and
 * 0 at or below it, so null-space directions are dropped rather than
 * amplified into noise.
 */
function invSqrtSymFlat(M, options, caller) {
  const n = M.n;
  const { values, vectors } = eigSym(toNested(M.data, n, n));
  // eigSym returns descending values, so the extremes bound the magnitude
  const scale = Math.max(Math.abs(values[0]), Math.abs(values[n - 1]));
  if (scale === 0) {
    return new Float64Array(n * n); // the zero matrix has no invertible part
  }
  const rcond = options.rcond !== undefined ? options.rcond : n * Number.EPSILON;
  const cutoff = rcond * scale;
  // Roundoff puts tiny negatives on a semidefinite matrix's spectrum; a
  // clearly negative eigenvalue instead means the input is indefinite.
  const negativeBound = -1e-10 * scale;
  for (const v of values) {
    if (v < negativeBound) {
      throw new Error(`${caller}: matrix is not positive semidefinite (eigenvalue ${v})`);
    }
  }
  const w = values.map((v) => (v > cutoff ? 1 / Math.sqrt(v) : 0));
  const V = fromNested(vectors, 'vectors').data;
  const out = new Float64Array(n * n);
  for (let i = 0; i <= n - 1; i++) {
    for (let j = i; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += V[i * n + k] * w[k] * V[j * n + k];
      out[i * n + j] = s;
      out[j * n + i] = s;
    }
  }
  return out;
}

/**
 * Inverse square root of a symmetric positive semidefinite matrix:
 * the symmetric W with W A W = I on A's range, and W = 0 on its null space.
 *
 * @param {Array<Array<number>>} A - Symmetric positive semidefinite matrix
 * @param {Object} [options]
 * @param {number} [options.rcond] - Relative eigenvalue cutoff below which a
 *   direction is treated as null; default n * eps
 * @returns {Array<Array<number>>} Symmetric n x n inverse square root
 * @throws {Error} When A is not symmetric or has a clearly negative eigenvalue
 */
export function invSqrtSym(A, options = {}) {
  const M = fromNested(A, 'A');
  assertSymmetric(M, 'invSqrtSym');
  return toNested(invSqrtSymFlat(M, options, 'invSqrtSym'), M.n, M.n);
}

/**
 * Generalized symmetric eigenproblem A x = lambda B x, for symmetric A and
 * symmetric positive (semi)definite B.
 *
 * When B is positive definite this is the textbook Cholesky reduction:
 * B = L L^T, eigendecompose C = L^-1 A L^-T, then x = L^-T y. The returned
 * eigenvectors are B-orthonormal (x^T B x = 1), matching LAPACK/scipy's
 * `eigh(A, B)`.
 *
 * When B is only semidefinite the Cholesky factorization does not exist —
 * scipy raises here — so the reduction falls back to B's truncated inverse
 * square root (see invSqrtSym). Note what this solves: with P the orthogonal
 * projector onto range(B), the returned pairs satisfy
 *
 *     P A x = lambda B x
 *
 * the eigenproblem of A restricted to range(B), which is the most that is
 * defined when B is singular — off that range A x = lambda B x generally has
 * no solution at all. Null directions come back as zero eigenvectors with
 * zero eigenvalues, and the remaining vectors are scaled to unit euclidean
 * length, since B-orthonormality is undefined once B is singular. `definite`
 * reports which route ran, so callers needing scipy's strictness can check it
 * (or pass `strict`).
 *
 * @param {Array<Array<number>>} A - Symmetric matrix
 * @param {Array<Array<number>>} B - Symmetric positive (semi)definite matrix
 * @param {Object} [options]
 * @param {boolean} [options.strict=false] - Throw instead of falling back when
 *   B is not positive definite
 * @param {number} [options.rcond] - Relative eigenvalue cutoff for the
 *   semidefinite fallback; default n * eps
 * @returns {{values: Array<number>, vectors: Array<Array<number>>, definite: boolean}}
 *   values[i] descending; vectors' column i is the eigenvector for values[i]
 * @throws {Error} When A or B is not symmetric, sizes disagree, or `strict`
 *   is set and B is not positive definite
 */
export function eigSymGeneralized(A, B, options = {}) {
  const MA = fromNested(A, 'A');
  const MB = fromNested(B, 'B');
  assertSymmetric(MA, 'eigSymGeneralized (A)');
  assertSymmetric(MB, 'eigSymGeneralized (B)');
  if (MA.n !== MB.n) {
    throw new Error(
      `eigSymGeneralized: A and B must have the same size (got ${MA.n} and ${MB.n})`,
    );
  }
  const n = MA.n;

  let L = null;
  try {
    L = choleskyFlat(MB, 'eigSymGeneralized');
  } catch (err) {
    if (options.strict === true) throw err;
  }

  if (L !== null) {
    // C = L^-1 A L^-T: solve L Z = A, then L C^T = Z^T. C is symmetric up
    // to roundoff, which eigSym validates strictly, so average it.
    const Z = solveLowerFlat(L, n, MA.data, n);
    const C = symmetrizeFlat(solveLowerFlat(L, n, transposeFlat(Z, n), n), n);
    const { values, vectors } = eigSym(toNested(C, n, n));
    // x = L^-T y, which is B-orthonormal because y is orthonormal
    const X = solveLowerTransposeFlat(L, n, fromNested(vectors, 'vectors').data, n);
    return { values, vectors: toNested(normalizeColumnsFlat(X, n, false), n, n), definite: true };
  }

  const W = invSqrtSymFlat(MB, options, 'eigSymGeneralized');
  const M = symmetrizeFlat(matmulFlat(matmulFlat(W, MA.data, n), W, n), n);
  const { values, vectors } = eigSym(toNested(M, n, n));
  const X = matmulFlat(W, fromNested(vectors, 'vectors').data, n);
  return { values, vectors: toNested(normalizeColumnsFlat(X, n, true), n, n), definite: false };
}
