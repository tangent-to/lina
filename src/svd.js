/**
 * Singular value decomposition by one-sided Jacobi (Hestenes), plus the
 * SVD-derived utilities pinv, rank and cond.
 *
 * One-sided Jacobi orthogonalizes pairs of columns of A directly; it is
 * simple, unconditionally convergent, and computes even tiny singular
 * values to high relative accuracy — a good fit for the suite's target
 * sizes. For m < n the problem is transposed internally.
 */

import { fromNested, toNested, vecFrom } from './_mat.js';

/**
 * Thin SVD: A = U diag(s) V^T with U m×k, s length k, V n×k, k = min(m, n).
 * Singular values are non-negative and descending.
 *
 * @param {Array<Array<number>>} A - Matrix (any shape)
 * @param {Object} [options]
 * @param {number} [options.maxSweeps=60] - Maximum Jacobi sweeps
 * @param {number} [options.tol=1e-15] - Column-pair orthogonality tolerance
 * @returns {{U: Array<Array<number>>, s: Array<number>, V: Array<Array<number>>}}
 */
export function svd(A, options = {}) {
  const M = fromNested(A);
  const transposed = M.m < M.n;
  const m = transposed ? M.n : M.m;
  const n = transposed ? M.m : M.n;

  // Column-major working copy W (m×n, m >= n) for cache-friendly column ops
  const W = new Float64Array(m * n);
  if (transposed) {
    // W[:, j] = A^T[:, j] = A[j, :]
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < m; i++) {
        W[j * m + i] = M.data[j * M.n + i];
      }
    }
  } else {
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < m; i++) {
        W[j * m + i] = M.data[i * M.n + j];
      }
    }
  }

  // V accumulates the right rotations (n×n, column-major)
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;

  const maxSweeps = options.maxSweeps || 60;
  const tol = options.tol || 1e-15;

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let rotated = false;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        let alpha = 0;
        let beta = 0;
        let gam = 0;
        const cp = p * m;
        const cq = q * m;
        for (let i = 0; i < m; i++) {
          const wp = W[cp + i];
          const wq = W[cq + i];
          alpha += wp * wp;
          beta += wq * wq;
          gam += wp * wq;
        }

        if (Math.abs(gam) <= tol * Math.sqrt(alpha * beta) || alpha === 0 || beta === 0) {
          continue;
        }
        rotated = true;

        // Jacobi rotation that zeroes the (p, q) inner product.
        // sign(0) must be +1: zeta = 0 (equal column norms) needs a 45°
        // rotation, not a no-op.
        const zeta = (beta - alpha) / (2 * gam);
        const sgn = zeta >= 0 ? 1 : -1;
        const t = sgn / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = c * t;

        for (let i = 0; i < m; i++) {
          const wp = W[cp + i];
          const wq = W[cq + i];
          W[cp + i] = c * wp - s * wq;
          W[cq + i] = s * wp + c * wq;
        }
        for (let i = 0; i < n; i++) {
          const vp = V[p * n + i];
          const vq = V[q * n + i];
          V[p * n + i] = c * vp - s * vq;
          V[q * n + i] = s * vp + c * vq;
        }
      }
    }

    if (!rotated) break;
  }

  // Singular values = column norms; U columns = normalized W columns
  const sv = new Array(n);
  for (let j = 0; j < n; j++) {
    let norm = 0;
    for (let i = 0; i < m; i++) norm += W[j * m + i] * W[j * m + i];
    sv[j] = Math.sqrt(norm);
  }

  // Sort descending
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => sv[j] - sv[i]);
  const s = order.map((j) => sv[j]);

  const Umat = new Float64Array(m * n); // row-major m×n
  const Vmat = new Float64Array(n * n); // row-major n×n
  for (let col = 0; col < n; col++) {
    const src = order[col];
    const norm = sv[src];
    if (norm > 0) {
      for (let i = 0; i < m; i++) Umat[i * n + col] = W[src * m + i] / norm;
    } // zero singular value: leave U column zero (rank-deficient case)
    for (let i = 0; i < n; i++) Vmat[i * n + col] = V[src * n + i];
  }

  const Uout = toNested(Umat, m, n);
  const Vout = toNested(Vmat, n, n);

  // Undo the internal transpose: A^T = U s V^T  =>  A = V s U^T
  return transposed ? { U: Vout, s, V: Uout } : { U: Uout, s, V: Vout };
}

/**
 * Numerical rank via SVD.
 *
 * @param {Array<Array<number>>} A - Matrix
 * @param {number} [tol] - Threshold; default max(m,n) * eps * s[0] (numpy convention)
 * @returns {number}
 */
export function rank(A, tol) {
  const { s } = svd(A);
  const m = A.length;
  const n = A[0].length;
  const threshold = tol !== undefined ? tol : Math.max(m, n) * Number.EPSILON * (s[0] || 0);
  return s.filter((x) => x > threshold).length;
}

/**
 * Condition number (2-norm): s_max / s_min. Infinity when singular.
 *
 * @param {Array<Array<number>>} A - Matrix
 * @returns {number}
 */
export function cond(A) {
  const { s } = svd(A);
  const smin = s[s.length - 1];
  return smin > 0 ? s[0] / smin : Infinity;
}

/**
 * Moore-Penrose pseudoinverse via SVD, with numpy's default cutoff.
 * Solves rank-deficient least squares: x = pinv(A) b is the minimum-norm
 * solution.
 *
 * @param {Array<Array<number>>} A - Matrix (any shape)
 * @param {number} [rcond] - Relative cutoff; default max(m,n) * eps
 * @returns {Array<Array<number>>} n×m pseudoinverse
 */
export function pinv(A, rcond) {
  const { U, s, V } = svd(A);
  const m = U.length;
  const n = V.length;
  const k = s.length;
  const cutoff = (rcond !== undefined ? rcond : Math.max(m, n) * Number.EPSILON) * (s[0] || 0);

  // pinv = V diag(1/s) U^T, dropping singular values below the cutoff
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = new Array(m).fill(0);
  }
  for (let j = 0; j < k; j++) {
    if (s[j] <= cutoff) continue;
    const invS = 1 / s[j];
    for (let i = 0; i < n; i++) {
      const vij = V[i][j] * invS;
      if (vij === 0) continue;
      const row = out[i];
      for (let l = 0; l < m; l++) {
        row[l] += vij * U[l][j];
      }
    }
  }
  return out;
}

/**
 * Minimum-norm least squares via the pseudoinverse (works for any rank).
 *
 * @param {Array<Array<number>>} A - m×n matrix
 * @param {Array<number>} b - Vector of length m
 * @returns {Array<number>} x of length n
 */
export function pinvSolve(A, b) {
  const m = A.length;
  vecFrom(b, m, 'b');
  const P = pinv(A);
  const n = P.length;
  const x = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < m; j++) sum += P[i][j] * b[j];
    x[i] = sum;
  }
  return x;
}
