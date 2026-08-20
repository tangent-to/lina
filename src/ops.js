/**
 * Basic dense matrix operations: multiply, transpose, identity, diagonal,
 * norms, and trace.
 *
 * Public APIs accept and return nested row-major matrices. Most kernels run
 * on flat Float64Array storage (see _mat.js); `matmul` is the exception and
 * reads nested rows directly, which measured faster on both V8 and
 * JavaScriptCore once the conversion is counted.
 */

import { fromNested, shapeOfNested, toNested, vecFrom } from './_mat.js';

/**
 * Largest dimension for which matmul reads nested rows directly instead of
 * converting to flat storage. Set from the JavaScriptCore crossover, which
 * sits between 128 and 160; V8 prefers nested at every size measured, so this
 * gives up ~1.2x there to avoid a 0.64-0.80x regression in Safari. Consumers
 * profile at a maximum dimension of 100, well inside the nested path.
 */
const NESTED_MATMUL_MAX_DIM = 128;

/**
 * Matrix product A B, or matrix-vector product A b.
 *
 * @param {Array<Array<number>>} A - m x n nested matrix
 * @param {Array<Array<number>>|Array<number>} B - n x p nested matrix, or
 *   vector of length n
 * @returns {Array<Array<number>>|Array<number>} m x p nested matrix, or
 *   vector of length m
 */
export function matmul(A, B) {
  const { m, n } = shapeOfNested(A, 'A');

  if (Array.isArray(B) && !Array.isArray(B[0])) {
    const x = vecFrom(B, n, 'B');
    const y = new Array(m);
    for (let i = 0; i < m; i++) {
      const Ai = A[i];
      let s = 0;
      for (let j = 0; j < n; j++) s += Ai[j] * x[j];
      y[i] = s;
    }
    return y;
  }

  const { m: bm, n: p } = shapeOfNested(B, 'B');
  if (bm !== n) {
    throw new Error(`matmul: dimension mismatch (A is ${m}x${n}, B is ${bm}x${p})`);
  }

  if (Math.max(m, n, p) <= NESTED_MATMUL_MAX_DIM) {
    // Same i-k-j order, reading the caller's rows directly. Below this size the
    // round trip through flat storage costs more than it saves, and on small
    // matrices — where consumers spend the overwhelming majority of their
    // calls — the conversion dominates the arithmetic outright (measured 3.7x
    // on 3x3 under V8, 2.3x under JavaScriptCore).
    const C = new Array(m);
    for (let i = 0; i < m; i++) {
      const Ci = new Array(p).fill(0);
      const Ai = A[i];
      for (let k = 0; k < n; k++) {
        const aik = Ai[k];
        if (aik === 0) continue;
        const Bk = B[k];
        for (let j = 0; j < p; j++) {
          Ci[j] += aik * Bk[j];
        }
      }
      C[i] = Ci;
    }
    return C;
  }

  // Above it the engines disagree: V8 still prefers nested by ~1.2x, but
  // JavaScriptCore inverts hard (0.80x at n=160, 0.64x at n=256), so the flat
  // kernel is the safe choice for the sizes where the conversion is amortised
  // anyway. Both kernels produce bit-identical results.
  const a = fromNested(A, 'A').data;
  const b = fromNested(B, 'B').data;
  const c = new Float64Array(m * p);
  for (let i = 0; i < m; i++) {
    for (let k = 0; k < n; k++) {
      const aik = a[i * n + k];
      if (aik === 0) continue;
      for (let j = 0; j < p; j++) {
        c[i * p + j] += aik * b[k * p + j];
      }
    }
  }
  return toNested(c, m, p);
}

/**
 * Matrix transpose.
 *
 * @param {Array<Array<number>>} A - m x n nested matrix
 * @returns {Array<Array<number>>} n x m nested matrix
 */
export function transpose(A) {
  const { data, m, n } = fromNested(A, 'A');
  const t = new Float64Array(n * m);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      t[j * m + i] = data[i * n + j];
    }
  }
  return toNested(t, n, m);
}

/**
 * Identity matrix of size n.
 *
 * @param {number} n - Dimension (positive integer)
 * @returns {Array<Array<number>>}
 */
export function identity(n) {
  if (!Number.isInteger(n) || n <= 0) {
    throw new Error(`identity: n must be a positive integer (got ${n})`);
  }
  const I = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(n).fill(0);
    row[i] = 1;
    I[i] = row;
  }
  return I;
}

/**
 * Build a diagonal matrix from a vector, or extract the diagonal of a matrix.
 *
 * @param {Array<number>|Array<Array<number>>} x - Vector (returns an n x n
 *   nested matrix) or nested matrix (returns its diagonal, length min(m, n))
 * @returns {Array<Array<number>>|Array<number>}
 */
export function diag(x) {
  if (!Array.isArray(x) || x.length === 0) {
    throw new Error('diag: argument must be a non-empty vector or nested matrix');
  }
  if (Array.isArray(x[0])) {
    const { data, m, n } = fromNested(x, 'A');
    const len = Math.min(m, n);
    const d = new Array(len);
    for (let i = 0; i < len; i++) d[i] = data[i * n + i];
    return d;
  }
  const n = x.length;
  const D = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(n).fill(0);
    row[i] = x[i];
    D[i] = row;
  }
  return D;
}

/**
 * Matrix or vector norm.
 *
 * For a nested matrix: 'fro' (Frobenius), 1 (max column abs sum), or
 * Infinity (max row abs sum). For a vector: 'fro' or 2 (euclidean),
 * 1 (abs sum), or Infinity (max abs).
 *
 * @param {Array<Array<number>>|Array<number>} A - Nested matrix or vector
 * @param {'fro'|1|2|Infinity} [kind='fro'] - Norm kind
 * @returns {number}
 */
export function norm(A, kind = 'fro') {
  if (!Array.isArray(A) || A.length === 0) {
    throw new Error('norm: argument must be a non-empty vector or nested matrix');
  }

  if (!Array.isArray(A[0])) {
    if (kind === 'fro' || kind === 2) {
      let s = 0;
      for (let i = 0; i < A.length; i++) s += A[i] * A[i];
      return Math.sqrt(s);
    }
    if (kind === 1) {
      let s = 0;
      for (let i = 0; i < A.length; i++) s += Math.abs(A[i]);
      return s;
    }
    if (kind === Infinity) {
      let s = 0;
      for (let i = 0; i < A.length; i++) s = Math.max(s, Math.abs(A[i]));
      return s;
    }
    throw new Error(`norm: unsupported vector norm kind ${String(kind)}`);
  }

  const { data, m, n } = fromNested(A, 'A');
  if (kind === 'fro') {
    let s = 0;
    for (let i = 0; i < data.length; i++) s += data[i] * data[i];
    return Math.sqrt(s);
  }
  if (kind === 1) {
    let best = 0;
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let i = 0; i < m; i++) s += Math.abs(data[i * n + j]);
      best = Math.max(best, s);
    }
    return best;
  }
  if (kind === Infinity) {
    let best = 0;
    for (let i = 0; i < m; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += Math.abs(data[i * n + j]);
      best = Math.max(best, s);
    }
    return best;
  }
  throw new Error(`norm: unsupported matrix norm kind ${String(kind)} (use 'fro', 1, or Infinity)`);
}

/**
 * Sum of the diagonal of a square matrix.
 *
 * @param {Array<Array<number>>} A - Square nested matrix
 * @returns {number}
 */
export function trace(A) {
  const { data, m, n } = fromNested(A, 'A');
  if (m !== n) {
    throw new Error(`trace: matrix must be square (got ${m}x${n})`);
  }
  let s = 0;
  for (let i = 0; i < n; i++) s += data[i * n + i];
  return s;
}
