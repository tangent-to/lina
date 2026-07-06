/**
 * Type declarations for @tangent.to/lina.
 *
 * The runtime is plain JavaScript (see index.js and the modules it re-exports);
 * these declarations give the public API explicit types for editors, `deno doc`
 * and JSR slow-types. Matrices are `number[][]` (nested row-major), vectors are
 * `number[]`; decompositions return plain objects.
 *
 * @module
 */

// --- lu.js ---

/**
 * LU factorization with partial pivoting: P A = L U.
 * L unit lower triangular, U upper triangular, P a permutation matrix.
 */
export function lu(A: number[][]): { L: number[][]; U: number[][]; P: number[][] };

/**
 * Solve A x = b via LU factorization with partial pivoting. `b` is a vector
 * (length n) or a nested n×k matrix of right-hand sides (solved column by
 * column); the result matches: a vector or a nested n×k matrix.
 */
export function solve(A: number[][], b: number[] | number[][]): number[] | number[][];

/** Determinant via LU factorization. Returns 0 for singular matrices. */
export function det(A: number[][]): number;

/** Matrix inverse via solve(A, I). */
export function inv(A: number[][]): number[][];

// --- cholesky.js ---

/**
 * Cholesky factorization of a symmetric positive definite matrix.
 * Returns lower triangular L with A = L L^T.
 */
export function cholesky(A: number[][]): number[][];

/** Solve A x = b given the Cholesky factor L of A (A = L L^T). */
export function choleskySolve(L: number[][], b: number[]): number[];

/** Test positive definiteness by attempting a Cholesky factorization. */
export function isPositiveDefinite(A: number[][]): boolean;

// --- qr.js ---

/**
 * QR decomposition A = Q R via Householder reflections. `mode` is 'reduced'
 * (default; Q is m×min(m,n), R is min(m,n)×n) or 'full' (Q is m×m, R is m×n).
 */
export function qr(
  A: number[][],
  options?: { mode?: 'reduced' | 'full' },
): { Q: number[][]; R: number[][] };

/**
 * Least-squares solution of A x ≈ b via reduced QR (requires m >= n and full
 * column rank). residualNorm = ||A x - b||_2.
 */
export function lstsq(A: number[][], b: number[]): { x: number[]; residualNorm: number };

// --- svd.js ---

/**
 * Thin SVD: A = U diag(s) V^T with U m×k, s length k, V n×k, k = min(m, n).
 * Singular values are non-negative and descending.
 */
export function svd(
  A: number[][],
  options?: { maxSweeps?: number; tol?: number },
): { U: number[][]; s: number[]; V: number[][] };

/** Numerical rank via SVD. Default tol = max(m,n) * eps * s[0]. */
export function rank(A: number[][], tol?: number): number;

/** Condition number (2-norm): s_max / s_min. Infinity when singular. */
export function cond(A: number[][]): number;

/** Moore-Penrose pseudoinverse via SVD (numpy's default cutoff). n×m matrix. */
export function pinv(A: number[][], rcond?: number): number[][];

/** Minimum-norm least squares via the pseudoinverse (works for any rank). */
export function pinvSolve(A: number[][], b: number[]): number[];

// --- eigsym.js ---

/**
 * Eigendecomposition of a symmetric matrix: A = V diag(values) V^T.
 * values[i] descending; column i of vectors is the eigenvector for values[i].
 */
export function eigSym(
  A: number[][],
  options?: { maxSweeps?: number; tol?: number },
): { values: number[]; vectors: number[][] };

// --- ops.js ---

/** Matrix product A B, or matrix-vector product A b. */
export function matmul(A: number[][], B: number[][] | number[]): number[][] | number[];

/** Matrix transpose (m×n to n×m). */
export function transpose(A: number[][]): number[][];

/** Identity matrix of size n. */
export function identity(n: number): number[][];

/**
 * Build a diagonal matrix from a vector (n×n), or extract the diagonal of a
 * matrix (length min(m, n)).
 */
export function diag(x: number[] | number[][]): number[][] | number[];

/**
 * Matrix or vector norm. Matrix: 'fro', 1 (max column abs sum), or Infinity
 * (max row abs sum). Vector: 'fro'/2 (euclidean), 1 (abs sum), Infinity (max abs).
 */
export function norm(A: number[][] | number[], kind?: 'fro' | 1 | 2 | number): number;

/** Sum of the diagonal of a square matrix. */
export function trace(A: number[][]): number;

/** Default namespace object bundling every named export. */
declare const _default: {
  lu: typeof lu;
  solve: typeof solve;
  det: typeof det;
  inv: typeof inv;
  cholesky: typeof cholesky;
  choleskySolve: typeof choleskySolve;
  isPositiveDefinite: typeof isPositiveDefinite;
  qr: typeof qr;
  lstsq: typeof lstsq;
  svd: typeof svd;
  pinv: typeof pinv;
  pinvSolve: typeof pinvSolve;
  rank: typeof rank;
  cond: typeof cond;
  eigSym: typeof eigSym;
  matmul: typeof matmul;
  transpose: typeof transpose;
  identity: typeof identity;
  diag: typeof diag;
  norm: typeof norm;
  trace: typeof trace;
};
export default _default;
