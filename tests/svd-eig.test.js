import { describe, expect, it } from 'vitest';
import { eigSym } from '../src/eigsym.js';
import { cond, pinv, pinvSolve, rank, svd } from '../src/svd.js';

/** Multiply nested matrices (test helper). */
function mm(A, B) {
  const m = A.length;
  const k = B.length;
  const n = B[0].length;
  const C = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < m; i++) {
    for (let l = 0; l < k; l++) {
      const a = A[i][l];
      for (let j = 0; j < n; j++) C[i][j] += a * B[l][j];
    }
  }
  return C;
}
const T = (A) => A[0].map((_, j) => A.map((row) => row[j]));
const maxAbsDiff = (A, B) =>
  Math.max(...A.flatMap((row, i) => row.map((v, j) => Math.abs(v - B[i][j]))));

/** Deterministic pseudo-random matrix (LCG). */
function randMat(m, n, seed = 1) {
  let s = seed >>> 0;
  const next = () => ((s = (1103515245 * s + 12345) >>> 0), s / 4294967296 - 0.5);
  return Array.from({ length: m }, () => Array.from({ length: n }, () => next() * 4));
}

function checkSvd(A, tol = 1e-10) {
  const { U, s, V } = svd(A);
  const k = s.length;
  // Reconstruction
  const S = Array.from({ length: k }, (_, i) =>
    Array.from({ length: k }, (_, j) => (i === j ? s[i] : 0)));
  const rec = mm(mm(U, S), T(V));
  expect(maxAbsDiff(rec, A)).toBeLessThan(tol);
  // Orthonormal columns
  expect(maxAbsDiff(mm(T(U), U), identity(k))).toBeLessThan(tol);
  expect(maxAbsDiff(mm(T(V), V), identity(k))).toBeLessThan(tol);
  // Descending non-negative
  for (let i = 0; i < k; i++) {
    expect(s[i]).toBeGreaterThanOrEqual(0);
    if (i > 0) expect(s[i]).toBeLessThanOrEqual(s[i - 1] + 1e-14);
  }
}
const identity = (n) =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

describe('svd', () => {
  it('decomposes square, tall and wide matrices', () => {
    checkSvd(randMat(5, 5, 1));
    checkSvd(randMat(8, 3, 2));
    checkSvd(randMat(3, 8, 3));
  });

  it('matches known singular values', () => {
    // [[3, 0], [0, -2]] has singular values [3, 2]
    const { s } = svd([[3, 0], [0, -2]]);
    expect(s[0]).toBeCloseTo(3, 12);
    expect(s[1]).toBeCloseTo(2, 12);
    // [[1, 1], [1, 1]] has singular values [2, 0]
    const r = svd([[1, 1], [1, 1]]);
    expect(r.s[0]).toBeCloseTo(2, 12);
    expect(r.s[1]).toBeCloseTo(0, 12);
  });

  it('handles rank-deficient matrices', () => {
    const A = [[1, 2, 3], [2, 4, 6], [1, 1, 1]]; // rank 2
    const { s } = svd(A);
    expect(s[2]).toBeLessThan(1e-12);
    expect(rank(A)).toBe(2);
  });

  it('computes rank and cond', () => {
    expect(rank(identity(4))).toBe(4);
    expect(cond(identity(4))).toBeCloseTo(1, 12);
    expect(cond([[1, 0], [0, 1e-8]])).toBeCloseTo(1e8, 4);
    expect(cond([[1, 1], [1, 1]])).toBe(Infinity);
  });

  it('pinv satisfies the Moore-Penrose identities', () => {
    const A = randMat(6, 4, 5);
    const P = pinv(A);
    expect(maxAbsDiff(mm(mm(A, P), A), A)).toBeLessThan(1e-9);
    expect(maxAbsDiff(mm(mm(P, A), P), P)).toBeLessThan(1e-9);
    // Rank-deficient
    const B = [[1, 2], [2, 4], [3, 6]]; // rank 1
    const PB = pinv(B);
    expect(maxAbsDiff(mm(mm(B, PB), B), B)).toBeLessThan(1e-10);
  });

  it('pinvSolve returns the minimum-norm least-squares solution', () => {
    // Consistent overdetermined system
    const A = [[1, 0], [0, 1], [1, 1]];
    const x = pinvSolve(A, [1, 2, 3]);
    expect(x[0]).toBeCloseTo(1, 10);
    expect(x[1]).toBeCloseTo(2, 10);
  });
});

describe('eigSym', () => {
  it('diagonalizes a known symmetric matrix', () => {
    // [[2, 1], [1, 2]] has eigenvalues 3 and 1
    const { values, vectors } = eigSym([[2, 1], [1, 2]]);
    expect(values[0]).toBeCloseTo(3, 12);
    expect(values[1]).toBeCloseTo(1, 12);
    const inv2 = 1 / Math.SQRT2;
    expect(Math.abs(vectors[0][0])).toBeCloseTo(inv2, 10);
    expect(Math.abs(vectors[1][0])).toBeCloseTo(inv2, 10);
  });

  it('reconstructs A = V diag(values) V^T on random symmetric matrices', () => {
    const B = randMat(6, 6, 7);
    const A = mm(T(B), B); // symmetric PSD
    const { values, vectors } = eigSym(A);
    const D = values.map((v, i) => values.map((_, j) => (i === j ? v : 0)));
    expect(maxAbsDiff(mm(mm(vectors, D), T(vectors)), A)).toBeLessThan(1e-9);
    expect(maxAbsDiff(mm(T(vectors), vectors), identity(6))).toBeLessThan(1e-10);
    // PSD: all eigenvalues >= 0, descending
    for (let i = 0; i < 6; i++) {
      expect(values[i]).toBeGreaterThan(-1e-10);
      if (i > 0) expect(values[i]).toBeLessThanOrEqual(values[i - 1] + 1e-12);
    }
  });

  it('agrees with svd on symmetric PSD matrices', () => {
    const B = randMat(5, 5, 11);
    const A = mm(T(B), B);
    const { values } = eigSym(A);
    const { s } = svd(A);
    for (let i = 0; i < 5; i++) {
      expect(values[i]).toBeCloseTo(s[i], 8);
    }
  });

  it('rejects non-symmetric and non-square input', () => {
    expect(() => eigSym([[1, 2], [3, 4]])).toThrow(/not symmetric/);
    expect(() => eigSym([[1, 2, 3], [4, 5, 6]])).toThrow(/square/);
  });

  it('handles diagonal and identity matrices exactly', () => {
    const { values } = eigSym([[5, 0, 0], [0, -2, 0], [0, 0, 3]]);
    expect(values[0]).toBeCloseTo(5, 14);
    expect(values[1]).toBeCloseTo(3, 14);
    expect(values[2]).toBeCloseTo(-2, 14);
  });
});
