import { describe, expect, it } from 'vitest';
import { eigSym, eigSymGeneralized, invSqrtSym } from '../src/eigsym.js';
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

/** Symmetric positive definite matrix of size n (diagonally dominant). */
function spd(n, seed) {
  const R = randMat(n, n, seed);
  const A = mm(T(R), R);
  for (let i = 0; i < n; i++) A[i][i] += n;
  return A;
}

/** Worst relative residual ||A x - lambda B x|| / ||A x|| over the given columns. */
function genResidual(A, B, values, vectors, columns) {
  const n = A.length;
  let worst = 0;
  for (const c of columns) {
    const x = vectors.map((row) => row[c]);
    let num = 0;
    let den = 0;
    for (let i = 0; i < n; i++) {
      let ax = 0;
      let bx = 0;
      for (let j = 0; j < n; j++) {
        ax += A[i][j] * x[j];
        bx += B[i][j] * x[j];
      }
      const d = ax - values[c] * bx;
      num += d * d;
      den += ax * ax;
    }
    worst = Math.max(worst, Math.sqrt(num) / Math.max(Math.sqrt(den), 1e-300));
  }
  return worst;
}

describe('eigSymGeneralized', () => {
  it('reduces to eigSym when B is the identity', () => {
    const A = spd(5, 3);
    const gen = eigSymGeneralized(A, identity(5));
    const plain = eigSym(A);
    expect(gen.definite).toBe(true);
    for (let i = 0; i < 5; i++) {
      expect(gen.values[i]).toBeCloseTo(plain.values[i], 10);
    }
    expect(maxAbsDiff(gen.vectors, plain.vectors)).toBeLessThan(1e-10);
  });

  it('solves a known 2x2 problem', () => {
    // A x = lambda B x with A = [[2,1],[1,3]], B = diag(2,1):
    // det(A - lambda B) = 2 lambda^2 - 8 lambda + 5, roots (4 +/- sqrt(6)) / 2
    const { values } = eigSymGeneralized([[2, 1], [1, 3]], [[2, 0], [0, 1]]);
    expect(values[0]).toBeCloseTo((4 + Math.sqrt(6)) / 2, 12);
    expect(values[1]).toBeCloseTo((4 - Math.sqrt(6)) / 2, 12);
  });

  it('satisfies A x = lambda B x with B-orthonormal vectors', () => {
    for (const [n, seed] of [[4, 5], [9, 13], [15, 21]]) {
      const A = mm(T(randMat(n, n, seed)), randMat(n, n, seed)); // symmetric
      const B = spd(n, seed + 1);
      const { values, vectors, definite } = eigSymGeneralized(A, B);
      expect(definite).toBe(true);
      const all = Array.from({ length: n }, (_, i) => i);
      expect(genResidual(A, B, values, vectors, all)).toBeLessThan(1e-9);
      // x^T B x = I
      expect(maxAbsDiff(mm(mm(T(vectors), B), vectors), identity(n))).toBeLessThan(1e-9);
      // descending
      for (let i = 1; i < n; i++) {
        expect(values[i]).toBeLessThanOrEqual(values[i - 1] + 1e-12);
      }
    }
  });

  it('falls back to the problem projected onto range(B) when B is singular', () => {
    // B has rank 2 of 3; the third direction carries no information
    const R = randMat(3, 2, 31);
    const B = mm(R, T(R));
    const A = spd(3, 33);
    const { values, vectors, definite } = eigSymGeneralized(A, B);
    expect(definite).toBe(false);

    // Orthogonal projector onto range(B), from B's own eigenbasis
    const { values: bv, vectors: bvec } = eigSym(B);
    const cutoff = bv[0] * 1e-10;
    const P = Array.from({ length: 3 }, () => new Array(3).fill(0));
    for (let k = 0; k < 3; k++) {
      if (bv[k] <= cutoff) continue;
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) P[i][j] += bvec[i][k] * bvec[j][k];
      }
    }

    // On range(B) the solution is exact: P A x = lambda B x. Off it, the
    // equation has no solution and the null direction is returned as zero.
    const live = values.map((v, i) => (Math.abs(v) > 1e-8 ? i : -1)).filter((i) => i >= 0);
    expect(live.length).toBe(2);
    expect(genResidual(mm(P, A), B, values, vectors, live)).toBeLessThan(1e-8);
    const nullCol = values.findIndex((v) => Math.abs(v) <= 1e-8);
    expect(Math.max(...vectors.map((row) => Math.abs(row[nullCol])))).toBeLessThan(1e-12);
  });

  it('throws on a singular B when strict is set', () => {
    const B = [[1, 0], [0, 0]];
    expect(() => eigSymGeneralized([[2, 1], [1, 3]], B, { strict: true }))
      .toThrow(/not positive definite/);
    expect(eigSymGeneralized([[2, 1], [1, 3]], B).definite).toBe(false);
  });

  it('rejects non-symmetric or mismatched input', () => {
    const I2 = identity(2);
    expect(() => eigSymGeneralized([[1, 2], [3, 4]], I2)).toThrow(/not symmetric/);
    expect(() => eigSymGeneralized(I2, [[1, 2], [3, 4]])).toThrow(/not symmetric/);
    expect(() => eigSymGeneralized(identity(3), I2)).toThrow(/same size/);
  });
});

describe('invSqrtSym', () => {
  it('inverts the square root of a positive definite matrix', () => {
    const A = spd(6, 41);
    const W = invSqrtSym(A);
    // W A W = I, and W is symmetric
    expect(maxAbsDiff(mm(mm(W, A), W), identity(6))).toBeLessThan(1e-10);
    expect(maxAbsDiff(W, T(W))).toBeLessThan(1e-12);
    // W W = inv(A)
    expect(maxAbsDiff(mm(mm(W, W), A), identity(6))).toBeLessThan(1e-10);
  });

  it('is exact on diagonal matrices', () => {
    expect(invSqrtSym([[4, 0], [0, 9]])).toEqual([[0.5, 0], [0, 1 / 3]]);
  });

  it('drops null directions instead of amplifying them', () => {
    const W = invSqrtSym([[4, 0], [0, 0]]);
    expect(W).toEqual([[0.5, 0], [0, 0]]);
    // Roundoff-level negatives are treated as zero, not as an error
    expect(invSqrtSym([[4, 0], [0, -1e-18]])).toEqual([[0.5, 0], [0, 0]]);
    expect(invSqrtSym([[0, 0], [0, 0]])).toEqual([[0, 0], [0, 0]]);
  });

  it('rejects indefinite and non-symmetric input', () => {
    expect(() => invSqrtSym([[1, 0], [0, -5]])).toThrow(/not positive semidefinite/);
    expect(() => invSqrtSym([[1, 2], [3, 4]])).toThrow(/not symmetric/);
  });
});
