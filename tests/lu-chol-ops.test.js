import { describe, expect, it } from 'vitest';
import { det, inv, lu, luFactor, luFactorSolve, solve } from '../src/lu.js';
import { cholesky, choleskySolve, isPositiveDefinite } from '../src/cholesky.js';
import { diag, identity, matmul, norm, trace, transpose } from '../src/ops.js';

/** Maximum absolute entry difference between two nested matrices. */
function maxDiff(A, B) {
  let d = 0;
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < A[i].length; j++) {
      d = Math.max(d, Math.abs(A[i][j] - B[i][j]));
    }
  }
  return d;
}

/** Maximum absolute entry difference between two vectors. */
function maxVecDiff(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
  return d;
}

describe('lu', () => {
  const A5 = [
    [0, 2, 1, 4, 1],
    [3, 1, 2, 0, 5],
    [1, 4, 0, 2, 2],
    [2, 0, 3, 1, 1],
    [4, 1, 1, 3, 0],
  ];

  it('reconstructs PA = LU on a 5x5 requiring pivoting', () => {
    const { L, U, P } = lu(A5);
    expect(maxDiff(matmul(P, A5), matmul(L, U))).toBeLessThan(1e-12);
  });

  it('returns unit lower triangular L and upper triangular U', () => {
    const { L, U } = lu(A5);
    for (let i = 0; i < 5; i++) {
      expect(L[i][i]).toBe(1);
      for (let j = i + 1; j < 5; j++) expect(L[i][j]).toBe(0);
      for (let j = 0; j < i; j++) expect(U[i][j]).toBe(0);
    }
  });

  it('returns a valid permutation matrix P', () => {
    const { P } = lu(A5);
    for (let i = 0; i < 5; i++) {
      let rowSum = 0;
      let colSum = 0;
      for (let j = 0; j < 5; j++) {
        expect(P[i][j] === 0 || P[i][j] === 1).toBe(true);
        rowSum += P[i][j];
        colSum += P[j][i];
      }
      expect(rowSum).toBe(1);
      expect(colSum).toBe(1);
    }
  });
});

describe('luFactor', () => {
  const A5 = [
    [0, 2, 1, 4, 1],
    [3, 1, 2, 0, 5],
    [1, 4, 0, 2, 2],
    [2, 0, 3, 1, 1],
    [4, 1, 1, 3, 0],
  ];

  it('packs the same L/U values as lu() into flat combined storage', () => {
    const { L, U } = lu(A5);
    const { lu: a, perm, n, singular } = luFactor(A5);
    expect(n).toBe(5);
    expect(singular).toBe(false);
    expect(perm.length).toBe(5);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const expected = j < i ? L[i][j] : U[i][j];
        expect(a[i * n + j]).toBe(expected);
      }
    }
  });

  it('luFactorSolve matches solve() for a factorization', () => {
    const b = [1, 2, 3, 4, 5];
    const fac = luFactor(A5);
    const x = luFactorSolve(fac, b);
    expect(maxVecDiff(Array.from(x), solve(A5, b))).toBeLessThan(1e-12);
  });

  it('flags a singular matrix', () => {
    const { singular } = luFactor([
      [1, 2],
      [2, 4],
    ]);
    expect(singular).toBe(true);
  });
});

describe('solve', () => {
  const A = [
    [2, 0, 1],
    [1, 1, 0],
    [0, 3, 1],
  ];

  it('matches a hand-solved 3x3 system', () => {
    // A [1, 2, 3] = [5, 3, 9]
    const x = solve(A, [5, 3, 9]);
    expect(maxVecDiff(x, [1, 2, 3])).toBeLessThan(1e-12);
  });

  it('solves multiple right-hand sides column by column', () => {
    const X = [
      [1, 0],
      [2, -1],
      [3, 2],
    ];
    const B = matmul(A, X);
    const got = solve(A, B);
    expect(maxDiff(got, X)).toBeLessThan(1e-12);
  });

  it('throws mentioning "singular" for a singular matrix', () => {
    const S = [
      [1, 2],
      [2, 4],
    ];
    expect(() => solve(S, [1, 1])).toThrow(/singular/);
  });

  it('rejects a right-hand side of the wrong length', () => {
    expect(() => solve(A, [1, 2])).toThrow(/length 3/);
  });

  it('rejects a right-hand side matrix with the wrong row count', () => {
    expect(() => solve(A, [[1], [2]])).toThrow(/3 rows/);
  });
});

describe('det', () => {
  it('computes known determinants', () => {
    expect(det([[1, 2], [3, 4]])).toBeCloseTo(-2, 12);
    expect(det(identity(3))).toBeCloseTo(1, 12);
  });

  it('tracks the permutation sign', () => {
    expect(det([[0, 1], [1, 0]])).toBeCloseTo(-1, 12);
  });

  it('returns 0 for a singular matrix instead of throwing', () => {
    expect(det([[1, 2], [2, 4]])).toBe(0);
  });
});

describe('inv', () => {
  it('satisfies inv(A) A = I', () => {
    const A = [
      [4, 2, 1],
      [-1, 3, 0],
      [2, 5, 7],
    ];
    expect(maxDiff(matmul(inv(A), A), identity(3))).toBeLessThan(1e-10);
  });
});

describe('cholesky', () => {
  const SPD = [
    [4, 2, -1],
    [2, 5, 3],
    [-1, 3, 6],
  ];

  it('reconstructs a known SPD matrix as L L^T', () => {
    const L = cholesky(SPD);
    expect(maxDiff(matmul(L, transpose(L)), SPD)).toBeLessThan(1e-12);
    for (let i = 0; i < 3; i++) {
      for (let j = i + 1; j < 3; j++) expect(L[i][j]).toBe(0);
    }
  });

  it('handles A^T A + I built with ops', () => {
    const B = [
      [1, 2, 0],
      [-1, 1, 3],
      [2, 0, 1],
    ];
    const A = matmul(transpose(B), B).map((row, i) => row.map((v, j) => (i === j ? v + 1 : v)));
    const L = cholesky(A);
    expect(maxDiff(matmul(L, transpose(L)), A)).toBeLessThan(1e-12);
  });

  it('throws mentioning "not positive definite" with the pivot index', () => {
    expect(() => cholesky([[1, 2], [2, 1]])).toThrow(/not positive definite.*pivot 1/);
  });

  it('rejects non-symmetric input', () => {
    expect(() => cholesky([[1, 2], [0, 1]])).toThrow(/symmetric/);
  });

  it('choleskySolve matches solve', () => {
    const b = [1, -2, 3];
    const L = cholesky(SPD);
    expect(maxVecDiff(choleskySolve(L, b), solve(SPD, b))).toBeLessThan(1e-12);
  });
});

describe('isPositiveDefinite', () => {
  it('is true for an SPD matrix', () => {
    expect(isPositiveDefinite([[2, 1], [1, 2]])).toBe(true);
  });

  it('is false for indefinite and semidefinite matrices', () => {
    expect(isPositiveDefinite([[1, 2], [2, 1]])).toBe(false);
    expect(isPositiveDefinite([[0, 0], [0, 0]])).toBe(false);
  });
});

describe('ops', () => {
  it('matmul matches hand values', () => {
    const C = matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]);
    expect(C).toEqual([[19, 22], [43, 50]]);
  });

  it('matmul with a vector returns a vector', () => {
    expect(matmul([[1, 2], [3, 4]], [5, 6])).toEqual([17, 39]);
  });

  it('transpose matches hand values', () => {
    expect(transpose([[1, 2, 3], [4, 5, 6]])).toEqual([[1, 4], [2, 5], [3, 6]]);
  });

  it('identity builds the n x n identity', () => {
    expect(identity(2)).toEqual([[1, 0], [0, 1]]);
    expect(() => identity(0)).toThrow(/positive integer/);
  });

  it('diag builds from a vector and extracts from a matrix', () => {
    expect(diag([1, 2, 3])).toEqual([[1, 0, 0], [0, 2, 0], [0, 0, 3]]);
    expect(diag([[1, 2], [3, 4]])).toEqual([1, 4]);
    expect(diag([[1, 2, 3], [4, 5, 6]])).toEqual([1, 5]);
  });

  it('trace matches hand values and requires square input', () => {
    expect(trace([[1, 2], [3, 4]])).toBe(5);
    expect(() => trace([[1, 2, 3], [4, 5, 6]])).toThrow(/square/);
  });

  it('matrix norms match hand values', () => {
    const M = [
      [1, -2],
      [3, 4],
    ];
    expect(norm(M)).toBeCloseTo(Math.sqrt(30), 12);
    expect(norm(M, 'fro')).toBeCloseTo(Math.sqrt(30), 12);
    expect(norm(M, 1)).toBe(6);
    expect(norm(M, Infinity)).toBe(7);
  });

  it('vector norms match hand values', () => {
    expect(norm([3, -4])).toBe(5);
    expect(norm([3, -4], 2)).toBe(5);
    expect(norm([3, -4], 1)).toBe(7);
    expect(norm([3, -4], Infinity)).toBe(4);
  });

  it('rejects dimension mismatches', () => {
    expect(() => matmul([[1, 2], [3, 4]], [[1, 2], [3, 4], [5, 6]])).toThrow(/mismatch/);
    expect(() => matmul([[1, 2], [3, 4]], [1, 2, 3])).toThrow(/length 2/);
  });

  it('rejects non-rectangular input via _mat', () => {
    expect(() => transpose([[1, 2], [3]])).toThrow(/rectangular/);
    expect(() => solve([[1, 2], [3]], [1, 2])).toThrow(/rectangular/);
    expect(() => cholesky([[1, 2], [3]])).toThrow(/rectangular/);
  });
});
