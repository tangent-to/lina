import { describe, it, expect } from 'vitest';
import { qr, lstsq } from '../src/qr.js';

/** Multiply nested matrices. */
function matmul(A, B) {
  const m = A.length;
  const p = B.length;
  const n = B[0].length;
  const C = [];
  for (let i = 0; i < m; i++) {
    const row = new Array(n).fill(0);
    for (let t = 0; t < p; t++) {
      for (let j = 0; j < n; j++) row[j] += A[i][t] * B[t][j];
    }
    C.push(row);
  }
  return C;
}

/** Transpose a nested matrix. */
function transpose(A) {
  return A[0].map((_, j) => A.map((row) => row[j]));
}

/** Max absolute elementwise difference between two nested matrices. */
function maxDiff(A, B) {
  let d = 0;
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < A[0].length; j++) d = Math.max(d, Math.abs(A[i][j] - B[i][j]));
  }
  return d;
}

/** n x n identity. */
function eye(n) {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
}

const square = [
  [4, 1, -2, 2],
  [1, 2, 0, 1],
  [-2, 0, 3, -2],
  [2, 1, -2, -1],
];

const tall = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 10],
  [2, -1, 0],
  [3, 3, 3],
  [-1, 4, 2],
];

const wide = [
  [1, 2, 3, 4, 5, 6],
  [7, 8, 10, -1, 2, 0],
  [3, -2, 1, 5, 4, -3],
];

function hilbert(n) {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => 1 / (i + j + 1)));
}

describe('qr', () => {
  it.each([
    ['square', square],
    ['tall 6x3', tall],
    ['wide 3x6', wide],
  ])('reconstructs A = QR (reduced) for %s matrix', (_name, A) => {
    const { Q, R } = qr(A);
    expect(maxDiff(matmul(Q, R), A)).toBeLessThan(1e-12);
  });

  it.each([
    ['square', square],
    ['tall 6x3', tall],
    ['wide 3x6', wide],
  ])('reconstructs A = QR (full) for %s matrix', (_name, A) => {
    const { Q, R } = qr(A, { mode: 'full' });
    expect(maxDiff(matmul(Q, R), A)).toBeLessThan(1e-12);
  });

  it.each([
    ['square', square],
    ['tall 6x3', tall],
    ['wide 3x6', wide],
  ])('has orthonormal columns Q^T Q = I (reduced) for %s matrix', (_name, A) => {
    const { Q } = qr(A);
    const k = Math.min(A.length, A[0].length);
    expect(maxDiff(matmul(transpose(Q), Q), eye(k))).toBeLessThan(1e-12);
  });

  it.each([
    ['square', square],
    ['tall 6x3', tall],
  ])('has orthogonal Q Q^T = I (full) for %s matrix', (_name, A) => {
    const { Q } = qr(A, { mode: 'full' });
    expect(maxDiff(matmul(Q, transpose(Q)), eye(A.length))).toBeLessThan(1e-12);
  });

  it.each([
    ['square reduced', square, {}],
    ['tall reduced', tall, {}],
    ['wide reduced', wide, {}],
    ['square full', square, { mode: 'full' }],
    ['tall full', tall, { mode: 'full' }],
    ['wide full', wide, { mode: 'full' }],
  ])('produces upper-triangular R (%s)', (_name, A, options) => {
    const { R } = qr(A, options);
    for (let i = 0; i < R.length; i++) {
      for (let j = 0; j < Math.min(i, R[0].length); j++) {
        expect(Math.abs(R[i][j])).toBeLessThan(1e-10);
      }
    }
  });

  it('returns reduced shapes: Q m x min(m,n), R min(m,n) x n', () => {
    const t = qr(tall);
    expect(t.Q.length).toBe(6);
    expect(t.Q[0].length).toBe(3);
    expect(t.R.length).toBe(3);
    expect(t.R[0].length).toBe(3);

    const w = qr(wide);
    expect(w.Q.length).toBe(3);
    expect(w.Q[0].length).toBe(3);
    expect(w.R.length).toBe(3);
    expect(w.R[0].length).toBe(6);

    const s = qr(square);
    expect(s.Q.length).toBe(4);
    expect(s.Q[0].length).toBe(4);
    expect(s.R.length).toBe(4);
    expect(s.R[0].length).toBe(4);
  });

  it('returns full shapes: Q m x m, R m x n', () => {
    const t = qr(tall, { mode: 'full' });
    expect(t.Q.length).toBe(6);
    expect(t.Q[0].length).toBe(6);
    expect(t.R.length).toBe(6);
    expect(t.R[0].length).toBe(3);

    const w = qr(wide, { mode: 'full' });
    expect(w.Q.length).toBe(3);
    expect(w.Q[0].length).toBe(3);
    expect(w.R.length).toBe(3);
    expect(w.R[0].length).toBe(6);
  });

  it('reconstructs the nearly singular Hilbert 5x5 matrix to 1e-10', () => {
    const H = hilbert(5);
    const { Q, R } = qr(H);
    expect(maxDiff(matmul(Q, R), H)).toBeLessThan(1e-10);
    expect(maxDiff(matmul(transpose(Q), Q), eye(5))).toBeLessThan(1e-12);
  });

  it('rejects invalid inputs', () => {
    expect(() => qr([])).toThrow();
    expect(() => qr([[1, 2], [3]])).toThrow(/rectangular/);
    expect(() => qr(square, { mode: 'economy' })).toThrow(/mode/);
  });
});

describe('lstsq', () => {
  it('matches closed-form OLS for a line fit through 5 points', () => {
    const xs = [0, 1, 2, 3, 4];
    const ys = [1.1, 1.9, 3.2, 3.9, 5.1];
    const A = xs.map((x) => [1, x]);

    // Closed-form OLS: b = Sxy / Sxx, a = ybar - b * xbar.
    const nPts = xs.length;
    const xbar = xs.reduce((s, v) => s + v, 0) / nPts;
    const ybar = ys.reduce((s, v) => s + v, 0) / nPts;
    let sxx = 0;
    let sxy = 0;
    for (let i = 0; i < nPts; i++) {
      sxx += (xs[i] - xbar) * (xs[i] - xbar);
      sxy += (xs[i] - xbar) * (ys[i] - ybar);
    }
    const slope = sxy / sxx;
    const intercept = ybar - slope * xbar;

    const { x, residualNorm } = lstsq(A, ys);
    expect(Math.abs(x[0] - intercept)).toBeLessThan(1e-10);
    expect(Math.abs(x[1] - slope)).toBeLessThan(1e-10);

    // residualNorm = ||A x - b||_2, checked against a direct computation.
    let rr = 0;
    for (let i = 0; i < nPts; i++) {
      const r = intercept + slope * xs[i] - ys[i];
      rr += r * r;
    }
    expect(Math.abs(residualNorm - Math.sqrt(rr))).toBeLessThan(1e-10);
  });

  it('returns a near-zero residual for a consistent square system', () => {
    const b = [1, -2, 3, 0.5];
    const { x, residualNorm } = lstsq(square, b);
    expect(residualNorm).toBeLessThan(1e-12);
    // Verify A x = b directly.
    for (let i = 0; i < 4; i++) {
      const s = square[i].reduce((acc, aij, j) => acc + aij * x[j], 0);
      expect(Math.abs(s - b[i])).toBeLessThan(1e-10);
    }
  });

  it('throws a rank-deficient error for collinear columns', () => {
    const A = [
      [1, 2],
      [2, 4],
      [3, 6],
    ];
    expect(() => lstsq(A, [1, 2, 3])).toThrow(/rank deficient/);
  });

  it('rejects dimension mismatches', () => {
    expect(() => lstsq(tall, [1, 2, 3])).toThrow(/length 6/);
    expect(() => lstsq(wide, [1, 2, 3])).toThrow(/rows/);
    expect(() => lstsq([[1, 2], [3]], [1, 2])).toThrow(/rectangular/);
  });
});
