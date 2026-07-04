/**
 * Internal matrix representation.
 *
 * Public lina APIs accept and return row-major nested arrays
 * (Array<Array<number>>) — the tangent suite's boundary currency.
 * Internally, algorithms run on flat Float64Array storage for speed.
 * This module is the only place that converts between the two.
 */

/**
 * Convert a nested row-major matrix to flat storage.
 * Validates rectangularity and finite dimensions.
 *
 * @param {Array<Array<number>>} A - Nested matrix
 * @param {string} [name='A'] - Argument name for error messages
 * @returns {{data: Float64Array, m: number, n: number}}
 */
export function fromNested(A, name = 'A') {
  if (!Array.isArray(A) || A.length === 0 || !Array.isArray(A[0])) {
    throw new Error(`${name} must be a non-empty array of rows`);
  }
  const m = A.length;
  const n = A[0].length;
  if (n === 0) {
    throw new Error(`${name} must have at least one column`);
  }
  const data = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    const row = A[i];
    if (!Array.isArray(row) || row.length !== n) {
      throw new Error(`${name} is not rectangular (row ${i} has length ${row?.length}, expected ${n})`);
    }
    for (let j = 0; j < n; j++) {
      data[i * n + j] = row[j];
    }
  }
  return { data, m, n };
}

/**
 * Convert flat storage back to a nested row-major matrix.
 *
 * @param {Float64Array|Array<number>} data - Flat row-major values
 * @param {number} m - Rows
 * @param {number} n - Columns
 * @returns {Array<Array<number>>}
 */
export function toNested(data, m, n) {
  const A = new Array(m);
  for (let i = 0; i < m; i++) {
    const row = new Array(n);
    for (let j = 0; j < n; j++) {
      row[j] = data[i * n + j];
    }
    A[i] = row;
  }
  return A;
}

/**
 * Validate a vector argument and copy it to a Float64Array.
 *
 * @param {Array<number>} b - Vector
 * @param {number} expected - Required length
 * @param {string} [name='b'] - Argument name for error messages
 * @returns {Float64Array}
 */
export function vecFrom(b, expected, name = 'b') {
  if (!Array.isArray(b) || b.length !== expected) {
    throw new Error(`${name} must be an array of length ${expected} (got ${b?.length})`);
  }
  return Float64Array.from(b);
}

/**
 * Check symmetry to a tolerance; throws when violated.
 *
 * @param {{data: Float64Array, m: number, n: number}} M - Flat matrix
 * @param {string} caller - Function name for the error message
 * @param {number} [tol=1e-10] - Relative tolerance
 */
export function assertSymmetric(M, caller, tol = 1e-10) {
  const { data, m, n } = M;
  if (m !== n) {
    throw new Error(`${caller}: matrix must be square (got ${m}x${n})`);
  }
  let scale = 0;
  for (let i = 0; i < data.length; i++) scale = Math.max(scale, Math.abs(data[i]));
  const bound = tol * Math.max(1, scale);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs(data[i * n + j] - data[j * n + i]) > bound) {
        throw new Error(`${caller}: matrix is not symmetric (A[${i}][${j}] != A[${j}][${i}])`);
      }
    }
  }
}
