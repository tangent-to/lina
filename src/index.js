/**
 * @tangent.to/lina - Linear algebra for JavaScript (ESM)
 *
 * LU, QR, Cholesky, SVD, symmetric eigendecomposition and least squares
 * on plain nested arrays, computed on flat Float64Array storage
 * internally. numpy/scipy.linalg-validated. MIT infrastructure of the
 * tangent suite.
 */

export { det, inv, lu, solve } from './lu.js';
export { cholesky, choleskySolve, isPositiveDefinite } from './cholesky.js';
export { lstsq, qr } from './qr.js';
export { cond, pinv, pinvSolve, rank, svd } from './svd.js';
export { eigSym } from './eigsym.js';
export { diag, identity, matmul, norm, trace, transpose } from './ops.js';

import { det, inv, lu, solve } from './lu.js';
import { cholesky, choleskySolve, isPositiveDefinite } from './cholesky.js';
import { lstsq, qr } from './qr.js';
import { cond, pinv, pinvSolve, rank, svd } from './svd.js';
import { eigSym } from './eigsym.js';
import { diag, identity, matmul, norm, trace, transpose } from './ops.js';

export default {
  lu,
  solve,
  det,
  inv,
  cholesky,
  choleskySolve,
  isPositiveDefinite,
  qr,
  lstsq,
  svd,
  pinv,
  pinvSolve,
  rank,
  cond,
  eigSym,
  matmul,
  transpose,
  identity,
  diag,
  norm,
  trace,
};
