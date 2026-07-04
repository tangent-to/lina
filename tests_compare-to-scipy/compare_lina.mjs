#!/usr/bin/env node
/**
 * Helper: run lina operations on matrices supplied by the Python driver.
 * Reads a JSON spec {op, A, b?} and prints a JSON result.
 */

import { readFileSync } from 'node:fs';

const spec = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const lina = await import('../src/index.js');

const { op, A, b } = spec;
let out;

switch (op) {
  case 'svd': {
    const { U, s, V } = lina.svd(A);
    out = { U, s, V };
    break;
  }
  case 'eigSym': {
    out = lina.eigSym(A);
    break;
  }
  case 'solve':
    out = { x: lina.solve(A, b) };
    break;
  case 'lstsq': {
    const r = lina.lstsq(A, b);
    out = { x: r.x, residualNorm: r.residualNorm };
    break;
  }
  case 'det':
    out = { det: lina.det(A) };
    break;
  case 'inv':
    out = { inv: lina.inv(A) };
    break;
  case 'cholesky':
    out = { L: lina.cholesky(A) };
    break;
  case 'qr': {
    const { Q, R } = lina.qr(A);
    out = { Q, R };
    break;
  }
  case 'pinv':
    out = { pinv: lina.pinv(A) };
    break;
  case 'rank_cond':
    out = { rank: lina.rank(A), cond: lina.cond(A) };
    break;
  default:
    console.error(`unknown op: ${op}`);
    process.exit(1);
}

process.stdout.write(JSON.stringify(out));
