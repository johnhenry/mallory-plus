/**
 * Reference-speed linalg surface (issue #26) — decisions recorded on the
 * issue, restated here so the rationale travels with the code:
 *
 * 1. **Ships now, before any native WASM kernel exists.** mallory-math
 *    already has working (if unaccelerated, nested-array) LU/QR/Cholesky/
 *    symmetric-eigendecomposition/SVD/solve/rref/rank/nullSpace/
 *    leastSquares/pseudoInverse/norms — wiring them through here turns a
 *    hard gap into a usable path immediately, at small-matrix scale, while
 *    native kernels are built behind the same shapes later. Every export
 *    below is explicitly labeled reference-speed in its own doc comment;
 *    none of this is a performance claim.
 * 2. **First native-kernel candidates, when that work starts**: `solve`
 *    and matmul-adjacent paths (`leastSquares`'s normal-equations solve,
 *    `pseudoInverse`'s SVD-based reconstruction) — these dominate real
 *    workloads per the issue. `Tensor.matmul` is already native; solve
 *    is not.
 * 3. **Crossover threshold between this and a future native kernel**: not
 *    applicable yet — there IS no native kernel to cross over to. This
 *    module is the only implementation until one lands; when it does, the
 *    plan is to keep this reference path as a fallback/correctness oracle
 *    (mirroring how tensor-autograd's DualNumber cross-check and the NumPy
 *    oracle already work in this repo) rather than deleting it.
 *
 * Every function converts its 2-D Tensor argument(s) to a mallory-math
 * `Matrix<number>` (via `toMatrix`), delegates to `MatrixMath`, and
 * converts the result(s) back (via `fromMatrix`/`fromVector`) — the same
 * copy-at-the-edge pattern `matrix.ts` (issue #14) already established.
 */
import { MatrixMath } from "mallory-math";
import type { Tensor } from "mallory-tensor-core";
import { fromMatrix, fromVector, toMatrix, toVector } from "./matrix.ts";

export interface LUDecomposition {
  L: Tensor;
  U: Tensor;
  P: Tensor;
  sign: number;
}

export interface QRDecomposition {
  Q: Tensor;
  R: Tensor;
}

export interface EigenDecomposition {
  values: Tensor;
  vectors: Tensor;
}

export interface SVDDecomposition {
  U: Tensor;
  S: Tensor;
  V: Tensor;
}

/** `A = P⁻¹·L·U` via partial pivoting. Reference-speed. */
export function lu(a: Tensor): LUDecomposition {
  const { L, U, P, sign } = MatrixMath.lu(toMatrix(a));
  return { L: fromMatrix(L), U: fromMatrix(U), P: fromMatrix(P), sign };
}

/** Solve `A·x = b` via LU decomposition. `b` is a 1-D Tensor. Reference-speed. */
export function solve(a: Tensor, b: Tensor): Tensor {
  return fromVector(MatrixMath.solve(toMatrix(a), toVector(b)));
}

/**
 * Determinant, via the existing {@link lu}: `sign * product(diag(U))`.
 * Reference-speed (issue #67). On a singular/near-singular `A`, one of
 * `U`'s diagonal entries is ~0 (that's what "singular" means for an LU
 * factorization), so this naturally returns ~0 rather than needing special
 * handling.
 */
export function det(a: Tensor): number {
  const { U, sign } = lu(a);
  const n = U.shape[0] as number;
  let product = sign;
  for (let i = 0; i < n; i++) {
    product *= U.at(i, i) as number;
  }
  return product;
}

/**
 * Square matrix inverse, via `n` calls to the existing {@link solve} — one
 * unit basis vector per column, assembled back into a matrix. Reference-
 * speed; the honest cost here specifically is `n` independent LU
 * factorizations (one per `solve` call) rather than one factorization
 * reused across all `n` columns — simplicity over micro-optimizing a path
 * with no measured bottleneck yet, consistent with this module's existing
 * "reference-speed, not a performance claim" disclosure.
 *
 * On a singular/near-singular `A`: does NOT throw. `MatrixMath.solve`'s
 * own documented convention is to write `0` into a solution's component at
 * a near-zero pivot rather than divide by it — `inv` inherits that
 * silently-degenerate-but-defined behavior column by column, matching
 * {@link solve}'s contract exactly rather than inventing new error
 * semantics here. Use {@link det} first to check invertibility if that
 * matters to the caller.
 */
export function inv(a: Tensor): Tensor {
  const n = a.shape[0] as number;
  if (a.shape.length !== 2 || a.shape[1] !== n) {
    throw new RangeError(`inv: expected a square matrix, got shape [${a.shape.join(", ")}]`);
  }
  const columns: number[][] = [];
  for (let i = 0; i < n; i++) {
    const e = new Array<number>(n).fill(0);
    e[i] = 1;
    columns.push(toVector(solve(a, fromVector(e))));
  }
  // columns[j][i] is the i-th component of the j-th solve (column j of the
  // inverse) -- transpose into row-major for fromMatrix.
  const rows: number[][] = Array.from({ length: n }, (_, i) => columns.map((col) => col[i] as number));
  return fromMatrix(rows);
}

/** Reduced row echelon form (Gauss-Jordan elimination). Reference-speed. */
export function rref(a: Tensor): Tensor {
  return fromMatrix(MatrixMath.rref(toMatrix(a)));
}

/** Matrix rank (number of nonzero rows in its RREF). Reference-speed. */
export function rank(a: Tensor): number {
  return MatrixMath.rank(toMatrix(a));
}

/** A basis for the null space (kernel) of `a`, as a 2-D Tensor whose rows are basis vectors. Reference-speed. */
export function nullSpace(a: Tensor): Tensor {
  return fromMatrix(MatrixMath.nullSpace(toMatrix(a)));
}

/** QR decomposition via modified Gram-Schmidt: `A = Q·R` (Q orthonormal). Reference-speed. */
export function qr(a: Tensor): QRDecomposition {
  const { Q, R } = MatrixMath.qr(toMatrix(a));
  return { Q: fromMatrix(Q), R: fromMatrix(R) };
}

/** Cholesky decomposition of a symmetric positive-definite matrix: `A = L·Lᵀ`. Throws if `a` isn't positive-definite. Reference-speed. */
export function cholesky(a: Tensor): Tensor {
  return fromMatrix(MatrixMath.cholesky(toMatrix(a)));
}

/** Eigenvalues (descending) and orthonormal eigenvectors of a SYMMETRIC matrix, via cyclic Jacobi rotation. Reference-speed. */
export function eigSymmetric(a: Tensor, options: { maxSweeps?: number } = {}): EigenDecomposition {
  const { values, vectors } = MatrixMath.eigenSymmetric(toMatrix(a), options.maxSweeps);
  return { values: fromVector(values), vectors: fromMatrix(vectors) };
}

/** Singular value decomposition `A = U·Σ·Vᵀ` (via the eigendecomposition of `AᵀA`). `S` is 1-D (the diagonal), descending. Reference-speed. */
export function svd(a: Tensor): SVDDecomposition {
  const { U, S, V } = MatrixMath.svd(toMatrix(a));
  return { U: fromMatrix(U), S: fromVector(S), V: fromMatrix(V) };
}

/** Least-squares solution of an overdetermined system `A·x ≈ b` (normal equations). Reference-speed. */
export function leastSquares(a: Tensor, b: Tensor): Tensor {
  return fromVector(MatrixMath.leastSquares(toMatrix(a), toVector(b)));
}

/** The Moore-Penrose pseudo-inverse `A⁺` (via SVD). Reference-speed. */
export function pseudoInverse(a: Tensor): Tensor {
  return fromMatrix(MatrixMath.pseudoInverse(toMatrix(a)));
}

/** Frobenius norm (root of the sum of squared entries). Reference-speed. */
export function frobeniusNorm(a: Tensor): number {
  return MatrixMath.frobeniusNorm(toMatrix(a));
}

/** Spectral (2-)norm: the largest singular value. Reference-speed. */
export function spectralNorm(a: Tensor): number {
  return MatrixMath.spectralNorm(toMatrix(a));
}

/** 2-norm condition number `σ_max / σ_min`. Reference-speed. */
export function conditionNumber(a: Tensor): number {
  return MatrixMath.conditionNumber(toMatrix(a));
}
