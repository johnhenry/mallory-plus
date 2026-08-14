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
import { ComplexNumber, MatrixMath } from "mallory-math";
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

// ---- general (non-symmetric) eigenvalues (issue #68) -----------------------
//
// EIGENVALUES ONLY -- eigenvectors are explicitly out of v1 scope. Getting
// them from the real Schur form this algorithm produces is a second,
// separable algorithm (back-substitution in the quasi-triangular form, then
// transforming back through the Hessenberg similarity); a real, non-
// symmetric matrix's eigenvalues can be complex (conjugate pairs), which is
// why this returns `ComplexNumber[]` rather than reusing `eigSymmetric`'s
// real-valued `EigenDecomposition` shape.
//
// Deliberately placed HERE rather than in mallory-math's MatrixMath
// (alongside eigenSymmetric, which would be the "by the book" home): this
// algorithm needs real numerical testing that mallory-plus already has
// infrastructure for (the NumPy oracle harness, numpy.linalg.eigvals as the
// reference) and mallory-math does not. A disclosed, deliberate deviation
// from the eigSymmetric precedent, not an accidental one.
//
// Algorithm: Hessenberg reduction (Householder similarity transforms) then
// the classical (single, Wilkinson-)shifted QR algorithm with deflation --
// NOT the implicit-double-shift Francis algorithm LAPACK's dgeev uses
// (which avoids ever forming complex arithmetic). This is the simpler,
// slower, easier-to-verify-correct version: each QR step reuses the
// EXISTING `qr()` export above (an explicit O(n^3) decomposition per
// iteration, not exploiting Hessenberg structure via Givens rotations) --
// completely consistent with this whole module's "reference-speed, not a
// performance claim" convention, and the canonical-implementation rule
// (never re-derive QR math that already exists in this file).

const EIG_EPS = 1e-10;
const EIG_MAX_ITERATIONS_PER_DEFLATION = 100;

/** Two-sided Householder similarity reduction to upper Hessenberg form: returns H where `H = Qᵀ·A·Q` for some orthogonal Q (Q itself is discarded -- not needed for eigenvalues-only). */
function toHessenberg(a: readonly (readonly number[])[]): number[][] {
  const n = a.length;
  const h: number[][] = a.map((row) => [...row]);
  for (let k = 0; k < n - 2; k++) {
    // Householder vector zeroing h[k+2..n, k] (everything below the subdiagonal in column k).
    let alpha = 0;
    for (let i = k + 1; i < n; i++) alpha += (h[i]![k] as number) ** 2;
    alpha = Math.sqrt(alpha);
    if (alpha < EIG_EPS) continue; // column already effectively zero below the subdiagonal
    if ((h[k + 1]![k] as number) > 0) alpha = -alpha; // sign choice for numerical stability
    const v = new Array<number>(n).fill(0);
    v[k + 1] = (h[k + 1]![k] as number) - alpha;
    for (let i = k + 2; i < n; i++) v[i] = h[i]![k] as number;
    let vNormSq = 0;
    for (let i = k + 1; i < n; i++) vNormSq += (v[i] as number) ** 2;
    if (vNormSq < EIG_EPS * EIG_EPS) continue;

    // Apply the reflector as a SIMILARITY transform: H <- P·H·P (P = I - 2vvᵀ/vᵀv, symmetric+orthogonal, so Pᵀ=P).
    // Left multiply: H <- P·H (affects rows k+1..n).
    for (let j = 0; j < n; j++) {
      let dot = 0;
      for (let i = k + 1; i < n; i++) dot += (v[i] as number) * (h[i]![j] as number);
      const factor = (2 * dot) / vNormSq;
      for (let i = k + 1; i < n; i++) (h[i] as number[])[j] = (h[i]![j] as number) - factor * (v[i] as number);
    }
    // Right multiply: H <- H·P (affects columns k+1..n).
    for (let i = 0; i < n; i++) {
      let dot = 0;
      for (let j = k + 1; j < n; j++) dot += (h[i]![j] as number) * (v[j] as number);
      const factor = (2 * dot) / vNormSq;
      for (let j = k + 1; j < n; j++) (h[i] as number[])[j] = (h[i]![j] as number) - factor * (v[j] as number);
    }
  }
  return h;
}

/** Eigenvalues of an isolated real 2x2 block via the quadratic formula -- real pair or a complex-conjugate pair, whichever the discriminant says. */
function eig2x2(a: number, b: number, c: number, d: number): [ComplexNumber, ComplexNumber] {
  const trace = a + d;
  const det2 = a * d - b * c;
  const discriminant = trace * trace - 4 * det2;
  if (discriminant >= 0) {
    const sq = Math.sqrt(discriminant);
    return [new ComplexNumber((trace + sq) / 2, 0), new ComplexNumber((trace - sq) / 2, 0)];
  }
  const sq = Math.sqrt(-discriminant);
  return [new ComplexNumber(trace / 2, sq / 2), new ComplexNumber(trace / 2, -sq / 2)];
}

function matMul(a: readonly (readonly number[])[], b: readonly (readonly number[])[]): number[][] {
  const n = a.length;
  const out: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < n; k++) {
      const aik = a[i]![k] as number;
      if (aik === 0) continue;
      for (let j = 0; j < n; j++) (out[i] as number[])[j] = (out[i]![j] as number) + aik * (b[k]![j] as number);
    }
  }
  return out;
}

/**
 * Eigenvalues of a general (not necessarily symmetric) real square matrix,
 * via Hessenberg reduction + shifted-QR deflation (see the section header
 * above for the full design). Returns exactly `n` eigenvalues, in
 * deflation order (bottom-right-most first) -- NOT sorted by magnitude,
 * matching this being an eigenvalue LIST, not a canonical ordering claim.
 *
 * Eigenvectors are explicitly out of v1 scope (see the section header).
 */
export function eigGeneral(a: Tensor): ComplexNumber[] {
  const n = a.shape[0] as number;
  if (a.shape.length !== 2 || a.shape[1] !== n) {
    throw new RangeError(`eigGeneral: expected a square matrix, got shape [${a.shape.join(", ")}]`);
  }
  if (n === 1) return [new ComplexNumber(toMatrix(a)[0]![0] as number, 0)];

  let h = toHessenberg(toMatrix(a));
  const eigenvalues: ComplexNumber[] = [];
  let m = n; // active submatrix is h[0..m, 0..m]
  let iterationsSinceLastDeflation = 0;

  while (m > 0) {
    if (m === 1) {
      eigenvalues.push(new ComplexNumber(h[0]![0] as number, 0));
      m = 0;
      break;
    }

    const sub = (i: number, j: number): number => h[i]![j] as number;
    const scale = Math.abs(sub(m - 2, m - 2)) + Math.abs(sub(m - 1, m - 1));
    if (Math.abs(sub(m - 1, m - 2)) < EIG_EPS * (scale || 1)) {
      // Trailing 1x1 has deflated off cleanly.
      eigenvalues.push(new ComplexNumber(sub(m - 1, m - 1), 0));
      m -= 1;
      iterationsSinceLastDeflation = 0;
      continue;
    }

    const isolated2x2 =
      m === 2 ||
      Math.abs(sub(m - 2, m - 3)) < EIG_EPS * (Math.abs(sub(m - 3, m - 3)) + Math.abs(sub(m - 2, m - 2)) || 1);
    if (isolated2x2) {
      const [l1, l2] = eig2x2(sub(m - 2, m - 2), sub(m - 2, m - 1), sub(m - 1, m - 2), sub(m - 1, m - 1));
      eigenvalues.push(l1, l2);
      m -= 2;
      iterationsSinceLastDeflation = 0;
      continue;
    }

    iterationsSinceLastDeflation++;
    if (iterationsSinceLastDeflation > EIG_MAX_ITERATIONS_PER_DEFLATION) {
      throw new Error(
        `eigGeneral: did not converge after ${EIG_MAX_ITERATIONS_PER_DEFLATION} QR iterations on a size-${m} active block -- the matrix may be too ill-conditioned for this reference-speed algorithm`,
      );
    }

    // Wilkinson shift: the eigenvalue of the trailing 2x2 block closest to
    // h[m-1][m-1] -- standard choice, converges much faster than an
    // unshifted or Rayleigh-quotient shift near a real eigenvalue.
    const [w1, w2] = eig2x2(sub(m - 2, m - 2), sub(m - 2, m - 1), sub(m - 1, m - 2), sub(m - 1, m - 1));
    const target = sub(m - 1, m - 1);
    const shift =
      Math.abs(w1.re - target) < Math.abs(w2.re - target) && w1.im === 0 ? w1.re : w2.im === 0 ? w2.re : target;

    const active = h.slice(0, m).map((row) => row.slice(0, m));
    for (let i = 0; i < m; i++) (active[i] as number[])[i] = (active[i]![i] as number) - shift;
    const { Q, R } = qr(fromMatrix(active));
    const qArr = toMatrix(Q);
    const rArr = toMatrix(R);
    const rq = matMul(rArr, qArr);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        (h[i] as number[])[j] = (rq[i]![j] as number) + (i === j ? shift : 0);
      }
    }
  }

  return eigenvalues;
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
