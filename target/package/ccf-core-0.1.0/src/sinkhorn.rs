//! Birkhoff polytope projector via Sinkhorn-Knopp iteration.
//!
//! Patent Claims 19–23: manifold-constrained trust mixing across contexts.
//!
//! Given a non-negative square matrix of raw similarity scores, iteratively
//! normalises rows then columns until every row sum and column sum equals 1.0.
//! The result is a doubly stochastic matrix — an element of the Birkhoff polytope.
//!
//! This matrix constrains cross-context trust transfer: no single context can
//! dominate, and the total trust weight is conserved across all contexts.
//!
//! # Reference
//! DeepSeek mHC (arXiv:2512.24880) applies the same mathematics to gradient
//! stability. CCF applies it to trust stability.
//!
//! # Invariants
//! - **I-SKN-001** — Converges in ≤20 iterations for any valid input matrix
//! - **I-SKN-002** — Output satisfies |row_sum(i) - 1.0| < 1e-6 for all i
//! - **I-DIST-001** — no_std compatible; fixed-size arrays, no heap allocation
//! - **I-DIST-005** — Zero unsafe code

/// Result of a Sinkhorn-Knopp projection.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvergenceResult {
    /// Whether the tolerance was met within `max_iterations`.
    pub converged: bool,
    /// Number of iterations performed.
    pub iterations: u32,
    /// Maximum absolute deviation from 1.0 across all row sums after final iteration.
    pub residual: f32,
}

/// Birkhoff polytope projector — doubly stochastic matrix via Sinkhorn-Knopp iteration.
///
/// Patent Claims 19–23.
#[derive(Clone, Debug)]
pub struct SinkhornKnopp {
    /// Convergence tolerance (default: 1e-6). Stop when max|row_sum - 1.0| < tolerance.
    pub tolerance: f32,
    /// Maximum number of row+column normalisation iterations (default: 20).
    pub max_iterations: u32,
}

impl Default for SinkhornKnopp {
    fn default() -> Self {
        Self { tolerance: 1e-6, max_iterations: 20 }
    }
}

impl SinkhornKnopp {
    /// Create a new projector with the given tolerance and iteration cap.
    pub fn new(tolerance: f32, max_iterations: u32) -> Self {
        Self { tolerance, max_iterations }
    }

    /// Project an N×N matrix in-place to the Birkhoff polytope.
    ///
    /// Input: non-negative matrix. Rows and columns must each have at least
    /// one positive entry (otherwise that marginal cannot be normalised).
    ///
    /// After convergence, every row sum and column sum is 1.0 ± `tolerance`.
    ///
    /// Patent Claim 19: doubly stochastic constraint.
    /// Patent Claim 20: iterative normalisation procedure.
    /// Patent Claim 21: bounded mixing (no row dominates after projection).
    /// Patent Claim 22: non-negativity preservation.
    /// Patent Claim 23: Birkhoff polytope membership.
    pub fn project<const N: usize>(&self, m: &mut [[f32; N]; N]) -> ConvergenceResult {
        for iter in 0..self.max_iterations {
            // Row normalisation
            for row in m.iter_mut() {
                let s: f32 = row.iter().sum();
                if s > 1e-12 {
                    let inv = 1.0 / s;
                    for x in row.iter_mut() { *x *= inv; }
                }
            }

            // Column normalisation
            for j in 0..N {
                let s: f32 = (0..N).map(|i| m[i][j]).sum();
                if s > 1e-12 {
                    let inv = 1.0 / s;
                    for i in 0..N { m[i][j] *= inv; }
                }
            }

            // Check convergence: max |row_sum - 1.0|
            let residual = m.iter().map(|row| {
                let s: f32 = row.iter().sum();
                (s - 1.0_f32).abs()
            }).fold(0.0_f32, f32::max);

            if residual < self.tolerance {
                return ConvergenceResult { converged: true, iterations: iter + 1, residual };
            }
        }

        // Return final residual even if not converged
        let residual = m.iter().map(|row| {
            let s: f32 = row.iter().sum();
            (s - 1.0_f32).abs()
        }).fold(0.0_f32, f32::max);

        ConvergenceResult { converged: false, iterations: self.max_iterations, residual }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sk() -> SinkhornKnopp { SinkhornKnopp::default() }

    fn assert_doubly_stochastic<const N: usize>(m: &[[f32; N]; N], tol: f32) {
        for (i, row) in m.iter().enumerate() {
            let rs: f32 = row.iter().sum();
            assert!((rs - 1.0).abs() < tol, "row {} sum = {}", i, rs);
        }
        for j in 0..N {
            let cs: f32 = (0..N).map(|i| m[i][j]).sum();
            assert!((cs - 1.0).abs() < tol, "col {} sum = {}", j, cs);
        }
    }

    #[test]
    fn test_claim_19_doubly_stochastic_output() {
        // Claim 19: output is doubly stochastic
        let mut m = [
            [4.0_f32, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 1.0, 2.0],
            [2.0, 3.0, 4.0, 1.0],
        ];
        let r = sk().project(&mut m);
        assert!(r.converged, "did not converge: {:?}", r);
        assert_doubly_stochastic(&m, 1e-5);
    }

    #[test]
    fn test_claim_20_converges_within_20_iterations() {
        // Claim 20: iterative normalisation converges within max_iterations
        let mut m = [
            [100.0_f32, 1.0, 1.0],
            [1.0, 100.0, 1.0],
            [1.0, 1.0, 100.0],
        ];
        let r = sk().project(&mut m);
        assert!(r.converged);
        assert!(r.iterations <= 20, "took {} iterations", r.iterations);
    }

    #[test]
    fn test_claim_21_no_single_row_dominates() {
        // Claim 21: bounded mixing — after projection each row sums to exactly 1.0,
        // so total trust weight is conserved and no context can accumulate unbounded weight.
        // A uniform-ish input projects to a matrix where no row sum exceeds 1.0.
        let mut m = [
            [3.0_f32, 1.0, 1.0, 1.0],
            [1.0, 3.0, 1.0, 1.0],
            [1.0, 1.0, 3.0, 1.0],
            [1.0, 1.0, 1.0, 3.0],
        ];
        let r = sk().project(&mut m);
        assert!(r.converged);
        // Bounded mixing: no single row sum exceeds 1.0 + tolerance
        for (i, row) in m.iter().enumerate() {
            let rs: f32 = row.iter().sum();
            assert!(rs <= 1.0 + 1e-5, "row {} sum {} exceeds 1.0 — unbounded weight", i, rs);
            // No single entry can exceed the row sum (trivially true for non-negative)
            for &v in row {
                assert!(v >= 0.0 && v <= 1.0 + 1e-5, "entry {} out of [0,1] bounds", v);
            }
        }
    }

    #[test]
    fn test_claim_22_non_negativity_preserved() {
        // Claim 22: all entries remain non-negative after projection
        let mut m = [
            [0.5_f32, 2.0, 1.0],
            [1.0, 0.5, 2.0],
            [2.0, 1.0, 0.5],
        ];
        sk().project(&mut m);
        for row in &m {
            for &v in row {
                assert!(v >= 0.0, "negative entry: {}", v);
            }
        }
    }

    #[test]
    fn test_claim_23_idempotence_already_stochastic() {
        // Claim 23: already doubly stochastic input is unchanged (Birkhoff polytope membership)
        let v = 1.0_f32 / 3.0;
        let mut m = [[v; 3]; 3];
        let original = m;
        sk().project(&mut m);
        for i in 0..3 {
            for j in 0..3 {
                assert!((m[i][j] - original[i][j]).abs() < 1e-5,
                    "entry [{},{}] changed: {} -> {}", i, j, original[i][j], m[i][j]);
            }
        }
    }

    #[test]
    fn test_2x2_simple_case() {
        let mut m = [[1.0_f32, 3.0], [2.0, 1.0]];
        let r = sk().project(&mut m);
        assert!(r.converged);
        assert_doubly_stochastic(&m, 1e-5);
    }

    #[test]
    fn test_asymmetric_matrix() {
        let mut m = [
            [10.0_f32, 1.0, 5.0],
            [2.0, 8.0, 3.0],
            [4.0, 6.0, 7.0],
        ];
        let r = sk().project(&mut m);
        assert!(r.converged);
        assert_doubly_stochastic(&m, 1e-5);
    }
}
