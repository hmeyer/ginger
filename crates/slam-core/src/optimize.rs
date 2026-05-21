//! Dense Levenberg–Marquardt with an optional Huber robustifier.
//!
//! Motion-only BA ([`crate::tracking`]) is a small *dense* 6-DOF
//! problem, so a well-tested dense LM is the right foundation; the
//! block-sparse Schur structure for local BA lives separately in
//! [`crate::local_ba`] as an algorithmic/layout change on top — the
//! numerics here do not change.
//!
//! Huber is applied per scalar residual via IRLS weighting
//! (`w = 1` for `|r| <= delta`, else `sqrt(delta/|r|)`), which turns the
//! Gauss–Newton normal equations into the robustified ones without a
//! separate M-estimator code path.

use nalgebra::{DMatrix, DVector};

/// A nonlinear least-squares problem: residual vector and its Jacobian
/// `∂r/∂params` at a parameter point.
pub trait LeastSquaresProblem {
    fn residuals(&self, params: &DVector<f64>) -> DVector<f64>;
    fn jacobian(&self, params: &DVector<f64>) -> DMatrix<f64>;
}

/// Solver settings. `huber_delta = None` is plain least squares.
#[derive(Clone, Copy, Debug)]
pub struct LmOptions {
    pub max_iters: usize,
    pub gradient_tol: f64,
    pub step_tol: f64,
    pub huber_delta: Option<f64>,
}

impl Default for LmOptions {
    fn default() -> Self {
        Self {
            max_iters: 100,
            gradient_tol: 1e-10,
            step_tol: 1e-12,
            huber_delta: None,
        }
    }
}

/// Outcome of a solve.
#[derive(Clone, Debug)]
pub struct LmReport {
    pub params: DVector<f64>,
    pub cost: f64,
    pub iters: usize,
    pub converged: bool,
}

fn huber_weight(r: f64, delta: Option<f64>) -> f64 {
    match delta {
        Some(d) if r.abs() > d => (d / r.abs()).sqrt(),
        _ => 1.0,
    }
}

/// Robustified squared cost `½ Σ ρ(r_i)`.
fn cost(r: &DVector<f64>, delta: Option<f64>) -> f64 {
    r.iter()
        .map(|&ri| match delta {
            Some(d) if ri.abs() > d => d * (ri.abs() - 0.5 * d),
            _ => 0.5 * ri * ri,
        })
        .sum()
}

/// Minimize `½ Σ ρ(r_i(params))` from `x0`. Standard damped LM: solve
/// `(JᵀWJ + λ·diag(JᵀWJ)) δ = -JᵀW r`, accept if the cost drops (and
/// shrink λ), otherwise grow λ and retry.
pub fn levenberg_marquardt(
    problem: &dyn LeastSquaresProblem,
    x0: DVector<f64>,
    opts: LmOptions,
) -> LmReport {
    let mut x = x0;
    let mut lambda = 1e-3;
    let mut r = problem.residuals(&x);
    let mut f = cost(&r, opts.huber_delta);

    for it in 0..opts.max_iters {
        let j = problem.jacobian(&x);
        // IRLS: fold the Huber weights into r and the rows of J.
        let w: Vec<f64> = r
            .iter()
            .map(|&ri| huber_weight(ri, opts.huber_delta))
            .collect();
        let mut jw = j.clone();
        let mut rw = r.clone();
        for i in 0..r.len() {
            rw[i] *= w[i];
            for c in 0..jw.ncols() {
                jw[(i, c)] *= w[i];
            }
        }
        let jt = jw.transpose();
        let h = &jt * &jw; // JᵀWJ
        let g = &jt * &rw; // JᵀW r
        let grad_norm = g.amax();
        if grad_norm < opts.gradient_tol {
            return LmReport {
                params: x,
                cost: f,
                iters: it,
                converged: true,
            };
        }

        // Inner loop: adapt λ until the damped step reduces the cost.
        let mut accepted = false;
        for _ in 0..30 {
            let mut a = h.clone();
            for d in 0..a.nrows() {
                a[(d, d)] += lambda * h[(d, d)].max(1e-12);
            }
            let Some(chol) = a.clone().cholesky() else {
                lambda *= 10.0;
                continue;
            };
            let delta = chol.solve(&(-&g));
            if delta.norm() < opts.step_tol {
                return LmReport {
                    params: x,
                    cost: f,
                    iters: it,
                    converged: true,
                };
            }
            let x_new = &x + &delta;
            let r_new = problem.residuals(&x_new);
            let f_new = cost(&r_new, opts.huber_delta);
            if f_new < f {
                x = x_new;
                r = r_new;
                f = f_new;
                lambda = (lambda * 0.3).max(1e-12);
                accepted = true;
                break;
            }
            lambda *= 10.0;
        }
        if !accepted {
            return LmReport {
                params: x,
                cost: f,
                iters: it + 1,
                converged: false,
            };
        }
    }
    LmReport {
        params: x,
        cost: f,
        iters: opts.max_iters,
        converged: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// y = a·exp(b·x) — a classic nonlinear least-squares fit.
    struct ExpFit {
        xs: Vec<f64>,
        ys: Vec<f64>,
    }
    impl LeastSquaresProblem for ExpFit {
        fn residuals(&self, p: &DVector<f64>) -> DVector<f64> {
            let (a, b) = (p[0], p[1]);
            DVector::from_iterator(
                self.xs.len(),
                self.xs
                    .iter()
                    .zip(&self.ys)
                    .map(|(&x, &y)| a * (b * x).exp() - y),
            )
        }
        fn jacobian(&self, p: &DVector<f64>) -> DMatrix<f64> {
            let (a, b) = (p[0], p[1]);
            DMatrix::from_fn(self.xs.len(), 2, |i, c| {
                let x = self.xs[i];
                if c == 0 {
                    (b * x).exp()
                } else {
                    a * x * (b * x).exp()
                }
            })
        }
    }

    #[test]
    fn recovers_known_exponential() {
        let (a_true, b_true) = (2.5, -0.7);
        let xs: Vec<f64> = (0..40).map(|i| i as f64 * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| a_true * (b_true * x).exp()).collect();
        let r = levenberg_marquardt(
            &ExpFit { xs, ys },
            DVector::from_vec(vec![1.0, 0.0]),
            LmOptions::default(),
        );
        assert!(r.converged, "did not converge: {r:?}");
        assert!((r.params[0] - a_true).abs() < 1e-6);
        assert!((r.params[1] - b_true).abs() < 1e-6);
    }

    #[test]
    fn huber_resists_an_outlier() {
        // Linear data y = 3x + 1 with one gross outlier.
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let mut ys: Vec<f64> = xs.iter().map(|&x| 3.0 * x + 1.0).collect();
        ys[10] += 500.0;

        struct LinFit {
            xs: Vec<f64>,
            ys: Vec<f64>,
        }
        impl LeastSquaresProblem for LinFit {
            fn residuals(&self, p: &DVector<f64>) -> DVector<f64> {
                DVector::from_iterator(
                    self.xs.len(),
                    self.xs
                        .iter()
                        .zip(&self.ys)
                        .map(|(&x, &y)| p[0] * x + p[1] - y),
                )
            }
            fn jacobian(&self, _p: &DVector<f64>) -> DMatrix<f64> {
                DMatrix::from_fn(
                    self.xs.len(),
                    2,
                    |i, c| if c == 0 { self.xs[i] } else { 1.0 },
                )
            }
        }
        let prob = LinFit {
            xs: xs.clone(),
            ys: ys.clone(),
        };
        let plain = levenberg_marquardt(
            &prob,
            DVector::from_vec(vec![0.0, 0.0]),
            LmOptions::default(),
        );
        let robust = levenberg_marquardt(
            &prob,
            DVector::from_vec(vec![0.0, 0.0]),
            LmOptions {
                huber_delta: Some(1.0),
                ..LmOptions::default()
            },
        );
        // Robust slope is much closer to the true 3.0 than plain LS.
        let e_plain = (plain.params[0] - 3.0).abs();
        let e_robust = (robust.params[0] - 3.0).abs();
        assert!(e_robust < 0.05, "robust slope off: {}", robust.params[0]);
        assert!(
            e_robust < e_plain,
            "Huber did not help: {e_robust} vs {e_plain}"
        );
    }
}
