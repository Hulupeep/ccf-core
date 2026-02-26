//! Python FFI bindings via PyO3.
//!
//! Exposes the CCF pipeline to Python using fixed 6-dimensional feature vectors.
//! For custom sensor dimensions, use the Rust API directly.
//!
//! # Building the Python extension
//!
//! ```bash
//! pip install maturin
//! maturin develop --features python-ffi
//! ```
//!
//! # Usage
//!
//! ```python
//! from ccf_core import CoherenceField, Personality, SocialPhase, PhaseSpace
//!
//! personality = Personality(curiosity_drive=0.6, startle_sensitivity=0.5, recovery_speed=0.5)
//! field = CoherenceField()
//! ps = PhaseSpace()
//! phase = SocialPhase.ShyObserver
//!
//! # feature_vec: 6 floats in [0.0, 1.0] — brightness, noise, presence, motion, orientation, time
//! features = [0.8, 0.0, 1.0, 0.0, 1.0, 0.5]
//! field.positive_interaction(features, personality, tick=0, alone=False)
//! coherence = field.effective_coherence(0.9, features)
//! phase = SocialPhase.classify(coherence, 0.2, phase, ps)
//! print(phase.led_tint())        # [r, g, b]
//! print(phase.expression_scale()) # 0.0–1.0
//! ```

#![allow(non_snake_case)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::accumulator::CoherenceField;
use crate::phase::{
    Personality as RustPersonality, PhaseSpace as RustPhaseSpace, SocialPhase as RustSocialPhase,
};
use crate::vocabulary::{ContextKey, SensorVocabulary};

/// Dimensionality of the Python-facing feature vector.
/// Matches the mBot2 6-sensor vocabulary. Use the Rust API for other dimensions.
const PY_DIM: usize = 6;

// ── Internal vocabulary wrapper ──────────────────────────────────────────────

/// Internal vocabulary type for the Python API.
/// Stores features quantised to u16 for stable hashing.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PyVocab([u16; PY_DIM]);

impl SensorVocabulary<PY_DIM> for PyVocab {
    fn to_feature_vec(&self) -> [f32; PY_DIM] {
        let mut out = [0.0f32; PY_DIM];
        for (i, &q) in self.0.iter().enumerate() {
            out[i] = q as f32 / 65535.0;
        }
        out
    }
}

fn features_to_key(features: &[f32]) -> PyResult<ContextKey<PyVocab, PY_DIM>> {
    if features.len() != PY_DIM {
        return Err(PyValueError::new_err(format!(
            "feature_vec must have exactly {PY_DIM} elements, got {}",
            features.len()
        )));
    }
    let mut q = [0u16; PY_DIM];
    for (i, &f) in features.iter().enumerate() {
        q[i] = (f.clamp(0.0, 1.0) * 65535.0) as u16;
    }
    Ok(ContextKey::new(PyVocab(q)))
}

// ── Personality ───────────────────────────────────────────────────────────────

/// Personality modulators — tune how trust builds and erodes.
///
/// All parameters are bounded to [0.0, 1.0].
#[pyclass(name = "Personality")]
#[derive(Clone)]
pub struct PyPersonality {
    inner: RustPersonality,
}

#[pymethods]
impl PyPersonality {
    /// Create a new Personality.
    ///
    /// Args:
    ///     curiosity_drive: Higher = explores new contexts more eagerly (default 0.5)
    ///     startle_sensitivity: Lower = more resilient to negative events (default 0.5)
    ///     recovery_speed: Higher = rebuilds trust faster after disruption (default 0.5)
    #[new]
    #[pyo3(signature = (curiosity_drive=0.5, startle_sensitivity=0.5, recovery_speed=0.5))]
    pub fn new(curiosity_drive: f32, startle_sensitivity: f32, recovery_speed: f32) -> Self {
        Self {
            inner: RustPersonality {
                curiosity_drive: curiosity_drive.clamp(0.0, 1.0),
                startle_sensitivity: startle_sensitivity.clamp(0.0, 1.0),
                recovery_speed: recovery_speed.clamp(0.0, 1.0),
            },
        }
    }

    /// Returns the curiosity drive modulator in [0.0, 1.0].
    #[getter]
    pub fn curiosity_drive(&self) -> f32 {
        self.inner.curiosity_drive
    }
    /// Returns the startle sensitivity modulator in [0.0, 1.0].
    #[getter]
    pub fn startle_sensitivity(&self) -> f32 {
        self.inner.startle_sensitivity
    }
    /// Returns the recovery speed modulator in [0.0, 1.0].
    #[getter]
    pub fn recovery_speed(&self) -> f32 {
        self.inner.recovery_speed
    }

    /// Python repr string.
    pub fn __repr__(&self) -> String {
        format!(
            "Personality(curiosity_drive={:.2}, startle_sensitivity={:.2}, recovery_speed={:.2})",
            self.inner.curiosity_drive,
            self.inner.startle_sensitivity,
            self.inner.recovery_speed,
        )
    }
}

// ── PhaseSpace ────────────────────────────────────────────────────────────────

/// Phase transition thresholds with Schmitt trigger hysteresis.
///
/// Use `PhaseSpace()` for default thresholds.
#[pyclass(name = "PhaseSpace")]
#[derive(Clone)]
pub struct PyPhaseSpace {
    inner: RustPhaseSpace,
}

#[pymethods]
impl PyPhaseSpace {
    /// Construct a PhaseSpace with default Schmitt trigger thresholds.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: RustPhaseSpace::new(),
        }
    }

    /// Python repr string.
    pub fn __repr__(&self) -> &'static str {
        "PhaseSpace()"
    }
}

// ── SocialPhase ───────────────────────────────────────────────────────────────

fn rust_to_py(p: RustSocialPhase) -> PySocialPhase {
    PySocialPhase { inner: p }
}

/// Four-quadrant social phase classifier.
///
/// Phases:
///     ShyObserver       — low coherence, low tension    (cautious, watching)
///     StartledRetreat   — low coherence, high tension   (withdraw, minimal output)
///     QuietlyBeloved    — high coherence, low tension   (full expressiveness)
///     ProtectiveGuardian — high coherence, high tension (alert but grounded)
#[pyclass(name = "SocialPhase")]
#[derive(Clone)]
pub struct PySocialPhase {
    inner: RustSocialPhase,
}

#[pymethods]
impl PySocialPhase {
    /// Classify a phase from coherence and tension values.
    ///
    /// Args:
    ///     coherence: effective coherence in [0.0, 1.0] from CoherenceField
    ///     tension:   tension in [0.0, 1.0] from your homeostasis / task layer
    ///     current:   the current phase (used for Schmitt trigger hysteresis)
    ///     space:     PhaseSpace with transition thresholds
    #[staticmethod]
    pub fn classify(
        coherence: f32,
        tension: f32,
        current: &PySocialPhase,
        space: &PyPhaseSpace,
    ) -> Self {
        rust_to_py(RustSocialPhase::classify(
            coherence,
            tension,
            current.inner,
            &space.inner,
        ))
    }

    /// ShyObserver class attribute.
    #[classattr]
    pub fn ShyObserver() -> Self {
        Self {
            inner: RustSocialPhase::ShyObserver,
        }
    }

    /// StartledRetreat class attribute.
    #[classattr]
    pub fn StartledRetreat() -> Self {
        Self {
            inner: RustSocialPhase::StartledRetreat,
        }
    }

    /// QuietlyBeloved class attribute.
    #[classattr]
    pub fn QuietlyBeloved() -> Self {
        Self {
            inner: RustSocialPhase::QuietlyBeloved,
        }
    }

    /// ProtectiveGuardian class attribute.
    #[classattr]
    pub fn ProtectiveGuardian() -> Self {
        Self {
            inner: RustSocialPhase::ProtectiveGuardian,
        }
    }

    /// LED tint for this phase as [r, g, b] bytes.
    pub fn led_tint(&self) -> [u8; 3] {
        self.inner.led_tint()
    }

    /// Expression scale for this phase: 0.0 (minimal) to 1.0 (full).
    pub fn expression_scale(&self) -> f32 {
        self.inner.expression_scale()
    }

    /// Python repr string.
    pub fn __repr__(&self) -> &'static str {
        match self.inner {
            RustSocialPhase::ShyObserver => "SocialPhase.ShyObserver",
            RustSocialPhase::StartledRetreat => "SocialPhase.StartledRetreat",
            RustSocialPhase::QuietlyBeloved => "SocialPhase.QuietlyBeloved",
            RustSocialPhase::ProtectiveGuardian => "SocialPhase.ProtectiveGuardian",
        }
    }

    /// Python equality comparison.
    pub fn __eq__(&self, other: &PySocialPhase) -> bool {
        self.inner == other.inner
    }
}

// ── CoherenceField ────────────────────────────────────────────────────────────

/// Context-keyed trust accumulator.
///
/// Maintains an independent trust history per sensory context.
/// All interaction methods take a 6-element feature vector representing
/// the current sensory state (values in [0.0, 1.0]).
///
/// Example::
///
///     field = CoherenceField()
///     personality = Personality(curiosity_drive=0.7)
///     features = [0.8, 0.0, 1.0, 0.0, 1.0, 0.5]  # bright, quiet, close, still, upright, day
///
///     for tick in range(50):
///         field.positive_interaction(features, personality, tick=tick, alone=False)
///
///     print(field.effective_coherence(0.9, features))  # → ~0.7
#[pyclass(name = "CoherenceField")]
pub struct PyCoherenceField {
    inner: CoherenceField<PyVocab, PY_DIM>,
}

#[pymethods]
impl PyCoherenceField {
    /// Create a new empty coherence field.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: CoherenceField::new(),
        }
    }

    /// Record a positive interaction in the given sensory context.
    ///
    /// Args:
    ///     feature_vec: 6 floats in [0.0, 1.0] — current sensory state
    ///     personality: Personality modulating the trust delta
    ///     tick:        monotonic tick counter (u64)
    ///     alone:       True if no external stimulus (passive presence only)
    pub fn positive_interaction(
        &mut self,
        feature_vec: Vec<f32>,
        personality: &PyPersonality,
        tick: u64,
        alone: bool,
    ) -> PyResult<()> {
        let key = features_to_key(&feature_vec)?;
        self.inner
            .positive_interaction(&key, &personality.inner, tick, alone);
        Ok(())
    }

    /// Record a negative interaction (startle, aversive event) in the given context.
    ///
    /// Args:
    ///     feature_vec: 6 floats in [0.0, 1.0] — current sensory state
    ///     personality: Personality modulating the drop magnitude
    ///     tick:        monotonic tick counter (u64)
    pub fn negative_interaction(
        &mut self,
        feature_vec: Vec<f32>,
        personality: &PyPersonality,
        tick: u64,
    ) -> PyResult<()> {
        let key = features_to_key(&feature_vec)?;
        self.inner
            .negative_interaction(&key, &personality.inner, tick);
        Ok(())
    }

    /// Read the effective coherence for a sensory context.
    ///
    /// Applies the minimum gate: both accumulated trust and the instant reading
    /// must be high for the result to be high.
    ///
    /// Args:
    ///     instant:     raw instant sensor reading in [0.0, 1.0]
    ///     feature_vec: 6 floats in [0.0, 1.0] — current sensory state
    ///
    /// Returns:
    ///     Effective coherence in [0.0, 1.0]
    pub fn effective_coherence(&self, instant: f32, feature_vec: Vec<f32>) -> PyResult<f32> {
        let key = features_to_key(&feature_vec)?;
        Ok(self.inner.effective_coherence(instant, &key))
    }

    /// Python repr string.
    pub fn __repr__(&self) -> &'static str {
        "CoherenceField()"
    }
}

// ── Module entry point ────────────────────────────────────────────────────────

/// CCF — Contextual Coherence Fields Python bindings.
///
/// Exposes the CCF pipeline for earned relational trust in autonomous agents.
/// Feature vector dimension is fixed at 6 (mBot2 vocabulary).
/// For custom dimensions use the Rust API directly.
#[pymodule]
pub fn ccf_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPersonality>()?;
    m.add_class::<PyPhaseSpace>()?;
    m.add_class::<PySocialPhase>()?;
    m.add_class::<PyCoherenceField>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("FEATURE_DIM", PY_DIM)?;
    Ok(())
}
