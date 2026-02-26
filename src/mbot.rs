//! mBot2 reference sensor vocabulary.
//!
//! The [mBot2](https://www.makeblock.com/mbot2) is a programmable robot by Makeblock,
//! retailing for around **$50–$80 USD**. It runs a CyberPi microcontroller with onboard
//! light, sound, and motion sensors.
//!
//! This module provides a ready-to-use [`SensorVocabulary`] implementation for the mBot2's
//! six onboard sensor dimensions. It is the vocabulary used in the CCF reference demo —
//! the same robot that demonstrates emergent social behaviour through accumulated experience,
//! entirely on-device with no cloud or ML model required.
//!
//! # Why this exists
//!
//! The mBot2 demo is the existence proof for what ccf-core makes possible: a sub-$100
//! robot that develops genuine context-sensitive social behaviour. Not scripted. Not
//! rule-based. The trust field accumulates through real interaction, and the robot's
//! responses emerge from what it has actually experienced in each environment.
//!
//! This module ships as a concrete reference so you can see exactly what a production
//! `SensorVocabulary` implementation looks like. Your own hardware vocabulary follows
//! the same pattern — just swap in your sensor dimensions.
//!
//! # See also
//!
//! - `examples/mbot2.rs` — full simulated CCF loop for the mBot2
//! - [`SensorVocabulary`] — the trait to implement for your own hardware

use crate::vocabulary::{ContextKey, SensorVocabulary};

/// mBot2 sensor vocabulary — 6-dimensional context for the CyberPi microcontroller.
///
/// Covers the six onboard sensor dimensions most relevant to social behaviour:
/// ambient light, ambient sound, nearby presence, robot motion, orientation, and
/// time of day (set by the host application).
///
/// ```rust
/// use ccf_core::mbot::{MbotSensors, BrightnessBand, NoiseBand,
///     PresenceSignature, MotionContext, Orientation, TimePeriod};
/// use ccf_core::vocabulary::ContextKey;
///
/// let key = ContextKey::new(MbotSensors {
///     brightness:  BrightnessBand::Bright,
///     noise:       NoiseBand::Quiet,
///     presence:    PresenceSignature::Close,
///     motion:      MotionContext::Static,
///     orientation: Orientation::Upright,
///     time_period: TimePeriod::Day,
/// });
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MbotSensors {
    /// Ambient light level (CyberPi light sensor).
    pub brightness: BrightnessBand,
    /// Ambient sound level (CyberPi microphone).
    pub noise: NoiseBand,
    /// Nearby presence signature (proximity / IR sensor).
    pub presence: PresenceSignature,
    /// Robot motion context (derived from wheel encoders).
    pub motion: MotionContext,
    /// Robot orientation relative to starting heading (IMU).
    pub orientation: Orientation,
    /// Time of day period (set by host application or RTC).
    pub time_period: TimePeriod,
}

/// Ambient light level — quantised from the CyperPi light sensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BrightnessBand {
    /// Very low ambient light (night, dark room).
    Dark,
    /// Moderate ambient light (indoor daytime, lamp).
    Dim,
    /// High ambient light (bright room, direct sunlight).
    Bright,
}

/// Ambient sound level — quantised from the CyberPi microphone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NoiseBand {
    /// Very low ambient noise (silent room).
    Quiet,
    /// Moderate ambient noise (background conversation, music).
    Moderate,
    /// High ambient noise (crowd, machinery, shouting).
    Loud,
}

/// Nearby presence signature — person or object detection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PresenceSignature {
    /// No person detected in sensor range.
    Absent,
    /// Person detected at a distance (outer detection zone).
    Far,
    /// Person detected in close proximity (inner detection zone).
    Close,
}

/// Robot motion context — derived from wheel encoder state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MotionContext {
    /// Robot is stationary (encoders at rest).
    Static,
    /// Robot is moving slowly (below speed threshold).
    Slow,
    /// Robot is moving quickly (above speed threshold).
    Fast,
}

/// Robot orientation relative to its starting heading (IMU pitch/roll).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Orientation {
    /// Robot is upright — tilt within acceptable range.
    Upright,
    /// Robot is tilted beyond the upright threshold (picked up, on slope).
    Tilted,
}

/// Time of day period — set by the host application or RTC.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TimePeriod {
    /// Daytime hours (context: active, well-lit environments typical).
    Day,
    /// Evening hours (context: winding down, lower light typical).
    Evening,
    /// Night-time hours (context: quiet, dark environments typical).
    Night,
}

impl SensorVocabulary<6> for MbotSensors {
    fn to_feature_vec(&self) -> [f32; 6] {
        let b = match self.brightness {
            BrightnessBand::Dark   => 0.0,
            BrightnessBand::Dim    => 0.5,
            BrightnessBand::Bright => 1.0,
        };
        let n = match self.noise {
            NoiseBand::Quiet    => 0.0,
            NoiseBand::Moderate => 0.5,
            NoiseBand::Loud     => 1.0,
        };
        let p = match self.presence {
            PresenceSignature::Absent => 0.0,
            PresenceSignature::Far    => 0.5,
            PresenceSignature::Close  => 1.0,
        };
        let m = match self.motion {
            MotionContext::Static => 0.0,
            MotionContext::Slow   => 0.5,
            MotionContext::Fast   => 1.0,
        };
        let o = match self.orientation {
            Orientation::Upright => 0.0,
            Orientation::Tilted  => 1.0,
        };
        let t = match self.time_period {
            TimePeriod::Day     => 0.0,
            TimePeriod::Evening => 0.5,
            TimePeriod::Night   => 1.0,
        };
        [b, n, p, m, o, t]
    }
}

/// Type alias for the canonical mBot2 context key.
pub type MbotContextKey = ContextKey<MbotSensors, 6>;
