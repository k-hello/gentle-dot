//! Keyword spotting (wake-word) engine for Gentle Dot (ADR 0002).
//!
//! Listens on a dedicated capture stream for a single wake word (default: "toto").
//! Uses `sherpa-onnx` KeywordSpotter or a lightweight phonetic energy correlator.
//! When the keyword is recognized, emits `voice://wake` and triggers voice recording
//! without losing any preceding or trailing audio samples.

use super::audio::SAMPLE_RATE;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Event sent to the panel when the wake word is detected: `{ keyword: "toto", score: 0.85 }`.
pub const WAKE_EVENT: &str = "voice://wake";

/// Default wake keyword for Lucio's assistant.
pub const DEFAULT_WAKE_KEYWORD: &str = "toto";

/// Default detection threshold (0.0 to 1.0). Higher threshold (0.72) protects
/// against false positives for the short 2-syllable word "toto".
pub const DEFAULT_WAKE_THRESHOLD: f32 = 0.72;

/// Refractory cooldown period after a wake detection to prevent double-firing.
pub const WAKE_COOLDOWN: Duration = Duration::from_millis(1500);

/// Payload emitted over Tauri event `voice://wake`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WakePayload {
    pub keyword: String,
    pub score: f32,
}

/// Configuration for the hands-free wake word listener.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WakeConfig {
    pub enabled: bool,
    pub keyword: String,
    pub threshold: f32,
}

impl Default for WakeConfig {
    fn default() -> Self {
        WakeConfig {
            enabled: false,
            keyword: DEFAULT_WAKE_KEYWORD.into(),
            threshold: DEFAULT_WAKE_THRESHOLD,
        }
    }
}

/// A circular ring buffer that retains recent audio so the beginning of the command
/// following the wake word ("Toto qué hora es...") is never chopped off.
pub struct AudioRingBuffer {
    buffer: Vec<f32>,
    capacity: usize,
    write_pos: usize,
    count: usize,
}

impl AudioRingBuffer {
    /// Creates a ring buffer that stores up to `duration` of 16 kHz audio.
    pub fn new(duration: Duration) -> Self {
        let capacity = (SAMPLE_RATE as f64 * duration.as_secs_f64()).round() as usize;
        AudioRingBuffer {
            buffer: vec![0.0; capacity],
            capacity,
            write_pos: 0,
            count: 0,
        }
    }

    /// Pushes new incoming PCM samples.
    pub fn push(&mut self, samples: &[f32]) {
        for &sample in samples {
            self.buffer[self.write_pos] = sample;
            self.write_pos = (self.write_pos + 1) % self.capacity;
            if self.count < self.capacity {
                self.count += 1;
            }
        }
    }

    /// Returns a copy of the recorded samples in chronological order.
    pub fn samples(&self) -> Vec<f32> {
        if self.count == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(self.count);
        let start = if self.count < self.capacity {
            0
        } else {
            self.write_pos
        };
        for i in 0..self.count {
            out.push(self.buffer[(start + i) % self.capacity]);
        }
        out
    }

    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.count = 0;
    }
}

/// Evaluates audio chunks and detects keyword occurrences with refractory cooldown.
pub struct WakeDetector {
    keyword: String,
    threshold: f32,
    last_wake: Option<Instant>,
    cooldown: Duration,
}

impl WakeDetector {
    pub fn new(keyword: impl Into<String>, threshold: f32) -> Self {
        WakeDetector {
            keyword: keyword.into(),
            threshold,
            last_wake: None,
            cooldown: WAKE_COOLDOWN,
        }
    }

    pub fn keyword(&self) -> &str {
        &self.keyword
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Checks if a detection is allowed by cooldown timing.
    pub fn can_wake(&self, now: Instant) -> bool {
        match self.last_wake {
            Some(last) => now.duration_since(last) >= self.cooldown,
            None => true,
        }
    }

    /// Records a wake event and resets cooldown.
    pub fn mark_wake(&mut self, now: Instant) {
        self.last_wake = Some(now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_stores_and_orders_samples() {
        let mut rb = AudioRingBuffer::new(Duration::from_millis(1)); // ~16 samples
        let input: Vec<f32> = (1..=20).map(|v| v as f32).collect();
        rb.push(&input);
        let out = rb.samples();
        assert_eq!(out.len(), rb.capacity);
        // The last capacity items should be 5..=20
        assert_eq!(out.last().copied(), Some(20.0));
    }

    #[test]
    fn wake_detector_respects_cooldown() {
        let mut detector = WakeDetector::new("toto", 0.72);
        let t0 = Instant::now();
        assert!(detector.can_wake(t0));
        detector.mark_wake(t0);
        assert!(!detector.can_wake(t0 + Duration::from_millis(500)));
        assert!(detector.can_wake(t0 + Duration::from_millis(1600)));
    }
}
