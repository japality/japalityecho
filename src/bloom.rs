/// A counting Bloom filter with 4-bit counters (max value 15) packed two per byte.
///
/// Provides approximate membership testing and minimum-count queries for u64 keys
/// using multiple independent hash functions built from bit-mixing.

#[derive(Debug, Clone)]
pub struct CountingBloomFilter {
    /// Packed 4-bit counters: low nibble = even slot, high nibble = odd slot.
    counters: Vec<u8>,
    num_hashes: usize,
    num_slots: usize,
}

impl CountingBloomFilter {
    /// Create a new counting Bloom filter sized for `expected_items` with the
    /// given target `false_positive_rate`.
    ///
    /// Optimal parameters:
    ///   m = -n * ln(p) / (ln2)^2
    ///   k = (m / n) * ln2
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let n = (expected_items.max(1)) as f64;
        let p = false_positive_rate.clamp(1e-15, 1.0 - 1e-15);
        let ln2 = core::f64::consts::LN_2;

        let m = (-n * p.ln() / (ln2 * ln2)).ceil() as usize;
        let m = m.max(1);

        let k = ((m as f64 / n) * ln2).ceil() as usize;
        let k = k.clamp(1, 32);

        // Each byte holds two 4-bit counters.
        let bytes_needed = (m + 1) / 2;

        Self {
            counters: vec![0u8; bytes_needed],
            num_hashes: k,
            num_slots: m,
        }
    }

    /// Insert a key, incrementing all `num_hashes` counter positions.
    /// Counters saturate at 15 (4-bit max).
    pub fn insert(&mut self, key: u64) {
        for i in 0..self.num_hashes {
            let slot = self.hash(key, i);
            let current = self.get_counter(slot);
            if current < 15 {
                self.set_counter(slot, current + 1);
            }
        }
    }

    /// Query the approximate count of a key: the minimum counter value across
    /// all hash positions.
    pub fn count(&self, key: u64) -> u8 {
        let mut min = u8::MAX;
        for i in 0..self.num_hashes {
            let slot = self.hash(key, i);
            let val = self.get_counter(slot);
            min = min.min(val);
            if min == 0 {
                return 0;
            }
        }
        min
    }

    /// Total heap memory used by the counter array, in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.counters.len()
    }

    // --- internal helpers ---

    /// Compute slot index for `key` using hash function `seed`.
    /// Uses a murmur-style bit-mixing cascade.
    #[inline]
    fn hash(&self, key: u64, seed: usize) -> usize {
        let mut h = key ^ (seed as u64).wrapping_mul(0x517cc1b727220a95);
        h = h.wrapping_mul(0xbf58476d1ce4e5b9);
        h ^= h >> 31;
        h = h.wrapping_mul(0x94d049bb133111eb);
        h ^= h >> 31;
        (h as usize) % self.num_slots
    }

    /// Read the 4-bit counter at `slot`.
    #[inline]
    fn get_counter(&self, slot: usize) -> u8 {
        let byte_idx = slot / 2;
        let byte = self.counters[byte_idx];
        if slot % 2 == 0 {
            byte & 0x0F
        } else {
            byte >> 4
        }
    }

    /// Write a 4-bit value into the counter at `slot`.
    #[inline]
    fn set_counter(&mut self, slot: usize, value: u8) {
        let byte_idx = slot / 2;
        let byte = &mut self.counters[byte_idx];
        if slot % 2 == 0 {
            *byte = (*byte & 0xF0) | (value & 0x0F);
        } else {
            *byte = (*byte & 0x0F) | ((value & 0x0F) << 4);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_filter_returns_zero() {
        let bf = CountingBloomFilter::new(1000, 0.01);
        assert_eq!(bf.count(42), 0);
        assert_eq!(bf.count(0), 0);
        assert_eq!(bf.count(u64::MAX), 0);
    }

    #[test]
    fn inserted_key_has_positive_count() {
        let mut bf = CountingBloomFilter::new(1000, 0.01);
        bf.insert(42);
        assert!(bf.count(42) >= 1);
    }

    #[test]
    fn multiple_inserts_increase_count() {
        let mut bf = CountingBloomFilter::new(10_000, 0.01);
        for _ in 0..5 {
            bf.insert(99);
        }
        let c = bf.count(99);
        assert!(c >= 5, "expected at least 5, got {c}");
    }

    #[test]
    fn counters_saturate_at_15() {
        let mut bf = CountingBloomFilter::new(1000, 0.01);
        for _ in 0..100 {
            bf.insert(7);
        }
        assert_eq!(bf.count(7), 15);
    }

    #[test]
    fn low_false_positive_rate() {
        let n = 10_000usize;
        let mut bf = CountingBloomFilter::new(n, 0.01);
        for i in 0..n as u64 {
            bf.insert(i);
        }
        let mut false_positives = 0u64;
        let test_range = n as u64..n as u64 + 100_000;
        let test_count = test_range.clone().count() as f64;
        for key in test_range {
            if bf.count(key) > 0 {
                false_positives += 1;
            }
        }
        let fpr = false_positives as f64 / test_count;
        assert!(fpr < 0.05, "FPR too high: {fpr:.4}");
    }

    #[test]
    fn memory_is_reasonable() {
        let bf = CountingBloomFilter::new(10_000_000, 0.01);
        let mb = bf.memory_bytes() as f64 / (1024.0 * 1024.0);
        // ~12 MB expected for 10M items at 1% FPR with 4-bit counters
        assert!(mb < 60.0, "memory too high: {mb:.1} MB");
        assert!(mb > 1.0, "memory suspiciously low: {mb:.1} MB");
    }
}
