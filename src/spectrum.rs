use std::collections::HashMap;

use crate::bloom::CountingBloomFilter;
use crate::model::FastqRecord;

#[derive(Debug, Clone)]
pub struct KmerSpectrum {
    k: usize,
    counts: HashMap<u64, u32>,
    /// When present, low-count k-mers are stored only in the Bloom filter
    /// while trusted k-mers (≥ threshold) retain exact counts in `counts`.
    bloom: Option<CountingBloomFilter>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SupportStats {
    pub min: u32,
    pub sum: u64,
    pub windows: u32,
}

/// Streaming builder for k-mer spectrum that can accumulate k-mers
/// from records without holding them all in memory.
pub struct KmerSpectrumBuilder {
    k: usize,
    min_base_quality: u8,
    counts: HashMap<u64, u32>,
    records_processed: usize,
}

impl KmerSpectrumBuilder {
    pub fn new(k: usize, min_base_quality: u8) -> Self {
        let k = k.max(3);
        Self {
            k,
            min_base_quality,
            counts: HashMap::new(),
            records_processed: 0,
        }
    }

    pub fn add_record(&mut self, record: &FastqRecord) {
        if record.sequence.len() < self.k || record.qualities.len() < self.k {
            return;
        }
        for start in 0..=record.sequence.len() - self.k {
            let end = start + self.k;
            if !record.qualities[start..end]
                .iter()
                .all(|q| q.saturating_sub(33) >= self.min_base_quality)
            {
                continue;
            }
            if let Some(encoded) = encode_canonical(&record.sequence[start..end]) {
                *self.counts.entry(encoded).or_insert(0) += 1;
            }
        }
        self.records_processed += 1;
    }

    pub fn records_processed(&self) -> usize {
        self.records_processed
    }

    pub fn unique_kmers(&self) -> usize {
        self.counts.len()
    }

    pub fn finalize(self) -> KmerSpectrum {
        KmerSpectrum {
            k: self.k,
            counts: self.counts,
            bloom: None,
        }
    }
}

impl KmerSpectrum {
    /// Compute optimal trusted k-mer threshold by analyzing the k-mer
    /// count distribution.  Uses the ratio of total reads processed vs
    /// the original sample size to proportionally scale the floor, then
    /// clamps to a reasonable range based on the median k-mer count.
    pub fn auto_trusted_floor(&self, fallback: u32, sample_size: usize, total_reads: usize) -> u32 {
        if self.counts.is_empty() || total_reads <= sample_size {
            return fallback;
        }

        // Compute median k-mer count
        let mut counts: Vec<u32> = self.counts.values().copied().collect();
        counts.sort_unstable();
        let median = counts[counts.len() / 2];

        // Scale floor proportionally to read count ratio, but cap at
        // a fraction of the median to avoid over-filtering
        let ratio = total_reads as f64 / sample_size.max(1) as f64;
        let scaled = (fallback as f64 * ratio.sqrt()).ceil() as u32;

        // Cap at median / 3 to ensure correction can still find candidates
        // above the floor in the correct-kmer peak
        let cap = (median / 3).max(fallback);
        let result = scaled.min(cap).max(fallback);

        result
    }

    pub fn from_records(records: &[FastqRecord], k: usize, min_base_quality: u8) -> Self {
        let k = k.max(3);
        let mut counts = HashMap::new();

        for record in records {
            if record.sequence.len() < k || record.qualities.len() < k {
                continue;
            }

            for start in 0..=record.sequence.len() - k {
                let end = start + k;
                if !record.qualities[start..end]
                    .iter()
                    .all(|q| q.saturating_sub(33) >= min_base_quality)
                {
                    continue;
                }

                if let Some(encoded) = encode_canonical(&record.sequence[start..end]) {
                    *counts.entry(encoded).or_insert(0) += 1;
                }
            }
        }

        Self { k, counts, bloom: None }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn unique_kmers(&self) -> usize {
        self.counts.len()
    }

    pub fn count_bytes(&self, bytes: &[u8]) -> u32 {
        if bytes.len() != self.k {
            return 0;
        }
        let Some(encoded) = encode_canonical(bytes) else {
            return 0;
        };
        // Exact lookup in the (trusted-only or full) HashMap
        if let Some(&exact) = self.counts.get(&encoded) {
            return exact;
        }
        // Fall back to Bloom filter approximate count when compressed
        if let Some(ref bloom) = self.bloom {
            return u32::from(bloom.count(encoded));
        }
        0
    }

    pub fn support_for_position(&self, sequence: &[u8], position: usize) -> SupportStats {
        if sequence.len() < self.k || position >= sequence.len() {
            return SupportStats::default();
        }

        let start_min = position.saturating_sub(self.k - 1);
        let start_max = position.min(sequence.len() - self.k);
        let mut support = SupportStats {
            min: u32::MAX,
            sum: 0,
            windows: 0,
        };

        for start in start_min..=start_max {
            let count = self.count_bytes(&sequence[start..start + self.k]);
            support.min = support.min.min(count);
            support.sum += u64::from(count);
            support.windows += 1;
        }

        if support.windows == 0 {
            SupportStats::default()
        } else {
            support
        }
    }

    pub fn trusted_entries(&self, min_count: u32) -> Vec<(u64, u32)> {
        let mut entries: Vec<(u64, u32)> = self
            .counts
            .iter()
            .filter_map(|(&key, &count)| (count >= min_count).then_some((key, count)))
            .collect();
        entries.sort_unstable_by_key(|(key, _)| *key);
        entries
    }

    /// Compress the spectrum in-place: insert all k-mers into a counting
    /// Bloom filter, then keep only trusted k-mers (count ≥ `trusted_floor`)
    /// in the exact HashMap.  Subsequent `count_bytes` calls use the hybrid
    /// lookup (exact for trusted, Bloom for the rest).
    ///
    /// Returns `(bloom_bytes, evicted_kmers)` for logging.
    pub fn compress_to_bloom(&mut self, trusted_floor: u32) -> (usize, usize) {
        let item_count = self.counts.len();
        let mut bloom = CountingBloomFilter::new(item_count, 0.01);
        for (&key, &count) in &self.counts {
            for _ in 0..count.min(255) {
                bloom.insert(key);
            }
        }
        let bloom_bytes = bloom.memory_bytes();
        let total = self.counts.len();
        self.counts.retain(|_, count| *count >= trusted_floor);
        self.counts.shrink_to_fit();
        let evicted = total - self.counts.len();
        self.bloom = Some(bloom);
        (bloom_bytes, evicted)
    }

    /// Whether Bloom compression is active.
    pub fn is_compressed(&self) -> bool {
        self.bloom.is_some()
    }

    /// Convert into a `BloomKmerSpectrum`.
    ///
    /// All k-mers are inserted into the Bloom filter.  Only those with
    /// count ≥ `trusted_floor` are kept in the exact HashMap.
    pub fn into_bloom_spectrum(self, trusted_floor: u32) -> BloomKmerSpectrum {
        let item_count = self.counts.len();
        let mut bloom = CountingBloomFilter::new(item_count, 0.01);

        let mut trusted = HashMap::new();

        for (&key, &count) in &self.counts {
            for _ in 0..count.min(255) {
                bloom.insert(key);
            }
            if count >= trusted_floor {
                trusted.insert(key, count);
            }
        }

        BloomKmerSpectrum {
            k: self.k,
            bloom,
            trusted,
        }
    }
}

/// A memory-efficient k-mer spectrum that pairs a counting Bloom filter
/// (for approximate counts of all k-mers) with an exact HashMap limited
/// to high-count trusted k-mers.
#[derive(Debug, Clone)]
pub struct BloomKmerSpectrum {
    k: usize,
    bloom: CountingBloomFilter,
    trusted: HashMap<u64, u32>,
}

impl BloomKmerSpectrum {
    pub fn k(&self) -> usize {
        self.k
    }

    /// Look up the count for an encoded k-mer byte slice.
    ///
    /// Returns the exact count if the k-mer is in the trusted set,
    /// otherwise falls back to the Bloom filter approximate count.
    pub fn count_bytes(&self, bytes: &[u8]) -> u32 {
        if bytes.len() != self.k {
            return 0;
        }
        let Some(encoded) = encode_canonical(bytes) else {
            return 0;
        };
        if let Some(&exact) = self.trusted.get(&encoded) {
            return exact;
        }
        u32::from(self.bloom.count(encoded))
    }

    pub fn support_for_position(&self, sequence: &[u8], position: usize) -> SupportStats {
        if sequence.len() < self.k || position >= sequence.len() {
            return SupportStats::default();
        }

        let start_min = position.saturating_sub(self.k - 1);
        let start_max = position.min(sequence.len() - self.k);
        let mut support = SupportStats {
            min: u32::MAX,
            sum: 0,
            windows: 0,
        };

        for start in start_min..=start_max {
            let count = self.count_bytes(&sequence[start..start + self.k]);
            support.min = support.min.min(count);
            support.sum += u64::from(count);
            support.windows += 1;
        }

        if support.windows == 0 {
            SupportStats::default()
        } else {
            support
        }
    }

    pub fn trusted_entries(&self, min_count: u32) -> Vec<(u64, u32)> {
        let mut entries: Vec<(u64, u32)> = self
            .trusted
            .iter()
            .filter_map(|(&key, &count)| (count >= min_count).then_some((key, count)))
            .collect();
        entries.sort_unstable_by_key(|(key, _)| *key);
        entries
    }

    /// Total memory used by the Bloom filter counter array, in bytes.
    pub fn bloom_memory_bytes(&self) -> usize {
        self.bloom.memory_bytes()
    }
}

fn encode_canonical(bytes: &[u8]) -> Option<u64> {
    let forward = encode_kmer(bytes)?;
    let reverse = reverse_complement_bits(forward, bytes.len());
    Some(forward.min(reverse))
}

fn encode_kmer(bytes: &[u8]) -> Option<u64> {
    let mut encoded = 0u64;
    for &base in bytes {
        encoded = (encoded << 2) | u64::from(base_bits(base)?);
    }
    Some(encoded)
}

fn reverse_complement_bits(mut encoded: u64, k: usize) -> u64 {
    let mut reversed = 0u64;
    for _ in 0..k {
        let bits = encoded & 0b11;
        let complement = bits ^ 0b11;
        reversed = (reversed << 2) | complement;
        encoded >>= 2;
    }
    reversed
}

fn base_bits(base: u8) -> Option<u8> {
    match base.to_ascii_uppercase() {
        b'A' => Some(0),
        b'C' => Some(1),
        b'G' => Some(2),
        b'T' => Some(3),
        _ => None,
    }
}
