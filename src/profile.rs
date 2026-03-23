use std::collections::{HashMap, HashSet};

use crate::model::{
    AdapterCandidate, AutoProfile, BackendPreference, BarcodeHint, ExecutionPlan, ExperimentType,
    FastqRecord, Platform, QualityProfile,
};

const KNOWN_ADAPTERS: &[(&str, &str)] = &[
    ("illumina-universal", "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"),
    ("illumina-short", "AGATCGGAAGAGC"),
    ("nextera-transposase", "CTGTCTCTTATACACATCT"),
];

const ATAC_NEXTERA_SCORE_MIN: f64 = 0.01;
const ILLUMINA_ADAPTER_SCORE_MIN: f64 = 0.001;

/// Minimum combined RNA-Seq evidence score (out of 1.0) to classify as RNA-Seq.
/// Each signal contributes a weighted partial score; a composite ≥ this threshold
/// triggers RNA-Seq classification even when no single signal is overwhelming.
const RNA_EVIDENCE_THRESHOLD: f64 = 0.40;

pub fn infer_auto_profile(records: &[FastqRecord]) -> AutoProfile {
    if records.is_empty() {
        return AutoProfile {
            platform: Platform::Unknown,
            experiment: ExperimentType::Unknown,
            read_count: 0,
            mean_read_length: 0.0,
            median_read_length: 0,
            length_stddev: 0.0,
            quality_profile: QualityProfile {
                mean_phred: 0.0,
                tail_drop: 0.0,
                cycle_means: Vec::new(),
            },
            adapter_candidates: Vec::new(),
            barcode_hint: None,
            notes: vec!["No reads available for profiling".to_string()],
        };
    }

    let lengths: Vec<usize> = records.iter().map(FastqRecord::len).collect();
    let mean_read_length = mean_usize(&lengths);
    let median_read_length = median_usize(&lengths);
    let length_stddev = stddev_usize(&lengths, mean_read_length);

    let max_cycle = lengths.iter().copied().max().unwrap_or(0);
    let mut cycle_sums = vec![0u64; max_cycle];
    let mut cycle_counts = vec![0u64; max_cycle];
    let mut total_phred = 0u64;
    let mut total_bases = 0u64;
    let mut poly_tail_hits = 0usize;

    for record in records {
        for (cycle, phred) in record
            .phred_scores()
            .take(record.sequence.len())
            .enumerate()
        {
            cycle_sums[cycle] += u64::from(phred);
            cycle_counts[cycle] += 1;
            total_phred += u64::from(phred);
            total_bases += 1;
        }

        if has_homopolymer_tail(&record.sequence, b'A')
            || has_homopolymer_tail(&record.sequence, b'T')
        {
            poly_tail_hits += 1;
        }
    }

    let cycle_means: Vec<f64> = cycle_sums
        .iter()
        .zip(cycle_counts.iter())
        .map(|(&sum, &count)| {
            if count == 0 {
                0.0
            } else {
                sum as f64 / count as f64
            }
        })
        .collect();

    let mean_phred = if total_bases == 0 {
        0.0
    } else {
        total_phred as f64 / total_bases as f64
    };
    let tail_drop = tail_drop(&cycle_means);
    let quality_profile = QualityProfile {
        mean_phred,
        tail_drop,
        cycle_means,
    };

    let barcode_hint = detect_barcode_hint(records, median_read_length);
    let adapter_candidates = detect_adapter_candidates(records);
    let leading_cycle_bias = estimate_leading_cycle_bias(records, 12);
    let sequence_duplication_rate = estimate_sequence_duplication_rate(records);
    let gc_content_bias = estimate_gc_content_bias(records, 10);
    let kmer_diversity = estimate_kmer_diversity(records, 15);
    let platform = detect_platform(
        mean_read_length,
        length_stddev,
        mean_phred,
        tail_drop,
        &adapter_candidates,
    );
    let poly_tail_rate = poly_tail_hits as f64 / records.len() as f64;
    let experiment = detect_experiment(
        platform,
        mean_read_length,
        &adapter_candidates,
        barcode_hint.as_ref(),
        poly_tail_rate,
        leading_cycle_bias,
        sequence_duplication_rate,
        gc_content_bias,
        kmer_diversity,
    );

    let mut notes = vec![format!(
        "Sampled {} reads with mean length {:.1} and mean phred {:.1}",
        records.len(),
        mean_read_length,
        mean_phred
    )];

    if let Some(hint) = &barcode_hint {
        notes.push(format!(
            "Detected barcode/poly-T structure with {:.1}% support",
            hint.poly_t_rate * 100.0
        ));
    }
    if let Some(top_adapter) = adapter_candidates.first() {
        notes.push(format!(
            "Top adapter candidate {} supported by {} reads",
            top_adapter.name, top_adapter.support
        ));
    }
    if poly_tail_rate > 0.15 {
        notes.push(format!(
            "Observed homopolymer tail signal in {:.1}% of sampled reads",
            poly_tail_rate * 100.0
        ));
    }
    if leading_cycle_bias >= 0.05 {
        notes.push(format!(
            "Observed {:.1}% leading-cycle composition bias in the first 12 cycles",
            leading_cycle_bias * 100.0
        ));
    }
    if sequence_duplication_rate >= 0.10 {
        notes.push(format!(
            "Observed {:.1}% approximate read duplication in the sampled profile",
            sequence_duplication_rate * 100.0
        ));
    }
    if gc_content_bias >= 0.04 {
        notes.push(format!(
            "Observed {:.1}% GC-content bias in the first 10 cycles (RNA-Seq indicator)",
            gc_content_bias * 100.0
        ));
    }
    if kmer_diversity < 0.85 {
        notes.push(format!(
            "k-mer diversity {:.3} (low diversity suggests transcriptomic origin)",
            kmer_diversity
        ));
    }

    AutoProfile {
        platform,
        experiment,
        read_count: records.len(),
        mean_read_length,
        median_read_length,
        length_stddev,
        quality_profile,
        adapter_candidates,
        barcode_hint,
        notes,
    }
}

pub fn build_execution_plan(
    profile: &AutoProfile,
    requested_backend: BackendPreference,
) -> ExecutionPlan {
    let kmer_size = match profile.platform {
        Platform::Ont | Platform::PacBio => 11,
        _ if profile.mean_read_length < 75.0 => 13,
        _ => 15,
    };

    let trim_min_quality = match profile.platform {
        Platform::Ont => 10,
        Platform::PacBio => 15,
        _ => 20,
    };

    let trusted_kmer_min_count = match profile.experiment {
        ExperimentType::SingleCell10xV2 | ExperimentType::SingleCell10xV3 => 3,
        ExperimentType::LongRead => 2,
        _ => 4,
    };

    let minimum_output_length = match profile.experiment {
        ExperimentType::SingleCell10xV2 | ExperimentType::SingleCell10xV3 => 20,
        ExperimentType::LongRead => 200,
        _ => 35,
    };

    let zero_copy_candidate = !matches!(requested_backend, BackendPreference::Cpu)
        && matches!(profile.platform, Platform::Illumina | Platform::Mgi)
        && profile.mean_read_length >= 75.0;

    let overlap_depth = if zero_copy_candidate { 3 } else { 1 };

    let mut notes = vec![format!(
        "Planner chose k={} and Q{} as the context-aware repair threshold",
        kmer_size, trim_min_quality
    )];
    notes.push(format!(
        "Trusted k-mer floor set to {} and minimum output length to {}",
        trusted_kmer_min_count, minimum_output_length
    ));
    if zero_copy_candidate {
        notes.push(
            "Input profile is compatible with pinned-memory / zero-copy GPU transfer planning"
                .to_string(),
        );
    } else {
        notes.push("Execution plan stays host-resident in this build".to_string());
    }

    ExecutionPlan {
        requested_backend,
        trim_min_quality,
        kmer_size,
        trusted_kmer_min_count,
        minimum_output_length,
        zero_copy_candidate,
        overlap_depth,
        adapter_candidates: profile.adapter_candidates.clone(),
        barcode_hint: profile.barcode_hint.clone(),
        notes,
    }
}

fn detect_platform(
    mean_read_length: f64,
    length_stddev: f64,
    mean_phred: f64,
    tail_drop: f64,
    adapters: &[AdapterCandidate],
) -> Platform {
    let illumina_adapter_signal = strongest_adapter_score(adapters, "illumina")
        .max(strongest_adapter_score(adapters, "nextera"));
    // Long-read: require BOTH mean > 800 AND stddev > 250
    // (stddev alone can trigger on mixed short+long contamination)
    if mean_read_length > 800.0 && length_stddev > 250.0 {
        if mean_phred < 20.0 {
            Platform::Ont
        } else {
            Platform::PacBio
        }
    } else if mean_read_length > 800.0 {
        // High mean length alone is still a strong indicator
        if mean_phred < 20.0 {
            Platform::Ont
        } else {
            Platform::PacBio
        }
    } else if mean_read_length >= 35.0 {
        if tail_drop >= 6.0 || illumina_adapter_signal >= ILLUMINA_ADAPTER_SCORE_MIN {
            Platform::Illumina
        } else if tail_drop <= 3.5 && mean_phred >= 28.0 {
            Platform::Mgi
        } else {
            Platform::Illumina
        }
    } else {
        Platform::Unknown
    }
}

fn detect_experiment(
    platform: Platform,
    mean_read_length: f64,
    adapters: &[AdapterCandidate],
    barcode_hint: Option<&BarcodeHint>,
    poly_tail_rate: f64,
    leading_cycle_bias: f64,
    sequence_duplication_rate: f64,
    gc_content_bias: f64,
    kmer_diversity: f64,
) -> ExperimentType {
    let nextera_signal = strongest_adapter_score(adapters, "nextera");
    if matches!(platform, Platform::Ont | Platform::PacBio) {
        ExperimentType::LongRead
    } else if let Some(hint) = barcode_hint {
        if hint.umi_bases >= 12 {
            ExperimentType::SingleCell10xV3
        } else {
            ExperimentType::SingleCell10xV2
        }
    } else if nextera_signal >= ATAC_NEXTERA_SCORE_MIN && mean_read_length < 120.0 {
        ExperimentType::AtacSeq
    } else if rna_seq_evidence_score(
        poly_tail_rate,
        leading_cycle_bias,
        sequence_duplication_rate,
        gc_content_bias,
        kmer_diversity,
    ) >= RNA_EVIDENCE_THRESHOLD
    {
        ExperimentType::RnaSeq
    } else {
        ExperimentType::Wgs
    }
}

/// Compute a composite RNA-Seq evidence score in [0, 1].
///
/// Five independent signals each contribute a weighted partial score:
///   1. Poly-A/T tail rate   — weight 0.30  (strong mRNA indicator)
///   2. Leading-cycle bias   — weight 0.20  (random-hexamer priming artifact)
///   3. Sequence duplication — weight 0.20  (high-expression transcript copies)
///   4. GC-content bias      — weight 0.15  (priming / transcript GC skew)
///   5. Low k-mer diversity  — weight 0.15  (transcriptome << genome complexity)
///
/// Each signal is converted to a 0–1 scale via a sigmoid-like ramp so that
/// partial evidence (e.g. poly-tail at 10% instead of the old hard 18%)
/// still contributes proportionally.
fn rna_seq_evidence_score(
    poly_tail_rate: f64,
    leading_cycle_bias: f64,
    sequence_duplication_rate: f64,
    gc_content_bias: f64,
    kmer_diversity: f64,
) -> f64 {
    // Signal 1: poly-A/T tail rate — ramp from 0.05 (floor) to 0.20 (saturated)
    let poly_score = smooth_ramp(poly_tail_rate, 0.05, 0.20);

    // Signal 2: leading-cycle composition bias — ramp from 0.03 to 0.10
    let bias_score = smooth_ramp(leading_cycle_bias, 0.03, 0.10);

    // Signal 3: sequence duplication rate — ramp from 0.08 to 0.25
    let dup_score = smooth_ramp(sequence_duplication_rate, 0.08, 0.25);

    // Signal 4: GC-content bias in first 10 cycles — ramp from 0.02 to 0.08
    let gc_score = smooth_ramp(gc_content_bias, 0.02, 0.08);

    // Signal 5: low k-mer diversity (inverted: lower diversity = higher score)
    // Typical WGS k-mer diversity ≈ 0.95+; RNA-Seq ≈ 0.70–0.90
    let kmer_score = smooth_ramp(1.0 - kmer_diversity, 0.05, 0.30);

    // Primary signals (poly-tail and duplication) are biologically specific to
    // RNA-Seq.  Secondary signals (cycle bias, GC skew, k-mer diversity) can
    // arise from non-biological sources (e.g. synthetic/low-complexity data).
    // Require at least some primary evidence before incorporating secondary
    // signals to avoid false positives.
    let primary = 0.40 * poly_score + 0.30 * dup_score;
    if primary < 0.01 {
        return primary;
    }

    let secondary = 0.15 * bias_score + 0.08 * gc_score + 0.07 * kmer_score;
    primary + secondary
}

/// Smooth ramp function: returns 0.0 when x <= lo, 1.0 when x >= hi,
/// and linearly interpolates in between.
fn smooth_ramp(x: f64, lo: f64, hi: f64) -> f64 {
    ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
}

fn detect_barcode_hint(records: &[FastqRecord], median_read_length: usize) -> Option<BarcodeHint> {
    let mut poly_t_hits = 0usize;
    let poly_t_run = if median_read_length <= 40 { 6 } else { 8 };

    for record in records {
        let upper = record.sequence.len().min(50);
        if upper < 28 {
            continue;
        }

        for start in 14..=upper.saturating_sub(poly_t_run) {
            if record.sequence[start..start + poly_t_run]
                .iter()
                .all(|base| base.to_ascii_uppercase() == b'T')
            {
                poly_t_hits += 1;
                break;
            }
        }
    }

    let rate = poly_t_hits as f64 / records.len() as f64;
    // Require strong poly-T signal: ≥40% for standard reads, ≥15% for very short reads.
    let minimum_rate = if median_read_length <= 40 { 0.15 } else { 0.40 };
    if rate < minimum_rate {
        return None;
    }

    let umi_bases = if median_read_length >= 28 { 12 } else { 10 };
    Some(BarcodeHint {
        prefix_bases: 16,
        umi_bases,
        poly_t_rate: rate,
    })
}

fn strongest_adapter_score(adapters: &[AdapterCandidate], needle: &str) -> f64 {
    adapters
        .iter()
        .filter(|candidate| candidate.name.contains(needle))
        .map(|candidate| candidate.score)
        .fold(0.0, f64::max)
}

fn estimate_sequence_duplication_rate(records: &[FastqRecord]) -> f64 {
    if records.is_empty() {
        return 0.0;
    }

    let unique = records
        .iter()
        .map(|record| record.sequence.as_slice())
        .collect::<HashSet<_>>()
        .len();
    1.0 - unique as f64 / records.len() as f64
}

fn estimate_leading_cycle_bias(records: &[FastqRecord], max_cycles: usize) -> f64 {
    if records.is_empty() || max_cycles == 0 {
        return 0.0;
    }

    let usable_cycles = records
        .iter()
        .map(FastqRecord::len)
        .min()
        .unwrap_or(0)
        .min(max_cycles);
    if usable_cycles == 0 {
        return 0.0;
    }

    let mut overall_counts = [0u64; 5];
    let mut cycle_counts = vec![[0u64; 5]; usable_cycles];

    for record in records {
        for &base in &record.sequence {
            overall_counts[base_bucket(base)] += 1;
        }
        for (cycle, &base) in record.sequence.iter().take(usable_cycles).enumerate() {
            cycle_counts[cycle][base_bucket(base)] += 1;
        }
    }

    let total_bases = overall_counts.iter().sum::<u64>() as f64;
    if total_bases == 0.0 {
        return 0.0;
    }

    let overall_freqs = overall_counts.map(|count| count as f64 / total_bases);
    cycle_counts
        .iter()
        .map(|counts| {
            let cycle_total = counts.iter().sum::<u64>() as f64;
            if cycle_total == 0.0 {
                return 0.0;
            }

            counts
                .iter()
                .enumerate()
                .map(|(index, &count)| {
                    let frequency = count as f64 / cycle_total;
                    (frequency - overall_freqs[index]).abs()
                })
                .sum::<f64>()
                / 2.0
        })
        .sum::<f64>()
        / usable_cycles as f64
}

fn base_bucket(base: u8) -> usize {
    match base.to_ascii_uppercase() {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        b'T' => 3,
        _ => 4,
    }
}

/// Measure GC-content deviation in the first `max_cycles` positions relative
/// to the overall GC-content of the sampled reads.  Random-hexamer priming
/// in RNA-Seq introduces a characteristic GC skew in the first ~10 cycles.
fn estimate_gc_content_bias(records: &[FastqRecord], max_cycles: usize) -> f64 {
    if records.is_empty() || max_cycles == 0 {
        return 0.0;
    }

    let mut overall_gc = 0u64;
    let mut overall_total = 0u64;
    let mut cycle_gc = vec![0u64; max_cycles];
    let mut cycle_total = vec![0u64; max_cycles];

    for record in records {
        for &base in &record.sequence {
            let upper = base.to_ascii_uppercase();
            overall_total += 1;
            if upper == b'G' || upper == b'C' {
                overall_gc += 1;
            }
        }
        for (i, &base) in record.sequence.iter().take(max_cycles).enumerate() {
            let upper = base.to_ascii_uppercase();
            cycle_total[i] += 1;
            if upper == b'G' || upper == b'C' {
                cycle_gc[i] += 1;
            }
        }
    }

    if overall_total == 0 {
        return 0.0;
    }
    let overall_gc_frac = overall_gc as f64 / overall_total as f64;

    let mut bias_sum = 0.0;
    let mut usable = 0usize;
    for i in 0..max_cycles {
        if cycle_total[i] == 0 {
            continue;
        }
        let cycle_gc_frac = cycle_gc[i] as f64 / cycle_total[i] as f64;
        bias_sum += (cycle_gc_frac - overall_gc_frac).abs();
        usable += 1;
    }

    if usable == 0 {
        0.0
    } else {
        bias_sum / usable as f64
    }
}

/// Estimate k-mer diversity as the fraction of observed k-mers out of the
/// theoretical maximum for the sampled reads.  Transcriptomic data has lower
/// diversity because reads come from a limited set of transcripts, whereas
/// WGS reads span the whole genome.
///
/// We use 15-mers and cap the sample at 50 000 reads to keep it fast.
fn estimate_kmer_diversity(records: &[FastqRecord], k: usize) -> f64 {
    if records.is_empty() || k == 0 {
        return 1.0;
    }

    let cap = records.len().min(50_000);
    let mut seen = HashSet::new();
    let mut total_kmers = 0u64;

    for record in &records[..cap] {
        if record.sequence.len() < k {
            continue;
        }
        for window in record.sequence.windows(k) {
            total_kmers += 1;
            seen.insert(window);
        }
    }

    if total_kmers == 0 {
        return 1.0;
    }
    seen.len() as f64 / total_kmers as f64
}

fn detect_adapter_candidates(records: &[FastqRecord]) -> Vec<AdapterCandidate> {
    let mut candidates = Vec::new();
    let mut seen_sequences = HashSet::new();

    for (name, adapter) in KNOWN_ADAPTERS {
        let support = records
            .iter()
            .filter(|record| adapter_tail_support(&record.sequence, adapter.as_bytes()).is_some())
            .count();

        if support > 0 && seen_sequences.insert((*adapter).to_string()) {
            candidates.push(AdapterCandidate {
                name: (*name).to_string(),
                sequence: (*adapter).to_string(),
                support,
                score: support as f64 / records.len() as f64,
            });
        }
    }

    let mut suffix_counts: HashMap<String, usize> = HashMap::new();
    for record in records {
        if record.sequence.len() < 12 {
            continue;
        }

        let tail_phred = record
            .qualities
            .iter()
            .rev()
            .take(12)
            .map(|quality| quality.saturating_sub(33) as f64)
            .sum::<f64>()
            / 12.0;
        if tail_phred > 25.0 {
            continue;
        }

        let suffix = String::from_utf8_lossy(&record.sequence[record.sequence.len() - 12..])
            .to_ascii_uppercase();
        if suffix_complexity_ok(&suffix) {
            *suffix_counts.entry(suffix).or_insert(0) += 1;
        }
    }

    let de_novo_threshold = (records.len() / 25).max(3);
    let mut de_novo: Vec<(String, usize)> = suffix_counts
        .into_iter()
        .filter(|(_, count)| {
            *count >= de_novo_threshold && (*count as f64 / records.len() as f64) < 0.80
        })
        .collect();
    de_novo.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    for (sequence, support) in de_novo.into_iter().take(2) {
        if seen_sequences.insert(sequence.clone()) {
            candidates.push(AdapterCandidate {
                name: "de_novo_suffix".to_string(),
                sequence,
                support,
                score: support as f64 / records.len() as f64,
            });
        }
    }

    candidates.sort_by(|a, b| {
        b.support
            .cmp(&a.support)
            .then_with(|| b.score.total_cmp(&a.score))
            .then_with(|| a.name.cmp(&b.name))
    });
    candidates
}

fn adapter_tail_support(sequence: &[u8], adapter: &[u8]) -> Option<usize> {
    if sequence.len() < 8 || adapter.len() < 8 {
        return None;
    }

    let tail_start = sequence.len().saturating_sub(adapter.len().max(28));
    let tail = &sequence[tail_start..];
    let mut best_overlap = None;

    for start in 0..tail.len() {
        let overlap = (tail.len() - start).min(adapter.len());
        if overlap < 8 {
            continue;
        }

        let matches = tail[start..start + overlap]
            .iter()
            .zip(adapter.iter())
            .filter(|(lhs, rhs)| lhs.to_ascii_uppercase() == rhs.to_ascii_uppercase())
            .count();
        let mismatches = overlap - matches;
        let allowed = ((overlap as f64) * 0.15).ceil() as usize;
        if mismatches <= allowed && matches * 100 >= overlap * 85 {
            best_overlap = Some(best_overlap.unwrap_or(0).max(overlap));
        }
    }

    best_overlap
}

fn suffix_complexity_ok(sequence: &str) -> bool {
    let unique: HashSet<char> = sequence.chars().collect();
    unique.len() >= 3
}

fn has_homopolymer_tail(sequence: &[u8], base: u8) -> bool {
    let tail = &sequence[sequence.len().saturating_sub(20)..];
    let mut run = 0usize;
    for nucleotide in tail {
        if nucleotide.to_ascii_uppercase() == base {
            run += 1;
            if run >= 8 {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

fn tail_drop(cycle_means: &[f64]) -> f64 {
    if cycle_means.len() < 10 {
        return 0.0;
    }
    let head = mean_f64(&cycle_means[..cycle_means.len().min(10)]);
    let tail = mean_f64(&cycle_means[cycle_means.len().saturating_sub(10)..]);
    head - tail
}

fn mean_usize(values: &[usize]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<usize>() as f64 / values.len() as f64
}

fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn median_usize(values: &[usize]) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn stddev_usize(values: &[usize], mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| {
            let diff = *value as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}
