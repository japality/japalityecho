use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

use crate::fastq::{sample_paired_records, sample_records};
use crate::model::{
    BackendPreference, EvaluationAggregateSummary, EvaluationMetricSummary, EvaluationReport,
    EvaluationScenarioSummary, FastqRecord, PairedEvaluationAggregateSummary,
    PairedEvaluationReport, PairedEvaluationScenarioSummary, ReadPair,
};
use crate::process::{ProcessOptions, process_file, process_files};

const EVALUATION_SUITE_NAME: &str = "synthetic_truth_suite_v1";
const PAIRED_EVALUATION_SUITE_NAME: &str = "synthetic_truth_paired_suite_v1";
const CORE_SEQUENCE: &str = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
const MATE2_CORE_SEQUENCE: &str = "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";
const ADAPTER_SEQUENCE: &str = "AGATCGGAAGAGC";
const ERROR_POSITION: usize = 18;
const RIGHT_ERROR_POSITION: usize = 11;
const SHORT_INSERT_LENGTH: usize = 20;

#[derive(Debug, Clone)]
pub struct EvaluationOptions {
    pub sample_size: usize,
    pub batch_reads: usize,
    pub backend_preference: BackendPreference,
    pub min_quality_override: Option<u8>,
    pub reads_per_scenario: usize,
}

pub fn evaluate_synthetic_truth(options: &EvaluationOptions) -> Result<EvaluationReport> {
    let suite = SyntheticTruthSuite::build(options.reads_per_scenario.max(1));
    let temp_dir = EvaluationTempDir::new()?;
    let input_path = temp_dir.path.join("synthetic.fastq");
    let output_path = temp_dir.path.join("synthetic.out.fastq");
    write_synthetic_truth_dataset(&input_path, options.reads_per_scenario.max(1))?;

    let process = process_file(
        &input_path,
        &output_path,
        &ProcessOptions {
            sample_size: options.sample_size,
            batch_reads: options.batch_reads,
            backend_preference: options.backend_preference,
            forced_adapter: None,
            min_quality_override: options.min_quality_override,
            kmer_size_override: None,
                    forced_platform: None,
                    forced_experiment: None,
        },
    )?;
    let output_records = sample_records(&output_path, usize::MAX)?;
    let outputs_by_header: HashMap<_, _> = output_records
        .into_iter()
        .map(|record| (record.header.clone(), record))
        .collect();

    let mut scenarios = Vec::with_capacity(suite.scenarios.len());
    let mut aggregate = AggregateAccumulator::default();
    for scenario in &suite.scenarios {
        let summary = score_scenario(scenario, &outputs_by_header);
        aggregate.absorb(&summary);
        scenarios.push(summary);
    }

    let aggregate = aggregate.into_summary();
    let mut notes = vec![
        "Deterministic synthetic truth suite covers clean retention, repair, trim, repair-plus-trim, and short-fragment discard".to_string(),
        "Strict exact-match rate reports end-to-end sequence correctness after correction and trimming".to_string(),
        "Synthetic truth is manuscript-friendly for regression figures, but should be paired with public biological datasets for external validity".to_string(),
    ];
    notes.push(format!(
        "Requested backend {} executed as '{}'",
        options.backend_preference, process.backend_used
    ));
    notes.push(format!(
        "Evaluation used {} reads across {} scenarios",
        aggregate.total_reads,
        suite.scenarios.len()
    ));

    Ok(EvaluationReport {
        suite_name: EVALUATION_SUITE_NAME.to_string(),
        requested_backend: options.backend_preference,
        reads_per_scenario: suite.reads_per_scenario,
        total_scenarios: suite.scenarios.len(),
        aggregate,
        scenarios,
        process,
        notes,
    })
}

pub fn evaluate_paired_synthetic_truth(
    options: &EvaluationOptions,
) -> Result<PairedEvaluationReport> {
    let suite = PairedSyntheticTruthSuite::build(options.reads_per_scenario.max(1));
    let temp_dir = EvaluationTempDir::new()?;
    let input1 = temp_dir.path.join("synthetic_r1.fastq");
    let input2 = temp_dir.path.join("synthetic_r2.fastq");
    let output1 = temp_dir.path.join("synthetic_r1.out.fastq");
    let output2 = temp_dir.path.join("synthetic_r2.out.fastq");
    write_paired_synthetic_truth_dataset(&input1, &input2, options.reads_per_scenario.max(1))?;

    let process = process_files(
        &input1,
        &output1,
        Some(&input2),
        Some(&output2),
        &ProcessOptions {
            sample_size: options.sample_size,
            batch_reads: options.batch_reads,
            backend_preference: options.backend_preference,
            forced_adapter: None,
            min_quality_override: options.min_quality_override,
            kmer_size_override: None,
                    forced_platform: None,
                    forced_experiment: None,
        },
    )?;
    let output_pairs = sample_paired_records(&output1, &output2, usize::MAX)?;
    let outputs_by_key: HashMap<_, _> = output_pairs
        .into_iter()
        .map(|pair| (pair_header_key(&pair.left.header), pair))
        .collect();

    let mut scenarios = Vec::with_capacity(suite.scenarios.len());
    let mut aggregate = PairedAggregateAccumulator::default();
    for scenario in &suite.scenarios {
        let summary = score_paired_scenario(scenario, &outputs_by_key);
        aggregate.absorb(&summary);
        scenarios.push(summary);
    }

    let aggregate = aggregate.into_summary();
    let mut notes = vec![
        "Deterministic paired-end truth suite exercises synchronized retention, mixed repair/trim cleanup, and pair-level discard behavior".to_string(),
        "Pair exact-match rate requires both mates to match truth after preprocessing".to_string(),
        "Paired synthetic truth directly probes mate synchronization, which is important for manuscript claims beyond single-end cleanup".to_string(),
    ];
    notes.push(format!(
        "Requested backend {} executed as '{}'",
        options.backend_preference, process.backend_used
    ));
    notes.push(format!(
        "Evaluation used {} pairs across {} scenarios",
        aggregate.total_pairs,
        suite.scenarios.len()
    ));

    Ok(PairedEvaluationReport {
        suite_name: PAIRED_EVALUATION_SUITE_NAME.to_string(),
        requested_backend: options.backend_preference,
        pairs_per_scenario: suite.pairs_per_scenario,
        total_scenarios: suite.scenarios.len(),
        aggregate,
        scenarios,
        process,
        notes,
    })
}

pub(crate) fn write_synthetic_truth_dataset(
    path: &Path,
    reads_per_scenario: usize,
) -> Result<usize> {
    let suite = SyntheticTruthSuite::build(reads_per_scenario.max(1));
    let input_records = suite.input_records();
    write_fastq_records(path, &input_records)?;
    Ok(input_records.len())
}

pub(crate) fn write_paired_synthetic_truth_dataset(
    path1: &Path,
    path2: &Path,
    pairs_per_scenario: usize,
) -> Result<usize> {
    let suite = PairedSyntheticTruthSuite::build(pairs_per_scenario.max(1));
    let input_pairs = suite.input_pairs();
    write_fastq_pairs(path1, path2, &input_pairs)?;
    Ok(input_pairs.len())
}

#[derive(Debug, Clone)]
struct SyntheticTruthSuite {
    reads_per_scenario: usize,
    scenarios: Vec<SyntheticScenario>,
}

impl SyntheticTruthSuite {
    fn build(reads_per_scenario: usize) -> Self {
        Self {
            reads_per_scenario,
            scenarios: vec![
                clean_retention_scenario(reads_per_scenario),
                single_error_repair_scenario(reads_per_scenario),
                adapter_trim_scenario(reads_per_scenario),
                repair_then_trim_scenario(reads_per_scenario),
                discard_short_insert_scenario(reads_per_scenario),
            ],
        }
    }

    fn input_records(&self) -> Vec<FastqRecord> {
        self.scenarios
            .iter()
            .flat_map(|scenario| scenario.reads.iter().map(|read| read.input.clone()))
            .collect()
    }
}

#[derive(Debug, Clone)]
struct PairedSyntheticTruthSuite {
    pairs_per_scenario: usize,
    scenarios: Vec<PairedSyntheticScenario>,
}

impl PairedSyntheticTruthSuite {
    fn build(pairs_per_scenario: usize) -> Self {
        Self {
            pairs_per_scenario,
            scenarios: vec![
                paired_clean_retention_scenario(pairs_per_scenario),
                paired_left_repair_right_trim_scenario(pairs_per_scenario),
                paired_left_trim_right_repair_scenario(pairs_per_scenario),
                paired_both_repair_trim_scenario(pairs_per_scenario),
                paired_discard_short_insert_scenario(pairs_per_scenario),
            ],
        }
    }

    fn input_pairs(&self) -> Vec<ReadPair> {
        self.scenarios
            .iter()
            .flat_map(|scenario| {
                scenario
                    .pairs
                    .iter()
                    .map(|pair| ReadPair::new(pair.left_input.clone(), pair.right_input.clone()))
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
struct SyntheticScenario {
    name: &'static str,
    description: &'static str,
    reads: Vec<SyntheticTruthRead>,
}

#[derive(Debug, Clone)]
struct SyntheticTruthRead {
    input: FastqRecord,
    expected_output_sequence: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
struct PairedSyntheticScenario {
    name: &'static str,
    description: &'static str,
    pairs: Vec<SyntheticTruthPair>,
}

#[derive(Debug, Clone)]
struct SyntheticTruthPair {
    left_input: FastqRecord,
    right_input: FastqRecord,
    expected_left_output: Option<Vec<u8>>,
    expected_right_output: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, Default)]
struct MetricCounts {
    true_positive: usize,
    false_positive: usize,
    false_negative: usize,
}

impl MetricCounts {
    fn observe_unit(&mut self, expected_positive: bool, observed_positive: bool) {
        match (expected_positive, observed_positive) {
            (true, true) => self.true_positive += 1,
            (false, true) => self.false_positive += 1,
            (true, false) => self.false_negative += 1,
            (false, false) => {}
        }
    }

    fn into_summary(self) -> EvaluationMetricSummary {
        let precision =
            ratio_or_perfect(self.true_positive, self.true_positive + self.false_positive);
        let recall = ratio_or_perfect(self.true_positive, self.true_positive + self.false_negative);
        let f1 = if (precision + recall) == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        EvaluationMetricSummary {
            true_positive: self.true_positive,
            false_positive: self.false_positive,
            false_negative: self.false_negative,
            precision,
            recall,
            f1,
        }
    }
}

#[derive(Debug, Default)]
struct AggregateAccumulator {
    total_reads: usize,
    expected_retained_reads: usize,
    observed_retained_reads: usize,
    exact_match_reads: usize,
    retention: MetricCounts,
    trimming: MetricCounts,
    correction: MetricCounts,
}

impl AggregateAccumulator {
    fn absorb(&mut self, scenario: &EvaluationScenarioSummary) {
        self.total_reads += scenario.reads;
        self.expected_retained_reads += scenario.expected_retained_reads;
        self.observed_retained_reads += scenario.observed_retained_reads;
        self.exact_match_reads += scenario.exact_match_reads;
        absorb_metric(&mut self.retention, &scenario.retention);
        absorb_metric(&mut self.trimming, &scenario.trimming);
        absorb_metric(&mut self.correction, &scenario.correction);
    }

    fn into_summary(self) -> EvaluationAggregateSummary {
        EvaluationAggregateSummary {
            total_reads: self.total_reads,
            expected_retained_reads: self.expected_retained_reads,
            observed_retained_reads: self.observed_retained_reads,
            exact_match_reads: self.exact_match_reads,
            exact_match_rate: ratio_or_perfect(
                self.exact_match_reads,
                self.expected_retained_reads,
            ),
            retention: self.retention.into_summary(),
            trimming: self.trimming.into_summary(),
            correction: self.correction.into_summary(),
        }
    }
}

#[derive(Debug, Default)]
struct PairedAggregateAccumulator {
    total_pairs: usize,
    expected_retained_pairs: usize,
    observed_retained_pairs: usize,
    exact_match_pairs: usize,
    retention: MetricCounts,
    trimming: MetricCounts,
    correction: MetricCounts,
}

impl PairedAggregateAccumulator {
    fn absorb(&mut self, scenario: &PairedEvaluationScenarioSummary) {
        self.total_pairs += scenario.pairs;
        self.expected_retained_pairs += scenario.expected_retained_pairs;
        self.observed_retained_pairs += scenario.observed_retained_pairs;
        self.exact_match_pairs += scenario.exact_match_pairs;
        absorb_metric(&mut self.retention, &scenario.retention);
        absorb_metric(&mut self.trimming, &scenario.trimming);
        absorb_metric(&mut self.correction, &scenario.correction);
    }

    fn into_summary(self) -> PairedEvaluationAggregateSummary {
        PairedEvaluationAggregateSummary {
            total_pairs: self.total_pairs,
            expected_retained_pairs: self.expected_retained_pairs,
            observed_retained_pairs: self.observed_retained_pairs,
            exact_match_pairs: self.exact_match_pairs,
            exact_match_rate: ratio_or_perfect(
                self.exact_match_pairs,
                self.expected_retained_pairs,
            ),
            retention: self.retention.into_summary(),
            trimming: self.trimming.into_summary(),
            correction: self.correction.into_summary(),
        }
    }
}

fn absorb_metric(target: &mut MetricCounts, summary: &EvaluationMetricSummary) {
    target.true_positive += summary.true_positive;
    target.false_positive += summary.false_positive;
    target.false_negative += summary.false_negative;
}

fn score_scenario(
    scenario: &SyntheticScenario,
    outputs_by_header: &HashMap<String, FastqRecord>,
) -> EvaluationScenarioSummary {
    let mut expected_retained_reads = 0usize;
    let mut observed_retained_reads = 0usize;
    let mut exact_match_reads = 0usize;
    let mut retention = MetricCounts::default();
    let mut trimming = MetricCounts::default();
    let mut correction = MetricCounts::default();

    for read in &scenario.reads {
        let output = outputs_by_header.get(&read.input.header);
        let expected_retained = read.expected_output_sequence.is_some();
        let observed_retained = output.is_some();
        let expected_trim = read
            .expected_output_sequence
            .as_ref()
            .is_some_and(|truth| truth.len() < read.input.sequence.len());
        let observed_trim =
            output.is_some_and(|record| record.sequence.len() < read.input.sequence.len());

        if expected_retained {
            expected_retained_reads += 1;
        }
        if observed_retained {
            observed_retained_reads += 1;
        }

        retention.observe_unit(expected_retained, observed_retained);
        trimming.observe_unit(expected_trim, observed_trim);

        if let (Some(truth), Some(output)) = (&read.expected_output_sequence, output) {
            if output.sequence == *truth {
                exact_match_reads += 1;
            }
            score_correction_bases(
                &read.input.sequence,
                truth,
                &output.sequence,
                &mut correction,
            );
        }
    }

    EvaluationScenarioSummary {
        name: scenario.name.to_string(),
        description: scenario.description.to_string(),
        reads: scenario.reads.len(),
        expected_retained_reads,
        observed_retained_reads,
        exact_match_reads,
        exact_match_rate: ratio_or_perfect(exact_match_reads, expected_retained_reads),
        retention: retention.into_summary(),
        trimming: trimming.into_summary(),
        correction: correction.into_summary(),
        notes: vec![scenario.description.to_string()],
    }
}

fn score_paired_scenario(
    scenario: &PairedSyntheticScenario,
    outputs_by_key: &HashMap<String, ReadPair>,
) -> PairedEvaluationScenarioSummary {
    let mut expected_retained_pairs = 0usize;
    let mut observed_retained_pairs = 0usize;
    let mut exact_match_pairs = 0usize;
    let mut retention = MetricCounts::default();
    let mut trimming = MetricCounts::default();
    let mut correction = MetricCounts::default();

    for pair in &scenario.pairs {
        let output = outputs_by_key.get(&pair_header_key(&pair.left_input.header));
        let expected_retained =
            pair.expected_left_output.is_some() && pair.expected_right_output.is_some();
        let observed_retained = output.is_some();
        let expected_trim = pair
            .expected_left_output
            .as_ref()
            .is_some_and(|truth| truth.len() < pair.left_input.sequence.len())
            || pair
                .expected_right_output
                .as_ref()
                .is_some_and(|truth| truth.len() < pair.right_input.sequence.len());
        let observed_trim = output.is_some_and(|pair_output| {
            pair_output.left.sequence.len() < pair.left_input.sequence.len()
                || pair_output.right.sequence.len() < pair.right_input.sequence.len()
        });

        if expected_retained {
            expected_retained_pairs += 1;
        }
        if observed_retained {
            observed_retained_pairs += 1;
        }

        retention.observe_unit(expected_retained, observed_retained);
        trimming.observe_unit(expected_trim, observed_trim);

        if let (Some(expected_left), Some(expected_right), Some(output_pair)) = (
            &pair.expected_left_output,
            &pair.expected_right_output,
            output,
        ) {
            if output_pair.left.sequence == *expected_left
                && output_pair.right.sequence == *expected_right
            {
                exact_match_pairs += 1;
            }
            score_correction_bases(
                &pair.left_input.sequence,
                expected_left,
                &output_pair.left.sequence,
                &mut correction,
            );
            score_correction_bases(
                &pair.right_input.sequence,
                expected_right,
                &output_pair.right.sequence,
                &mut correction,
            );
        }
    }

    PairedEvaluationScenarioSummary {
        name: scenario.name.to_string(),
        description: scenario.description.to_string(),
        pairs: scenario.pairs.len(),
        expected_retained_pairs,
        observed_retained_pairs,
        exact_match_pairs,
        exact_match_rate: ratio_or_perfect(exact_match_pairs, expected_retained_pairs),
        retention: retention.into_summary(),
        trimming: trimming.into_summary(),
        correction: correction.into_summary(),
        notes: vec![scenario.description.to_string()],
    }
}

fn score_correction_bases(
    observed_sequence: &[u8],
    truth_sequence: &[u8],
    output_sequence: &[u8],
    counts: &mut MetricCounts,
) {
    let compare_len = observed_sequence
        .len()
        .min(truth_sequence.len())
        .min(output_sequence.len());
    for index in 0..compare_len {
        let observed = observed_sequence[index];
        let truth = truth_sequence[index];
        let output = output_sequence[index];
        let needs_correction = observed != truth;

        if needs_correction {
            if output == truth {
                counts.true_positive += 1;
            } else {
                counts.false_negative += 1;
                if output != observed {
                    counts.false_positive += 1;
                }
            }
        } else if output != truth {
            counts.false_positive += 1;
        }
    }
}

fn clean_retention_scenario(reads_per_scenario: usize) -> SyntheticScenario {
    SyntheticScenario {
        name: "clean_retention",
        description: "High-quality genomic reads should pass unchanged",
        reads: (0..reads_per_scenario)
            .map(|index| SyntheticTruthRead {
                input: FastqRecord::new(
                    format!("clean-{index}"),
                    CORE_SEQUENCE.as_bytes().to_vec(),
                    high_quality(CORE_SEQUENCE.len()),
                ),
                expected_output_sequence: Some(CORE_SEQUENCE.as_bytes().to_vec()),
            })
            .collect(),
    }
}

fn single_error_repair_scenario(reads_per_scenario: usize) -> SyntheticScenario {
    let observed = mutate_base(CORE_SEQUENCE, ERROR_POSITION);
    SyntheticScenario {
        name: "single_error_repair",
        description: "A low-quality substitution inside the genomic body should be repaired",
        reads: (0..reads_per_scenario)
            .map(|index| SyntheticTruthRead {
                input: FastqRecord::new(
                    format!("repair-{index}"),
                    observed.clone(),
                    quality_with_low_position(CORE_SEQUENCE.len(), ERROR_POSITION),
                ),
                expected_output_sequence: Some(CORE_SEQUENCE.as_bytes().to_vec()),
            })
            .collect(),
    }
}

fn adapter_trim_scenario(reads_per_scenario: usize) -> SyntheticScenario {
    let observed = format!("{CORE_SEQUENCE}{ADAPTER_SEQUENCE}").into_bytes();
    SyntheticScenario {
        name: "adapter_trim",
        description: "Adapter-tailed reads should be trimmed back to the genomic insert",
        reads: (0..reads_per_scenario)
            .map(|index| SyntheticTruthRead {
                input: FastqRecord::new(
                    format!("trim-{index}"),
                    observed.clone(),
                    quality_with_low_adapter(CORE_SEQUENCE.len(), None),
                ),
                expected_output_sequence: Some(CORE_SEQUENCE.as_bytes().to_vec()),
            })
            .collect(),
    }
}

fn repair_then_trim_scenario(reads_per_scenario: usize) -> SyntheticScenario {
    let repaired_prefix = mutate_base(CORE_SEQUENCE, ERROR_POSITION);
    let observed = format!(
        "{}{}",
        String::from_utf8_lossy(&repaired_prefix),
        ADAPTER_SEQUENCE
    )
    .into_bytes();
    SyntheticScenario {
        name: "repair_then_trim",
        description: "A low-quality substitution plus adapter tail should be repaired and then trimmed",
        reads: (0..reads_per_scenario)
            .map(|index| SyntheticTruthRead {
                input: FastqRecord::new(
                    format!("repair-trim-{index}"),
                    observed.clone(),
                    quality_with_low_adapter(CORE_SEQUENCE.len(), Some(ERROR_POSITION)),
                ),
                expected_output_sequence: Some(CORE_SEQUENCE.as_bytes().to_vec()),
            })
            .collect(),
    }
}

fn discard_short_insert_scenario(reads_per_scenario: usize) -> SyntheticScenario {
    let short_insert = &CORE_SEQUENCE[..SHORT_INSERT_LENGTH];
    let observed = format!("{short_insert}{ADAPTER_SEQUENCE}").into_bytes();
    SyntheticScenario {
        name: "discard_short_insert",
        description: "Reads that become shorter than the minimum output length after trimming should be discarded",
        reads: (0..reads_per_scenario)
            .map(|index| SyntheticTruthRead {
                input: FastqRecord::new(
                    format!("discard-{index}"),
                    observed.clone(),
                    quality_with_low_adapter(short_insert.len(), None),
                ),
                expected_output_sequence: None,
            })
            .collect(),
    }
}

fn paired_clean_retention_scenario(pairs_per_scenario: usize) -> PairedSyntheticScenario {
    PairedSyntheticScenario {
        name: "paired_clean_retention",
        description: "High-quality mate pairs should pass unchanged and stay synchronized",
        pairs: (0..pairs_per_scenario)
            .map(|index| SyntheticTruthPair {
                left_input: FastqRecord::new(
                    format!("paired-clean-{index}/1"),
                    CORE_SEQUENCE.as_bytes().to_vec(),
                    high_quality(CORE_SEQUENCE.len()),
                ),
                right_input: FastqRecord::new(
                    format!("paired-clean-{index}/2"),
                    MATE2_CORE_SEQUENCE.as_bytes().to_vec(),
                    high_quality(MATE2_CORE_SEQUENCE.len()),
                ),
                expected_left_output: Some(CORE_SEQUENCE.as_bytes().to_vec()),
                expected_right_output: Some(MATE2_CORE_SEQUENCE.as_bytes().to_vec()),
            })
            .collect(),
    }
}

fn paired_left_repair_right_trim_scenario(pairs_per_scenario: usize) -> PairedSyntheticScenario {
    let left_observed = mutate_base(CORE_SEQUENCE, ERROR_POSITION);
    let right_observed = format!("{MATE2_CORE_SEQUENCE}{ADAPTER_SEQUENCE}").into_bytes();
    PairedSyntheticScenario {
        name: "paired_left_repair_right_trim",
        description: "One mate should be repaired while the other mate trims an adapter tail",
        pairs: (0..pairs_per_scenario)
            .map(|index| SyntheticTruthPair {
                left_input: FastqRecord::new(
                    format!("paired-lr-rt-{index}/1"),
                    left_observed.clone(),
                    quality_with_low_position(CORE_SEQUENCE.len(), ERROR_POSITION),
                ),
                right_input: FastqRecord::new(
                    format!("paired-lr-rt-{index}/2"),
                    right_observed.clone(),
                    quality_with_low_adapter(MATE2_CORE_SEQUENCE.len(), None),
                ),
                expected_left_output: Some(CORE_SEQUENCE.as_bytes().to_vec()),
                expected_right_output: Some(MATE2_CORE_SEQUENCE.as_bytes().to_vec()),
            })
            .collect(),
    }
}

fn paired_left_trim_right_repair_scenario(pairs_per_scenario: usize) -> PairedSyntheticScenario {
    let left_observed = format!("{CORE_SEQUENCE}{ADAPTER_SEQUENCE}").into_bytes();
    let right_observed = mutate_base(MATE2_CORE_SEQUENCE, RIGHT_ERROR_POSITION);
    PairedSyntheticScenario {
        name: "paired_left_trim_right_repair",
        description: "The opposite mate configuration should also preserve synchronized cleanup",
        pairs: (0..pairs_per_scenario)
            .map(|index| SyntheticTruthPair {
                left_input: FastqRecord::new(
                    format!("paired-lt-rr-{index}/1"),
                    left_observed.clone(),
                    quality_with_low_adapter(CORE_SEQUENCE.len(), None),
                ),
                right_input: FastqRecord::new(
                    format!("paired-lt-rr-{index}/2"),
                    right_observed.clone(),
                    quality_with_low_position(MATE2_CORE_SEQUENCE.len(), RIGHT_ERROR_POSITION),
                ),
                expected_left_output: Some(CORE_SEQUENCE.as_bytes().to_vec()),
                expected_right_output: Some(MATE2_CORE_SEQUENCE.as_bytes().to_vec()),
            })
            .collect(),
    }
}

fn paired_both_repair_trim_scenario(pairs_per_scenario: usize) -> PairedSyntheticScenario {
    let left_observed = format!(
        "{}{}",
        String::from_utf8_lossy(&mutate_base(CORE_SEQUENCE, ERROR_POSITION)),
        ADAPTER_SEQUENCE
    )
    .into_bytes();
    let right_observed = format!(
        "{}{}",
        String::from_utf8_lossy(&mutate_base(MATE2_CORE_SEQUENCE, RIGHT_ERROR_POSITION)),
        ADAPTER_SEQUENCE
    )
    .into_bytes();
    PairedSyntheticScenario {
        name: "paired_both_repair_trim",
        description: "Both mates should support simultaneous repair and adapter trimming",
        pairs: (0..pairs_per_scenario)
            .map(|index| SyntheticTruthPair {
                left_input: FastqRecord::new(
                    format!("paired-both-{index}/1"),
                    left_observed.clone(),
                    quality_with_low_adapter(CORE_SEQUENCE.len(), Some(ERROR_POSITION)),
                ),
                right_input: FastqRecord::new(
                    format!("paired-both-{index}/2"),
                    right_observed.clone(),
                    quality_with_low_adapter(MATE2_CORE_SEQUENCE.len(), Some(RIGHT_ERROR_POSITION)),
                ),
                expected_left_output: Some(CORE_SEQUENCE.as_bytes().to_vec()),
                expected_right_output: Some(MATE2_CORE_SEQUENCE.as_bytes().to_vec()),
            })
            .collect(),
    }
}

fn paired_discard_short_insert_scenario(pairs_per_scenario: usize) -> PairedSyntheticScenario {
    let right_short_insert = &MATE2_CORE_SEQUENCE[..SHORT_INSERT_LENGTH];
    let right_observed = format!("{right_short_insert}{ADAPTER_SEQUENCE}").into_bytes();
    PairedSyntheticScenario {
        name: "paired_discard_short_insert",
        description: "If one mate becomes too short after trimming, the pair should be discarded together",
        pairs: (0..pairs_per_scenario)
            .map(|index| SyntheticTruthPair {
                left_input: FastqRecord::new(
                    format!("paired-discard-{index}/1"),
                    CORE_SEQUENCE.as_bytes().to_vec(),
                    high_quality(CORE_SEQUENCE.len()),
                ),
                right_input: FastqRecord::new(
                    format!("paired-discard-{index}/2"),
                    right_observed.clone(),
                    quality_with_low_adapter(right_short_insert.len(), None),
                ),
                expected_left_output: None,
                expected_right_output: None,
            })
            .collect(),
    }
}

fn high_quality(len: usize) -> Vec<u8> {
    vec![b'I'; len]
}

fn quality_with_low_position(len: usize, low_position: usize) -> Vec<u8> {
    let mut qualities = high_quality(len);
    if low_position < qualities.len() {
        qualities[low_position] = b'!';
    }
    qualities
}

fn quality_with_low_adapter(prefix_len: usize, low_position: Option<usize>) -> Vec<u8> {
    let mut qualities = high_quality(prefix_len + ADAPTER_SEQUENCE.len());
    if let Some(low_position) = low_position.filter(|position| *position < prefix_len) {
        qualities[low_position] = b'!';
    }
    for quality in &mut qualities[prefix_len..] {
        *quality = b'!';
    }
    qualities
}

fn mutate_base(sequence: &str, position: usize) -> Vec<u8> {
    let mut sequence = sequence.as_bytes().to_vec();
    if position < sequence.len() {
        sequence[position] = match sequence[position] {
            b'A' => b'C',
            b'C' => b'G',
            b'G' => b'T',
            _ => b'A',
        };
    }
    sequence
}

fn pair_header_key(header: &str) -> String {
    let token = header.split_whitespace().next().unwrap_or(header);
    token
        .strip_suffix("/1")
        .or_else(|| token.strip_suffix("/2"))
        .unwrap_or(token)
        .to_string()
}

fn ratio_or_perfect(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        1.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn write_fastq_records(path: &Path, records: &[FastqRecord]) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for record in records {
        writer.write_all(b"@")?;
        writer.write_all(record.header.as_bytes())?;
        writer.write_all(b"\n")?;
        writer.write_all(&record.sequence)?;
        writer.write_all(b"\n+\n")?;
        writer.write_all(&record.qualities)?;
        writer.write_all(b"\n")?;
    }
    writer
        .flush()
        .with_context(|| format!("failed to flush {}", path.display()))?;
    Ok(())
}

fn write_fastq_pairs(path1: &Path, path2: &Path, pairs: &[ReadPair]) -> Result<()> {
    let file1 =
        File::create(path1).with_context(|| format!("failed to create {}", path1.display()))?;
    let file2 =
        File::create(path2).with_context(|| format!("failed to create {}", path2.display()))?;
    let mut writer1 = BufWriter::new(file1);
    let mut writer2 = BufWriter::new(file2);
    for pair in pairs {
        write_fastq_record(&mut writer1, &pair.left)?;
        write_fastq_record(&mut writer2, &pair.right)?;
    }
    writer1
        .flush()
        .with_context(|| format!("failed to flush {}", path1.display()))?;
    writer2
        .flush()
        .with_context(|| format!("failed to flush {}", path2.display()))?;
    Ok(())
}

fn write_fastq_record<W: Write>(writer: &mut W, record: &FastqRecord) -> Result<()> {
    writer.write_all(b"@")?;
    writer.write_all(record.header.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.write_all(&record.sequence)?;
    writer.write_all(b"\n+\n")?;
    writer.write_all(&record.qualities)?;
    writer.write_all(b"\n")?;
    Ok(())
}

#[derive(Debug)]
struct EvaluationTempDir {
    path: PathBuf,
}

impl EvaluationTempDir {
    fn new() -> Result<Self> {
        let unique = format!(
            "japalityecho-eval-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .context("system clock is before unix epoch")?
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        fs::create_dir_all(&path)
            .with_context(|| format!("failed to create {}", path.display()))?;
        Ok(Self { path })
    }
}

impl Drop for EvaluationTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
