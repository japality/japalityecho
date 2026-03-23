use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::backend::{BackendResolution, HeterogeneousScheduler};
use crate::fastq::{
    discover_mate_file, read_batches, read_batches_with_read_ahead, read_paired_batches,
    read_paired_batches_with_read_ahead, sample_paired_records,
    sample_records, write_processed_record_pairs, write_processed_records,
};
use crate::gpu::prepare_acceleration_preview;
use crate::model::{
    AdapterCandidate, AutoProfile, BackendPreference, BenchmarkComparisonReport,
    BenchmarkComparisonSummary, BenchmarkReport, BenchmarkRound, BenchmarkSessionMode,
    BenchmarkSummary, ExecutionPlan, ExperimentType, FastqRecord, InspectReport, Platform,
    ProcessReport, ReadBatch, ReadPair, ThroughputSummary,
};
use crate::profile::{build_execution_plan, infer_auto_profile};
use crate::dbg::DeBruijnGraph;
use crate::spectrum::{KmerSpectrum, KmerSpectrumBuilder};

#[derive(Debug, Clone)]
pub struct ProcessOptions {
    pub sample_size: usize,
    pub batch_reads: usize,
    pub backend_preference: BackendPreference,
    pub forced_adapter: Option<String>,
    pub min_quality_override: Option<u8>,
    pub kmer_size_override: Option<usize>,
    pub forced_platform: Option<Platform>,
    pub forced_experiment: Option<ExperimentType>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkOptions {
    pub process: ProcessOptions,
    pub rounds: usize,
    pub session_mode: BenchmarkSessionMode,
}

#[derive(Debug, Clone)]
pub struct BenchmarkComparisonOptions {
    pub process: ProcessOptions,
    pub rounds: usize,
}

pub fn inspect_file(
    input: &Path,
    sample_size: usize,
    backend_preference: BackendPreference,
) -> Result<(AutoProfile, ExecutionPlan)> {
    let report = inspect_inputs(input, None, sample_size, backend_preference)?;
    Ok((report.auto_profile, report.execution_plan))
}

pub fn inspect_inputs(
    input1: &Path,
    input2: Option<&Path>,
    sample_size: usize,
    backend_preference: BackendPreference,
) -> Result<InspectReport> {
    inspect_inputs_with_overrides(input1, input2, sample_size, backend_preference, None, None)
}

pub fn inspect_inputs_with_overrides(
    input1: &Path,
    input2: Option<&Path>,
    sample_size: usize,
    backend_preference: BackendPreference,
    forced_platform: Option<Platform>,
    forced_experiment: Option<ExperimentType>,
) -> Result<InspectReport> {
    // Check for auto-discovered mate to report in notes
    let discovered_mate = if input2.is_none() {
        discover_mate_file(input1)
    } else {
        None
    };

    let context = prepare_run_context(
        input1,
        input2,
        sample_size,
        backend_preference,
        forced_platform,
        forced_experiment,
    )?;
    let mut notes = context.resolution.notes.clone();
    if let Some(ref input2_path) = input2 {
        notes.push(format!(
            "Paired-end inspection uses synchronized mates {} and {}",
            input1.display(),
            input2_path.display()
        ));
    } else if let Some(ref mate_path) = discovered_mate {
        notes.push(format!(
            "Auto-discovered paired mate {} for improved detection",
            mate_path.display()
        ));
    } else {
        notes.push(format!("Single-end inspection uses {}", input1.display()));
    }

    let paired = input2.is_some() || discovered_mate.is_some();
    Ok(InspectReport {
        paired_end: paired,
        auto_profile: context.auto_profile,
        execution_plan: context.execution_plan,
        accelerator_scaffold: context.resolution.accelerator_scaffold,
        accelerator_runtime: context.resolution.accelerator_runtime,
        notes,
    })
}

pub fn process_file(
    input: &Path,
    output: &Path,
    options: &ProcessOptions,
) -> Result<ProcessReport> {
    process_files(input, output, None, None, options)
}

pub fn benchmark_file(input: &Path, options: &BenchmarkOptions) -> Result<BenchmarkReport> {
    benchmark_files(input, None, options)
}

pub fn benchmark_compare_file(
    input: &Path,
    options: &BenchmarkComparisonOptions,
) -> Result<BenchmarkComparisonReport> {
    benchmark_compare_files(input, None, options)
}

pub fn benchmark_files(
    input1: &Path,
    input2: Option<&Path>,
    options: &BenchmarkOptions,
) -> Result<BenchmarkReport> {
    let paired_end = input2.is_some();
    let rounds = options.rounds.max(1);
    let output_plan = BenchmarkOutputPlan::prepare()?;
    let session_mode = options.session_mode;
    let mut benchmark_rounds = Vec::with_capacity(rounds);
    let mut setup_us = 0u64;
    let mut notes = vec![
        format!(
            "Benchmark mode '{}' runs {} rounds",
            session_mode, rounds
        ),
        "Benchmark timing reflects end-to-end preprocessing rather than an isolated kernel microbenchmark"
            .to_string(),
        output_plan.describe_sink(),
    ];

    match session_mode {
        BenchmarkSessionMode::ColdStart => {
            notes.push(
                "Each round prepares a fresh run context and backend, so timings include session cold-start cost"
                    .to_string(),
            );
            for round_index in 0..rounds {
                let (output1, output2) = output_plan.round_outputs(round_index, paired_end);
                let process_result = process_files(
                    input1,
                    &output1,
                    input2,
                    output2.as_deref(),
                    &options.process,
                );
                output_plan.cleanup_round_outputs(&output1, output2.as_deref());
                let process = process_result?;
                benchmark_rounds.push(BenchmarkRound {
                    round_index: round_index + 1,
                    process,
                });
            }
        }
        BenchmarkSessionMode::ReuseSession => {
            let setup_started = Instant::now();
            let prepared = prepare_process_session(input1, input2, &options.process)?;
            setup_us = duration_as_us(setup_started.elapsed());
            notes.push(format!(
                "Prepared one reusable backend/session in {}us before timed rounds",
                setup_us
            ));
            notes.push(
                "Subsequent rounds reuse the same backend instance so device-resident cache and scratch reuse remain active"
                    .to_string(),
            );
            for round_index in 0..rounds {
                let (output1, output2) = output_plan.round_outputs(round_index, paired_end);
                let process_result = process_with_prepared_session(
                    input1,
                    &output1,
                    input2,
                    output2.as_deref(),
                    &options.process,
                    &prepared,
                );
                output_plan.cleanup_round_outputs(&output1, output2.as_deref());
                let process = process_result?;
                benchmark_rounds.push(BenchmarkRound {
                    round_index: round_index + 1,
                    process,
                });
            }
        }
    }

    let summary = summarize_benchmark(&benchmark_rounds, session_mode, setup_us);
    if !summary.backend_consistent {
        notes.push(format!(
            "Backend varied across rounds: {}",
            summary.backends.join(", ")
        ));
    }
    if let Some(steady_state_wall_clock_us) = summary.steady_state_average_wall_clock_us {
        notes.push(format!(
            "Warm-up round {:.3}s vs steady-state average {:.3}s",
            summary.warmup_wall_clock_us as f64 / 1_000_000.0,
            steady_state_wall_clock_us / 1_000_000.0
        ));
    }
    if let Some(warmup_penalty_pct) = summary.warmup_penalty_pct {
        notes.push(format!(
            "Warm-up penalty relative to steady-state average: {warmup_penalty_pct:.1}%"
        ));
    }
    if let Some(steady_state_transfer) = summary.steady_state_average_transfer_bytes {
        notes.push(format!(
            "Warm-up transfer {}B vs steady-state average {:.1}B",
            summary.warmup_transfer_bytes, steady_state_transfer
        ));
    }

    Ok(BenchmarkReport {
        paired_end,
        summary,
        rounds: benchmark_rounds,
        notes,
    })
}

pub fn benchmark_compare_files(
    input1: &Path,
    input2: Option<&Path>,
    options: &BenchmarkComparisonOptions,
) -> Result<BenchmarkComparisonReport> {
    let paired_end = input2.is_some();
    let cold_start = benchmark_files(
        input1,
        input2,
        &BenchmarkOptions {
            process: options.process.clone(),
            rounds: options.rounds,
            session_mode: BenchmarkSessionMode::ColdStart,
        },
    )?;
    let reuse_session = benchmark_files(
        input1,
        input2,
        &BenchmarkOptions {
            process: options.process.clone(),
            rounds: options.rounds,
            session_mode: BenchmarkSessionMode::ReuseSession,
        },
    )?;
    let summary = summarize_benchmark_comparison(&cold_start, &reuse_session);
    let mut notes = vec![
        format!(
            "Compared benchmark session modes over {} rounds per mode",
            summary.rounds
        ),
        "Comparison runs cold_start first and then reuse_session; keep filesystem conditions consistent when tracking regressions"
            .to_string(),
    ];
    if let Some(raw_speedup) = summary.raw_speedup {
        notes.push(format!(
            "Reuse-session average wall-clock speedup over cold-start: {raw_speedup:.3}x"
        ));
    }
    if let Some(amortized_speedup) = summary.amortized_speedup {
        notes.push(format!(
            "Reuse-session amortized speedup after including one-time setup: {amortized_speedup:.3}x"
        ));
    }
    if let Some(transfer_savings_pct) = summary.steady_state_transfer_savings_pct {
        notes.push(format!(
            "Reuse-session steady-state transfer reduction vs cold-start: {transfer_savings_pct:.1}%"
        ));
    }
    if !summary.backend_match {
        notes.push(format!(
            "Compared modes resolved to different backends: cold_start={} reuse_session={}",
            summary.cold_start_backends.join(", "),
            summary.reuse_session_backends.join(", ")
        ));
    }

    Ok(BenchmarkComparisonReport {
        paired_end,
        cold_start,
        reuse_session,
        summary,
        notes,
    })
}

pub fn process_files(
    input1: &Path,
    output1: &Path,
    input2: Option<&Path>,
    output2: Option<&Path>,
    options: &ProcessOptions,
) -> Result<ProcessReport> {
    let prepared = prepare_process_session(input1, input2, options)?;
    process_with_prepared_session(input1, output1, input2, output2, options, &prepared)
}

struct PreparedProcessSession {
    paired_end: bool,
    context: RunContext,
    spectrum: KmerSpectrum,
    dbg: DeBruijnGraph,
    resolution: BackendResolution,
    backend: Box<dyn crate::backend::ExecutionBackend>,
    backend_notes: Vec<String>,
}

fn prepare_process_session(
    input1: &Path,
    input2: Option<&Path>,
    options: &ProcessOptions,
) -> Result<PreparedProcessSession> {
    let paired_end = input2.is_some();
    let mut context = prepare_run_context(
        input1,
        input2,
        options.sample_size,
        options.backend_preference,
        options.forced_platform,
        options.forced_experiment,
    )?;

    if let Some(adapter) = &options.forced_adapter {
        context.execution_plan.adapter_candidates.insert(
            0,
            AdapterCandidate {
                name: "user_forced".to_string(),
                sequence: adapter.to_ascii_uppercase(),
                support: context.sample_records.len(),
                score: 1.0,
            },
        );
    }

    if let Some(min_quality) = options.min_quality_override {
        context.execution_plan.trim_min_quality = min_quality;
        context.execution_plan.notes.push(format!(
            "Overrode planner quality threshold with user-provided Q{}",
            min_quality
        ));
    }

    if let Some(k) = options.kmer_size_override {
        context.execution_plan.kmer_size = k;
        context.execution_plan.notes.push(format!(
            "Overrode planner k-mer size with user-provided k={}",
            k
        ));
    }

    // Build k-mer spectrum from ALL reads (two-pass architecture):
    // Pass 1 (already done above): sampled 100K reads for platform/adapter detection
    // Pass 2 (here): stream ALL reads to build complete k-mer spectrum
    let kmer_k = context.execution_plan.kmer_size;
    let min_bq = context.execution_plan.trim_min_quality.saturating_sub(2);
    #[allow(unused_assignments)]
    let mut full_scan_total_reads = 0usize;
    let mut spectrum = {
        let mut builder = KmerSpectrumBuilder::new(kmer_k, min_bq);
        if let Some(input2) = input2 {
            read_paired_batches(input1, input2, 50_000, |batch| {
                for pair in &batch.pairs {
                    builder.add_record(&pair.left);
                    builder.add_record(&pair.right);
                }
                Ok(())
            })?;
        } else {
            read_batches(input1, 50_000, |batch| {
                for record in &batch.records {
                    builder.add_record(record);
                }
                Ok(())
            })?;
        }
        full_scan_total_reads = builder.records_processed();
        let unique_kmers = builder.unique_kmers();
        context.execution_plan.notes.push(format!(
            "Full-spectrum: built from {} reads ({} unique k-mers) vs {} sampled",
            full_scan_total_reads, unique_kmers, context.sample_records.len()
        ));
        builder.finalize()
    };
    // Auto-calibrate trusted_kmer_min_count from the full spectrum
    let old_floor = context.execution_plan.trusted_kmer_min_count;
    let sample_size = context.sample_records.len();
    let auto_floor = spectrum.auto_trusted_floor(old_floor, sample_size, full_scan_total_reads);
    if auto_floor != old_floor {
        context.execution_plan.trusted_kmer_min_count = auto_floor;
        context.execution_plan.notes.push(format!(
            "Auto-calibrated trusted floor: {} -> {} (full spectrum from {} reads vs {} sampled)",
            old_floor, auto_floor, full_scan_total_reads, sample_size
        ));
    }
    let dbg = DeBruijnGraph::from_spectrum(
        &spectrum,
        context.execution_plan.trusted_kmer_min_count,
    );
    context.execution_plan.notes.push(format!(
        "DBG: {} nodes, {} edges (second-pass multi-base correction)",
        dbg.node_count(),
        dbg.edge_count(),
    ));
    let scheduler = HeterogeneousScheduler::new(options.backend_preference);
    let resolution = scheduler.resolve(&context.execution_plan);
    let backend_instance = scheduler.instantiate(&resolution, &context.execution_plan, &spectrum);
    // Compress spectrum via Bloom filter: keep exact counts only for
    // trusted k-mers, approximate counts for the long tail.
    let (bloom_bytes, evicted) =
        spectrum.compress_to_bloom(context.execution_plan.trusted_kmer_min_count);
    context.execution_plan.notes.push(format!(
        "Bloom compression: {:.1} MB filter, {} trusted (exact) + {} evicted (approximate)",
        bloom_bytes as f64 / (1024.0 * 1024.0),
        spectrum.unique_kmers(),
        evicted,
    ));
    Ok(PreparedProcessSession {
        paired_end,
        context,
        spectrum,
        dbg,
        resolution,
        backend: backend_instance.backend,
        backend_notes: backend_instance.notes,
    })
}

fn process_with_prepared_session(
    input1: &Path,
    output1: &Path,
    input2: Option<&Path>,
    output2: Option<&Path>,
    options: &ProcessOptions,
    prepared: &PreparedProcessSession,
) -> Result<ProcessReport> {
    match (input2, output2) {
        (Some(_), None) => bail!("paired-end processing requires --output2"),
        (None, Some(_)) => bail!("--output2 requires --input2"),
        _ => {}
    }

    if let Some(input2) = input2 {
        let output2 = output2.expect("paired output path validated");
        process_paired(
            input1,
            output1,
            input2,
            output2,
            options,
            &prepared.context,
            &prepared.spectrum,
            &prepared.dbg,
            prepared.backend.as_ref(),
            &prepared.resolution,
            &prepared.backend_notes,
        )
    } else {
        process_single(
            input1,
            output1,
            options,
            prepared.paired_end,
            &prepared.context,
            &prepared.spectrum,
            &prepared.dbg,
            prepared.backend.as_ref(),
            &prepared.resolution,
            &prepared.backend_notes,
        )
    }
}

#[derive(Debug, Clone)]
struct RunContext {
    sample_records: Vec<FastqRecord>,
    auto_profile: AutoProfile,
    execution_plan: ExecutionPlan,
    resolution: BackendResolution,
}

#[derive(Debug, Default)]
struct ThroughputAccumulator {
    input_bases: usize,
    output_bases: usize,
    cumulative_submit_us: u64,
    cumulative_wait_us: u64,
    cumulative_end_to_end_us: u64,
    cumulative_overlap_us: u64,
    max_batch_end_to_end_us: u64,
}

impl ThroughputAccumulator {
    fn observe(
        &mut self,
        input_bases: usize,
        output_bases: usize,
        timing: &crate::backend::BatchTiming,
    ) {
        self.input_bases += input_bases;
        self.output_bases += output_bases;
        self.cumulative_submit_us += timing.submit_us;
        self.cumulative_wait_us += timing.wait_us;
        self.cumulative_end_to_end_us += timing.end_to_end_us;
        self.cumulative_overlap_us += timing.overlap_us;
        self.max_batch_end_to_end_us = self.max_batch_end_to_end_us.max(timing.end_to_end_us);
    }

    fn finalize(
        &self,
        wall_clock: Duration,
        input_reads: usize,
        output_reads: usize,
        batches_processed: usize,
    ) -> ThroughputSummary {
        let wall_clock_us = duration_as_us(wall_clock);
        let wall_clock_secs = wall_clock.as_secs_f64();
        let safe_rate = |value: usize| {
            if wall_clock_secs > 0.0 {
                value as f64 / wall_clock_secs
            } else {
                0.0
            }
        };
        let safe_avg = |value: u64| {
            if batches_processed > 0 {
                value as f64 / batches_processed as f64
            } else {
                0.0
            }
        };

        ThroughputSummary {
            wall_clock_us,
            input_bases: self.input_bases,
            output_bases: self.output_bases,
            input_reads_per_sec: safe_rate(input_reads),
            output_reads_per_sec: safe_rate(output_reads),
            input_bases_per_sec: safe_rate(self.input_bases),
            output_bases_per_sec: safe_rate(self.output_bases),
            batches_per_sec: if wall_clock_secs > 0.0 {
                batches_processed as f64 / wall_clock_secs
            } else {
                0.0
            },
            cumulative_submit_us: self.cumulative_submit_us,
            cumulative_wait_us: self.cumulative_wait_us,
            cumulative_end_to_end_us: self.cumulative_end_to_end_us,
            cumulative_overlap_us: self.cumulative_overlap_us,
            max_batch_end_to_end_us: self.max_batch_end_to_end_us,
            average_submit_us: safe_avg(self.cumulative_submit_us),
            average_wait_us: safe_avg(self.cumulative_wait_us),
            average_end_to_end_us: safe_avg(self.cumulative_end_to_end_us),
        }
    }
}

struct BenchmarkOutputPlan {
    root: Option<PathBuf>,
}

impl BenchmarkOutputPlan {
    fn prepare() -> Result<Self> {
        #[cfg(unix)]
        {
            Ok(Self { root: None })
        }

        #[cfg(not(unix))]
        {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "japalityecho-benchmark-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&root).with_context(|| {
                format!("failed to create benchmark workspace {}", root.display())
            })?;
            Ok(Self { root: Some(root) })
        }
    }

    fn describe_sink(&self) -> String {
        match &self.root {
            Some(root) => format!(
                "Benchmark outputs are written into temporary files under {} and removed after each round",
                root.display()
            ),
            None => {
                "Benchmark outputs are discarded to /dev/null so persisted FASTQ writes do not dominate the measurement"
                    .to_string()
            }
        }
    }

    fn round_outputs(&self, round_index: usize, paired_end: bool) -> (PathBuf, Option<PathBuf>) {
        match &self.root {
            Some(root) => {
                let output1 = root.join(format!("round-{}-r1.fastq", round_index + 1));
                let output2 =
                    paired_end.then(|| root.join(format!("round-{}-r2.fastq", round_index + 1)));
                (output1, output2)
            }
            None => (
                PathBuf::from("/dev/null"),
                paired_end.then(|| PathBuf::from("/dev/null")),
            ),
        }
    }

    fn cleanup_round_outputs(&self, output1: &Path, output2: Option<&Path>) {
        if self.root.is_some() {
            let _ = fs::remove_file(output1);
            if let Some(output2) = output2 {
                let _ = fs::remove_file(output2);
            }
        }
    }
}

impl Drop for BenchmarkOutputPlan {
    fn drop(&mut self) {
        if let Some(root) = &self.root {
            let _ = fs::remove_dir_all(root);
        }
    }
}

fn prepare_run_context(
    input1: &Path,
    input2: Option<&Path>,
    sample_size: usize,
    backend_preference: BackendPreference,
    forced_platform: Option<Platform>,
    forced_experiment: Option<ExperimentType>,
) -> Result<RunContext> {
    // Auto-discover R2 mate when not explicitly provided.
    let discovered_mate = if input2.is_none() {
        discover_mate_file(input1)
    } else {
        None
    };
    let auto_discovered = discovered_mate.is_some();
    let effective_input2 = input2
        .map(Path::to_path_buf)
        .or(discovered_mate);

    // When R2 is explicitly provided via --input2, use synchronized paired
    // sampling (existing behavior). When R2 is auto-discovered, profile R1
    // at full sample size to preserve detection accuracy, then check R2
    // separately for poly-tail boost.
    let (profile_records, sample_records) = if let Some(ref input2_path) = input2.map(Path::to_path_buf) {
        let sample_pairs = sample_size.max(2).div_ceil(2);
        let pairs = sample_paired_records(input1, input2_path, sample_pairs)?;
        if pairs.is_empty() {
            bail!(
                "input FASTQ contains no paired reads: {} and {}",
                input1.display(),
                input2_path.display()
            );
        }
        let r1_records: Vec<_> = pairs.iter().map(|pair| pair.left.clone()).collect();
        let all_records = flatten_pairs(&pairs);
        (r1_records, all_records)
    } else {
        let records = sample_records(input1, sample_size.max(1))?;
        if records.is_empty() {
            bail!("input FASTQ contains no reads: {}", input1.display());
        }
        (records.clone(), records)
    };

    let mut auto_profile = infer_auto_profile(&profile_records);

    // When R2 is available (auto-discovered or explicit) and R1 profiling
    // did not detect RNA-seq, check R2 for poly-A/T tails. R2 of poly-A
    // selected libraries often starts with a poly-T run that R1 lacks.
    if auto_profile.experiment == ExperimentType::Wgs {
        let r2_path = if auto_discovered {
            effective_input2.as_deref()
        } else {
            input2
        };
        if let Some(r2) = r2_path {
            let r2_records = crate::fastq::sample_records(r2, sample_size / 2)?;
            if !r2_records.is_empty() {
                let r2_poly = count_poly_tail_fraction(&r2_records);
                if r2_poly > auto_profile.poly_tail_rate {
                    // Re-run profiling with combined R1+R2 for RNA-seq scoring.
                    // Only override experiment — keep R1's structural metrics.
                    let mut combined = profile_records.clone();
                    combined.extend(r2_records);
                    let combined_profile = infer_auto_profile(&combined);
                    if combined_profile.experiment == ExperimentType::RnaSeq {
                        auto_profile.experiment = ExperimentType::RnaSeq;
                        auto_profile.notes.push(format!(
                            "R2 mate poly-tail rate {:.1}% boosted RNA-seq detection (R1 was {:.1}%)",
                            r2_poly * 100.0,
                            auto_profile.poly_tail_rate * 100.0,
                        ));
                    }
                }
            }
        }
    }

    // Apply manual overrides if provided
    if let Some(platform) = forced_platform {
        auto_profile.platform = platform;
    }
    if let Some(experiment) = forced_experiment {
        auto_profile.experiment = experiment;
    }

    let execution_plan = build_execution_plan(&auto_profile, backend_preference);
    let scheduler = HeterogeneousScheduler::new(backend_preference);
    let resolution = scheduler.resolve(&execution_plan);

    Ok(RunContext {
        sample_records,
        auto_profile,
        execution_plan,
        resolution,
    })
}

/// Count the fraction of reads with a poly-A or poly-T tail.
fn count_poly_tail_fraction(records: &[FastqRecord]) -> f64 {
    if records.is_empty() {
        return 0.0;
    }
    let hits = records
        .iter()
        .filter(|r| {
            crate::profile::has_homopolymer_tail(&r.sequence, b'A')
                || crate::profile::has_homopolymer_tail(&r.sequence, b'T')
        })
        .count();
    hits as f64 / records.len() as f64
}

fn process_single(
    input: &Path,
    output: &Path,
    options: &ProcessOptions,
    paired_end: bool,
    context: &RunContext,
    spectrum: &KmerSpectrum,
    dbg: &DeBruijnGraph,
    backend: &dyn crate::backend::ExecutionBackend,
    resolution: &BackendResolution,
    backend_notes: &[String],
) -> Result<ProcessReport> {
    let started_at = Instant::now();
    let output_file =
        File::create(output).with_context(|| format!("failed to create {}", output.display()))?;
    let mut writer = BufWriter::new(output_file);

    let mut input_reads = 0usize;
    let mut output_reads = 0usize;
    let mut discarded_reads = 0usize;
    let mut corrected_bases = 0usize;
    let mut indel_corrections = 0usize;
    let mut trimmed_reads = 0usize;
    let mut trimmed_bases = 0usize;
    let mut batches_processed = 0usize;
    let mut acceleration_previews = Vec::new();
    let mut acceleration_executions = Vec::new();
    let mut throughput = ThroughputAccumulator::default();
    let mut notes = resolution.notes.clone();
    notes.extend(backend_notes.iter().cloned());
    let read_ahead_depth = context.execution_plan.overlap_depth.max(2);
    notes.push(format!(
        "Background batch read-ahead enabled with depth {} for overlapped FASTQ ingestion",
        read_ahead_depth
    ));
    let plan = Arc::new(context.execution_plan.clone());
    let spectrum = Arc::new(spectrum.clone());

    let mut batch_reader =
        read_batches_with_read_ahead(input, options.batch_reads.max(1), read_ahead_depth);
    let overlap_depth = context.execution_plan.overlap_depth.max(2);
    let mut pending_batches: std::collections::VecDeque<Box<dyn crate::backend::PendingProcessedBatch>> =
        std::collections::VecDeque::with_capacity(overlap_depth);
    while let Some(batch) = batch_reader.next_batch()? {
        if batches_processed == 0 {
            if let Some(scaffold) = &resolution.accelerator_scaffold {
                acceleration_previews.push(prepare_acceleration_preview(
                    "single",
                    &batch,
                    &context.execution_plan,
                    scaffold,
                ));
            }
        }
        let submitted = backend.submit_batch(batch, Arc::clone(&plan), Arc::clone(&spectrum));
        pending_batches.push_back(submitted);
        while pending_batches.len() >= overlap_depth {
            let oldest = pending_batches.pop_front().unwrap();
            let processed = apply_dbg_to_batch(oldest.wait(), dbg, context.execution_plan.trim_min_quality);
            handle_single_processed_batch(
                processed,
                &mut writer,
                context.execution_plan.minimum_output_length,
                &mut notes,
                &mut acceleration_executions,
                &mut input_reads,
                &mut output_reads,
                &mut discarded_reads,
                &mut corrected_bases,
                &mut indel_corrections,
                &mut trimmed_reads,
                &mut trimmed_bases,
                &mut batches_processed,
                &mut throughput,
            )?;
        }
    }
    while let Some(remaining) = pending_batches.pop_front() {
        let processed = apply_dbg_to_batch(remaining.wait(), dbg, context.execution_plan.trim_min_quality);
        handle_single_processed_batch(
            processed,
            &mut writer,
            context.execution_plan.minimum_output_length,
            &mut notes,
            &mut acceleration_executions,
            &mut input_reads,
            &mut output_reads,
            &mut discarded_reads,
            &mut corrected_bases,
            &mut indel_corrections,
            &mut trimmed_reads,
            &mut trimmed_bases,
            &mut batches_processed,
            &mut throughput,
        )?;
    }

    writer.flush()?;
    let throughput = throughput.finalize(
        started_at.elapsed(),
        input_reads,
        output_reads,
        batches_processed,
    );

    notes.push(format!(
        "Used {}-mer sampling to repair low-confidence positions before trimming",
        spectrum.k()
    ));
    notes.push(format!(
        "Executor resolved to '{}' (requested '{}')",
        backend.name(),
        options.backend_preference
    ));
    notes.push(format!(
        "Wall clock {:.3}s | input {:.1} reads/s {:.3} Mbases/s | output {:.1} reads/s {:.3} Mbases/s",
        throughput.wall_clock_us as f64 / 1_000_000.0,
        throughput.input_reads_per_sec,
        throughput.input_bases_per_sec / 1_000_000.0,
        throughput.output_reads_per_sec,
        throughput.output_bases_per_sec / 1_000_000.0
    ));
    notes.push(format!(
        "Cumulative batch host timing: submit={}us wait={}us overlap={}us max_end_to_end={}us",
        throughput.cumulative_submit_us,
        throughput.cumulative_wait_us,
        throughput.cumulative_overlap_us,
        throughput.max_batch_end_to_end_us
    ));

    Ok(ProcessReport {
        paired_end,
        input_pairs: None,
        output_pairs: None,
        discarded_pairs: None,
        input_reads,
        output_reads,
        discarded_reads,
        corrected_bases,
        indel_corrections,
        trimmed_reads,
        trimmed_bases,
        batches_processed,
        backend_used: backend.name().to_string(),
        zero_copy_candidate: context.execution_plan.zero_copy_candidate,
        accelerator_scaffold: resolution.accelerator_scaffold.clone(),
        accelerator_runtime: resolution.accelerator_runtime.clone(),
        acceleration_previews,
        acceleration_executions,
        auto_profile: context.auto_profile.clone(),
        execution_plan: context.execution_plan.clone(),
        throughput,
        notes,
    })
}

fn process_paired(
    input1: &Path,
    output1: &Path,
    input2: &Path,
    output2: &Path,
    options: &ProcessOptions,
    context: &RunContext,
    spectrum: &KmerSpectrum,
    dbg: &DeBruijnGraph,
    backend: &dyn crate::backend::ExecutionBackend,
    resolution: &BackendResolution,
    backend_notes: &[String],
) -> Result<ProcessReport> {
    let started_at = Instant::now();
    let output_file1 =
        File::create(output1).with_context(|| format!("failed to create {}", output1.display()))?;
    let output_file2 =
        File::create(output2).with_context(|| format!("failed to create {}", output2.display()))?;
    let mut writer1 = BufWriter::new(output_file1);
    let mut writer2 = BufWriter::new(output_file2);

    let mut input_pairs = 0usize;
    let mut output_pairs = 0usize;
    let mut discarded_pairs = 0usize;
    let mut input_reads = 0usize;
    let mut output_reads = 0usize;
    let mut discarded_reads = 0usize;
    let mut corrected_bases = 0usize;
    let mut indel_corrections = 0usize;
    let mut trimmed_reads = 0usize;
    let mut trimmed_bases = 0usize;
    let mut batches_processed = 0usize;
    let mut acceleration_previews = Vec::new();
    let mut acceleration_executions = Vec::new();
    let mut throughput = ThroughputAccumulator::default();
    let mut notes = resolution.notes.clone();
    notes.extend(backend_notes.iter().cloned());
    notes.push(format!(
        "Paired-end mode keeps mates synchronized; if either mate falls below {} bases, the pair is discarded",
        context.execution_plan.minimum_output_length
    ));
    let read_ahead_depth = context.execution_plan.overlap_depth.max(2);
    notes.push(format!(
        "Background paired batch read-ahead enabled with depth {} for overlapped FASTQ ingestion",
        read_ahead_depth
    ));
    let plan = Arc::new(context.execution_plan.clone());
    let spectrum = Arc::new(spectrum.clone());

    let mut pair_batch_reader = read_paired_batches_with_read_ahead(
        input1,
        input2,
        options.batch_reads.max(1),
        read_ahead_depth,
    );
    while let Some(pair_batch) = pair_batch_reader.next_batch()? {
        let pair_count = pair_batch.pairs.len();
        let total_pair_bases = pair_batch.total_bases;
        let (left_records, right_records) = split_pair_batch(pair_batch.pairs);
        let left_batch = ReadBatch::new(batches_processed, left_records);
        let right_batch = ReadBatch::new(batches_processed, right_records);

        if batches_processed == 0 {
            if let Some(scaffold) = &resolution.accelerator_scaffold {
                acceleration_previews.push(prepare_acceleration_preview(
                    "mate1",
                    &left_batch,
                    &context.execution_plan,
                    scaffold,
                ));
                acceleration_previews.push(prepare_acceleration_preview(
                    "mate2",
                    &right_batch,
                    &context.execution_plan,
                    scaffold,
                ));
            }
        }

        let pending_left =
            backend.submit_batch(left_batch, Arc::clone(&plan), Arc::clone(&spectrum));
        let pending_right =
            backend.submit_batch(right_batch, Arc::clone(&plan), Arc::clone(&spectrum));
        let processed_left = apply_dbg_to_batch(pending_left.wait(), dbg, context.execution_plan.trim_min_quality);
        let processed_right = apply_dbg_to_batch(pending_right.wait(), dbg, context.execution_plan.trim_min_quality);
        let left_input_bases = processed_left.layout.total_bases;
        let right_input_bases = processed_right.layout.total_bases;
        let left_timing = processed_left.timing.clone();
        let right_timing = processed_right.timing.clone();

        if batches_processed == 0 {
            notes.push(format!(
                "First paired batch layout: {} pairs / {} bases / R1 {} bytes / R2 {} bytes",
                pair_count,
                total_pair_bases,
                processed_left.layout.host_bytes,
                processed_right.layout.host_bytes
            ));
            notes.extend(processed_left.transfer_plan.notes.clone());
        }
        if let Some(execution) = processed_left.acceleration_execution.clone() {
            acceleration_executions.push(execution);
        }
        if let Some(execution) = processed_right.acceleration_execution.clone() {
            acceleration_executions.push(execution);
        }

        input_pairs += pair_count;
        input_reads += processed_left.stats.processed_reads + processed_right.stats.processed_reads;
        corrected_bases +=
            processed_left.stats.corrected_bases + processed_right.stats.corrected_bases;
        indel_corrections +=
            processed_left.stats.indel_corrections + processed_right.stats.indel_corrections;
        trimmed_reads += processed_left.stats.trimmed_reads + processed_right.stats.trimmed_reads;
        trimmed_bases += processed_left.stats.trimmed_bases + processed_right.stats.trimmed_bases;
        batches_processed += 1;
        let mut kept_left_bases = 0usize;
        let mut kept_right_bases = 0usize;

        let kept_pairs: Vec<_> = processed_left
            .records
            .into_iter()
            .zip(processed_right.records.into_iter())
            .filter_map(|(left, right)| {
                let keep = left.len() >= context.execution_plan.minimum_output_length
                    && right.len() >= context.execution_plan.minimum_output_length;
                if keep {
                    output_pairs += 1;
                    output_reads += 2;
                    kept_left_bases += left.len();
                    kept_right_bases += right.len();
                    Some((left, right))
                } else {
                    discarded_pairs += 1;
                    discarded_reads += 2;
                    None
                }
            })
            .collect();
        throughput.observe(left_input_bases, kept_left_bases, &left_timing);
        throughput.observe(right_input_bases, kept_right_bases, &right_timing);

        write_processed_record_pairs(&mut writer1, &mut writer2, &kept_pairs)?;
    }

    writer1.flush()?;
    writer2.flush()?;
    let throughput = throughput.finalize(
        started_at.elapsed(),
        input_reads,
        output_reads,
        batches_processed * 2,
    );

    notes.push(format!(
        "Used {}-mer sampling to repair low-confidence positions across both mates",
        spectrum.k()
    ));
    notes.push(format!(
        "Executor resolved to '{}' (requested '{}')",
        backend.name(),
        options.backend_preference
    ));
    notes.push(format!(
        "Wall clock {:.3}s | input {:.1} reads/s {:.3} Mbases/s | output {:.1} reads/s {:.3} Mbases/s",
        throughput.wall_clock_us as f64 / 1_000_000.0,
        throughput.input_reads_per_sec,
        throughput.input_bases_per_sec / 1_000_000.0,
        throughput.output_reads_per_sec,
        throughput.output_bases_per_sec / 1_000_000.0
    ));
    notes.push(format!(
        "Cumulative batch host timing: submit={}us wait={}us overlap={}us max_end_to_end={}us",
        throughput.cumulative_submit_us,
        throughput.cumulative_wait_us,
        throughput.cumulative_overlap_us,
        throughput.max_batch_end_to_end_us
    ));

    Ok(ProcessReport {
        paired_end: true,
        input_pairs: Some(input_pairs),
        output_pairs: Some(output_pairs),
        discarded_pairs: Some(discarded_pairs),
        input_reads,
        output_reads,
        discarded_reads,
        corrected_bases,
        indel_corrections,
        trimmed_reads,
        trimmed_bases,
        batches_processed,
        backend_used: backend.name().to_string(),
        zero_copy_candidate: context.execution_plan.zero_copy_candidate,
        accelerator_scaffold: resolution.accelerator_scaffold.clone(),
        accelerator_runtime: resolution.accelerator_runtime.clone(),
        acceleration_previews,
        acceleration_executions,
        auto_profile: context.auto_profile.clone(),
        execution_plan: context.execution_plan.clone(),
        throughput,
        notes,
    })
}

/// Apply De Bruijn graph second-pass correction to all records in a batch.
///
/// Scans each record for remaining low-quality regions and uses graph path-finding
/// to correct multi-base errors that single-position correction could not fix.
fn apply_dbg_to_batch(
    mut batch: crate::backend::ProcessedBatch,
    dbg: &DeBruijnGraph,
    min_quality: u8,
) -> crate::backend::ProcessedBatch {
    if dbg.is_empty() {
        return batch;
    }
    for record in &mut batch.records {
        let limit = record.sequence.len();
        let corrected = dbg.correct_sequence_regions(
            &mut record.sequence,
            &record.qualities,
            min_quality,
            limit,
        );
        for pos in corrected {
            if !record.corrected_positions.contains(&pos) {
                record.corrected_positions.push(pos);
                batch.stats.corrected_bases += 1;
            }
        }
    }
    batch
}

fn handle_single_processed_batch(
    processed: crate::backend::ProcessedBatch,
    writer: &mut BufWriter<File>,
    minimum_output_length: usize,
    notes: &mut Vec<String>,
    acceleration_executions: &mut Vec<crate::model::AccelerationExecution>,
    input_reads: &mut usize,
    output_reads: &mut usize,
    discarded_reads: &mut usize,
    corrected_bases: &mut usize,
    indel_corrections: &mut usize,
    trimmed_reads: &mut usize,
    trimmed_bases: &mut usize,
    batches_processed: &mut usize,
    throughput: &mut ThroughputAccumulator,
) -> Result<()> {
    let input_bases = processed.layout.total_bases;
    let timing = processed.timing.clone();
    if *batches_processed == 0 {
        notes.push(format!(
            "First batch layout: {} reads / {} bases / {} bytes in {}",
            processed.layout.reads,
            processed.layout.total_bases,
            processed.layout.host_bytes,
            processed.layout.preferred_memory
        ));
        notes.extend(processed.transfer_plan.notes.clone());
    }
    if let Some(execution) = processed.acceleration_execution.clone() {
        acceleration_executions.push(execution);
    }

    *input_reads += processed.stats.processed_reads;
    *corrected_bases += processed.stats.corrected_bases;
    *indel_corrections += processed.stats.indel_corrections;
    *trimmed_reads += processed.stats.trimmed_reads;
    *trimmed_bases += processed.stats.trimmed_bases;
    *batches_processed += 1;

    let kept_records: Vec<_> = processed
        .records
        .into_iter()
        .filter(|record| {
            let keep = record.len() >= minimum_output_length;
            if keep {
                *output_reads += 1;
            } else {
                *discarded_reads += 1;
            }
            keep
        })
        .collect();
    let kept_bases = kept_records.iter().map(|record| record.len()).sum();
    throughput.observe(input_bases, kept_bases, &timing);

    write_processed_records(writer, &kept_records)
}

fn duration_as_us(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

fn summarize_benchmark(
    rounds: &[BenchmarkRound],
    session_mode: BenchmarkSessionMode,
    setup_us: u64,
) -> BenchmarkSummary {
    let round_count = rounds.len().max(1);
    let mut backends = Vec::new();
    let mut average_wall_clock_us = 0.0f64;
    let mut best_wall_clock_us = u64::MAX;
    let mut worst_wall_clock_us = 0u64;
    let mut average_input_reads_per_sec = 0.0f64;
    let mut average_output_reads_per_sec = 0.0f64;
    let mut average_input_bases_per_sec = 0.0f64;
    let mut average_output_bases_per_sec = 0.0f64;
    let mut average_cumulative_overlap_us = 0.0f64;
    let mut best_cumulative_overlap_us = 0u64;

    for round in rounds {
        if !backends
            .iter()
            .any(|backend| backend == &round.process.backend_used)
        {
            backends.push(round.process.backend_used.clone());
        }
        let throughput = &round.process.throughput;
        average_wall_clock_us += throughput.wall_clock_us as f64;
        best_wall_clock_us = best_wall_clock_us.min(throughput.wall_clock_us);
        worst_wall_clock_us = worst_wall_clock_us.max(throughput.wall_clock_us);
        average_input_reads_per_sec += throughput.input_reads_per_sec;
        average_output_reads_per_sec += throughput.output_reads_per_sec;
        average_input_bases_per_sec += throughput.input_bases_per_sec;
        average_output_bases_per_sec += throughput.output_bases_per_sec;
        average_cumulative_overlap_us += throughput.cumulative_overlap_us as f64;
        best_cumulative_overlap_us =
            best_cumulative_overlap_us.max(throughput.cumulative_overlap_us);
    }

    let warmup_wall_clock_us = rounds
        .first()
        .map(|round| round.process.throughput.wall_clock_us)
        .unwrap_or(0);
    let warmup_transfer_bytes = rounds
        .first()
        .map(|round| total_transfer_bytes(&round.process))
        .unwrap_or(0);

    let steady_state_rounds = rounds.get(1..).unwrap_or(&[]);
    let steady_state_average_wall_clock_us = (!steady_state_rounds.is_empty()).then(|| {
        steady_state_rounds
            .iter()
            .map(|round| round.process.throughput.wall_clock_us as f64)
            .sum::<f64>()
            / steady_state_rounds.len() as f64
    });
    let steady_state_average_transfer_bytes = (!steady_state_rounds.is_empty()).then(|| {
        steady_state_rounds
            .iter()
            .map(|round| total_transfer_bytes(&round.process) as f64)
            .sum::<f64>()
            / steady_state_rounds.len() as f64
    });
    let warmup_penalty_pct = steady_state_average_wall_clock_us
        .filter(|steady_state| *steady_state > 0.0)
        .map(|steady_state| ((warmup_wall_clock_us as f64 - steady_state) / steady_state) * 100.0);

    BenchmarkSummary {
        session_mode,
        setup_us,
        rounds: rounds.len(),
        backends,
        backend_consistent: rounds
            .first()
            .map(|first| {
                rounds
                    .iter()
                    .all(|round| round.process.backend_used == first.process.backend_used)
            })
            .unwrap_or(true),
        average_wall_clock_us: average_wall_clock_us / round_count as f64,
        best_wall_clock_us: if best_wall_clock_us == u64::MAX {
            0
        } else {
            best_wall_clock_us
        },
        worst_wall_clock_us,
        average_input_reads_per_sec: average_input_reads_per_sec / round_count as f64,
        average_output_reads_per_sec: average_output_reads_per_sec / round_count as f64,
        average_input_bases_per_sec: average_input_bases_per_sec / round_count as f64,
        average_output_bases_per_sec: average_output_bases_per_sec / round_count as f64,
        average_cumulative_overlap_us: average_cumulative_overlap_us / round_count as f64,
        best_cumulative_overlap_us,
        warmup_wall_clock_us,
        steady_state_average_wall_clock_us,
        warmup_penalty_pct,
        warmup_transfer_bytes,
        steady_state_average_transfer_bytes,
    }
}

fn summarize_benchmark_comparison(
    cold_start: &BenchmarkReport,
    reuse_session: &BenchmarkReport,
) -> BenchmarkComparisonSummary {
    let cold_summary = &cold_start.summary;
    let reuse_summary = &reuse_session.summary;
    let reuse_amortized_average_wall_clock_us = reuse_summary.average_wall_clock_us
        + (reuse_summary.setup_us as f64 / reuse_summary.rounds.max(1) as f64);
    let average_wall_clock_delta_us =
        cold_summary.average_wall_clock_us - reuse_summary.average_wall_clock_us;
    let amortized_wall_clock_delta_us =
        cold_summary.average_wall_clock_us - reuse_amortized_average_wall_clock_us;

    BenchmarkComparisonSummary {
        rounds: cold_summary.rounds.min(reuse_summary.rounds),
        cold_start_backends: cold_summary.backends.clone(),
        reuse_session_backends: reuse_summary.backends.clone(),
        backend_match: cold_summary.backends == reuse_summary.backends,
        cold_start_average_wall_clock_us: cold_summary.average_wall_clock_us,
        reuse_session_average_wall_clock_us: reuse_summary.average_wall_clock_us,
        reuse_session_setup_us: reuse_summary.setup_us,
        reuse_session_amortized_average_wall_clock_us: reuse_amortized_average_wall_clock_us,
        average_wall_clock_delta_us,
        amortized_wall_clock_delta_us,
        raw_speedup: speedup_ratio(
            cold_summary.average_wall_clock_us,
            reuse_summary.average_wall_clock_us,
        ),
        amortized_speedup: speedup_ratio(
            cold_summary.average_wall_clock_us,
            reuse_amortized_average_wall_clock_us,
        ),
        input_bases_per_sec_uplift_pct: uplift_pct(
            cold_summary.average_input_bases_per_sec,
            reuse_summary.average_input_bases_per_sec,
        ),
        output_bases_per_sec_uplift_pct: uplift_pct(
            cold_summary.average_output_bases_per_sec,
            reuse_summary.average_output_bases_per_sec,
        ),
        average_overlap_delta_us: reuse_summary.average_cumulative_overlap_us
            - cold_summary.average_cumulative_overlap_us,
        steady_state_transfer_savings_bytes: match (
            cold_summary.steady_state_average_transfer_bytes,
            reuse_summary.steady_state_average_transfer_bytes,
        ) {
            (Some(cold), Some(reuse)) => Some(cold - reuse),
            _ => None,
        },
        steady_state_transfer_savings_pct: match (
            cold_summary.steady_state_average_transfer_bytes,
            reuse_summary.steady_state_average_transfer_bytes,
        ) {
            (Some(cold), Some(reuse)) if cold > 0.0 => Some(((cold - reuse) / cold) * 100.0),
            _ => None,
        },
    }
}

fn speedup_ratio(baseline: f64, candidate: f64) -> Option<f64> {
    (candidate > 0.0).then_some(baseline / candidate)
}

fn uplift_pct(baseline: f64, candidate: f64) -> Option<f64> {
    (baseline > 0.0).then_some(((candidate - baseline) / baseline) * 100.0)
}

fn total_transfer_bytes(report: &ProcessReport) -> usize {
    report
        .acceleration_executions
        .iter()
        .map(|execution| execution.transfer_bytes)
        .sum()
}

fn flatten_pairs(pairs: &[ReadPair]) -> Vec<FastqRecord> {
    let mut records = Vec::with_capacity(pairs.len() * 2);
    for pair in pairs {
        records.push(pair.left.clone());
        records.push(pair.right.clone());
    }
    records
}

fn split_pair_batch(pairs: Vec<ReadPair>) -> (Vec<FastqRecord>, Vec<FastqRecord>) {
    let mut left = Vec::with_capacity(pairs.len());
    let mut right = Vec::with_capacity(pairs.len());
    for pair in pairs {
        left.push(pair.left);
        right.push(pair.right);
    }
    (left, right)
}
