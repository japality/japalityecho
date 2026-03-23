use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::evaluate::{
    EvaluationOptions, evaluate_paired_synthetic_truth, evaluate_synthetic_truth,
    write_paired_synthetic_truth_dataset, write_synthetic_truth_dataset,
};
use crate::model::{
    BackendParitySummary, BackendPreference, BenchmarkComparisonSummary, BenchmarkSessionMode,
    BenchmarkSummary, EvaluationAggregateSummary, EvaluationReport,
    PairedEvaluationAggregateSummary, PairedEvaluationReport, PaperArtifactFile,
    PaperArtifactsReport, PublicationBundleReport,
};
use crate::process::{
    BenchmarkComparisonOptions, BenchmarkOptions, ProcessOptions, benchmark_compare_files,
    benchmark_files,
};
use crate::study::{
    StudyAnnotateOptions, StudyArtifactsOptions, StudyDownloadOptions, StudyFetchMetadataOptions,
    StudyIngestOptions, StudyManifestOptions, annotate_study_manifest, bootstrap_study_manifest,
    download_public_fastqs, fetch_public_metadata, generate_study_artifacts, ingest_study_results,
};

#[derive(Debug, Clone)]
pub struct PaperArtifactsOptions {
    pub output_dir: PathBuf,
    pub sample_size: usize,
    pub batch_reads: usize,
    pub reads_per_scenario: usize,
    pub benchmark_rounds: usize,
    pub accelerator_backend: BackendPreference,
    pub min_quality_override: Option<u8>,
}

#[derive(Debug, Clone)]
pub struct PublicationBundleOptions {
    pub input_path: PathBuf,
    pub output_dir: PathBuf,
    pub annotations_path: Option<PathBuf>,
    pub results_dir: Option<PathBuf>,
    pub default_baseline_name: Option<String>,
    pub sample_size: usize,
    pub batch_reads: usize,
    pub reads_per_scenario: usize,
    pub benchmark_rounds: usize,
    pub accelerator_backend: BackendPreference,
    pub min_quality_override: Option<u8>,
    pub base_url: String,
    pub geo_base_url: String,
    pub chunk_size: usize,
    pub cache_dir: Option<PathBuf>,
    pub retries: usize,
    pub resume_existing: bool,
    pub overwrite_existing_downloads: bool,
}

pub fn generate_paper_artifacts(options: &PaperArtifactsOptions) -> Result<PaperArtifactsReport> {
    let output_dir = options.output_dir.clone();
    let data_dir = output_dir.join("data");
    let figure_dir = output_dir.join("figures");
    fs::create_dir_all(&data_dir)
        .with_context(|| format!("failed to create {}", data_dir.display()))?;
    fs::create_dir_all(&figure_dir)
        .with_context(|| format!("failed to create {}", figure_dir.display()))?;

    let synthetic_input = data_dir.join("synthetic_truth.fastq");
    let paired_input1 = data_dir.join("synthetic_truth_paired_r1.fastq");
    let paired_input2 = data_dir.join("synthetic_truth_paired_r2.fastq");
    let total_reads = write_synthetic_truth_dataset(&synthetic_input, options.reads_per_scenario)?;
    let total_pairs = write_paired_synthetic_truth_dataset(
        &paired_input1,
        &paired_input2,
        options.reads_per_scenario,
    )?;

    let cpu_eval_options = EvaluationOptions {
        sample_size: options.sample_size,
        batch_reads: options.batch_reads,
        backend_preference: BackendPreference::Cpu,
        min_quality_override: options.min_quality_override,
        reads_per_scenario: options.reads_per_scenario,
    };
    let accelerator_eval_options = EvaluationOptions {
        sample_size: options.sample_size,
        batch_reads: options.batch_reads,
        backend_preference: options.accelerator_backend,
        min_quality_override: options.min_quality_override,
        reads_per_scenario: options.reads_per_scenario,
    };

    let cpu_evaluation = evaluate_synthetic_truth(&cpu_eval_options)?;
    let accelerator_evaluation = evaluate_synthetic_truth(&accelerator_eval_options)?;
    let paired_cpu_evaluation = evaluate_paired_synthetic_truth(&cpu_eval_options)?;
    let paired_accelerator_evaluation = evaluate_paired_synthetic_truth(&accelerator_eval_options)?;
    let parity = build_backend_parity_summary(
        &cpu_evaluation,
        &accelerator_evaluation,
        &paired_cpu_evaluation,
        &paired_accelerator_evaluation,
    );

    let cpu_process = ProcessOptions {
        sample_size: options.sample_size,
        batch_reads: options.batch_reads,
        backend_preference: BackendPreference::Cpu,
        forced_adapter: None,
        min_quality_override: options.min_quality_override,
        kmer_size_override: None,
                    forced_platform: None,
                    forced_experiment: None,
    };
    let accelerator_process = ProcessOptions {
        sample_size: options.sample_size,
        batch_reads: options.batch_reads,
        backend_preference: options.accelerator_backend,
        forced_adapter: None,
        min_quality_override: options.min_quality_override,
        kmer_size_override: None,
                    forced_platform: None,
                    forced_experiment: None,
    };

    let cpu_benchmark = benchmark_files(
        &synthetic_input,
        None,
        &BenchmarkOptions {
            process: cpu_process,
            rounds: options.benchmark_rounds.max(1),
            session_mode: BenchmarkSessionMode::ColdStart,
        },
    )?;
    let accelerator_cold_benchmark = benchmark_files(
        &synthetic_input,
        None,
        &BenchmarkOptions {
            process: accelerator_process.clone(),
            rounds: options.benchmark_rounds.max(1),
            session_mode: BenchmarkSessionMode::ColdStart,
        },
    )?;
    let accelerator_reuse_benchmark = benchmark_files(
        &synthetic_input,
        None,
        &BenchmarkOptions {
            process: accelerator_process.clone(),
            rounds: options.benchmark_rounds.max(1),
            session_mode: BenchmarkSessionMode::ReuseSession,
        },
    )?;
    let accelerator_compare = benchmark_compare_files(
        &synthetic_input,
        None,
        &BenchmarkComparisonOptions {
            process: accelerator_process,
            rounds: options.benchmark_rounds.max(1),
        },
    )?;

    let accelerator_label = options.accelerator_backend.label().to_string();
    let cpu_eval_json = data_dir.join("evaluation_cpu.json");
    let accel_eval_json = data_dir.join("evaluation_accelerator.json");
    let paired_cpu_eval_json = data_dir.join("evaluation_paired_cpu.json");
    let paired_accel_eval_json = data_dir.join("evaluation_paired_accelerator.json");
    let cpu_bench_json = data_dir.join("benchmark_cpu_cold.json");
    let accel_cold_json = data_dir.join("benchmark_accelerator_cold.json");
    let accel_reuse_json = data_dir.join("benchmark_accelerator_reuse.json");
    let accel_compare_json = data_dir.join("benchmark_accelerator_compare.json");
    let scenario_csv = data_dir.join("scenario_accuracy.csv");
    let aggregate_csv = data_dir.join("aggregate_accuracy.csv");
    let paired_scenario_csv = data_dir.join("paired_scenario_accuracy.csv");
    let paired_aggregate_csv = data_dir.join("paired_aggregate_accuracy.csv");
    let throughput_csv = data_dir.join("backend_throughput.csv");
    let transfer_csv = data_dir.join("transfer_efficiency.csv");
    let parity_csv = data_dir.join("backend_parity.csv");
    let exact_chart = figure_dir.join("scenario_exact_match.svg");
    let paired_exact_chart = figure_dir.join("paired_scenario_exact_match.svg");
    let aggregate_chart = figure_dir.join("aggregate_quality.svg");
    let throughput_chart = figure_dir.join("backend_throughput.svg");
    let transfer_chart = figure_dir.join("transfer_efficiency.svg");
    let parity_chart = figure_dir.join("backend_parity.svg");
    let summary_json = output_dir.join("paper_artifacts.json");

    write_json_pretty(&cpu_eval_json, &cpu_evaluation)?;
    write_json_pretty(&accel_eval_json, &accelerator_evaluation)?;
    write_json_pretty(&paired_cpu_eval_json, &paired_cpu_evaluation)?;
    write_json_pretty(&paired_accel_eval_json, &paired_accelerator_evaluation)?;
    write_json_pretty(&cpu_bench_json, &cpu_benchmark)?;
    write_json_pretty(&accel_cold_json, &accelerator_cold_benchmark)?;
    write_json_pretty(&accel_reuse_json, &accelerator_reuse_benchmark)?;
    write_json_pretty(&accel_compare_json, &accelerator_compare)?;

    write_scenario_accuracy_csv(&scenario_csv, &cpu_evaluation, &accelerator_evaluation)?;
    write_aggregate_accuracy_csv(&aggregate_csv, &cpu_evaluation, &accelerator_evaluation)?;
    write_paired_scenario_accuracy_csv(
        &paired_scenario_csv,
        &paired_cpu_evaluation,
        &paired_accelerator_evaluation,
    )?;
    write_paired_aggregate_accuracy_csv(
        &paired_aggregate_csv,
        &paired_cpu_evaluation,
        &paired_accelerator_evaluation,
    )?;
    write_backend_throughput_csv(
        &throughput_csv,
        &cpu_benchmark.summary,
        &accelerator_cold_benchmark.summary,
        &accelerator_reuse_benchmark.summary,
        options.accelerator_backend,
    )?;
    write_transfer_efficiency_csv(
        &transfer_csv,
        &accelerator_cold_benchmark.summary,
        &accelerator_reuse_benchmark.summary,
        &accelerator_compare.summary,
    )?;
    write_backend_parity_csv(
        &parity_csv,
        &cpu_evaluation,
        &accelerator_evaluation,
        &paired_cpu_evaluation,
        &paired_accelerator_evaluation,
        &accelerator_label,
    )?;

    write_svg(
        &exact_chart,
        &grouped_bar_chart_svg(
            "Synthetic truth exact-match rate by scenario",
            "Exact-match (%)",
            &cpu_evaluation
                .scenarios
                .iter()
                .map(|scenario| scenario.name.replace('_', " "))
                .collect::<Vec<_>>(),
            &[
                ChartSeries {
                    name: "CPU",
                    color: "#4E79A7",
                    values: cpu_evaluation
                        .scenarios
                        .iter()
                        .map(|scenario| scenario.exact_match_rate * 100.0)
                        .collect(),
                },
                ChartSeries {
                    name: accelerator_label.as_str(),
                    color: "#F28E2B",
                    values: accelerator_evaluation
                        .scenarios
                        .iter()
                        .map(|scenario| scenario.exact_match_rate * 100.0)
                        .collect(),
                },
            ],
        ),
    )?;
    write_svg(
        &paired_exact_chart,
        &grouped_bar_chart_svg(
            "Paired synthetic truth exact-match rate by scenario",
            "Pair exact-match (%)",
            &paired_cpu_evaluation
                .scenarios
                .iter()
                .map(|scenario| scenario.name.replace('_', " "))
                .collect::<Vec<_>>(),
            &[
                ChartSeries {
                    name: "CPU",
                    color: "#4E79A7",
                    values: paired_cpu_evaluation
                        .scenarios
                        .iter()
                        .map(|scenario| scenario.exact_match_rate * 100.0)
                        .collect(),
                },
                ChartSeries {
                    name: accelerator_label.as_str(),
                    color: "#F28E2B",
                    values: paired_accelerator_evaluation
                        .scenarios
                        .iter()
                        .map(|scenario| scenario.exact_match_rate * 100.0)
                        .collect(),
                },
            ],
        ),
    )?;
    write_svg(
        &aggregate_chart,
        &grouped_bar_chart_svg(
            "Aggregate synthetic accuracy metrics",
            "Score (%)",
            &[
                "SE exact".to_string(),
                "SE retention f1".to_string(),
                "SE trimming f1".to_string(),
                "SE correction f1".to_string(),
                "PE exact".to_string(),
                "PE retention f1".to_string(),
                "PE trimming f1".to_string(),
                "PE correction f1".to_string(),
            ],
            &[
                ChartSeries {
                    name: "CPU",
                    color: "#4E79A7",
                    values: vec![
                        cpu_evaluation.aggregate.exact_match_rate * 100.0,
                        cpu_evaluation.aggregate.retention.f1 * 100.0,
                        cpu_evaluation.aggregate.trimming.f1 * 100.0,
                        cpu_evaluation.aggregate.correction.f1 * 100.0,
                        paired_cpu_evaluation.aggregate.exact_match_rate * 100.0,
                        paired_cpu_evaluation.aggregate.retention.f1 * 100.0,
                        paired_cpu_evaluation.aggregate.trimming.f1 * 100.0,
                        paired_cpu_evaluation.aggregate.correction.f1 * 100.0,
                    ],
                },
                ChartSeries {
                    name: accelerator_label.as_str(),
                    color: "#F28E2B",
                    values: vec![
                        accelerator_evaluation.aggregate.exact_match_rate * 100.0,
                        accelerator_evaluation.aggregate.retention.f1 * 100.0,
                        accelerator_evaluation.aggregate.trimming.f1 * 100.0,
                        accelerator_evaluation.aggregate.correction.f1 * 100.0,
                        paired_accelerator_evaluation.aggregate.exact_match_rate * 100.0,
                        paired_accelerator_evaluation.aggregate.retention.f1 * 100.0,
                        paired_accelerator_evaluation.aggregate.trimming.f1 * 100.0,
                        paired_accelerator_evaluation.aggregate.correction.f1 * 100.0,
                    ],
                },
            ],
        ),
    )?;
    write_svg(
        &throughput_chart,
        &single_series_bar_chart_svg(
            "Backend throughput summary",
            "Input throughput (Mbases/s)",
            &[
                "cpu cold".to_string(),
                format!("{} cold", accelerator_label),
                format!("{} reuse", accelerator_label),
            ],
            &[
                cpu_benchmark.summary.average_input_bases_per_sec / 1_000_000.0,
                accelerator_cold_benchmark
                    .summary
                    .average_input_bases_per_sec
                    / 1_000_000.0,
                accelerator_reuse_benchmark
                    .summary
                    .average_input_bases_per_sec
                    / 1_000_000.0,
            ],
            "#59A14F",
        ),
    )?;
    write_svg(
        &transfer_chart,
        &single_series_bar_chart_svg(
            "Accelerator transfer efficiency",
            "Transfer bytes",
            &[
                "cold warmup".to_string(),
                "cold steady".to_string(),
                "reuse warmup".to_string(),
                "reuse steady".to_string(),
            ],
            &[
                accelerator_cold_benchmark.summary.warmup_transfer_bytes as f64,
                accelerator_cold_benchmark
                    .summary
                    .steady_state_average_transfer_bytes
                    .unwrap_or(0.0),
                accelerator_reuse_benchmark.summary.warmup_transfer_bytes as f64,
                accelerator_reuse_benchmark
                    .summary
                    .steady_state_average_transfer_bytes
                    .unwrap_or(0.0),
            ],
            "#E15759",
        ),
    )?;
    write_svg(
        &parity_chart,
        &grouped_bar_chart_svg(
            "CPU vs accelerator parity across single-end and paired-end metrics",
            "Score (%)",
            &[
                "SE exact".to_string(),
                "SE retain".to_string(),
                "SE trim".to_string(),
                "SE correct".to_string(),
                "PE exact".to_string(),
                "PE retain".to_string(),
                "PE trim".to_string(),
                "PE correct".to_string(),
            ],
            &[
                ChartSeries {
                    name: "CPU",
                    color: "#4E79A7",
                    values: parity_chart_values_cpu(&cpu_evaluation, &paired_cpu_evaluation),
                },
                ChartSeries {
                    name: accelerator_label.as_str(),
                    color: "#F28E2B",
                    values: parity_chart_values_accelerator(
                        &accelerator_evaluation,
                        &paired_accelerator_evaluation,
                    ),
                },
            ],
        ),
    )?;

    let artifacts = vec![
        artifact(
            "json",
            &summary_json,
            "Top-level paper artifact bundle summary",
        ),
        artifact(
            "json",
            &cpu_eval_json,
            "CPU synthetic truth evaluation report",
        ),
        artifact(
            "json",
            &accel_eval_json,
            "Accelerator synthetic truth evaluation report",
        ),
        artifact(
            "json",
            &paired_cpu_eval_json,
            "CPU paired synthetic truth evaluation report",
        ),
        artifact(
            "json",
            &paired_accel_eval_json,
            "Accelerator paired synthetic truth evaluation report",
        ),
        artifact("json", &cpu_bench_json, "CPU cold-start benchmark report"),
        artifact(
            "json",
            &accel_cold_json,
            "Accelerator cold-start benchmark report",
        ),
        artifact(
            "json",
            &accel_reuse_json,
            "Accelerator reuse-session benchmark report",
        ),
        artifact(
            "json",
            &accel_compare_json,
            "Accelerator cold-vs-reuse comparison report",
        ),
        artifact(
            "csv",
            &scenario_csv,
            "Per-scenario single-end accuracy table for manuscript figures",
        ),
        artifact(
            "csv",
            &aggregate_csv,
            "Single-end aggregate accuracy table for manuscript summary",
        ),
        artifact(
            "csv",
            &paired_scenario_csv,
            "Per-scenario paired-end accuracy table for manuscript figures",
        ),
        artifact(
            "csv",
            &paired_aggregate_csv,
            "Paired-end aggregate accuracy table for manuscript summary",
        ),
        artifact("csv", &throughput_csv, "Backend throughput summary table"),
        artifact("csv", &transfer_csv, "Transfer-efficiency summary table"),
        artifact(
            "csv",
            &parity_csv,
            "CPU-versus-accelerator parity summary table",
        ),
        artifact(
            "svg",
            &exact_chart,
            "Single-end scenario exact-match figure",
        ),
        artifact(
            "svg",
            &paired_exact_chart,
            "Paired-end scenario exact-match figure",
        ),
        artifact("svg", &aggregate_chart, "Aggregate quality metric figure"),
        artifact("svg", &throughput_chart, "Backend throughput figure"),
        artifact("svg", &transfer_chart, "Transfer-efficiency figure"),
        artifact("svg", &parity_chart, "CPU-versus-accelerator parity figure"),
        artifact(
            "fastq",
            &synthetic_input,
            "Deterministic single-end synthetic truth FASTQ used for manuscript evaluation",
        ),
        artifact(
            "fastq",
            &paired_input1,
            "Deterministic paired-end synthetic truth FASTQ mate 1",
        ),
        artifact(
            "fastq",
            &paired_input2,
            "Deterministic paired-end synthetic truth FASTQ mate 2",
        ),
    ];

    let mut notes = vec![
        "Artifact bundle is deterministic and machine-readable, so the same figures can be regenerated during review or revision".to_string(),
        "Synthetic figures are manuscript-ready for methods validation, but public-dataset experiments are still required for a strong biological-results section".to_string(),
        format!(
            "Generated from {} single-end reads across {} single-end scenarios and {} paired-end pairs across {} paired scenarios",
            total_reads,
            cpu_evaluation.total_scenarios,
            total_pairs,
            paired_cpu_evaluation.total_scenarios
        ),
    ];
    notes.push(format!(
        "CPU evaluation backend='{}'; accelerator evaluation backend='{}'; paired accelerator backend='{}'",
        cpu_evaluation.process.backend_used,
        accelerator_evaluation.process.backend_used,
        paired_accelerator_evaluation.process.backend_used
    ));
    notes.push(format!(
        "Largest absolute CPU-versus-accelerator accuracy gap across bundled metrics: {:.4}",
        max_abs_parity_delta(&parity)
    ));
    if let Some(raw_speedup) = accelerator_compare.summary.raw_speedup {
        notes.push(format!(
            "Accelerator reuse-session raw speedup over cold-start: {raw_speedup:.3}x"
        ));
    }
    if let Some(transfer_savings_pct) = accelerator_compare
        .summary
        .steady_state_transfer_savings_pct
    {
        notes.push(format!(
            "Accelerator reuse-session steady-state transfer reduction: {transfer_savings_pct:.1}%"
        ));
    }

    let report = PaperArtifactsReport {
        output_dir: output_dir.display().to_string(),
        requested_accelerator_backend: options.accelerator_backend,
        reads_per_scenario: options.reads_per_scenario.max(1),
        benchmark_rounds: options.benchmark_rounds.max(1),
        cpu_evaluation,
        accelerator_evaluation,
        paired_cpu_evaluation,
        paired_accelerator_evaluation,
        parity,
        cpu_benchmark: cpu_benchmark.summary,
        accelerator_cold_benchmark: accelerator_cold_benchmark.summary,
        accelerator_reuse_benchmark: accelerator_reuse_benchmark.summary,
        accelerator_compare: accelerator_compare.summary,
        artifacts,
        notes,
    };
    write_json_pretty(&summary_json, &report)?;
    Ok(report)
}

pub fn generate_publication_bundle(
    options: &PublicationBundleOptions,
) -> Result<PublicationBundleReport> {
    let output_dir = options.output_dir.clone();
    let data_dir = output_dir.join("data");
    let paper_dir = output_dir.join("paper_artifacts");
    let study_dir = output_dir.join("study_artifacts");
    let download_root = output_dir.join("downloads");
    fs::create_dir_all(&data_dir)
        .with_context(|| format!("failed to create {}", data_dir.display()))?;

    let metadata_path = data_dir.join("public_metadata.tsv");
    let fetch_cache_dir = options
        .cache_dir
        .clone()
        .unwrap_or_else(|| data_dir.join("public_metadata.cache"));
    let fetch = fetch_public_metadata(&StudyFetchMetadataOptions {
        input_path: options.input_path.clone(),
        output_path: metadata_path.clone(),
        base_url: options.base_url.clone(),
        geo_base_url: options.geo_base_url.clone(),
        chunk_size: options.chunk_size,
        cache_dir: Some(fetch_cache_dir),
        retries: options.retries,
        resume_existing: options.resume_existing,
    })?;

    let download = download_public_fastqs(&StudyDownloadOptions {
        input_path: metadata_path.clone(),
        download_root: download_root.clone(),
        retries: options.retries,
        overwrite_existing: options.overwrite_existing_downloads,
    })?;

    let manifest_path = output_dir.join("study_manifest.tsv");
    let manifest = bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: metadata_path.clone(),
        output_path: manifest_path.clone(),
        default_baseline_name: options.default_baseline_name.clone(),
        download_root: download_root.clone(),
    })?;

    let mut final_manifest_path = manifest_path.clone();
    let annotate = if let Some(annotations_path) = options.annotations_path.as_ref() {
        let annotated_path = output_dir.join("study_manifest.annotated.tsv");
        let report = annotate_study_manifest(&StudyAnnotateOptions {
            manifest_path: final_manifest_path.clone(),
            annotations_path: annotations_path.clone(),
            output_path: annotated_path.clone(),
            overwrite_existing: false,
        })?;
        final_manifest_path = annotated_path;
        Some(report)
    } else {
        None
    };

    let ingest = if let Some(results_dir) = options.results_dir.as_ref() {
        let ingested_path = output_dir.join("study_manifest.publication.tsv");
        let report = ingest_study_results(&StudyIngestOptions {
            manifest_path: final_manifest_path.clone(),
            input_dir: results_dir.clone(),
            output_path: ingested_path.clone(),
            recursive: true,
            overwrite_existing: false,
        })?;
        final_manifest_path = ingested_path;
        Some(report)
    } else {
        None
    };

    let study_artifacts = generate_study_artifacts(&StudyArtifactsOptions {
        manifest_path: final_manifest_path.clone(),
        output_dir: study_dir,
        sample_size: options.sample_size,
        batch_reads: options.batch_reads,
        benchmark_rounds: options.benchmark_rounds,
        backend_preference: options.accelerator_backend,
        min_quality_override: options.min_quality_override,
    })?;
    let paper_artifacts = generate_paper_artifacts(&PaperArtifactsOptions {
        output_dir: paper_dir,
        sample_size: options.sample_size,
        batch_reads: options.batch_reads,
        reads_per_scenario: options.reads_per_scenario,
        benchmark_rounds: options.benchmark_rounds,
        accelerator_backend: options.accelerator_backend,
        min_quality_override: options.min_quality_override,
    })?;

    let manuscript_summary_path = output_dir.join("manuscript_summary.txt");
    write_publication_summary_text(
        &manuscript_summary_path,
        &fetch,
        &download,
        &manifest,
        annotate.as_ref(),
        ingest.as_ref(),
        &study_artifacts,
        &paper_artifacts,
    )?;

    let mut notes = vec![
        "Publication bundle orchestrates public-study bootstrap, local FASTQ materialization, synthetic manuscript validation, and manifest-driven case-study benchmarking into one reproducible directory".to_string(),
        format!(
            "Public metadata path: {} | download root: {} | final manifest: {}",
            metadata_path.display(),
            download_root.display(),
            final_manifest_path.display()
        ),
        format!(
            "Synthetic paper artifacts and study artifacts were written under {}",
            output_dir.display()
        ),
    ];
    if annotate.is_some() {
        notes.push("Manifest annotations were merged before study artifact generation".to_string());
    }
    if ingest.is_some() {
        notes.push(
            "Structured native-tool results were ingested before study artifact generation"
                .to_string(),
        );
    }

    let report = PublicationBundleReport {
        input_path: options.input_path.display().to_string(),
        output_dir: output_dir.display().to_string(),
        metadata_path: metadata_path.display().to_string(),
        download_root: download_root.display().to_string(),
        final_manifest_path: final_manifest_path.display().to_string(),
        manuscript_summary_path: manuscript_summary_path.display().to_string(),
        fetch,
        download,
        manifest,
        annotate,
        ingest,
        study_artifacts,
        paper_artifacts,
        notes,
    };
    write_json_pretty(&output_dir.join("publication_bundle.json"), &report)?;
    Ok(report)
}

fn build_backend_parity_summary(
    cpu: &EvaluationReport,
    accelerator: &EvaluationReport,
    paired_cpu: &PairedEvaluationReport,
    paired_accelerator: &PairedEvaluationReport,
) -> BackendParitySummary {
    BackendParitySummary {
        single_end_exact_match_delta: accelerator.aggregate.exact_match_rate
            - cpu.aggregate.exact_match_rate,
        single_end_retention_f1_delta: accelerator.aggregate.retention.f1
            - cpu.aggregate.retention.f1,
        single_end_trimming_f1_delta: accelerator.aggregate.trimming.f1 - cpu.aggregate.trimming.f1,
        single_end_correction_f1_delta: accelerator.aggregate.correction.f1
            - cpu.aggregate.correction.f1,
        paired_end_exact_match_delta: paired_accelerator.aggregate.exact_match_rate
            - paired_cpu.aggregate.exact_match_rate,
        paired_end_retention_f1_delta: paired_accelerator.aggregate.retention.f1
            - paired_cpu.aggregate.retention.f1,
        paired_end_trimming_f1_delta: paired_accelerator.aggregate.trimming.f1
            - paired_cpu.aggregate.trimming.f1,
        paired_end_correction_f1_delta: paired_accelerator.aggregate.correction.f1
            - paired_cpu.aggregate.correction.f1,
    }
}

fn max_abs_parity_delta(parity: &BackendParitySummary) -> f64 {
    [
        parity.single_end_exact_match_delta,
        parity.single_end_retention_f1_delta,
        parity.single_end_trimming_f1_delta,
        parity.single_end_correction_f1_delta,
        parity.paired_end_exact_match_delta,
        parity.paired_end_retention_f1_delta,
        parity.paired_end_trimming_f1_delta,
        parity.paired_end_correction_f1_delta,
    ]
    .into_iter()
    .map(f64::abs)
    .fold(0.0, f64::max)
}

fn parity_chart_values_cpu(
    cpu: &EvaluationReport,
    paired_cpu: &PairedEvaluationReport,
) -> Vec<f64> {
    vec![
        cpu.aggregate.exact_match_rate * 100.0,
        cpu.aggregate.retention.f1 * 100.0,
        cpu.aggregate.trimming.f1 * 100.0,
        cpu.aggregate.correction.f1 * 100.0,
        paired_cpu.aggregate.exact_match_rate * 100.0,
        paired_cpu.aggregate.retention.f1 * 100.0,
        paired_cpu.aggregate.trimming.f1 * 100.0,
        paired_cpu.aggregate.correction.f1 * 100.0,
    ]
}

fn parity_chart_values_accelerator(
    accelerator: &EvaluationReport,
    paired_accelerator: &PairedEvaluationReport,
) -> Vec<f64> {
    vec![
        accelerator.aggregate.exact_match_rate * 100.0,
        accelerator.aggregate.retention.f1 * 100.0,
        accelerator.aggregate.trimming.f1 * 100.0,
        accelerator.aggregate.correction.f1 * 100.0,
        paired_accelerator.aggregate.exact_match_rate * 100.0,
        paired_accelerator.aggregate.retention.f1 * 100.0,
        paired_accelerator.aggregate.trimming.f1 * 100.0,
        paired_accelerator.aggregate.correction.f1 * 100.0,
    ]
}

fn artifact(kind: &str, path: &Path, description: &str) -> PaperArtifactFile {
    PaperArtifactFile {
        kind: kind.to_string(),
        path: path.display().to_string(),
        description: description.to_string(),
    }
}

fn write_json_pretty<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
}

fn write_publication_summary_text(
    path: &Path,
    fetch: &crate::model::StudyFetchMetadataReport,
    download: &crate::model::StudyDownloadReport,
    manifest: &crate::model::StudyManifestBootstrapReport,
    annotate: Option<&crate::model::StudyAnnotateReport>,
    ingest: Option<&crate::model::StudyIngestReport>,
    study: &crate::model::StudyArtifactsReport,
    paper: &PaperArtifactsReport,
) -> Result<()> {
    let mut lines = vec![
        "JapalityECHO publication bundle summary".to_string(),
        "".to_string(),
        "Public-study acquisition".to_string(),
        format!(
            "- Requested {} unique accession(s); fetched {} run-level metadata row(s); unmatched {} accession(s).",
            fetch.summary.unique_accessions,
            fetch.summary.fetched_records,
            fetch.summary.unmatched_accessions
        ),
        format!(
            "- Materialized {} dataset(s) across {} FASTQ file(s): downloaded {}, resumed {}, skipped existing {}, failed {}.",
            download.summary.datasets,
            download.summary.requested_files,
            download.summary.downloaded_files,
            download.summary.resumed_files,
            download.summary.skipped_existing_files,
            download.summary.failed_files
        ),
        format!(
            "- Canonical manifest contains {} dataset(s), including {} paired-end case(s).",
            manifest.summary.datasets, manifest.summary.paired_datasets
        ),
        "".to_string(),
        "Synthetic methods validation".to_string(),
        format!(
            "- Accelerator backend '{}' reached single-end exact-match {:.2}% and paired-end exact-match {:.2}%.",
            paper.requested_accelerator_backend.label(),
            paper.accelerator_evaluation.aggregate.exact_match_rate * 100.0,
            paper
                .paired_accelerator_evaluation
                .aggregate
                .exact_match_rate
                * 100.0
        ),
        format!(
            "- Largest absolute CPU-versus-accelerator parity gap across bundled synthetic metrics was {:.4}.",
            max_abs_parity_delta(&paper.parity)
        ),
        format!(
            "- Accelerator cold-start throughput averaged {:.3} Mbases/s.",
            paper.accelerator_cold_benchmark.average_input_bases_per_sec / 1_000_000.0
        ),
        "".to_string(),
        "Public-study benchmarking".to_string(),
        format!(
            "- Study artifacts cover {} dataset(s) ({} paired-end) with average input throughput {:.3} Mbases/s.",
            study.aggregate.datasets,
            study.aggregate.paired_datasets,
            study.aggregate.average_input_bases_per_sec / 1_000_000.0
        ),
        format!(
            "- Mean trimmed-read fraction {:.3}, discarded-read fraction {:.3}, corrected bases per Mbase {:.3}.",
            study.aggregate.average_trimmed_read_fraction,
            study.aggregate.average_discarded_read_fraction,
            study.aggregate.average_corrected_bases_per_mbase
        ),
    ];
    if let Some(platform_accuracy) = study.detection.platform_accuracy {
        lines.push(format!(
            "- Zero-config platform detection accuracy on labeled study rows: {:.2}%.",
            platform_accuracy * 100.0
        ));
    }
    if let Some(experiment_accuracy) = study.detection.experiment_accuracy {
        lines.push(format!(
            "- Zero-config experiment detection accuracy on labeled study rows: {:.2}%.",
            experiment_accuracy * 100.0
        ));
    }
    if let Some(speedup) = study.comparison.average_input_speedup_vs_baseline {
        lines.push(format!(
            "- Average study-level throughput speedup versus supplied baselines: {:.3}x.",
            speedup
        ));
    }
    if let Some(delta) = study.comparison.average_alignment_rate_delta {
        lines.push(format!(
            "- Average alignment-rate delta versus supplied baselines: {delta:.4}."
        ));
    }
    if let Some(delta) = study.comparison.average_variant_f1_delta {
        lines.push(format!(
            "- Average variant-F1 delta versus supplied baselines: {delta:.4}."
        ));
    }
    if let Some(ratio) = study.comparison.average_assembly_n50_ratio {
        lines.push(format!(
            "- Average assembly N50 ratio versus supplied baselines: {ratio:.3}x."
        ));
    }
    if let Some(annotate) = annotate {
        lines.push("".to_string());
        lines.push("Manifest annotation merge".to_string());
        lines.push(format!(
            "- Annotation merge changed {} dataset(s) and filled {} field(s).",
            annotate.summary.datasets_changed, annotate.summary.fields_filled
        ));
    }
    if let Some(ingest) = ingest {
        lines.push("".to_string());
        lines.push("Structured result ingest".to_string());
        lines.push(format!(
            "- Ingest scanned {} file(s) and changed {} dataset(s).",
            ingest.summary.files_scanned, ingest.summary.datasets_changed
        ));
    }
    lines.push("".to_string());
    lines.push("Key output paths".to_string());
    lines.push(format!("- Fetch report status CSV: {}", fetch.status_path));
    lines.push(format!("- Download status CSV: {}", download.status_path));
    lines.push(format!("- Study artifact bundle: {}", study.output_dir));
    lines.push(format!(
        "- Synthetic paper artifact bundle: {}",
        paper.output_dir
    ));
    fs::write(path, lines.join("\n")).with_context(|| format!("failed to write {}", path.display()))
}

fn write_scenario_accuracy_csv(
    path: &Path,
    cpu: &EvaluationReport,
    accelerator: &EvaluationReport,
) -> Result<()> {
    let mut csv = String::from(
        "backend,scenario,exact_match_rate,retention_f1,trimming_f1,correction_f1,retention_precision,retention_recall,trimming_precision,trimming_recall,correction_precision,correction_recall\n",
    );
    append_scenario_csv_rows(&mut csv, "cpu", cpu);
    append_scenario_csv_rows(&mut csv, accelerator.requested_backend.label(), accelerator);
    fs::write(path, csv).with_context(|| format!("failed to write {}", path.display()))
}

fn append_scenario_csv_rows(csv: &mut String, backend: &str, report: &EvaluationReport) {
    for scenario in &report.scenarios {
        csv.push_str(&format!(
            "{backend},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            scenario.name,
            scenario.exact_match_rate,
            scenario.retention.f1,
            scenario.trimming.f1,
            scenario.correction.f1,
            scenario.retention.precision,
            scenario.retention.recall,
            scenario.trimming.precision,
            scenario.trimming.recall,
            scenario.correction.precision,
            scenario.correction.recall
        ));
    }
}

fn write_paired_scenario_accuracy_csv(
    path: &Path,
    cpu: &PairedEvaluationReport,
    accelerator: &PairedEvaluationReport,
) -> Result<()> {
    let mut csv = String::from(
        "backend,scenario,exact_match_rate,retention_f1,trimming_f1,correction_f1,retention_precision,retention_recall,trimming_precision,trimming_recall,correction_precision,correction_recall\n",
    );
    append_paired_scenario_csv_rows(&mut csv, "cpu", cpu);
    append_paired_scenario_csv_rows(&mut csv, accelerator.requested_backend.label(), accelerator);
    fs::write(path, csv).with_context(|| format!("failed to write {}", path.display()))
}

fn append_paired_scenario_csv_rows(
    csv: &mut String,
    backend: &str,
    report: &PairedEvaluationReport,
) {
    for scenario in &report.scenarios {
        csv.push_str(&format!(
            "{backend},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            scenario.name,
            scenario.exact_match_rate,
            scenario.retention.f1,
            scenario.trimming.f1,
            scenario.correction.f1,
            scenario.retention.precision,
            scenario.retention.recall,
            scenario.trimming.precision,
            scenario.trimming.recall,
            scenario.correction.precision,
            scenario.correction.recall
        ));
    }
}

fn write_aggregate_accuracy_csv(
    path: &Path,
    cpu: &EvaluationReport,
    accelerator: &EvaluationReport,
) -> Result<()> {
    let mut csv = String::from("backend,metric,value\n");
    append_aggregate_csv_rows(&mut csv, "cpu", &cpu.aggregate);
    append_aggregate_csv_rows(
        &mut csv,
        accelerator.requested_backend.label(),
        &accelerator.aggregate,
    );
    fs::write(path, csv).with_context(|| format!("failed to write {}", path.display()))
}

fn append_aggregate_csv_rows(
    csv: &mut String,
    backend: &str,
    aggregate: &EvaluationAggregateSummary,
) {
    for (metric, value) in [
        ("exact_match_rate", aggregate.exact_match_rate),
        ("retention_f1", aggregate.retention.f1),
        ("trimming_f1", aggregate.trimming.f1),
        ("correction_f1", aggregate.correction.f1),
    ] {
        csv.push_str(&format!("{backend},{metric},{value:.6}\n"));
    }
}

fn write_paired_aggregate_accuracy_csv(
    path: &Path,
    cpu: &PairedEvaluationReport,
    accelerator: &PairedEvaluationReport,
) -> Result<()> {
    let mut csv = String::from("backend,metric,value\n");
    append_paired_aggregate_csv_rows(&mut csv, "cpu", &cpu.aggregate);
    append_paired_aggregate_csv_rows(
        &mut csv,
        accelerator.requested_backend.label(),
        &accelerator.aggregate,
    );
    fs::write(path, csv).with_context(|| format!("failed to write {}", path.display()))
}

fn append_paired_aggregate_csv_rows(
    csv: &mut String,
    backend: &str,
    aggregate: &PairedEvaluationAggregateSummary,
) {
    for (metric, value) in [
        ("exact_match_rate", aggregate.exact_match_rate),
        ("retention_f1", aggregate.retention.f1),
        ("trimming_f1", aggregate.trimming.f1),
        ("correction_f1", aggregate.correction.f1),
    ] {
        csv.push_str(&format!("{backend},{metric},{value:.6}\n"));
    }
}

fn write_backend_throughput_csv(
    path: &Path,
    cpu: &BenchmarkSummary,
    accelerator_cold: &BenchmarkSummary,
    accelerator_reuse: &BenchmarkSummary,
    accelerator_backend: BackendPreference,
) -> Result<()> {
    let mut csv = String::from(
        "run,session_mode,requested_backend,backend_used,average_wall_clock_us,average_input_reads_per_sec,average_input_bases_per_sec,average_output_reads_per_sec,average_output_bases_per_sec\n",
    );
    append_benchmark_csv_row(&mut csv, "cpu_cold", BackendPreference::Cpu, cpu);
    append_benchmark_csv_row(
        &mut csv,
        "accelerator_cold",
        accelerator_backend,
        accelerator_cold,
    );
    append_benchmark_csv_row(
        &mut csv,
        "accelerator_reuse",
        accelerator_backend,
        accelerator_reuse,
    );
    fs::write(path, csv).with_context(|| format!("failed to write {}", path.display()))
}

fn append_benchmark_csv_row(
    csv: &mut String,
    run: &str,
    requested_backend: BackendPreference,
    summary: &BenchmarkSummary,
) {
    csv.push_str(&format!(
        "{run},{},{},{},{:.3},{:.6},{:.6},{:.6},{:.6}\n",
        summary.session_mode,
        requested_backend,
        summary.backends.join("+"),
        summary.average_wall_clock_us,
        summary.average_input_reads_per_sec,
        summary.average_input_bases_per_sec,
        summary.average_output_reads_per_sec,
        summary.average_output_bases_per_sec
    ));
}

fn write_transfer_efficiency_csv(
    path: &Path,
    accelerator_cold: &BenchmarkSummary,
    accelerator_reuse: &BenchmarkSummary,
    accelerator_compare: &BenchmarkComparisonSummary,
) -> Result<()> {
    let csv = format!(
        "run,warmup_transfer_bytes,steady_state_average_transfer_bytes,raw_speedup,amortized_speedup,steady_state_transfer_savings_pct\n\
cold,{},{:.6},,,\n\
reuse,{},{:.6},{},{},{}\n",
        accelerator_cold.warmup_transfer_bytes,
        accelerator_cold
            .steady_state_average_transfer_bytes
            .unwrap_or(0.0),
        accelerator_reuse.warmup_transfer_bytes,
        accelerator_reuse
            .steady_state_average_transfer_bytes
            .unwrap_or(0.0),
        option_csv(accelerator_compare.raw_speedup),
        option_csv(accelerator_compare.amortized_speedup),
        option_csv(accelerator_compare.steady_state_transfer_savings_pct),
    );
    fs::write(path, csv).with_context(|| format!("failed to write {}", path.display()))
}

fn write_backend_parity_csv(
    path: &Path,
    cpu: &EvaluationReport,
    accelerator: &EvaluationReport,
    paired_cpu: &PairedEvaluationReport,
    paired_accelerator: &PairedEvaluationReport,
    accelerator_label: &str,
) -> Result<()> {
    let mut csv = String::from(
        "suite,metric,accelerator_backend,cpu_value,accelerator_value,delta,absolute_delta\n",
    );
    append_parity_csv_row(
        &mut csv,
        "single_end",
        "exact_match_rate",
        accelerator_label,
        cpu.aggregate.exact_match_rate,
        accelerator.aggregate.exact_match_rate,
    );
    append_parity_csv_row(
        &mut csv,
        "single_end",
        "retention_f1",
        accelerator_label,
        cpu.aggregate.retention.f1,
        accelerator.aggregate.retention.f1,
    );
    append_parity_csv_row(
        &mut csv,
        "single_end",
        "trimming_f1",
        accelerator_label,
        cpu.aggregate.trimming.f1,
        accelerator.aggregate.trimming.f1,
    );
    append_parity_csv_row(
        &mut csv,
        "single_end",
        "correction_f1",
        accelerator_label,
        cpu.aggregate.correction.f1,
        accelerator.aggregate.correction.f1,
    );
    append_parity_csv_row(
        &mut csv,
        "paired_end",
        "exact_match_rate",
        accelerator_label,
        paired_cpu.aggregate.exact_match_rate,
        paired_accelerator.aggregate.exact_match_rate,
    );
    append_parity_csv_row(
        &mut csv,
        "paired_end",
        "retention_f1",
        accelerator_label,
        paired_cpu.aggregate.retention.f1,
        paired_accelerator.aggregate.retention.f1,
    );
    append_parity_csv_row(
        &mut csv,
        "paired_end",
        "trimming_f1",
        accelerator_label,
        paired_cpu.aggregate.trimming.f1,
        paired_accelerator.aggregate.trimming.f1,
    );
    append_parity_csv_row(
        &mut csv,
        "paired_end",
        "correction_f1",
        accelerator_label,
        paired_cpu.aggregate.correction.f1,
        paired_accelerator.aggregate.correction.f1,
    );
    fs::write(path, csv).with_context(|| format!("failed to write {}", path.display()))
}

fn append_parity_csv_row(
    csv: &mut String,
    suite: &str,
    metric: &str,
    accelerator_backend: &str,
    cpu_value: f64,
    accelerator_value: f64,
) {
    let delta = accelerator_value - cpu_value;
    csv.push_str(&format!(
        "{suite},{metric},{accelerator_backend},{cpu_value:.6},{accelerator_value:.6},{delta:.6},{:.6}\n",
        delta.abs()
    ));
}

fn option_csv(value: Option<f64>) -> String {
    value.map(|value| format!("{value:.6}")).unwrap_or_default()
}

fn write_svg(path: &Path, svg: &str) -> Result<()> {
    let mut file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    file.write_all(svg.as_bytes())
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

struct ChartSeries<'a> {
    name: &'a str,
    color: &'a str,
    values: Vec<f64>,
}

fn grouped_bar_chart_svg(
    title: &str,
    y_label: &str,
    categories: &[String],
    series: &[ChartSeries<'_>],
) -> String {
    let values = series
        .iter()
        .flat_map(|series| series.values.iter().copied())
        .collect::<Vec<_>>();
    let max_value = values.iter().copied().fold(0.0_f64, f64::max).max(1.0) * 1.1;
    let width = 960.0;
    let height = 560.0;
    let margin_left = 70.0;
    let margin_right = 30.0;
    let margin_top = 70.0;
    let margin_bottom = 120.0;
    let plot_width = width - margin_left - margin_right;
    let plot_height = height - margin_top - margin_bottom;
    let group_width = plot_width / categories.len().max(1) as f64;
    let bar_width = (group_width * 0.75) / series.len().max(1) as f64;

    let mut svg = svg_header(width, height, title);
    svg.push_str(&format!(
        "<text x=\"{}\" y=\"30\" font-size=\"24\" font-family=\"Arial\" font-weight=\"bold\" text-anchor=\"middle\">{}</text>",
        width / 2.0,
        escape_xml(title)
    ));
    svg.push_str(&format!(
        "<text x=\"20\" y=\"{}\" font-size=\"16\" font-family=\"Arial\" transform=\"rotate(-90 20,{})\">{}</text>",
        height / 2.0,
        height / 2.0,
        escape_xml(y_label)
    ));
    svg.push_str(&format!(
        "<line x1=\"{margin_left}\" y1=\"{margin_top}\" x2=\"{margin_left}\" y2=\"{}\" stroke=\"#333\" stroke-width=\"1.5\" />",
        margin_top + plot_height
    ));
    svg.push_str(&format!(
        "<line x1=\"{margin_left}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#333\" stroke-width=\"1.5\" />",
        margin_top + plot_height,
        margin_left + plot_width,
        margin_top + plot_height
    ));

    for tick in 0..=5 {
        let tick_value = max_value * tick as f64 / 5.0;
        let y = margin_top + plot_height - (tick_value / max_value) * plot_height;
        svg.push_str(&format!(
            "<line x1=\"{margin_left}\" y1=\"{y:.2}\" x2=\"{}\" y2=\"{y:.2}\" stroke=\"#ddd\" stroke-width=\"1\" />",
            margin_left + plot_width
        ));
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"{:.2}\" font-size=\"12\" font-family=\"Arial\" text-anchor=\"end\">{:.1}</text>",
            margin_left - 8.0,
            y + 4.0,
            tick_value
        ));
    }

    for (group_index, category) in categories.iter().enumerate() {
        let group_x = margin_left + group_index as f64 * group_width;
        for (series_index, series) in series.iter().enumerate() {
            let value = series.values.get(group_index).copied().unwrap_or(0.0);
            let bar_height = (value / max_value) * plot_height;
            let x = group_x + group_width * 0.125 + series_index as f64 * bar_width;
            let y = margin_top + plot_height - bar_height;
            svg.push_str(&format!(
                "<rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{bar_width:.2}\" height=\"{bar_height:.2}\" fill=\"{}\" />",
                series.color
            ));
            svg.push_str(&format!(
                "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"11\" font-family=\"Arial\" text-anchor=\"middle\">{:.1}</text>",
                x + bar_width / 2.0,
                y - 6.0,
                value
            ));
        }
        svg.push_str(&format!(
            "<text x=\"{:.2}\" y=\"{}\" font-size=\"12\" font-family=\"Arial\" text-anchor=\"middle\">{}</text>",
            group_x + group_width / 2.0,
            margin_top + plot_height + 20.0,
            escape_xml(category)
        ));
    }

    for (legend_index, series) in series.iter().enumerate() {
        let legend_x = margin_left + legend_index as f64 * 180.0;
        svg.push_str(&format!(
            "<rect x=\"{legend_x:.2}\" y=\"40\" width=\"16\" height=\"16\" fill=\"{}\" />",
            series.color
        ));
        svg.push_str(&format!(
            "<text x=\"{:.2}\" y=\"53\" font-size=\"13\" font-family=\"Arial\">{}</text>",
            legend_x + 24.0,
            escape_xml(series.name)
        ));
    }

    svg.push_str("</svg>");
    svg
}

fn single_series_bar_chart_svg(
    title: &str,
    y_label: &str,
    categories: &[String],
    values: &[f64],
    color: &str,
) -> String {
    grouped_bar_chart_svg(
        title,
        y_label,
        categories,
        &[ChartSeries {
            name: y_label,
            color,
            values: values.to_vec(),
        }],
    )
}

fn svg_header(width: f64, height: f64, title: &str) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\"><title>{}</title><rect width=\"100%\" height=\"100%\" fill=\"white\" />",
        escape_xml(title)
    )
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
