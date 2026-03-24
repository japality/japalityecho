use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use japalityecho::{
    BackendPreference, BenchmarkComparisonOptions, BenchmarkComparisonReport, BenchmarkOptions,
    BenchmarkReport, BenchmarkSessionMode, EvaluationMetricSummary, EvaluationOptions,
    EvaluationReport, HistoryRegressionStatus, HistoryReport, HistoryReportOptions, InspectReport,
    PaperArtifactsOptions, PaperArtifactsReport, ProcessOptions, ProcessReport,
    PublicationBundleOptions, PublicationBundleReport, StudyAnnotateOptions, StudyAnnotateReport,
    StudyArtifactsOptions, StudyArtifactsReport, StudyDiscoverOptions, StudyDownloadOptions,
    StudyDownloadReport, StudyFetchMetadataOptions, StudyFetchMetadataReport, StudyIngestOptions,
    StudyIngestReport, StudyManifestBootstrapReport, StudyManifestOptions, annotate_study_manifest,
    append_benchmark_comparison_history, append_benchmark_history, benchmark_compare_files,
    benchmark_files, bootstrap_study_manifest, discover_study_inventory, download_public_fastqs,
    evaluate_synthetic_truth, fetch_public_metadata, generate_paper_artifacts,
    generate_publication_bundle, generate_study_artifacts, ingest_study_results,
    inspect_inputs_with_overrides, process_files, read_history_report,
};

const HISTORY_REPORT_FAIL_EXIT_CODE: i32 = 2;

#[derive(Parser, Debug)]
#[command(
    name = "JapalityECHO",
    version,
    about = "Zero-config, context-aware NGS preprocessing with a heterogeneous execution plan"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Detect {
        input: PathBuf,
        #[arg(long)]
        input2: Option<PathBuf>,
        #[arg(long, default_value_t = 100_000)]
        sample_size: usize,
        #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
        backend: BackendArg,
        #[arg(long, help = "Override auto-detected platform (illumina, mgi, ont, pacbio)")]
        platform: Option<String>,
        #[arg(long, help = "Override auto-detected experiment type (wgs, rnaseq, 10xv3, 10xv2, atacseq, longread)")]
        experiment: Option<String>,
        #[arg(long)]
        json: bool,
    },
    Process {
        input: PathBuf,
        output: PathBuf,
        #[arg(long)]
        input2: Option<PathBuf>,
        #[arg(long)]
        output2: Option<PathBuf>,
        #[arg(long, default_value_t = 100_000)]
        sample_size: usize,
        #[arg(long, default_value_t = 50_000)]
        batch_reads: usize,
        #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
        backend: BackendArg,
        #[arg(long)]
        adapter: Option<String>,
        #[arg(long)]
        min_quality: Option<u8>,
        #[arg(long)]
        kmer_size: Option<usize>,
        #[arg(long, help = "Override auto-detected platform (illumina, mgi, ont, pacbio)")]
        platform: Option<String>,
        #[arg(long, help = "Override auto-detected experiment type (wgs, rnaseq, 10xv3, 10xv2, atacseq, longread)")]
        experiment: Option<String>,
        #[arg(long)]
        json: bool,
    },
    Benchmark {
        input: PathBuf,
        #[arg(long)]
        input2: Option<PathBuf>,
        #[arg(long, default_value_t = 100_000)]
        sample_size: usize,
        #[arg(long, default_value_t = 50_000)]
        batch_reads: usize,
        #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
        backend: BackendArg,
        #[arg(long)]
        adapter: Option<String>,
        #[arg(long)]
        min_quality: Option<u8>,
        #[arg(long)]
        kmer_size: Option<usize>,
        #[arg(long, default_value_t = 3)]
        rounds: usize,
        #[arg(long, value_enum, default_value_t = BenchmarkSessionModeArg::ColdStart)]
        session_mode: BenchmarkSessionModeArg,
        #[arg(long)]
        label: Option<String>,
        #[arg(long)]
        history: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    BenchmarkCompare {
        input: PathBuf,
        #[arg(long)]
        input2: Option<PathBuf>,
        #[arg(long, default_value_t = 100_000)]
        sample_size: usize,
        #[arg(long, default_value_t = 50_000)]
        batch_reads: usize,
        #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
        backend: BackendArg,
        #[arg(long)]
        adapter: Option<String>,
        #[arg(long)]
        min_quality: Option<u8>,
        #[arg(long)]
        kmer_size: Option<usize>,
        #[arg(long, default_value_t = 3)]
        rounds: usize,
        #[arg(long)]
        label: Option<String>,
        #[arg(long)]
        history: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    Evaluate {
        #[arg(long, default_value_t = 100_000)]
        sample_size: usize,
        #[arg(long, default_value_t = 32)]
        batch_reads: usize,
        #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
        backend: BackendArg,
        #[arg(long)]
        min_quality: Option<u8>,
        #[arg(long, default_value_t = 12)]
        reads_per_scenario: usize,
        #[arg(long)]
        json: bool,
    },
    PaperArtifacts {
        output_dir: PathBuf,
        #[arg(long, default_value_t = 100_000)]
        sample_size: usize,
        #[arg(long, default_value_t = 32)]
        batch_reads: usize,
        #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
        backend: BackendArg,
        #[arg(long)]
        min_quality: Option<u8>,
        #[arg(long, default_value_t = 12)]
        reads_per_scenario: usize,
        #[arg(long, default_value_t = 3)]
        benchmark_rounds: usize,
        #[arg(long)]
        json: bool,
    },
    PublicationBundle {
        input: PathBuf,
        output_dir: PathBuf,
        #[arg(long)]
        annotations: Option<PathBuf>,
        #[arg(long)]
        results_dir: Option<PathBuf>,
        #[arg(long)]
        default_baseline_name: Option<String>,
        #[arg(long, default_value_t = 100_000)]
        sample_size: usize,
        #[arg(long, default_value_t = 32)]
        batch_reads: usize,
        #[arg(long, default_value_t = 12)]
        reads_per_scenario: usize,
        #[arg(long, default_value_t = 3)]
        benchmark_rounds: usize,
        #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
        backend: BackendArg,
        #[arg(long)]
        min_quality: Option<u8>,
        #[arg(
            long,
            default_value = "https://www.ebi.ac.uk/ena/portal/api/filereport"
        )]
        base_url: String,
        #[arg(long, default_value = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi")]
        geo_base_url: String,
        #[arg(long, default_value_t = 200)]
        chunk_size: usize,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(long, default_value_t = 2)]
        retries: usize,
        #[arg(long)]
        resume: bool,
        #[arg(long)]
        overwrite_existing_downloads: bool,
        #[arg(long)]
        json: bool,
    },
    StudyArtifacts {
        manifest: PathBuf,
        output_dir: PathBuf,
        #[arg(long, default_value_t = 100_000)]
        sample_size: usize,
        #[arg(long, default_value_t = 32)]
        batch_reads: usize,
        #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
        backend: BackendArg,
        #[arg(long)]
        min_quality: Option<u8>,
        #[arg(long, default_value_t = 3)]
        benchmark_rounds: usize,
        #[arg(long)]
        json: bool,
    },
    StudyManifest {
        inventory: PathBuf,
        output_manifest: PathBuf,
        #[arg(long)]
        default_baseline_name: Option<String>,
        #[arg(long, default_value = "downloads")]
        download_root: PathBuf,
        #[arg(long)]
        json: bool,
    },
    StudyFetchMetadata {
        input: PathBuf,
        output_metadata: PathBuf,
        #[arg(
            long,
            default_value = "https://www.ebi.ac.uk/ena/portal/api/filereport"
        )]
        base_url: String,
        #[arg(long, default_value = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi")]
        geo_base_url: String,
        #[arg(long, default_value_t = 200)]
        chunk_size: usize,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(long, default_value_t = 2)]
        retries: usize,
        #[arg(long)]
        resume: bool,
        #[arg(long)]
        json: bool,
    },
    StudyDownload {
        input: PathBuf,
        download_root: PathBuf,
        #[arg(long, default_value_t = 2)]
        retries: usize,
        #[arg(long)]
        overwrite_existing: bool,
        #[arg(long)]
        json: bool,
    },
    StudyDiscover {
        input_dir: PathBuf,
        output_inventory: PathBuf,
        #[arg(long)]
        no_recursive: bool,
        #[arg(long)]
        json: bool,
    },
    StudyAnnotate {
        manifest: PathBuf,
        annotations: PathBuf,
        output_manifest: PathBuf,
        #[arg(long)]
        overwrite_existing: bool,
        #[arg(long)]
        json: bool,
    },
    StudyIngest {
        manifest: PathBuf,
        input_dir: PathBuf,
        output_manifest: PathBuf,
        #[arg(long)]
        no_recursive: bool,
        #[arg(long)]
        overwrite_existing: bool,
        #[arg(long)]
        json: bool,
    },
    HistoryReport {
        history: PathBuf,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long, conflicts_with = "baseline_latest_pass")]
        baseline_label: Option<String>,
        #[arg(long, conflicts_with = "baseline_label")]
        baseline_label_prefix: Option<String>,
        #[arg(long, conflicts_with = "baseline_label")]
        baseline_latest_pass: bool,
        #[arg(long)]
        max_wall_clock_regression_pct: Option<f64>,
        #[arg(long)]
        min_raw_speedup: Option<f64>,
        #[arg(long)]
        min_transfer_savings_pct: Option<f64>,
        #[arg(long, value_enum)]
        fail_on_status: Vec<HistoryFailStatusArg>,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendArg {
    Auto,
    Cpu,
    Cuda,
    Hip,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BenchmarkSessionModeArg {
    ColdStart,
    ReuseSession,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum HistoryFailStatusArg {
    Pass,
    Alert,
    NoBaseline,
    NotComparable,
}

impl From<BackendArg> for BackendPreference {
    fn from(value: BackendArg) -> Self {
        match value {
            BackendArg::Auto => BackendPreference::Auto,
            BackendArg::Cpu => BackendPreference::Cpu,
            BackendArg::Cuda => BackendPreference::Cuda,
            BackendArg::Hip => BackendPreference::Hip,
        }
    }
}

impl From<BenchmarkSessionModeArg> for BenchmarkSessionMode {
    fn from(value: BenchmarkSessionModeArg) -> Self {
        match value {
            BenchmarkSessionModeArg::ColdStart => BenchmarkSessionMode::ColdStart,
            BenchmarkSessionModeArg::ReuseSession => BenchmarkSessionMode::ReuseSession,
        }
    }
}

impl From<HistoryFailStatusArg> for HistoryRegressionStatus {
    fn from(value: HistoryFailStatusArg) -> Self {
        match value {
            HistoryFailStatusArg::Pass => HistoryRegressionStatus::Pass,
            HistoryFailStatusArg::Alert => HistoryRegressionStatus::Alert,
            HistoryFailStatusArg::NoBaseline => HistoryRegressionStatus::NoBaseline,
            HistoryFailStatusArg::NotComparable => HistoryRegressionStatus::NotComparable,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Detect {
            input,
            input2,
            sample_size,
            backend,
            platform,
            experiment,
            json,
        } => {
            let forced_platform = platform
                .as_deref()
                .map(|s| s.parse::<japalityecho::Platform>())
                .transpose()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let forced_experiment = experiment
                .as_deref()
                .map(|s| s.parse::<japalityecho::ExperimentType>())
                .transpose()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let report = inspect_inputs_with_overrides(
                &input,
                input2.as_deref(),
                sample_size,
                backend.into(),
                forced_platform,
                forced_experiment,
            )?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_detect_summary(&input, input2.as_ref(), &report);
            }
        }
        Command::Process {
            input,
            output,
            input2,
            output2,
            sample_size,
            batch_reads,
            backend,
            adapter,
            min_quality,
            kmer_size,
            platform,
            experiment,
            json,
        } => {
            let forced_platform = platform
                .as_deref()
                .map(|s| s.parse::<japalityecho::Platform>())
                .transpose()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let forced_experiment = experiment
                .as_deref()
                .map(|s| s.parse::<japalityecho::ExperimentType>())
                .transpose()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let report = process_files(
                &input,
                &output,
                input2.as_deref(),
                output2.as_deref(),
                &ProcessOptions {
                    sample_size,
                    batch_reads,
                    backend_preference: backend.into(),
                    forced_adapter: adapter,
                    min_quality_override: min_quality,
                    kmer_size_override: kmer_size,
                    forced_platform,
                    forced_experiment,
                },
            )?;

            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_process_summary(&output, output2.as_ref(), &report);
            }
        }
        Command::Benchmark {
            input,
            input2,
            sample_size,
            batch_reads,
            backend,
            adapter,
            min_quality,
            kmer_size,
            rounds,
            session_mode,
            label,
            history,
            json,
        } => {
            let options = BenchmarkOptions {
                process: ProcessOptions {
                    sample_size,
                    batch_reads,
                    backend_preference: backend.into(),
                    forced_adapter: adapter.clone(),
                    min_quality_override: min_quality,
                    kmer_size_override: kmer_size,
                    forced_platform: None,
                    forced_experiment: None,
                },
                rounds,
                session_mode: session_mode.into(),
            };
            let mut report = benchmark_files(&input, input2.as_deref(), &options)?;
            if let Some(history) = history.as_deref() {
                append_benchmark_history(
                    history,
                    label.as_deref(),
                    &input,
                    input2.as_deref(),
                    &options,
                    &mut report,
                )?;
            }

            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_benchmark_summary(&input, input2.as_ref(), &report);
            }
        }
        Command::BenchmarkCompare {
            input,
            input2,
            sample_size,
            batch_reads,
            backend,
            adapter,
            min_quality,
            kmer_size,
            rounds,
            label,
            history,
            json,
        } => {
            let options = BenchmarkComparisonOptions {
                process: ProcessOptions {
                    sample_size,
                    batch_reads,
                    backend_preference: backend.into(),
                    forced_adapter: adapter.clone(),
                    min_quality_override: min_quality,
                    kmer_size_override: kmer_size,
                    forced_platform: None,
                    forced_experiment: None,
                },
                rounds,
            };
            let mut report = benchmark_compare_files(&input, input2.as_deref(), &options)?;
            if let Some(history) = history.as_deref() {
                append_benchmark_comparison_history(
                    history,
                    label.as_deref(),
                    &input,
                    input2.as_deref(),
                    &options,
                    &mut report,
                )?;
            }

            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_benchmark_comparison_summary(&input, input2.as_ref(), &report);
            }
        }
        Command::Evaluate {
            sample_size,
            batch_reads,
            backend,
            min_quality,
            reads_per_scenario,
            json,
        } => {
            let report = evaluate_synthetic_truth(&EvaluationOptions {
                sample_size,
                batch_reads,
                backend_preference: backend.into(),
                min_quality_override: min_quality,
                reads_per_scenario,
            })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_evaluation_report(&report);
            }
        }
        Command::PaperArtifacts {
            output_dir,
            sample_size,
            batch_reads,
            backend,
            min_quality,
            reads_per_scenario,
            benchmark_rounds,
            json,
        } => {
            let report = generate_paper_artifacts(&PaperArtifactsOptions {
                output_dir: output_dir.clone(),
                sample_size,
                batch_reads,
                reads_per_scenario,
                benchmark_rounds,
                accelerator_backend: backend.into(),
                min_quality_override: min_quality,
            })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_paper_artifacts_report(&report);
            }
        }
        Command::PublicationBundle {
            input,
            output_dir,
            annotations,
            results_dir,
            default_baseline_name,
            sample_size,
            batch_reads,
            reads_per_scenario,
            benchmark_rounds,
            backend,
            min_quality,
            base_url,
            geo_base_url,
            chunk_size,
            cache_dir,
            retries,
            resume,
            overwrite_existing_downloads,
            json,
        } => {
            let report = generate_publication_bundle(&PublicationBundleOptions {
                input_path: input,
                output_dir,
                annotations_path: annotations,
                results_dir,
                default_baseline_name,
                sample_size,
                batch_reads,
                reads_per_scenario,
                benchmark_rounds,
                accelerator_backend: backend.into(),
                min_quality_override: min_quality,
                base_url,
                geo_base_url,
                chunk_size,
                cache_dir,
                retries,
                resume_existing: resume,
                overwrite_existing_downloads,
            })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_publication_bundle_report(&report);
            }
        }
        Command::StudyArtifacts {
            manifest,
            output_dir,
            sample_size,
            batch_reads,
            backend,
            min_quality,
            benchmark_rounds,
            json,
        } => {
            let report = generate_study_artifacts(&StudyArtifactsOptions {
                manifest_path: manifest,
                output_dir,
                sample_size,
                batch_reads,
                benchmark_rounds,
                backend_preference: backend.into(),
                min_quality_override: min_quality,
            })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_study_artifacts_report(&report);
            }
        }
        Command::StudyManifest {
            inventory,
            output_manifest,
            default_baseline_name,
            download_root,
            json,
        } => {
            let report = bootstrap_study_manifest(&StudyManifestOptions {
                inventory_path: inventory,
                output_path: output_manifest,
                default_baseline_name,
                download_root,
            })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_study_manifest_report(&report);
            }
        }
        Command::StudyFetchMetadata {
            input,
            output_metadata,
            base_url,
            geo_base_url,
            chunk_size,
            cache_dir,
            retries,
            resume,
            json,
        } => {
            let report = fetch_public_metadata(&StudyFetchMetadataOptions {
                input_path: input,
                output_path: output_metadata,
                base_url,
                geo_base_url,
                chunk_size,
                cache_dir,
                retries,
                resume_existing: resume,
            })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_study_fetch_metadata_report(&report);
            }
        }
        Command::StudyDownload {
            input,
            download_root,
            retries,
            overwrite_existing,
            json,
        } => {
            let report = download_public_fastqs(&StudyDownloadOptions {
                input_path: input,
                download_root,
                retries,
                overwrite_existing,
            })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_study_download_report(&report);
            }
        }
        Command::StudyDiscover {
            input_dir,
            output_inventory,
            no_recursive,
            json,
        } => {
            let report = discover_study_inventory(&StudyDiscoverOptions {
                input_dir,
                output_path: output_inventory,
                recursive: !no_recursive,
            })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_study_discover_report(&report);
            }
        }
        Command::StudyAnnotate {
            manifest,
            annotations,
            output_manifest,
            overwrite_existing,
            json,
        } => {
            let report = annotate_study_manifest(&StudyAnnotateOptions {
                manifest_path: manifest,
                annotations_path: annotations,
                output_path: output_manifest,
                overwrite_existing,
            })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_study_annotate_report(&report);
            }
        }
        Command::StudyIngest {
            manifest,
            input_dir,
            output_manifest,
            no_recursive,
            overwrite_existing,
            json,
        } => {
            let report = ingest_study_results(&StudyIngestOptions {
                manifest_path: manifest,
                input_dir,
                output_path: output_manifest,
                recursive: !no_recursive,
                overwrite_existing,
            })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_study_ingest_report(&report);
            }
        }
        Command::HistoryReport {
            history,
            limit,
            baseline_label,
            baseline_label_prefix,
            baseline_latest_pass,
            max_wall_clock_regression_pct,
            min_raw_speedup,
            min_transfer_savings_pct,
            fail_on_status,
            json,
        } => {
            let report = read_history_report(
                &history,
                &HistoryReportOptions {
                    limit,
                    baseline_label,
                    baseline_label_prefix,
                    baseline_latest_pass,
                    max_wall_clock_regression_pct,
                    min_raw_speedup,
                    min_transfer_savings_pct,
                },
            )?;
            let matched_fail_status = history_report_fail_status(&report, &fail_on_status);
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_history_report(&report);
                if let Some(status) = matched_fail_status {
                    println!(
                        "  exit policy: matched regression status {status}; exiting with code {HISTORY_REPORT_FAIL_EXIT_CODE}"
                    );
                }
            }
            if matched_fail_status.is_some() {
                std::process::exit(HISTORY_REPORT_FAIL_EXIT_CODE);
            }
        }
    }

    Ok(())
}

fn history_report_fail_status(
    report: &HistoryReport,
    fail_on_status: &[HistoryFailStatusArg],
) -> Option<HistoryRegressionStatus> {
    let regression = report.regression.as_ref()?;
    fail_on_status
        .iter()
        .copied()
        .map(HistoryRegressionStatus::from)
        .find(|status| *status == regression.status)
}

fn print_detect_summary(input: &PathBuf, input2: Option<&PathBuf>, report: &InspectReport) {
    println!("JapalityECHO inspect: {}", input.display());
    if let Some(input2) = input2 {
        println!("  mate2: {} | paired_end: true", input2.display());
    }
    println!(
        "  platform: {} | experiment: {} | sampled reads: {}",
        report.auto_profile.platform,
        report.auto_profile.experiment,
        report.auto_profile.read_count
    );
    println!(
        "  read length mean/median/stddev: {:.1} / {} / {:.1}",
        report.auto_profile.mean_read_length,
        report.auto_profile.median_read_length,
        report.auto_profile.length_stddev
    );
    println!(
        "  mean phred: {:.1} | tail drop: {:.1}",
        report.auto_profile.quality_profile.mean_phred,
        report.auto_profile.quality_profile.tail_drop
    );
    println!(
        "  execution plan: backend={} k={} min_q={} trusted_kmer_min={} min_output_len={}",
        report.execution_plan.requested_backend,
        report.execution_plan.kmer_size,
        report.execution_plan.trim_min_quality,
        report.execution_plan.trusted_kmer_min_count,
        report.execution_plan.minimum_output_length
    );

    if !report.auto_profile.adapter_candidates.is_empty() {
        println!("  adapter candidates:");
        for candidate in report.auto_profile.adapter_candidates.iter().take(3) {
            println!(
                "    - {} ({}) support={} score={:.3}",
                candidate.name, candidate.sequence, candidate.support, candidate.score
            );
        }
    }

    if let Some(barcode_hint) = &report.auto_profile.barcode_hint {
        println!(
            "  barcode hint: prefix={} umi={} poly_t_rate={:.1}%",
            barcode_hint.prefix_bases,
            barcode_hint.umi_bases,
            barcode_hint.poly_t_rate * 100.0
        );
    }

    if let Some(scaffold) = &report.accelerator_scaffold {
        println!(
            "  accelerator scaffold: {} from {} [{}]",
            scaffold.backend,
            scaffold.source_path,
            scaffold.entrypoints.join(", ")
        );
    }
    if let Some(runtime) = &report.accelerator_runtime {
        println!(
            "  accelerator runtime: {} status={} device={}",
            runtime.backend,
            runtime.status,
            runtime.device_name.as_deref().unwrap_or("n/a")
        );
    }

    for note in &report.execution_plan.notes {
        println!("  note: {}", note);
    }
    for note in &report.notes {
        println!("  note: {}", note);
    }
}

fn print_process_summary(output: &PathBuf, output2: Option<&PathBuf>, report: &ProcessReport) {
    println!("JapalityECHO process complete: {}", output.display());
    if let Some(output2) = output2 {
        println!("  mate2 output: {} | paired_end: true", output2.display());
    }
    println!(
        "  backend: {} | reads in/out/discarded: {} / {} / {}",
        report.backend_used, report.input_reads, report.output_reads, report.discarded_reads
    );
    if let (Some(input_pairs), Some(output_pairs), Some(discarded_pairs)) = (
        report.input_pairs,
        report.output_pairs,
        report.discarded_pairs,
    ) {
        println!(
            "  pairs in/out/discarded: {} / {} / {}",
            input_pairs, output_pairs, discarded_pairs
        );
    }
    println!(
        "  corrected bases: {} | indel corrections: {} | trimmed reads: {} | trimmed bases: {}",
        report.corrected_bases, report.indel_corrections, report.trimmed_reads, report.trimmed_bases
    );
    println!(
        "  batches: {} | zero-copy candidate: {}",
        report.batches_processed, report.zero_copy_candidate
    );
    println!(
        "  wall clock: {:.3}s | input {:.1} reads/s {:.3} Mbases/s | output {:.1} reads/s {:.3} Mbases/s",
        report.throughput.wall_clock_us as f64 / 1_000_000.0,
        report.throughput.input_reads_per_sec,
        report.throughput.input_bases_per_sec / 1_000_000.0,
        report.throughput.output_reads_per_sec,
        report.throughput.output_bases_per_sec / 1_000_000.0
    );
    println!(
        "  batch host timing: submit={}us wait={}us overlap={}us max_end_to_end={}us",
        report.throughput.cumulative_submit_us,
        report.throughput.cumulative_wait_us,
        report.throughput.cumulative_overlap_us,
        report.throughput.max_batch_end_to_end_us
    );
    println!(
        "  inferred platform/experiment: {} / {}",
        report.auto_profile.platform, report.auto_profile.experiment
    );
    if let Some(scaffold) = &report.accelerator_scaffold {
        println!(
            "  accelerator scaffold: {} from {} [{}]",
            scaffold.backend,
            scaffold.source_path,
            scaffold.entrypoints.join(", ")
        );
    }
    if let Some(runtime) = &report.accelerator_runtime {
        println!(
            "  accelerator runtime: {} status={} device={}",
            runtime.backend,
            runtime.status,
            runtime.device_name.as_deref().unwrap_or("n/a")
        );
    }
    for preview in &report.acceleration_previews {
        println!(
            "  acceleration preview [{}]: packed={}B q={}B offsets={}B grid={} block={} streams={}",
            preview.label,
            preview.packed_bases_bytes,
            preview.quality_bytes,
            preview.offset_bytes,
            preview.grid_size,
            preview.block_size,
            preview.stream_count
        );
    }
    for execution in &report.acceleration_executions {
        println!(
            "  acceleration execution [batch {} {}]: ok={} kernel={} host_pinned={}B device={}B transfer={}B submit={}us wait={}us overlap={}us",
            execution.batch_index,
            execution.stage,
            execution.successful,
            execution.kernel_name,
            execution.host_pinned_bytes,
            execution.device_bytes,
            execution.transfer_bytes,
            execution.submit_us,
            execution.wait_us,
            execution.overlap_us
        );
    }
    for note in &report.notes {
        println!("  note: {}", note);
    }
}

fn print_benchmark_summary(input: &PathBuf, input2: Option<&PathBuf>, report: &BenchmarkReport) {
    println!("JapalityECHO benchmark: {}", input.display());
    if let Some(input2) = input2 {
        println!("  mate2 input: {} | paired_end: true", input2.display());
    }
    println!(
        "  rounds: {} | mode: {} | setup: {}us",
        report.summary.rounds, report.summary.session_mode, report.summary.setup_us
    );
    println!(
        "  backends: {} | consistent: {}",
        report.summary.backends.join(", "),
        report.summary.backend_consistent
    );
    println!(
        "  avg/best/worst wall clock: {:.3}s / {:.3}s / {:.3}s",
        report.summary.average_wall_clock_us / 1_000_000.0,
        report.summary.best_wall_clock_us as f64 / 1_000_000.0,
        report.summary.worst_wall_clock_us as f64 / 1_000_000.0
    );
    println!(
        "  avg throughput: input {:.1} reads/s {:.3} Mbases/s | output {:.1} reads/s {:.3} Mbases/s",
        report.summary.average_input_reads_per_sec,
        report.summary.average_input_bases_per_sec / 1_000_000.0,
        report.summary.average_output_reads_per_sec,
        report.summary.average_output_bases_per_sec / 1_000_000.0
    );
    println!(
        "  avg overlap: {:.1}us | best overlap: {}us",
        report.summary.average_cumulative_overlap_us, report.summary.best_cumulative_overlap_us
    );
    if let Some(steady_state_wall_clock_us) = report.summary.steady_state_average_wall_clock_us {
        println!(
            "  warm-up vs steady-state: {:.3}s vs {:.3}s",
            report.summary.warmup_wall_clock_us as f64 / 1_000_000.0,
            steady_state_wall_clock_us / 1_000_000.0
        );
    }
    if let Some(warmup_penalty_pct) = report.summary.warmup_penalty_pct {
        println!("  warm-up penalty: {warmup_penalty_pct:.1}%");
    }
    if let Some(steady_state_transfer) = report.summary.steady_state_average_transfer_bytes {
        println!(
            "  warm-up transfer vs steady-state avg: {}B vs {:.1}B",
            report.summary.warmup_transfer_bytes, steady_state_transfer
        );
    }
    for round in &report.rounds {
        println!(
            "  round {}: backend={} wall={:.3}s input={:.3} Mbases/s overlap={}us",
            round.round_index,
            round.process.backend_used,
            round.process.throughput.wall_clock_us as f64 / 1_000_000.0,
            round.process.throughput.input_bases_per_sec / 1_000_000.0,
            round.process.throughput.cumulative_overlap_us
        );
    }
    for note in &report.notes {
        println!("  note: {}", note);
    }
}

fn print_benchmark_comparison_summary(
    input: &PathBuf,
    input2: Option<&PathBuf>,
    report: &BenchmarkComparisonReport,
) {
    println!("JapalityECHO benchmark compare: {}", input.display());
    if let Some(input2) = input2 {
        println!("  mate2 input: {} | paired_end: true", input2.display());
    }
    println!(
        "  rounds per mode: {} | backend match: {}",
        report.summary.rounds, report.summary.backend_match
    );
    println!(
        "  cold-start backends: {} | reuse-session backends: {}",
        report.summary.cold_start_backends.join(", "),
        report.summary.reuse_session_backends.join(", ")
    );
    println!(
        "  avg wall clock: cold {:.3}s | reuse {:.3}s | reuse amortized {:.3}s",
        report.summary.cold_start_average_wall_clock_us / 1_000_000.0,
        report.summary.reuse_session_average_wall_clock_us / 1_000_000.0,
        report.summary.reuse_session_amortized_average_wall_clock_us / 1_000_000.0
    );
    println!(
        "  reuse setup: {}us | wall-clock delta: {:.1}us | amortized delta: {:.1}us",
        report.summary.reuse_session_setup_us,
        report.summary.average_wall_clock_delta_us,
        report.summary.amortized_wall_clock_delta_us
    );
    if let Some(raw_speedup) = report.summary.raw_speedup {
        println!("  raw speedup: {raw_speedup:.3}x");
    }
    if let Some(amortized_speedup) = report.summary.amortized_speedup {
        println!("  amortized speedup: {amortized_speedup:.3}x");
    }
    if let Some(input_uplift_pct) = report.summary.input_bases_per_sec_uplift_pct {
        println!("  input throughput uplift: {input_uplift_pct:.1}%");
    }
    if let Some(output_uplift_pct) = report.summary.output_bases_per_sec_uplift_pct {
        println!("  output throughput uplift: {output_uplift_pct:.1}%");
    }
    println!(
        "  average overlap delta: {:.1}us",
        report.summary.average_overlap_delta_us
    );
    if let Some(transfer_savings_bytes) = report.summary.steady_state_transfer_savings_bytes {
        println!(
            "  steady-state transfer savings: {:.1}B",
            transfer_savings_bytes
        );
    }
    if let Some(transfer_savings_pct) = report.summary.steady_state_transfer_savings_pct {
        println!("  steady-state transfer savings pct: {transfer_savings_pct:.1}%");
    }
    for note in &report.notes {
        println!("  note: {}", note);
    }
}

fn print_evaluation_metric(label: &str, metric: &EvaluationMetricSummary) {
    println!(
        "  {label}: tp={} fp={} fn={} precision={:.3} recall={:.3} f1={:.3}",
        metric.true_positive,
        metric.false_positive,
        metric.false_negative,
        metric.precision,
        metric.recall,
        metric.f1
    );
}

fn print_evaluation_report(report: &EvaluationReport) {
    println!("JapalityECHO evaluation: {}", report.suite_name);
    println!(
        "  requested backend: {} | actual backend: {}",
        report.requested_backend, report.process.backend_used
    );
    println!(
        "  scenarios: {} | reads/scenario: {} | total reads: {}",
        report.total_scenarios, report.reads_per_scenario, report.aggregate.total_reads
    );
    println!(
        "  exact matches: {} / {} ({:.1}%)",
        report.aggregate.exact_match_reads,
        report.aggregate.expected_retained_reads,
        report.aggregate.exact_match_rate * 100.0
    );
    print_evaluation_metric("retention", &report.aggregate.retention);
    print_evaluation_metric("trimming", &report.aggregate.trimming);
    print_evaluation_metric("correction", &report.aggregate.correction);
    println!(
        "  throughput: wall {:.3}s | input {:.1} reads/s | output {:.1} reads/s",
        report.process.throughput.wall_clock_us as f64 / 1_000_000.0,
        report.process.throughput.input_reads_per_sec,
        report.process.throughput.output_reads_per_sec
    );
    for scenario in &report.scenarios {
        println!(
            "  scenario {}: exact {:.1}% | retain f1 {:.3} | trim f1 {:.3} | correct f1 {:.3}",
            scenario.name,
            scenario.exact_match_rate * 100.0,
            scenario.retention.f1,
            scenario.trimming.f1,
            scenario.correction.f1
        );
    }
    for note in &report.notes {
        println!("  note: {}", note);
    }
}

fn print_paper_artifacts_report(report: &PaperArtifactsReport) {
    println!("JapalityECHO paper artifacts: {}", report.output_dir);
    println!(
        "  accelerator backend: {} | reads/scenario: {} | benchmark rounds: {}",
        report.requested_accelerator_backend, report.reads_per_scenario, report.benchmark_rounds
    );
    println!(
        "  single-end exact: cpu {:.1}% | accelerator {:.1}%",
        report.cpu_evaluation.aggregate.exact_match_rate * 100.0,
        report.accelerator_evaluation.aggregate.exact_match_rate * 100.0
    );
    println!(
        "  paired-end exact: cpu {:.1}% | accelerator {:.1}%",
        report.paired_cpu_evaluation.aggregate.exact_match_rate * 100.0,
        report
            .paired_accelerator_evaluation
            .aggregate
            .exact_match_rate
            * 100.0
    );
    println!(
        "  cpu throughput: {:.3} Mbases/s | accelerator cold: {:.3} Mbases/s | accelerator reuse: {:.3} Mbases/s",
        report.cpu_benchmark.average_input_bases_per_sec / 1_000_000.0,
        report
            .accelerator_cold_benchmark
            .average_input_bases_per_sec
            / 1_000_000.0,
        report
            .accelerator_reuse_benchmark
            .average_input_bases_per_sec
            / 1_000_000.0
    );
    let max_parity_gap = [
        report.parity.single_end_exact_match_delta,
        report.parity.single_end_retention_f1_delta,
        report.parity.single_end_trimming_f1_delta,
        report.parity.single_end_correction_f1_delta,
        report.parity.paired_end_exact_match_delta,
        report.parity.paired_end_retention_f1_delta,
        report.parity.paired_end_trimming_f1_delta,
        report.parity.paired_end_correction_f1_delta,
    ]
    .into_iter()
    .map(f64::abs)
    .fold(0.0, f64::max);
    println!("  max cpu-accelerator accuracy gap: {max_parity_gap:.4}");
    if let Some(raw_speedup) = report.accelerator_compare.raw_speedup {
        println!("  reuse raw speedup: {raw_speedup:.3}x");
    }
    if let Some(transfer_savings_pct) = report.accelerator_compare.steady_state_transfer_savings_pct
    {
        println!("  reuse transfer savings: {transfer_savings_pct:.1}%");
    }
    for artifact in &report.artifacts {
        println!(
            "  artifact [{}]: {} ({})",
            artifact.kind, artifact.path, artifact.description
        );
    }
    for note in &report.notes {
        println!("  note: {}", note);
    }
}

fn print_publication_bundle_report(report: &PublicationBundleReport) {
    println!("JapalityECHO publication bundle: {}", report.output_dir);
    println!("  input: {}", report.input_path);
    println!(
        "  fetch rows: {} | downloaded FASTQs: {} | final datasets: {}",
        report.fetch.summary.fetched_records,
        report.download.summary.downloaded_files + report.download.summary.resumed_files,
        report.study_artifacts.aggregate.datasets
    );
    println!(
        "  manifest: {} | manuscript summary: {}",
        report.final_manifest_path, report.manuscript_summary_path
    );
    println!(
        "  study throughput: {:.3} Mbases/s | synthetic accelerator throughput: {:.3} Mbases/s",
        report.study_artifacts.aggregate.average_input_bases_per_sec / 1_000_000.0,
        report
            .paper_artifacts
            .accelerator_cold_benchmark
            .average_input_bases_per_sec
            / 1_000_000.0
    );
    if let Some(platform_accuracy) = report.study_artifacts.detection.platform_accuracy {
        println!("  platform accuracy: {:.1}%", platform_accuracy * 100.0);
    }
    if let Some(experiment_accuracy) = report.study_artifacts.detection.experiment_accuracy {
        println!("  experiment accuracy: {:.1}%", experiment_accuracy * 100.0);
    }
    if let Some(speedup) = report
        .study_artifacts
        .comparison
        .average_input_speedup_vs_baseline
    {
        println!("  baseline throughput speedup: {speedup:.3}x");
    }
    for note in &report.notes {
        println!("  note: {}", note);
    }
}

fn print_study_artifacts_report(report: &StudyArtifactsReport) {
    println!("JapalityECHO study artifacts: {}", report.output_dir);
    println!(
        "  manifest: {} | backend: {} | datasets: {} (paired: {})",
        report.manifest_path,
        report.requested_backend,
        report.aggregate.datasets,
        report.aggregate.paired_datasets
    );
    println!(
        "  avg throughput: {:.3} Mbases/s | avg trimmed: {:.1}% | avg discarded: {:.1}% | avg corrected density: {:.1}",
        report.aggregate.average_input_bases_per_sec / 1_000_000.0,
        report.aggregate.average_trimmed_read_fraction * 100.0,
        report.aggregate.average_discarded_read_fraction * 100.0,
        report.aggregate.average_corrected_bases_per_mbase
    );
    if report.comparison.datasets_with_baseline > 0 {
        println!(
            "  baseline datasets: {} | throughput comparisons: {}",
            report.comparison.datasets_with_baseline,
            report.comparison.datasets_with_baseline_throughput
        );
    }
    if let Some(platform_accuracy) = report.detection.platform_accuracy {
        println!("  platform accuracy: {:.1}%", platform_accuracy * 100.0);
    }
    if let Some(experiment_accuracy) = report.detection.experiment_accuracy {
        println!("  experiment accuracy: {:.1}%", experiment_accuracy * 100.0);
    }
    if let Some(speedup) = report.comparison.average_input_speedup_vs_baseline {
        println!("  avg baseline speedup: {speedup:.3}x");
    }
    if let Some(alignment_delta) = report.comparison.average_alignment_rate_delta {
        println!("  avg alignment delta: {alignment_delta:+.3}");
    }
    if let Some(variant_delta) = report.comparison.average_variant_f1_delta {
        println!("  avg variant-f1 delta: {variant_delta:+.3}");
    }
    if let Some(mean_coverage_ratio) = report.comparison.average_mean_coverage_ratio {
        println!("  avg mean-coverage ratio: {mean_coverage_ratio:.3}x");
    }
    if let Some(coverage_breadth_delta) = report.comparison.average_coverage_breadth_delta {
        println!("  avg coverage-breadth delta: {coverage_breadth_delta:+.3}");
    }
    if let Some(assembly_ratio) = report.comparison.average_assembly_n50_ratio {
        println!("  avg assembly-n50 ratio: {assembly_ratio:.3}x");
    }
    for dataset in &report.datasets {
        println!(
            "  dataset {}: backend {} | detect {} / {} | throughput {:.3} Mbases/s",
            dataset.dataset_id,
            dataset.process.backend_used,
            dataset.detected_platform,
            dataset.detected_experiment,
            dataset.benchmark.average_input_bases_per_sec / 1_000_000.0
        );
        if let Some(baseline) = &dataset.baseline {
            if let Some(speedup) = dataset.comparison.input_speedup_vs_baseline {
                println!(
                    "    baseline {} | throughput speedup {:.3}x",
                    baseline.name, speedup
                );
            } else {
                println!("    baseline {}", baseline.name);
            }
        }
    }
    for artifact in &report.artifacts {
        println!(
            "  artifact [{}]: {} ({})",
            artifact.kind, artifact.path, artifact.description
        );
    }
    for note in &report.notes {
        println!("  note: {}", note);
    }
}

fn print_study_manifest_report(report: &StudyManifestBootstrapReport) {
    println!("JapalityECHO study manifest: {}", report.output_path);
    println!("  inventory: {}", report.input_path);
    println!(
        "  datasets: {} (paired: {}) | generated ids: {}",
        report.summary.datasets,
        report.summary.paired_datasets,
        report.summary.datasets_with_generated_id
    );
    println!(
        "  provenance: accession {} | citation {} | expected platform {} | expected experiment {}",
        report.summary.datasets_with_accession,
        report.summary.datasets_with_citation,
        report.summary.datasets_with_expected_platform,
        report.summary.datasets_with_expected_experiment
    );
    println!(
        "  baseline names: {}",
        report.summary.datasets_with_baseline_name
    );
    println!(
        "  downstream metrics: {}",
        report.summary.datasets_with_downstream_metrics
    );
    println!("  provenance summary: {}", report.provenance_summary_path);
    if let Some(default_baseline_name) = &report.default_baseline_name {
        println!("  default baseline: {}", default_baseline_name);
    }
    for note in &report.notes {
        println!("  note: {}", note);
    }
}

fn print_study_fetch_metadata_report(report: &StudyFetchMetadataReport) {
    println!("JapalityECHO study fetch metadata: {}", report.output_path);
    println!(
        "  input: {} | source: {} | chunk size: {} | retries: {} | resume: {}",
        report.input_path, report.source, report.chunk_size, report.retries, report.resume_existing
    );
    println!(
        "  requested identifiers: {} (unique {}) | matched: {} | unmatched: {}",
        report.summary.requested_accessions,
        report.summary.unique_accessions,
        report.summary.matched_accessions,
        report.summary.unmatched_accessions
    );
    println!(
        "  resumed accessions: {} | resumed run rows: {}",
        report.summary.resumed_accessions, report.summary.resumed_records
    );
    println!(
        "  GEO bridge: {} accession(s) -> {} resolved public accession(s) | fallback resolutions: {}",
        report.summary.geo_bridge_accessions,
        report.summary.geo_bridge_resolved_accessions,
        report.summary.geo_bridge_fallback_accessions
    );
    println!(
        "  fetched run rows: {} | failed: {} | base url: {}",
        report.summary.fetched_records, report.summary.failed_accessions, report.base_url
    );
    println!(
        "  cache hits: {} | remote fetches: {} | retried chunks: {}",
        report.summary.cache_hits, report.summary.remote_fetches, report.summary.retried_chunks
    );
    println!(
        "  GEO base url: {} | cache dir: {} | status csv: {}",
        report.geo_base_url, report.cache_dir, report.status_path
    );
    if !report.failed_accessions.is_empty() {
        println!(
            "  failed accessions: {}",
            report.failed_accessions.join(", ")
        );
    }
    if !report.unmatched_accessions.is_empty() {
        println!(
            "  unmatched accessions: {}",
            report.unmatched_accessions.join(", ")
        );
    }
    for note in &report.notes {
        println!("  note: {}", note);
    }
}

fn print_study_download_report(report: &StudyDownloadReport) {
    println!("JapalityECHO study download root: {}", report.download_root);
    println!(
        "  input: {} | retries: {} | overwrite existing: {}",
        report.input_path, report.retries, report.overwrite_existing
    );
    println!(
        "  datasets: {} | requested FASTQs: {} | downloaded: {}",
        report.summary.datasets, report.summary.requested_files, report.summary.downloaded_files
    );
    println!(
        "  resumed: {} | skipped existing: {} | failed: {}",
        report.summary.resumed_files,
        report.summary.skipped_existing_files,
        report.summary.failed_files
    );
    println!(
        "  available bytes: {} | status csv: {}",
        report.summary.available_bytes, report.status_path
    );
    if !report.failed_destinations.is_empty() {
        println!(
            "  failed destinations: {}",
            report.failed_destinations.join(", ")
        );
    }
    for note in &report.notes {
        println!("  note: {}", note);
    }
}

fn print_study_discover_report(report: &japalityecho::StudyDiscoverReport) {
    println!("JapalityECHO study discover: {}", report.output_path);
    println!(
        "  input: {} | recursive: {} | files: {} | datasets: {}",
        report.input_dir, report.recursive, report.summary.fastq_files, report.summary.datasets
    );
    println!(
        "  paired: {} | single-end: {} | accession-labelled: {}",
        report.summary.paired_datasets,
        report.summary.single_end_datasets,
        report.summary.accession_labeled_datasets
    );
    for dataset in &report.datasets {
        println!(
            "  dataset {}: {}{}",
            dataset.dataset_id,
            dataset.input1,
            dataset
                .input2
                .as_ref()
                .map(|input2| format!(" | {}", input2))
                .unwrap_or_default()
        );
    }
    for note in &report.notes {
        println!("  note: {}", note);
    }
}

fn print_study_annotate_report(report: &StudyAnnotateReport) {
    println!("JapalityECHO study annotate: {}", report.output_path);
    println!(
        "  manifest: {} | annotations: {}",
        report.manifest_path, report.annotations_path
    );
    println!(
        "  rows: {} matched / {} unmatched | datasets changed: {}",
        report.summary.matched_rows, report.summary.unmatched_rows, report.summary.datasets_changed
    );
    println!(
        "  fields filled: {} | overwritten: {} | notes appended: {}",
        report.summary.fields_filled,
        report.summary.fields_overwritten,
        report.summary.notes_appended
    );
    println!(
        "  citation coverage: {} -> {} | baseline coverage: {} -> {}",
        report.summary.before.datasets_with_citation,
        report.summary.after.datasets_with_citation,
        report.summary.before.datasets_with_baseline_name,
        report.summary.after.datasets_with_baseline_name
    );
    println!("  summary csv: {}", report.summary_path);
    for note in &report.notes {
        println!("  note: {}", note);
    }
}

fn print_study_ingest_report(report: &StudyIngestReport) {
    println!("JapalityECHO study ingest: {}", report.output_path);
    println!(
        "  manifest: {} | input dir: {} | recursive: {}",
        report.manifest_path, report.input_dir, report.recursive
    );
    println!(
        "  files: {} scanned, {} structured (json {}, delimited {}, ignored {})",
        report.summary.files_scanned,
        report.summary.structured_files,
        report.summary.json_files,
        report.summary.delimited_files,
        report.summary.ignored_files
    );
    println!(
        "  records: {} ingested, {} matched, {} unmatched | generated rows: {}",
        report.summary.records_ingested,
        report.summary.matched_records,
        report.summary.unmatched_records,
        report.summary.generated_annotation_rows
    );
    println!(
        "  datasets changed: {} | fields filled: {} | overwritten: {} | notes appended: {}",
        report.summary.datasets_changed,
        report.summary.fields_filled,
        report.summary.fields_overwritten,
        report.summary.notes_appended
    );
    println!(
        "  generated annotations: {} | summary csv: {}",
        report.generated_annotations_path, report.summary_path
    );
    for note in &report.notes {
        println!("  note: {}", note);
    }
}

fn print_history_report(report: &HistoryReport) {
    println!("JapalityECHO history report: {}", report.source_path);
    println!(
        "  entries: total={} benchmark={} compare={}",
        report.total_entries, report.benchmark_entries, report.benchmark_compare_entries
    );
    println!(
        "  requested backends: {}",
        report
            .requested_backends
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    if !report.labels.is_empty() {
        println!("  labels: {}", report.labels.join(", "));
    }
    if let Some(latest) = &report.latest_entry {
        println!(
            "  latest: kind={} backend={} label={} wall={:.3}s",
            latest.kind,
            latest.requested_backend,
            latest.label.as_deref().unwrap_or("unlabeled"),
            latest.average_wall_clock_us / 1_000_000.0
        );
    }
    if let Some(raw_speedup) = report.comparison_stats.latest_raw_speedup {
        println!("  latest raw speedup: {raw_speedup:.3}x");
    }
    if let Some(amortized_speedup) = report.comparison_stats.latest_amortized_speedup {
        println!("  latest amortized speedup: {amortized_speedup:.3}x");
    }
    if let Some(avg_raw_speedup) = report.comparison_stats.average_raw_speedup {
        println!("  avg raw speedup: {avg_raw_speedup:.3}x");
    }
    if let Some(avg_transfer_savings_pct) = report.comparison_stats.average_transfer_savings_pct {
        println!("  avg transfer savings pct: {avg_transfer_savings_pct:.1}%");
    }
    if let Some(regression) = &report.regression {
        println!(
            "  regression: status={} baseline_source={}",
            regression.status, regression.baseline_source
        );
        if let Some(threshold) = regression.thresholds.max_wall_clock_regression_pct {
            println!("  threshold max wall regression: {threshold:.1}%");
        }
        if let Some(threshold) = regression.thresholds.min_raw_speedup {
            println!("  threshold min raw speedup: {threshold:.3}x");
        }
        if let Some(threshold) = regression.thresholds.min_transfer_savings_pct {
            println!("  threshold min transfer savings: {threshold:.1}%");
        }
        if let Some(baseline) = &regression.baseline_entry {
            println!(
                "  baseline: kind={} backend={} label={} wall={:.3}s",
                baseline.kind,
                baseline.requested_backend,
                baseline.label.as_deref().unwrap_or("unlabeled"),
                baseline.average_wall_clock_us / 1_000_000.0
            );
        }
        if let Some(wall_delta_us) = regression.wall_clock_delta_us {
            println!("  wall delta: {:.1}us", wall_delta_us);
        }
        if let Some(wall_delta_pct) = regression.wall_clock_delta_pct {
            println!("  wall delta pct: {wall_delta_pct:.1}%");
        }
        if let Some(raw_speedup_delta) = regression.raw_speedup_delta {
            println!("  raw speedup delta: {raw_speedup_delta:.3}x");
        }
        if let Some(transfer_delta_pct) = regression.transfer_savings_pct_delta {
            println!("  transfer savings delta pct: {transfer_delta_pct:.1}%");
        }
        for alert in &regression.alerts {
            println!("  alert: {}", alert);
        }
    }
    for entry in &report.recent_entries {
        println!(
            "  entry: kind={} backend={} label={} rounds={} wall={:.3}s",
            entry.kind,
            entry.requested_backend,
            entry.label.as_deref().unwrap_or("unlabeled"),
            entry.rounds,
            entry.average_wall_clock_us / 1_000_000.0
        );
    }
    for note in &report.notes {
        println!("  note: {}", note);
    }
}
