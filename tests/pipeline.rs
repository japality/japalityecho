use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::process::Command;
use std::thread;

use anyhow::Result;
use flate2::{Compression, write::GzEncoder};
use japalityecho::fastq::sample_records;
use japalityecho::gpu::CUDA_KERNEL_PATH;
use japalityecho::model::{
    AcceleratorRuntimeStatus, BackendPreference, BenchmarkSessionMode, ExperimentType, FastqRecord,
    Platform,
};
use japalityecho::process::{
    BenchmarkComparisonOptions, BenchmarkOptions, ProcessOptions, benchmark_compare_file,
    benchmark_file, inspect_inputs, process_file, process_files,
};
use japalityecho::profile::infer_auto_profile;
use japalityecho::{
    EvaluationOptions, HistoryReportOptions, PaperArtifactsOptions, StudyAnnotateOptions,
    StudyArtifactsOptions, StudyDiscoverOptions, StudyDownloadOptions, StudyFetchMetadataOptions,
    StudyIngestOptions, StudyManifestOptions, annotate_study_manifest,
    append_benchmark_comparison_history, append_benchmark_history, bootstrap_study_manifest,
    discover_study_inventory, download_public_fastqs, evaluate_paired_synthetic_truth,
    evaluate_synthetic_truth, fetch_public_metadata, generate_paper_artifacts,
    generate_study_artifacts, ingest_study_results, read_history_report,
};
use tempfile::tempdir;

#[test]
fn detects_single_cell_poly_t_signature() {
    let records: Vec<FastqRecord> = (0..64)
        .map(|index| {
            let sequence = b"ACGTACGTACGTACGTGATCGATCGATCTTTTTTTT".to_vec();
            FastqRecord::new(
                format!("sc-{index}"),
                sequence.clone(),
                vec![b'I'; sequence.len()],
            )
        })
        .collect();

    let profile = infer_auto_profile(&records);
    assert_eq!(profile.experiment, ExperimentType::SingleCell10xV3);
    assert!(profile.barcode_hint.is_some());
}

fn synthetic_dna_from_index(mut value: usize, len: usize) -> String {
    let alphabet = [b'A', b'C', b'G', b'T'];
    let mut sequence = Vec::with_capacity(len);
    for offset in 0..len {
        let digit = value % 4;
        sequence.push(alphabet[(digit + offset) % 4]);
        value = value / 4 + offset + 1;
    }
    String::from_utf8(sequence).expect("synthetic DNA is ASCII")
}

/// Pseudo-random DNA with uniform base composition (mimics real genomic reads).
fn random_dna(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            b"ACGT"[((state >> 33) as usize) % 4]
        })
        .collect()
}

fn illumina_style_qualities(len: usize) -> Vec<u8> {
    let mut qualities = vec![b'I'; len];
    for quality in qualities.iter_mut().skip(len.saturating_sub(10)) {
        *quality = b'!';
    }
    qualities
}

#[test]
fn detects_rna_seq_from_leading_bias_and_duplication_without_poly_t() {
    let suffixes: Vec<String> = (0..8)
        .map(|index| synthetic_dna_from_index(index, 64))
        .collect();
    let records: Vec<FastqRecord> = (0..128)
        .map(|index| {
            let sequence = format!("GGGCGGGCGGGC{}", suffixes[index % suffixes.len()]);
            FastqRecord::new(
                format!("rna-{index}"),
                sequence.as_bytes().to_vec(),
                illumina_style_qualities(sequence.len()),
            )
        })
        .collect();

    let profile = infer_auto_profile(&records);
    assert_eq!(profile.experiment, ExperimentType::RnaSeq);
}

#[test]
fn does_not_call_atac_from_trace_nextera_signal() {
    let nextera = b"CTGTCTCTTATACACATCT";
    let records: Vec<FastqRecord> = (0..2000)
        .map(|index| {
            let mut sequence = random_dna(76, (index + 100) as u64);
            if index == 0 {
                let start = sequence.len() - nextera.len();
                sequence[start..].copy_from_slice(nextera);
            }
            FastqRecord::new(
                format!("wgs-{index}"),
                sequence,
                illumina_style_qualities(76),
            )
        })
        .collect();

    let profile = infer_auto_profile(&records);
    assert_eq!(profile.experiment, ExperimentType::Wgs);
}

#[test]
fn detects_atac_from_strong_nextera_signal() {
    let nextera = b"CTGTCTCTTATACACATCT";
    let records: Vec<FastqRecord> = (0..256)
        .map(|index| {
            let mut sequence = random_dna(76, (index + 4000) as u64);
            if index < 24 {
                let start = sequence.len() - nextera.len();
                sequence[start..].copy_from_slice(nextera);
            }
            FastqRecord::new(
                format!("atac-{index}"),
                sequence,
                illumina_style_qualities(76),
            )
        })
        .collect();

    let profile = infer_auto_profile(&records);
    assert_eq!(profile.experiment, ExperimentType::AtacSeq);
}

#[test]
fn illumina_adapter_signal_overrides_flat_quality_mgi_guess() {
    let illumina = b"AGATCGGAAGAGC";
    let records: Vec<FastqRecord> = (0..2000)
        .map(|index| {
            let mut sequence = synthetic_dna_from_index(index + 8000, 36).into_bytes();
            if index < 5 {
                let start = sequence.len() - illumina.len();
                sequence[start..].copy_from_slice(illumina);
            }
            FastqRecord::new(format!("flat-{index}"), sequence, vec![b'I'; 36])
        })
        .collect();

    let profile = infer_auto_profile(&records);
    assert_eq!(profile.platform, Platform::Illumina);
}

#[test]
fn cpu_pipeline_repairs_errors_and_trims_adapters() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("input.fastq");
    let output = dir.path().join("output.fastq");
    let core = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let adapter = "AGATCGGAAGAGC";
    let mut fastq = String::new();

    for index in 0..16 {
        fastq.push_str(&format!(
            "@good-{index}\n{core}\n+\n{}\n",
            "I".repeat(core.len())
        ));
    }

    let mut corrupted = core.as_bytes().to_vec();
    corrupted[18] = b'T';
    let corrupted = String::from_utf8(corrupted)?;

    for index in 0..6 {
        let prefix_quality = "I".repeat(18);
        let suffix_quality = "I".repeat(core.len() - 19);
        let qualities = format!(
            "{prefix_quality}!{suffix_quality}{}",
            "!".repeat(adapter.len())
        );
        fastq.push_str(&format!(
            "@bad-{index}\n{}{}\n+\n{}\n",
            corrupted, adapter, qualities
        ));
    }

    fs::write(&input, fastq)?;

    let report = process_file(
        &input,
        &output,
        &ProcessOptions {
            sample_size: 64,
            batch_reads: 8,
            backend_preference: BackendPreference::Cpu,
            forced_adapter: None,
            min_quality_override: Some(20),
            kmer_size_override: None,
                    forced_platform: None,
                    forced_experiment: None,
        },
    )?;

    assert_eq!(report.backend_used, "cpu");
    assert!(report.corrected_bases >= 6);
    assert!(report.trimmed_reads >= 6);
    assert_eq!(report.discarded_reads, 0);
    assert!(report.throughput.wall_clock_us > 0);
    assert!(report.throughput.input_bases > 0);
    assert!(report.throughput.input_reads_per_sec > 0.0);
    assert!(report.notes.iter().any(|note| note.contains("read-ahead")));

    let output_records = sample_records(&output, 64)?;
    assert_eq!(output_records.len(), 22);
    assert!(
        output_records
            .iter()
            .all(|record| record.sequence.len() == core.len())
    );
    assert!(
        output_records
            .iter()
            .any(|record| String::from_utf8_lossy(&record.sequence) == core)
    );

    Ok(())
}

#[test]
fn paired_end_pipeline_keeps_mates_synchronized() -> Result<()> {
    let dir = tempdir()?;
    let input1 = dir.path().join("r1.fastq");
    let input2 = dir.path().join("r2.fastq");
    let output1 = dir.path().join("r1.out.fastq");
    let output2 = dir.path().join("r2.out.fastq");
    let core1 = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let core2 = "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";
    let adapter = "AGATCGGAAGAGC";
    let mut fastq1 = String::new();
    let mut fastq2 = String::new();

    for index in 0..16 {
        fastq1.push_str(&format!(
            "@pair-{index}/1\n{core1}\n+\n{}\n",
            "I".repeat(core1.len())
        ));
        fastq2.push_str(&format!(
            "@pair-{index}/2\n{core2}\n+\n{}\n",
            "I".repeat(core2.len())
        ));
    }

    let mut corrupted = core1.as_bytes().to_vec();
    corrupted[18] = b'T';
    let corrupted = String::from_utf8(corrupted)?;
    for index in 16..20 {
        let qualities1 = format!(
            "{}!{}{}",
            "I".repeat(18),
            "I".repeat(core1.len() - 19),
            "!".repeat(adapter.len())
        );
        let qualities2 = format!("{}{}", "I".repeat(core2.len()), "!".repeat(adapter.len()));
        fastq1.push_str(&format!(
            "@pair-{index}/1\n{}{}\n+\n{}\n",
            corrupted, adapter, qualities1
        ));
        fastq2.push_str(&format!(
            "@pair-{index}/2\n{}{}\n+\n{}\n",
            core2, adapter, qualities2
        ));
    }

    let discard_mate = format!("{}{}", &core2[..20], adapter);
    let discard_qualities = format!("{}{}", "I".repeat(20), "!".repeat(adapter.len()));
    fastq1.push_str(&format!(
        "@pair-20/1\n{core1}\n+\n{}\n",
        "I".repeat(core1.len())
    ));
    fastq2.push_str(&format!(
        "@pair-20/2\n{discard_mate}\n+\n{discard_qualities}\n"
    ));

    fs::write(&input1, fastq1)?;
    fs::write(&input2, fastq2)?;

    let report = process_files(
        &input1,
        &output1,
        Some(&input2),
        Some(&output2),
        &ProcessOptions {
            sample_size: 64,
            batch_reads: 8,
            backend_preference: BackendPreference::Cpu,
            forced_adapter: None,
            min_quality_override: Some(20),
            kmer_size_override: None,
                    forced_platform: None,
                    forced_experiment: None,
        },
    )?;

    assert!(report.paired_end);
    assert_eq!(report.input_pairs, Some(21));
    assert_eq!(report.output_pairs, Some(20));
    assert_eq!(report.discarded_pairs, Some(1));
    assert!(report.corrected_bases >= 4);
    assert!(report.throughput.wall_clock_us > 0);
    assert!(report.throughput.input_bases > 0);
    assert!(
        report
            .notes
            .iter()
            .any(|note| note.contains("paired batch read-ahead"))
    );

    let out1 = sample_records(&output1, 64)?;
    let out2 = sample_records(&output2, 64)?;
    assert_eq!(out1.len(), 20);
    assert_eq!(out2.len(), 20);
    assert!(
        out1.iter().zip(out2.iter()).all(|(left, right)| {
            left.header.split('/').next() == right.header.split('/').next()
        })
    );

    Ok(())
}

#[test]
fn benchmark_mode_summarizes_repeated_cpu_runs() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("bench.fastq");
    let core = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let adapter = "AGATCGGAAGAGC";
    let mut fastq = String::new();

    for index in 0..12 {
        fastq.push_str(&format!(
            "@bench-{index}\n{core}\n+\n{}\n",
            "I".repeat(core.len())
        ));
    }
    for index in 12..18 {
        let qualities = format!(
            "{}!{}{}",
            "I".repeat(18),
            "I".repeat(core.len() - 19),
            "!".repeat(adapter.len())
        );
        let mut corrupted = core.as_bytes().to_vec();
        corrupted[18] = b'T';
        let corrupted = String::from_utf8(corrupted)?;
        fastq.push_str(&format!(
            "@bench-{index}\n{}{}\n+\n{}\n",
            corrupted, adapter, qualities
        ));
    }

    fs::write(&input, fastq)?;
    let report = benchmark_file(
        &input,
        &BenchmarkOptions {
            process: ProcessOptions {
                sample_size: 32,
                batch_reads: 6,
                backend_preference: BackendPreference::Cpu,
                forced_adapter: None,
                min_quality_override: Some(20),
                kmer_size_override: None,
                    forced_platform: None,
                    forced_experiment: None,
            },
            rounds: 3,
            session_mode: BenchmarkSessionMode::ReuseSession,
        },
    )?;

    assert!(!report.paired_end);
    assert_eq!(
        report.summary.session_mode,
        BenchmarkSessionMode::ReuseSession
    );
    assert_eq!(report.summary.rounds, 3);
    assert_eq!(report.rounds.len(), 3);
    assert!(report.summary.setup_us > 0);
    assert!(report.summary.backend_consistent);
    assert_eq!(report.summary.backends, vec!["cpu".to_string()]);
    assert!(report.summary.average_wall_clock_us > 0.0);
    assert!(report.summary.average_input_bases_per_sec > 0.0);
    assert!(report.summary.best_wall_clock_us > 0);
    assert!(report.summary.worst_wall_clock_us >= report.summary.best_wall_clock_us);
    assert!(report.summary.steady_state_average_wall_clock_us.is_some());
    assert!(
        report
            .notes
            .iter()
            .any(|note| note.contains("reuse the same backend instance"))
    );
    assert!(
        report
            .rounds
            .iter()
            .all(|round| round.process.backend_used == "cpu")
    );

    Ok(())
}

#[test]
fn benchmark_compare_reports_mode_deltas() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("bench-compare.fastq");
    let core = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let adapter = "AGATCGGAAGAGC";
    let mut fastq = String::new();

    for index in 0..12 {
        fastq.push_str(&format!(
            "@cmp-{index}\n{core}\n+\n{}\n",
            "I".repeat(core.len())
        ));
    }
    for index in 12..18 {
        let qualities = format!(
            "{}!{}{}",
            "I".repeat(18),
            "I".repeat(core.len() - 19),
            "!".repeat(adapter.len())
        );
        let mut corrupted = core.as_bytes().to_vec();
        corrupted[18] = b'T';
        let corrupted = String::from_utf8(corrupted)?;
        fastq.push_str(&format!(
            "@cmp-{index}\n{}{}\n+\n{}\n",
            corrupted, adapter, qualities
        ));
    }

    fs::write(&input, fastq)?;
    let report = benchmark_compare_file(
        &input,
        &BenchmarkComparisonOptions {
            process: ProcessOptions {
                sample_size: 32,
                batch_reads: 6,
                backend_preference: BackendPreference::Cpu,
                forced_adapter: None,
                min_quality_override: Some(20),
                kmer_size_override: None,
                    forced_platform: None,
                    forced_experiment: None,
            },
            rounds: 3,
        },
    )?;

    assert!(!report.paired_end);
    assert_eq!(
        report.cold_start.summary.session_mode,
        BenchmarkSessionMode::ColdStart
    );
    assert_eq!(
        report.reuse_session.summary.session_mode,
        BenchmarkSessionMode::ReuseSession
    );
    assert_eq!(report.summary.rounds, 3);
    assert!(report.summary.backend_match);
    assert!(report.summary.reuse_session_setup_us > 0);
    assert!(report.summary.reuse_session_amortized_average_wall_clock_us > 0.0);
    assert!(report.summary.raw_speedup.is_some());
    assert!(
        report
            .notes
            .iter()
            .any(|note| note.contains("cold_start first and then reuse_session"))
    );

    Ok(())
}

#[test]
fn benchmark_history_appends_jsonl_records() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("bench-history.fastq");
    let history = dir.path().join("history").join("benchmarks.jsonl");
    let core = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let adapter = "AGATCGGAAGAGC";
    let mut fastq = String::new();

    for index in 0..12 {
        fastq.push_str(&format!(
            "@hist-{index}\n{core}\n+\n{}\n",
            "I".repeat(core.len())
        ));
    }
    for index in 12..18 {
        let qualities = format!(
            "{}!{}{}",
            "I".repeat(18),
            "I".repeat(core.len() - 19),
            "!".repeat(adapter.len())
        );
        let mut corrupted = core.as_bytes().to_vec();
        corrupted[18] = b'T';
        let corrupted = String::from_utf8(corrupted)?;
        fastq.push_str(&format!(
            "@hist-{index}\n{}{}\n+\n{}\n",
            corrupted, adapter, qualities
        ));
    }

    fs::write(&input, fastq)?;

    let benchmark_options = BenchmarkOptions {
        process: ProcessOptions {
            sample_size: 32,
            batch_reads: 6,
            backend_preference: BackendPreference::Cpu,
            forced_adapter: None,
            min_quality_override: Some(20),
            kmer_size_override: None,
                    forced_platform: None,
                    forced_experiment: None,
        },
        rounds: 2,
        session_mode: BenchmarkSessionMode::ReuseSession,
    };
    let mut benchmark = benchmark_file(&input, &benchmark_options)?;
    append_benchmark_history(
        &history,
        Some("cpu-reuse"),
        &input,
        None,
        &benchmark_options,
        &mut benchmark,
    )?;
    assert!(
        benchmark
            .notes
            .iter()
            .any(|note| note.contains("Wrote benchmark history entry"))
    );

    let comparison_options = BenchmarkComparisonOptions {
        process: ProcessOptions {
            sample_size: 32,
            batch_reads: 6,
            backend_preference: BackendPreference::Cpu,
            forced_adapter: None,
            min_quality_override: Some(20),
            kmer_size_override: None,
                    forced_platform: None,
                    forced_experiment: None,
        },
        rounds: 2,
    };
    let mut comparison = benchmark_compare_file(&input, &comparison_options)?;
    append_benchmark_comparison_history(
        &history,
        Some("cpu-compare"),
        &input,
        None,
        &comparison_options,
        &mut comparison,
    )?;
    assert!(
        comparison
            .notes
            .iter()
            .any(|note| note.contains("Wrote benchmark comparison history entry"))
    );

    let lines: Vec<_> = fs::read_to_string(&history)?
        .lines()
        .map(serde_json::from_str::<serde_json::Value>)
        .collect::<std::result::Result<_, _>>()?;
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["kind"], "benchmark");
    assert_eq!(lines[0]["label"], "cpu-reuse");
    assert_eq!(lines[0]["session_mode"], "reuse_session");
    assert_eq!(lines[1]["kind"], "benchmark_compare");
    assert_eq!(lines[1]["label"], "cpu-compare");
    assert_eq!(lines[1]["cold_start_mode"], "cold_start");
    assert_eq!(lines[1]["reuse_session_mode"], "reuse_session");
    assert_eq!(lines[1]["requested_backend"], "cpu");

    Ok(())
}

#[test]
fn history_report_summarizes_recorded_entries() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("history-report.fastq");
    let history = dir.path().join("history").join("benchmarks.jsonl");
    let core = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let adapter = "AGATCGGAAGAGC";
    let mut fastq = String::new();

    for index in 0..12 {
        fastq.push_str(&format!(
            "@hr-{index}\n{core}\n+\n{}\n",
            "I".repeat(core.len())
        ));
    }
    for index in 12..18 {
        let qualities = format!(
            "{}!{}{}",
            "I".repeat(18),
            "I".repeat(core.len() - 19),
            "!".repeat(adapter.len())
        );
        let mut corrupted = core.as_bytes().to_vec();
        corrupted[18] = b'T';
        let corrupted = String::from_utf8(corrupted)?;
        fastq.push_str(&format!(
            "@hr-{index}\n{}{}\n+\n{}\n",
            corrupted, adapter, qualities
        ));
    }

    fs::write(&input, fastq)?;
    let benchmark_options = BenchmarkOptions {
        process: ProcessOptions {
            sample_size: 32,
            batch_reads: 6,
            backend_preference: BackendPreference::Cpu,
            forced_adapter: None,
            min_quality_override: Some(20),
            kmer_size_override: None,
                    forced_platform: None,
                    forced_experiment: None,
        },
        rounds: 2,
        session_mode: BenchmarkSessionMode::ReuseSession,
    };
    let mut benchmark = benchmark_file(&input, &benchmark_options)?;
    append_benchmark_history(
        &history,
        Some("history-reuse"),
        &input,
        None,
        &benchmark_options,
        &mut benchmark,
    )?;

    let comparison_options = BenchmarkComparisonOptions {
        process: ProcessOptions {
            sample_size: 32,
            batch_reads: 6,
            backend_preference: BackendPreference::Cpu,
            forced_adapter: None,
            min_quality_override: Some(20),
            kmer_size_override: None,
                    forced_platform: None,
                    forced_experiment: None,
        },
        rounds: 2,
    };
    let mut comparison = benchmark_compare_file(&input, &comparison_options)?;
    append_benchmark_comparison_history(
        &history,
        Some("history-compare"),
        &input,
        None,
        &comparison_options,
        &mut comparison,
    )?;

    let report = read_history_report(
        &history,
        &HistoryReportOptions {
            limit: 10,
            ..HistoryReportOptions::default()
        },
    )?;
    assert_eq!(report.total_entries, 2);
    assert_eq!(report.benchmark_entries, 1);
    assert_eq!(report.benchmark_compare_entries, 1);
    assert!(report.labels.iter().any(|label| label == "history-reuse"));
    assert!(report.labels.iter().any(|label| label == "history-compare"));
    assert_eq!(report.requested_backends, vec![BackendPreference::Cpu]);
    assert!(report.latest_entry.is_some());
    assert_eq!(report.recent_entries.len(), 2);
    assert!(report.comparison_stats.latest_raw_speedup.is_some());
    assert!(
        report
            .notes
            .iter()
            .any(|note| note.contains("Loaded 2 history entries"))
    );

    Ok(())
}

#[test]
fn synthetic_evaluation_reports_publication_metrics() -> Result<()> {
    let report = evaluate_synthetic_truth(&EvaluationOptions {
        sample_size: 128,
        batch_reads: 8,
        backend_preference: BackendPreference::Cpu,
        min_quality_override: Some(20),
        reads_per_scenario: 6,
    })?;

    assert_eq!(report.suite_name, "synthetic_truth_suite_v1");
    assert_eq!(report.total_scenarios, 5);
    assert_eq!(report.reads_per_scenario, 6);
    assert_eq!(report.aggregate.total_reads, 30);
    assert!(report.aggregate.exact_match_rate >= 0.95);
    assert!(report.aggregate.retention.f1 >= 0.95);
    assert!(report.aggregate.trimming.f1 >= 0.95);
    assert!(report.aggregate.correction.recall >= 0.95);
    let discard = report
        .scenarios
        .iter()
        .find(|scenario| scenario.name == "discard_short_insert")
        .expect("discard scenario should exist");
    assert_eq!(discard.observed_retained_reads, 0);
    assert_eq!(discard.expected_retained_reads, 0);

    Ok(())
}

#[test]
fn paired_synthetic_evaluation_reports_publication_metrics() -> Result<()> {
    let report = evaluate_paired_synthetic_truth(&EvaluationOptions {
        sample_size: 128,
        batch_reads: 8,
        backend_preference: BackendPreference::Cpu,
        min_quality_override: Some(20),
        reads_per_scenario: 6,
    })?;

    assert_eq!(report.suite_name, "synthetic_truth_paired_suite_v1");
    assert_eq!(report.total_scenarios, 5);
    assert_eq!(report.pairs_per_scenario, 6);
    assert_eq!(report.aggregate.total_pairs, 30);
    assert!(report.aggregate.exact_match_rate >= 0.95);
    assert!(report.aggregate.retention.f1 >= 0.95);
    assert!(report.aggregate.trimming.f1 >= 0.95);
    assert!(report.aggregate.correction.recall >= 0.95);
    let discard = report
        .scenarios
        .iter()
        .find(|scenario| scenario.name == "paired_discard_short_insert")
        .expect("paired discard scenario should exist");
    assert_eq!(discard.observed_retained_pairs, 0);
    assert_eq!(discard.expected_retained_pairs, 0);

    Ok(())
}

#[test]
fn evaluate_cli_reports_json_metrics() -> Result<()> {
    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("evaluate")
        .arg("--backend")
        .arg("cpu")
        .arg("--reads-per-scenario")
        .arg("4")
        .arg("--json")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["suite_name"], "synthetic_truth_suite_v1");
    assert_eq!(report["total_scenarios"], 5);
    assert_eq!(report["reads_per_scenario"], 4);
    assert_eq!(report["aggregate"]["total_reads"], 20);
    assert_eq!(report["scenarios"][0]["name"], "clean_retention");

    Ok(())
}

#[test]
fn paper_artifacts_generate_reproducible_bundle() -> Result<()> {
    let dir = tempdir()?;
    let output_dir = dir.path().join("paper-artifacts");
    let report = generate_paper_artifacts(&PaperArtifactsOptions {
        output_dir: output_dir.clone(),
        sample_size: 128,
        batch_reads: 8,
        reads_per_scenario: 4,
        benchmark_rounds: 2,
        accelerator_backend: BackendPreference::Cpu,
        min_quality_override: Some(20),
    })?;

    assert_eq!(report.reads_per_scenario, 4);
    assert_eq!(report.benchmark_rounds, 2);
    assert!(report.artifacts.len() >= 10);
    assert_eq!(report.paired_cpu_evaluation.total_scenarios, 5);
    assert_eq!(report.paired_accelerator_evaluation.total_scenarios, 5);
    assert!(report.parity.paired_end_exact_match_delta.abs() <= 0.05);
    assert!(output_dir.join("paper_artifacts.json").exists());
    assert!(
        output_dir
            .join("data")
            .join("scenario_accuracy.csv")
            .exists()
    );
    assert!(
        output_dir
            .join("figures")
            .join("scenario_exact_match.svg")
            .exists()
    );
    assert!(
        output_dir
            .join("figures")
            .join("backend_throughput.svg")
            .exists()
    );
    assert!(
        output_dir
            .join("data")
            .join("paired_scenario_accuracy.csv")
            .exists()
    );
    assert!(output_dir.join("data").join("backend_parity.csv").exists());
    assert!(
        output_dir
            .join("figures")
            .join("paired_scenario_exact_match.svg")
            .exists()
    );
    assert!(
        output_dir
            .join("figures")
            .join("backend_parity.svg")
            .exists()
    );

    Ok(())
}

#[test]
fn paper_artifacts_cli_reports_json_bundle() -> Result<()> {
    let dir = tempdir()?;
    let output_dir = dir.path().join("paper-cli");

    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("paper-artifacts")
        .arg(&output_dir)
        .arg("--backend")
        .arg("cpu")
        .arg("--reads-per-scenario")
        .arg("4")
        .arg("--benchmark-rounds")
        .arg("2")
        .arg("--json")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["reads_per_scenario"], 4);
    assert_eq!(report["benchmark_rounds"], 2);
    assert_eq!(report["paired_cpu_evaluation"]["total_scenarios"], 5);
    assert_eq!(
        report["paired_accelerator_evaluation"]["total_scenarios"],
        5
    );
    assert!(report["parity"]["paired_end_exact_match_delta"].is_number());
    assert!(
        report["artifacts"]
            .as_array()
            .is_some_and(|items| !items.is_empty())
    );
    assert!(
        output_dir
            .join("figures")
            .join("aggregate_quality.svg")
            .exists()
    );
    assert!(
        output_dir
            .join("figures")
            .join("backend_parity.svg")
            .exists()
    );

    Ok(())
}

#[test]
fn study_manifest_bootstrap_generates_canonical_manifest() -> Result<()> {
    let dir = tempdir()?;
    let inventory = write_study_inventory_fixture(dir.path())?;
    let output_manifest = dir.path().join("generated_manifest.tsv");

    let report = bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: output_manifest.clone(),
        default_baseline_name: Some("fastp".to_string()),
        download_root: dir.path().join("downloads"),
    })?;

    assert_eq!(report.summary.datasets, 2);
    assert_eq!(report.summary.paired_datasets, 1);
    assert_eq!(report.summary.datasets_with_generated_id, 1);
    assert_eq!(report.summary.datasets_with_accession, 2);
    assert_eq!(report.summary.datasets_with_baseline_name, 2);
    assert_eq!(report.summary.datasets_with_downstream_metrics, 2);
    assert!(output_manifest.exists());
    assert!(
        dir.path()
            .join("generated_manifest.provenance_summary.csv")
            .exists()
    );

    let manifest_text = fs::read_to_string(&output_manifest)?;
    assert!(manifest_text.contains("dataset_id\taccession\tcitation\tinput1"));
    assert!(manifest_text.contains("SRR100001"));
    assert!(manifest_text.contains("sc_case"));
    assert!(manifest_text.contains("fastp"));
    assert!(manifest_text.contains("cutadapt"));

    let study_output = dir.path().join("study-from-bootstrap");
    let study_report = generate_study_artifacts(&StudyArtifactsOptions {
        manifest_path: output_manifest,
        output_dir: study_output.clone(),
        sample_size: 64,
        batch_reads: 4,
        benchmark_rounds: 2,
        backend_preference: BackendPreference::Cpu,
        min_quality_override: Some(20),
    })?;
    assert_eq!(study_report.aggregate.datasets, 2);
    assert_eq!(study_report.comparison.datasets_with_baseline, 2);
    assert!(study_output.join("study_artifacts.json").exists());

    Ok(())
}

#[test]
fn study_manifest_bootstraps_public_accession_export() -> Result<()> {
    let dir = tempdir()?;
    let inventory = write_public_accession_metadata_fixture(dir.path())?;
    let download_root = dir.path().join("downloads");
    let output_manifest = dir.path().join("public_manifest.tsv");

    let report = bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: output_manifest.clone(),
        default_baseline_name: Some("fastp".to_string()),
        download_root: download_root.clone(),
    })?;

    assert_eq!(report.summary.datasets, 2);
    assert_eq!(report.summary.paired_datasets, 1);
    assert_eq!(report.summary.datasets_with_generated_id, 2);
    assert_eq!(report.summary.datasets_with_accession, 2);
    assert_eq!(report.summary.datasets_with_citation, 2);
    assert_eq!(report.summary.datasets_with_expected_platform, 2);
    assert_eq!(report.summary.datasets_with_expected_experiment, 2);
    assert_eq!(report.summary.datasets_with_baseline_name, 2);
    assert!(output_manifest.exists());

    let manifest_text = fs::read_to_string(&output_manifest)?;
    assert!(manifest_text.contains("downloads/ERR300001/ERR300001.fastq.gz"));
    assert!(manifest_text.contains("downloads/SRR200001/SRR200001_1.fastq.gz"));
    assert!(manifest_text.contains("downloads/SRR200001/SRR200001_2.fastq.gz"));
    assert!(manifest_text.contains("Human WGS benchmarking study"));
    assert!(manifest_text.contains("PBMC single-cell atlas"));
    assert!(manifest_text.contains("fastp"));

    write_public_accession_download_fixture(&download_root)?;

    let study_output = dir.path().join("study-from-public-accessions");
    let study_report = generate_study_artifacts(&StudyArtifactsOptions {
        manifest_path: output_manifest,
        output_dir: study_output.clone(),
        sample_size: 64,
        batch_reads: 4,
        benchmark_rounds: 2,
        backend_preference: BackendPreference::Cpu,
        min_quality_override: Some(20),
    })?;
    assert_eq!(study_report.aggregate.datasets, 2);
    assert_eq!(study_report.comparison.datasets_with_baseline, 2);
    assert!(study_output.join("study_artifacts.json").exists());

    Ok(())
}

#[test]
fn study_manifest_public_accession_export_falls_back_to_layout_placeholders() -> Result<()> {
    let dir = tempdir()?;
    let inventory = write_public_accession_layout_only_fixture(dir.path())?;
    let output_manifest = dir.path().join("public_layout_manifest.tsv");

    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: output_manifest.clone(),
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;

    let manifest_text = fs::read_to_string(&output_manifest)?;
    assert!(manifest_text.contains("downloads/SRR400001/SRR400001_1.fastq.gz"));
    assert!(manifest_text.contains("downloads/SRR400001/SRR400001_2.fastq.gz"));
    assert!(manifest_text.contains("manifest assumes paired FASTQs"));

    Ok(())
}

#[test]
fn study_discover_builds_inventory_template() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let output_inventory = dir.path().join("discovered_inventory.tsv");

    let report = discover_study_inventory(&StudyDiscoverOptions {
        input_dir: input_dir.clone(),
        output_path: output_inventory.clone(),
        recursive: true,
    })?;

    assert_eq!(report.summary.files_scanned, 4);
    assert_eq!(report.summary.datasets, 2);
    assert_eq!(report.summary.paired_datasets, 1);
    assert_eq!(report.summary.single_end_datasets, 1);
    assert_eq!(report.summary.accession_labeled_datasets, 2);
    assert!(output_inventory.exists());

    let inventory_text = fs::read_to_string(&output_inventory)?;
    assert!(inventory_text.contains("ERR300001"));
    assert!(inventory_text.contains("SRR200001"));

    let canonical_manifest = dir.path().join("canonical_from_discovery.tsv");
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: output_inventory.clone(),
        output_path: canonical_manifest.clone(),
        default_baseline_name: Some("fastp".to_string()),
        download_root: dir.path().join("downloads"),
    })?;

    let study_output = dir.path().join("study-from-discovery");
    let study_report = generate_study_artifacts(&StudyArtifactsOptions {
        manifest_path: canonical_manifest,
        output_dir: study_output.clone(),
        sample_size: 64,
        batch_reads: 4,
        benchmark_rounds: 2,
        backend_preference: BackendPreference::Cpu,
        min_quality_override: Some(20),
    })?;
    assert_eq!(study_report.aggregate.datasets, 2);
    assert!(study_output.join("study_artifacts.json").exists());

    Ok(())
}

#[test]
fn study_discover_can_disable_recursive_scan() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let output_inventory = dir.path().join("non_recursive_inventory.tsv");

    let report = discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: output_inventory.clone(),
        recursive: false,
    })?;

    assert_eq!(report.summary.datasets, 0);
    assert_eq!(report.summary.fastq_files, 0);
    assert_eq!(report.summary.files_scanned, 1);
    assert!(output_inventory.exists());

    Ok(())
}

#[test]
fn study_discover_cli_reports_json_bundle() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let output_inventory = dir.path().join("discovered_inventory.tsv");

    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("study-discover")
        .arg(&input_dir)
        .arg(&output_inventory)
        .arg("--json")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["summary"]["files_scanned"], 4);
    assert_eq!(report["summary"]["datasets"], 2);
    assert_eq!(report["summary"]["paired_datasets"], 1);
    assert_eq!(report["summary"]["accession_labeled_datasets"], 2);
    assert!(output_inventory.exists());

    Ok(())
}

#[test]
fn study_manifest_cli_reports_json_bundle() -> Result<()> {
    let dir = tempdir()?;
    let inventory = write_study_inventory_fixture(dir.path())?;
    let output_manifest = dir.path().join("generated_manifest.tsv");

    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("study-manifest")
        .arg(&inventory)
        .arg(&output_manifest)
        .arg("--default-baseline-name")
        .arg("fastp")
        .arg("--json")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["summary"]["datasets"], 2);
    assert_eq!(report["summary"]["paired_datasets"], 1);
    assert_eq!(report["summary"]["datasets_with_generated_id"], 1);
    assert_eq!(report["summary"]["datasets_with_baseline_name"], 2);
    assert_eq!(report["default_baseline_name"], "fastp");
    assert!(output_manifest.exists());
    assert!(
        dir.path()
            .join("generated_manifest.provenance_summary.csv")
            .exists()
    );

    Ok(())
}

#[test]
fn study_manifest_cli_bootstraps_public_accession_export() -> Result<()> {
    let dir = tempdir()?;
    let inventory = write_public_accession_metadata_fixture(dir.path())?;
    let output_manifest = dir.path().join("public_manifest.tsv");

    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("study-manifest")
        .arg(&inventory)
        .arg(&output_manifest)
        .arg("--default-baseline-name")
        .arg("fastp")
        .arg("--download-root")
        .arg(dir.path().join("downloads"))
        .arg("--json")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["summary"]["datasets"], 2);
    assert_eq!(report["summary"]["paired_datasets"], 1);
    assert_eq!(report["summary"]["datasets_with_generated_id"], 2);
    assert_eq!(report["summary"]["datasets_with_expected_platform"], 2);
    assert_eq!(report["summary"]["datasets_with_expected_experiment"], 2);
    assert_eq!(report["default_baseline_name"], "fastp");
    assert!(output_manifest.exists());

    Ok(())
}

#[test]
fn study_fetch_metadata_writes_phase35_compatible_output() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("accessions.txt");
    fs::write(&input, "SRR200001\nSRR200002\nSRR200001\nSRR999999\n")?;
    let output_metadata = dir.path().join("fetched_metadata.tsv");
    let output_manifest = dir.path().join("public_manifest.tsv");

    let (base_url, server) = spawn_tsv_http_server(vec![sample_public_metadata_response()])?;
    let report = fetch_public_metadata(&StudyFetchMetadataOptions {
        input_path: input.clone(),
        output_path: output_metadata.clone(),
        base_url: base_url.clone(),
        geo_base_url: base_url,
        chunk_size: 10,
        cache_dir: None,
        retries: 2,
        resume_existing: false,
    })?;
    join_http_server(server)?;

    assert_eq!(report.summary.requested_accessions, 4);
    assert_eq!(report.summary.unique_accessions, 3);
    assert_eq!(report.summary.fetched_records, 2);
    assert_eq!(report.summary.matched_accessions, 2);
    assert_eq!(report.summary.unmatched_accessions, 1);
    assert_eq!(report.summary.failed_accessions, 0);
    assert_eq!(report.summary.cache_hits, 0);
    assert_eq!(report.summary.remote_fetches, 1);
    assert_eq!(report.summary.retried_chunks, 0);
    assert_eq!(report.unmatched_accessions, vec!["SRR999999".to_string()]);
    assert!(report.failed_accessions.is_empty());
    assert!(output_metadata.exists());
    assert!(Path::new(&report.status_path).exists());
    assert!(Path::new(&report.cache_dir).exists());

    let fetched = fs::read_to_string(&output_metadata)?;
    assert!(fetched.contains("run_accession\tstudy_accession\tstudy_title"));
    assert!(fetched.contains("SRR200001"));
    assert!(fetched.contains("SRR200002"));
    assert!(fetched.contains("query_accessions"));
    let status_csv = fs::read_to_string(&report.status_path)?;
    assert!(status_csv.contains("SRR200001,matched"));
    assert!(status_csv.contains("SRR999999,unmatched"));

    let manifest = bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: output_metadata,
        output_path: output_manifest.clone(),
        default_baseline_name: Some("fastp".to_string()),
        download_root: dir.path().join("downloads"),
    })?;
    assert_eq!(manifest.summary.datasets, 2);
    assert_eq!(manifest.summary.paired_datasets, 1);
    assert_eq!(manifest.summary.datasets_with_accession, 2);
    assert_eq!(manifest.summary.datasets_with_expected_platform, 2);
    assert!(output_manifest.exists());

    let rendered_manifest = fs::read_to_string(output_manifest)?;
    assert!(rendered_manifest.contains("downloads/SRR200001/SRR200001_1.fastq.gz"));
    assert!(rendered_manifest.contains("SRR200002.fastq.gz"));

    Ok(())
}

#[test]
fn study_fetch_metadata_falls_back_to_single_accession_requests_when_chunk_is_empty() -> Result<()>
{
    let dir = tempdir()?;
    let input = dir.path().join("accessions.txt");
    fs::write(&input, "SRR200001\nSRR200002\n")?;
    let output_metadata = dir.path().join("fallback_metadata.tsv");

    let full_body = sample_public_metadata_response();
    let mut lines = full_body.lines();
    let header = format!("{}\n", lines.next().unwrap_or_default());
    let rows: HashMap<String, String> = lines
        .map(|line| {
            let accession = line.split('\t').next().unwrap_or_default().to_string();
            (accession, line.to_string())
        })
        .collect();
    let (base_url, server) = spawn_tsv_http_server(vec![
        header.clone(),
        format!(
            "{}{}\n",
            header,
            rows.get("SRR200001").cloned().unwrap_or_default()
        ),
        format!(
            "{}{}\n",
            header,
            rows.get("SRR200002").cloned().unwrap_or_default()
        ),
    ])?;

    let report = fetch_public_metadata(&StudyFetchMetadataOptions {
        input_path: input,
        output_path: output_metadata.clone(),
        base_url: base_url.clone(),
        geo_base_url: base_url,
        chunk_size: 10,
        cache_dir: None,
        retries: 0,
        resume_existing: false,
    })?;
    join_http_server(server)?;

    assert_eq!(report.summary.requested_accessions, 2);
    assert_eq!(report.summary.matched_accessions, 2);
    assert_eq!(report.summary.fetched_records, 2);
    assert_eq!(report.summary.unmatched_accessions, 0);
    assert_eq!(report.summary.remote_fetches, 1);
    let fetched = fs::read_to_string(output_metadata)?;
    assert!(fetched.contains("SRR200001"));
    assert!(fetched.contains("SRR200002"));

    Ok(())
}

#[test]
fn study_fetch_metadata_expands_study_and_project_accessions() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("accessions.txt");
    fs::write(&input, "SRP300001\nPRJNA300001\nSRP300001\n")?;
    let output_metadata = dir.path().join("expanded_metadata.tsv");
    let output_manifest = dir.path().join("expanded_manifest.tsv");

    let (base_url, server) = spawn_tsv_http_server(vec![sample_public_study_expansion_response()])?;
    let report = fetch_public_metadata(&StudyFetchMetadataOptions {
        input_path: input,
        output_path: output_metadata.clone(),
        base_url: base_url.clone(),
        geo_base_url: base_url,
        chunk_size: 10,
        cache_dir: None,
        retries: 2,
        resume_existing: false,
    })?;
    join_http_server(server)?;

    assert_eq!(report.summary.requested_accessions, 3);
    assert_eq!(report.summary.unique_accessions, 2);
    assert_eq!(report.summary.fetched_records, 2);
    assert_eq!(report.summary.matched_accessions, 2);
    assert_eq!(report.summary.unmatched_accessions, 0);
    assert_eq!(report.summary.failed_accessions, 0);
    assert!(
        report
            .notes
            .iter()
            .any(|note| note.contains("expanded into 2 run-level metadata row"))
    );

    let fetched = fs::read_to_string(&output_metadata)?;
    assert!(fetched.contains("PRJNA300001"));
    assert!(fetched.contains("SRP300001;PRJNA300001"));
    assert!(fetched.contains("SRR300001"));
    assert!(fetched.contains("SRR300002"));

    let manifest = bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: output_metadata,
        output_path: output_manifest.clone(),
        default_baseline_name: Some("fastp".to_string()),
        download_root: dir.path().join("downloads"),
    })?;
    assert_eq!(manifest.summary.datasets, 2);
    assert_eq!(manifest.summary.paired_datasets, 2);
    assert_eq!(manifest.summary.datasets_with_accession, 2);
    assert!(output_manifest.exists());

    Ok(())
}

#[test]
fn study_fetch_metadata_resolves_geo_series_accession() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("accessions.txt");
    fs::write(&input, "GSE400001\n")?;
    let output_metadata = dir.path().join("geo_metadata.tsv");
    let output_manifest = dir.path().join("geo_manifest.tsv");

    let (ena_base_url, ena_server) =
        spawn_tsv_http_server(vec![sample_public_study_expansion_response()])?;
    let (geo_base_url, geo_server) = spawn_scripted_http_server(vec![HttpFixtureResponse::text(
        sample_geo_series_bridge_response(),
    )])?;
    let report = fetch_public_metadata(&StudyFetchMetadataOptions {
        input_path: input,
        output_path: output_metadata.clone(),
        base_url: ena_base_url,
        geo_base_url,
        chunk_size: 10,
        cache_dir: None,
        retries: 1,
        resume_existing: false,
    })?;
    join_http_server(geo_server)?;
    join_http_server(ena_server)?;

    assert_eq!(report.summary.requested_accessions, 1);
    assert_eq!(report.summary.unique_accessions, 1);
    assert_eq!(report.summary.geo_bridge_accessions, 1);
    assert_eq!(report.summary.geo_bridge_resolved_accessions, 2);
    assert_eq!(report.summary.matched_accessions, 1);
    assert_eq!(report.summary.fetched_records, 2);
    let fetched = fs::read_to_string(&output_metadata)?;
    assert!(fetched.contains("GSE400001"));
    let status_csv = fs::read_to_string(&report.status_path)?;
    assert!(status_csv.contains("GSE400001,matched,2,SRR300001;SRR300002"));
    assert!(status_csv.contains("SRP300001;PRJNA300001"));

    let manifest = bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: output_metadata,
        output_path: output_manifest.clone(),
        default_baseline_name: Some("fastp".to_string()),
        download_root: dir.path().join("downloads"),
    })?;
    assert_eq!(manifest.summary.datasets, 2);
    assert!(output_manifest.exists());

    Ok(())
}

#[test]
fn study_fetch_metadata_falls_back_to_full_geo_text_when_quick_text_has_no_public_accessions()
-> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("accessions.txt");
    fs::write(&input, "GSE400001\n")?;
    let output_metadata = dir.path().join("geo_fallback_metadata.tsv");

    let (ena_base_url, ena_server) =
        spawn_tsv_http_server(vec![sample_public_study_expansion_response()])?;
    let (geo_base_url, geo_server) = spawn_scripted_http_server(vec![
        HttpFixtureResponse::text(sample_geo_series_quick_without_public_accessions_response()),
        HttpFixtureResponse::text(sample_geo_series_bridge_response()),
    ])?;
    let report = fetch_public_metadata(&StudyFetchMetadataOptions {
        input_path: input,
        output_path: output_metadata.clone(),
        base_url: ena_base_url,
        geo_base_url,
        chunk_size: 10,
        cache_dir: None,
        retries: 1,
        resume_existing: false,
    })?;
    join_http_server(geo_server)?;
    join_http_server(ena_server)?;

    assert_eq!(report.summary.geo_bridge_accessions, 1);
    assert_eq!(report.summary.geo_bridge_resolved_accessions, 2);
    assert_eq!(report.summary.geo_bridge_fallback_accessions, 1);
    assert_eq!(report.summary.matched_accessions, 1);
    assert_eq!(report.summary.fetched_records, 2);
    assert!(
        report
            .notes
            .iter()
            .any(|note| note.contains("full-text fallback"))
    );
    let status_csv = fs::read_to_string(&report.status_path)?;
    assert!(status_csv.contains("geo_surface"));
    assert!(status_csv.contains(
        "GSE400001,matched,2,SRR300001;SRR300002,remote,1,,full_text_self,SRP300001;PRJNA300001"
    ));
    assert!(output_metadata.exists());

    Ok(())
}

#[test]
fn study_fetch_metadata_uses_cache_and_writes_status_bookkeeping() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("accessions.txt");
    let output_metadata = dir.path().join("cached_metadata.tsv");
    let cache_dir = dir.path().join("fetch_cache");
    fs::write(&input, "SRR200001\nSRR200002\n")?;

    let (base_url, server) = spawn_tsv_http_server(vec![sample_public_metadata_response()])?;
    let first_report = fetch_public_metadata(&StudyFetchMetadataOptions {
        input_path: input.clone(),
        output_path: output_metadata.clone(),
        base_url: base_url.clone(),
        geo_base_url: base_url.clone(),
        chunk_size: 10,
        cache_dir: Some(cache_dir.clone()),
        retries: 2,
        resume_existing: false,
    })?;
    join_http_server(server)?;

    assert_eq!(first_report.summary.cache_hits, 0);
    assert_eq!(first_report.summary.remote_fetches, 1);
    assert!(cache_dir.exists());

    let second_report = fetch_public_metadata(&StudyFetchMetadataOptions {
        input_path: input,
        output_path: output_metadata,
        base_url,
        geo_base_url: "http://unused.local/geo".to_string(),
        chunk_size: 10,
        cache_dir: Some(cache_dir),
        retries: 2,
        resume_existing: false,
    })?;

    assert_eq!(second_report.summary.cache_hits, 1);
    assert_eq!(second_report.summary.remote_fetches, 0);
    assert_eq!(second_report.summary.failed_accessions, 0);
    let status_csv = fs::read_to_string(&second_report.status_path)?;
    assert!(status_csv.contains("SRR200001,matched,1,SRR200001,cache,0,"));
    assert!(status_csv.contains("SRR200002,matched,1,SRR200002,cache,0,"));

    Ok(())
}

#[test]
fn study_download_materializes_public_fastqs() -> Result<()> {
    let dir = tempdir()?;
    let download_root = dir.path().join("downloads");
    let input = write_public_download_metadata_fixture("http://127.0.0.1:9", dir.path())?;

    let (base_url, server) = spawn_scripted_http_server(vec![
        HttpFixtureResponse::text(sample_download_single_fastq_response()),
        HttpFixtureResponse::text(sample_download_paired_r1_fastq_response()),
        HttpFixtureResponse::text(sample_download_paired_r2_fastq_response()),
    ])?;
    rewrite_public_download_metadata_urls(&input, &base_url)?;

    let report = download_public_fastqs(&StudyDownloadOptions {
        input_path: input,
        download_root: download_root.clone(),
        retries: 1,
        overwrite_existing: false,
    })?;
    join_http_server(server)?;

    assert_eq!(report.summary.datasets, 2);
    assert_eq!(report.summary.requested_files, 3);
    assert_eq!(report.summary.downloaded_files, 3);
    assert_eq!(report.summary.resumed_files, 0);
    assert_eq!(report.summary.skipped_existing_files, 0);
    assert_eq!(report.summary.failed_files, 0);
    assert!(
        download_root
            .join("ERR300001")
            .join("ERR300001.fastq")
            .exists()
    );
    assert!(
        download_root
            .join("SRR200001")
            .join("SRR200001_1.fastq")
            .exists()
    );
    assert!(
        download_root
            .join("SRR200001")
            .join("SRR200001_2.fastq")
            .exists()
    );
    let status_csv = fs::read_to_string(&report.status_path)?;
    assert!(status_csv.contains("downloaded"));
    assert!(status_csv.contains("ERR300001"));
    assert!(status_csv.contains("SRR200001"));

    Ok(())
}

#[test]
fn study_download_skips_existing_fastqs_on_resume() -> Result<()> {
    let dir = tempdir()?;
    let download_root = dir.path().join("downloads");
    let input = write_public_download_metadata_fixture("http://127.0.0.1:9", dir.path())?;

    let (base_url, first_server) = spawn_scripted_http_server(vec![
        HttpFixtureResponse::text(sample_download_single_fastq_response()),
        HttpFixtureResponse::text(sample_download_paired_r1_fastq_response()),
        HttpFixtureResponse::text(sample_download_paired_r2_fastq_response()),
    ])?;
    rewrite_public_download_metadata_urls(&input, &base_url)?;
    let first_report = download_public_fastqs(&StudyDownloadOptions {
        input_path: input.clone(),
        download_root: download_root.clone(),
        retries: 1,
        overwrite_existing: false,
    })?;
    join_http_server(first_server)?;
    assert_eq!(first_report.summary.downloaded_files, 3);

    let second_report = download_public_fastqs(&StudyDownloadOptions {
        input_path: input,
        download_root,
        retries: 1,
        overwrite_existing: false,
    })?;
    assert_eq!(second_report.summary.downloaded_files, 0);
    assert_eq!(second_report.summary.skipped_existing_files, 3);
    assert_eq!(second_report.summary.failed_files, 0);
    let status_csv = fs::read_to_string(&second_report.status_path)?;
    assert!(status_csv.contains("skipped_existing"));

    Ok(())
}

#[test]
fn study_fetch_metadata_cli_resolves_geo_sample_json_bundle() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("geo_accessions.txt");
    fs::write(&input, "GSM400001\n")?;
    let output_metadata = dir.path().join("gsm_metadata.tsv");

    let (ena_base_url, ena_server) =
        spawn_tsv_http_server(vec![sample_public_metadata_response()])?;
    let (geo_base_url, geo_server) = spawn_scripted_http_server(vec![HttpFixtureResponse::text(
        sample_geo_sample_bridge_response(),
    )])?;
    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("study-fetch-metadata")
        .arg(&input)
        .arg(&output_metadata)
        .arg("--base-url")
        .arg(&ena_base_url)
        .arg("--geo-base-url")
        .arg(&geo_base_url)
        .arg("--json")
        .output()?;
    join_http_server(geo_server)?;
    join_http_server(ena_server)?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["summary"]["requested_accessions"], 1);
    assert_eq!(report["summary"]["geo_bridge_accessions"], 1);
    assert_eq!(report["summary"]["geo_bridge_resolved_accessions"], 1);
    assert_eq!(report["summary"]["matched_accessions"], 1);
    assert_eq!(report["summary"]["fetched_records"], 1);
    assert_eq!(report["summary"]["failed_accessions"], 0);
    assert!(output_metadata.exists());

    Ok(())
}

#[test]
fn study_download_cli_reports_json_bundle() -> Result<()> {
    let dir = tempdir()?;
    let download_root = dir.path().join("downloads");
    let input = write_public_download_metadata_fixture("http://127.0.0.1:9", dir.path())?;

    let (base_url, server) = spawn_scripted_http_server(vec![
        HttpFixtureResponse::text(sample_download_single_fastq_response()),
        HttpFixtureResponse::text(sample_download_paired_r1_fastq_response()),
        HttpFixtureResponse::text(sample_download_paired_r2_fastq_response()),
    ])?;
    rewrite_public_download_metadata_urls(&input, &base_url)?;
    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("study-download")
        .arg(&input)
        .arg(&download_root)
        .arg("--json")
        .output()?;
    join_http_server(server)?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["summary"]["datasets"], 2);
    assert_eq!(report["summary"]["requested_files"], 3);
    assert_eq!(report["summary"]["downloaded_files"], 3);
    assert_eq!(report["summary"]["failed_files"], 0);
    assert!(report["status_path"].as_str().is_some());

    Ok(())
}

#[test]
fn publication_bundle_cli_emits_study_and_paper_artifacts() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("accessions.txt");
    fs::write(&input, "ERR300001\nSRR200001\n")?;
    let output_dir = dir.path().join("publication_bundle");

    let (base_url, server) = spawn_public_download_http_server()?;
    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("publication-bundle")
        .arg(&input)
        .arg(&output_dir)
        .arg("--base-url")
        .arg(format!("{base_url}/ena"))
        .arg("--geo-base-url")
        .arg(format!("{base_url}/geo"))
        .arg("--backend")
        .arg("cpu")
        .arg("--benchmark-rounds")
        .arg("1")
        .arg("--reads-per-scenario")
        .arg("4")
        .arg("--json")
        .output()?;
    join_http_server(server)?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["fetch"]["summary"]["fetched_records"], 2);
    assert_eq!(report["download"]["summary"]["downloaded_files"], 3);
    assert_eq!(report["study_artifacts"]["aggregate"]["datasets"], 2);
    assert_eq!(
        report["paper_artifacts"]["requested_accelerator_backend"],
        "cpu"
    );
    assert!(output_dir.join("publication_bundle.json").exists());
    assert!(output_dir.join("manuscript_summary.txt").exists());
    assert!(
        output_dir
            .join("study_artifacts")
            .join("study_artifacts.json")
            .exists()
    );
    assert!(
        output_dir
            .join("paper_artifacts")
            .join("paper_artifacts.json")
            .exists()
    );

    Ok(())
}

#[test]
fn study_fetch_metadata_cli_reports_geo_fallback_json_bundle() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("geo_fallback_accessions.txt");
    fs::write(&input, "GSE400001\n")?;
    let output_metadata = dir.path().join("geo_fallback_cli_metadata.tsv");

    let (ena_base_url, ena_server) =
        spawn_tsv_http_server(vec![sample_public_study_expansion_response()])?;
    let (geo_base_url, geo_server) = spawn_scripted_http_server(vec![
        HttpFixtureResponse::text(sample_geo_series_quick_without_public_accessions_response()),
        HttpFixtureResponse::text(sample_geo_series_bridge_response()),
    ])?;
    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("study-fetch-metadata")
        .arg(&input)
        .arg(&output_metadata)
        .arg("--base-url")
        .arg(&ena_base_url)
        .arg("--geo-base-url")
        .arg(&geo_base_url)
        .arg("--json")
        .output()?;
    join_http_server(geo_server)?;
    join_http_server(ena_server)?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["summary"]["geo_bridge_accessions"], 1);
    assert_eq!(report["summary"]["geo_bridge_resolved_accessions"], 2);
    assert_eq!(report["summary"]["geo_bridge_fallback_accessions"], 1);
    assert_eq!(report["summary"]["matched_accessions"], 1);
    let status_path = report["status_path"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing status_path"))?;
    let status_csv = fs::read_to_string(status_path)?;
    assert!(status_csv.contains("full_text_self"));

    Ok(())
}

#[test]
fn study_fetch_metadata_resume_reuses_existing_rows_and_statuses() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("accessions.txt");
    fs::write(&input, "SRR200001\nSRR999999\n")?;
    let output_metadata = dir.path().join("resumed_metadata.tsv");

    let (base_url, server) = spawn_tsv_http_server(vec![sample_public_metadata_response()])?;
    let first_report = fetch_public_metadata(&StudyFetchMetadataOptions {
        input_path: input.clone(),
        output_path: output_metadata.clone(),
        base_url: base_url.clone(),
        geo_base_url: base_url,
        chunk_size: 10,
        cache_dir: None,
        retries: 1,
        resume_existing: false,
    })?;
    join_http_server(server)?;

    assert_eq!(first_report.summary.fetched_records, 1);
    assert_eq!(first_report.summary.matched_accessions, 1);
    assert_eq!(first_report.summary.unmatched_accessions, 1);

    let resumed_report = fetch_public_metadata(&StudyFetchMetadataOptions {
        input_path: input,
        output_path: output_metadata.clone(),
        base_url: "http://unused.local/ena".to_string(),
        geo_base_url: "http://unused.local/geo".to_string(),
        chunk_size: 10,
        cache_dir: None,
        retries: 1,
        resume_existing: true,
    })?;

    assert!(resumed_report.resume_existing);
    assert_eq!(resumed_report.summary.resumed_accessions, 2);
    assert_eq!(resumed_report.summary.resumed_records, 1);
    assert_eq!(resumed_report.summary.remote_fetches, 0);
    assert_eq!(resumed_report.summary.cache_hits, 0);
    assert_eq!(resumed_report.summary.fetched_records, 1);
    assert_eq!(resumed_report.summary.matched_accessions, 1);
    assert_eq!(resumed_report.summary.unmatched_accessions, 1);
    assert!(resumed_report.failed_accessions.is_empty());
    let fetched = fs::read_to_string(&output_metadata)?;
    assert!(fetched.contains("SRR200001"));
    assert!(!fetched.contains("SRR200002"));
    let status_csv = fs::read_to_string(&resumed_report.status_path)?;
    assert!(status_csv.contains("SRR200001,matched"));
    assert!(status_csv.contains("SRR999999,unmatched"));

    Ok(())
}

#[test]
fn study_fetch_metadata_resume_fetches_only_new_accessions() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("accessions.txt");
    fs::write(&input, "SRR200001\n")?;
    let output_metadata = dir.path().join("append_metadata.tsv");

    let (first_base_url, first_server) =
        spawn_tsv_http_server(vec![sample_public_metadata_response()])?;
    let first_report = fetch_public_metadata(&StudyFetchMetadataOptions {
        input_path: input.clone(),
        output_path: output_metadata.clone(),
        base_url: first_base_url.clone(),
        geo_base_url: first_base_url,
        chunk_size: 10,
        cache_dir: None,
        retries: 1,
        resume_existing: false,
    })?;
    join_http_server(first_server)?;

    assert_eq!(first_report.summary.fetched_records, 1);
    fs::write(&input, "SRR200001\nSRR200002\n")?;

    let (second_base_url, second_server) =
        spawn_tsv_http_server(vec![sample_public_metadata_response()])?;
    let resumed_report = fetch_public_metadata(&StudyFetchMetadataOptions {
        input_path: input,
        output_path: output_metadata.clone(),
        base_url: second_base_url.clone(),
        geo_base_url: second_base_url,
        chunk_size: 10,
        cache_dir: None,
        retries: 1,
        resume_existing: true,
    })?;
    join_http_server(second_server)?;

    assert_eq!(resumed_report.summary.resumed_accessions, 1);
    assert_eq!(resumed_report.summary.resumed_records, 1);
    assert_eq!(resumed_report.summary.remote_fetches, 1);
    assert_eq!(resumed_report.summary.cache_hits, 0);
    assert_eq!(resumed_report.summary.matched_accessions, 2);
    assert_eq!(resumed_report.summary.fetched_records, 2);
    let fetched = fs::read_to_string(&output_metadata)?;
    assert!(fetched.contains("SRR200001"));
    assert!(fetched.contains("SRR200002"));
    let status_csv = fs::read_to_string(&resumed_report.status_path)?;
    assert!(status_csv.contains("SRR200001,matched"));
    assert!(status_csv.contains("SRR200002,matched"));

    Ok(())
}

#[test]
fn study_fetch_metadata_retries_failed_chunk_and_reports_attempts() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("accessions.txt");
    let output_metadata = dir.path().join("retried_metadata.tsv");
    let cache_dir = dir.path().join("fetch_cache");
    fs::write(&input, "SRR200001\n")?;

    let (base_url, server) = spawn_scripted_http_server(vec![
        HttpFixtureResponse::status(500, "temporary failure"),
        HttpFixtureResponse::tsv(sample_public_metadata_response()),
    ])?;
    let report = fetch_public_metadata(&StudyFetchMetadataOptions {
        input_path: input,
        output_path: output_metadata,
        base_url: base_url.clone(),
        geo_base_url: base_url,
        chunk_size: 10,
        cache_dir: Some(cache_dir),
        retries: 1,
        resume_existing: false,
    })?;
    join_http_server(server)?;

    assert_eq!(report.summary.fetched_records, 1);
    assert_eq!(report.summary.matched_accessions, 1);
    assert_eq!(report.summary.retried_chunks, 1);
    assert_eq!(report.summary.remote_fetches, 1);
    let status_csv = fs::read_to_string(&report.status_path)?;
    assert!(status_csv.contains("SRR200001,matched,1,SRR200001,remote,2,"));

    Ok(())
}

#[test]
fn study_fetch_metadata_records_failed_accessions_without_dropping_successes() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("accessions.txt");
    let output_metadata = dir.path().join("partial_metadata.tsv");
    let cache_dir = dir.path().join("fetch_cache");
    fs::write(&input, "SRR200001\nSRR999999\n")?;

    let (base_url, server) = spawn_scripted_http_server(vec![
        HttpFixtureResponse::tsv(sample_public_metadata_response()),
        HttpFixtureResponse::status(500, "persistent failure"),
    ])?;
    let report = fetch_public_metadata(&StudyFetchMetadataOptions {
        input_path: input,
        output_path: output_metadata.clone(),
        base_url: base_url.clone(),
        geo_base_url: base_url,
        chunk_size: 1,
        cache_dir: Some(cache_dir),
        retries: 0,
        resume_existing: false,
    })?;
    join_http_server(server)?;

    assert_eq!(report.summary.fetched_records, 1);
    assert_eq!(report.summary.matched_accessions, 1);
    assert_eq!(report.summary.failed_accessions, 1);
    assert_eq!(report.failed_accessions, vec!["SRR999999".to_string()]);
    assert!(output_metadata.exists());
    let status_csv = fs::read_to_string(&report.status_path)?;
    assert!(status_csv.contains("SRR200001,matched,1,SRR200001,remote,1,"));
    assert!(status_csv.contains("SRR999999,fetch_failed,0,,remote,1,"));

    Ok(())
}

#[test]
fn study_fetch_metadata_cli_reports_json_bundle() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("inventory.tsv");
    fs::write(
        &input,
        "dataset_id\taccession\nsample-a\tSRR200001\nsample-b\tSRR200002\n",
    )?;
    let output_metadata = dir.path().join("fetched_metadata.tsv");

    let (base_url, server) = spawn_tsv_http_server(vec![sample_public_metadata_response()])?;
    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("study-fetch-metadata")
        .arg(&input)
        .arg(&output_metadata)
        .arg("--base-url")
        .arg(&base_url)
        .arg("--json")
        .output()?;
    join_http_server(server)?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["summary"]["requested_accessions"], 2);
    assert_eq!(report["summary"]["matched_accessions"], 2);
    assert_eq!(report["summary"]["unmatched_accessions"], 0);
    assert_eq!(report["summary"]["failed_accessions"], 0);
    assert_eq!(report["summary"]["remote_fetches"], 1);
    assert_eq!(report["source"], "ena_filereport");
    assert!(report["cache_dir"].as_str().is_some());
    assert!(report["status_path"].as_str().is_some());
    assert!(output_metadata.exists());

    Ok(())
}

#[test]
fn study_fetch_metadata_cli_resume_reports_json_bundle() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("resume_inventory.tsv");
    fs::write(
        &input,
        "dataset_id\taccession\nsample-a\tSRR200001\nsample-b\tSRR999999\n",
    )?;
    let output_metadata = dir.path().join("resume_metadata.tsv");

    let (base_url, server) = spawn_tsv_http_server(vec![sample_public_metadata_response()])?;
    let first = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("study-fetch-metadata")
        .arg(&input)
        .arg(&output_metadata)
        .arg("--base-url")
        .arg(&base_url)
        .arg("--json")
        .output()?;
    join_http_server(server)?;
    assert!(first.status.success());

    let resumed = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("study-fetch-metadata")
        .arg(&input)
        .arg(&output_metadata)
        .arg("--base-url")
        .arg("http://unused.local/ena")
        .arg("--geo-base-url")
        .arg("http://unused.local/geo")
        .arg("--resume")
        .arg("--json")
        .output()?;
    assert!(resumed.status.success());

    let stdout = String::from_utf8(resumed.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["resume_existing"], true);
    assert_eq!(report["summary"]["resumed_accessions"], 2);
    assert_eq!(report["summary"]["resumed_records"], 1);
    assert_eq!(report["summary"]["remote_fetches"], 0);
    assert_eq!(report["summary"]["fetched_records"], 1);

    Ok(())
}

#[test]
fn study_fetch_metadata_cli_expands_study_accession_json_bundle() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("inventory.tsv");
    fs::write(
        &input,
        "study_accession\tbioproject\nSRP300001\tPRJNA300001\n",
    )?;
    let output_metadata = dir.path().join("expanded_metadata.tsv");

    let (base_url, server) = spawn_tsv_http_server(vec![sample_public_study_expansion_response()])?;
    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("study-fetch-metadata")
        .arg(&input)
        .arg(&output_metadata)
        .arg("--base-url")
        .arg(&base_url)
        .arg("--retries")
        .arg("1")
        .arg("--json")
        .output()?;
    join_http_server(server)?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["summary"]["requested_accessions"], 1);
    assert_eq!(report["summary"]["unique_accessions"], 1);
    assert_eq!(report["summary"]["matched_accessions"], 1);
    assert_eq!(report["summary"]["fetched_records"], 2);
    assert_eq!(report["summary"]["unmatched_accessions"], 0);
    assert_eq!(report["summary"]["failed_accessions"], 0);
    assert!(output_metadata.exists());

    Ok(())
}

#[test]
fn study_annotate_enriches_manifest_without_overwriting_existing_values() -> Result<()> {
    let dir = tempdir()?;
    let manifest = write_study_artifact_fixture(dir.path())?;
    let annotations = write_study_annotation_fixture(dir.path())?;
    let output_manifest = dir.path().join("annotated_manifest.tsv");

    let report = annotate_study_manifest(&StudyAnnotateOptions {
        manifest_path: manifest,
        annotations_path: annotations,
        output_path: output_manifest.clone(),
        overwrite_existing: false,
    })?;

    assert_eq!(report.summary.annotation_rows, 3);
    assert_eq!(report.summary.matched_rows, 2);
    assert_eq!(report.summary.unmatched_rows, 1);
    assert_eq!(report.summary.fields_overwritten, 0);
    assert_eq!(report.summary.after.datasets_with_citation, 2);
    assert_eq!(report.summary.after.datasets_with_downstream_metrics, 2);
    assert!(output_manifest.exists());
    assert!(
        dir.path()
            .join("annotated_manifest.annotation_summary.csv")
            .exists()
    );

    let manifest_text = fs::read_to_string(&output_manifest)?;
    assert!(manifest_text.contains("Example WGS benchmark"));
    assert!(!manifest_text.contains("Replacement WGS citation"));
    assert!(manifest_text.contains("0.978"));
    assert!(manifest_text.contains("Paired single-cell case | Curated single-cell note"));

    Ok(())
}

#[test]
fn study_annotate_cli_reports_json_bundle() -> Result<()> {
    let dir = tempdir()?;
    let manifest = write_study_artifact_fixture(dir.path())?;
    let annotations = write_study_annotation_fixture(dir.path())?;
    let output_manifest = dir.path().join("annotated_manifest.tsv");

    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("study-annotate")
        .arg(&manifest)
        .arg(&annotations)
        .arg(&output_manifest)
        .arg("--json")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["summary"]["matched_rows"], 2);
    assert_eq!(report["summary"]["unmatched_rows"], 1);
    assert_eq!(report["summary"]["after"]["datasets_with_citation"], 2);
    assert!(output_manifest.exists());
    assert!(
        dir.path()
            .join("annotated_manifest.annotation_summary.csv")
            .exists()
    );

    Ok(())
}

#[test]
fn study_annotate_can_overwrite_existing_values() -> Result<()> {
    let dir = tempdir()?;
    let manifest = write_study_artifact_fixture(dir.path())?;
    let annotations = write_study_annotation_fixture(dir.path())?;
    let output_manifest = dir.path().join("annotated_manifest_overwrite.tsv");

    let report = annotate_study_manifest(&StudyAnnotateOptions {
        manifest_path: manifest,
        annotations_path: annotations,
        output_path: output_manifest.clone(),
        overwrite_existing: true,
    })?;

    assert_eq!(report.summary.matched_rows, 2);
    assert!(report.summary.fields_overwritten > 0);

    let manifest_text = fs::read_to_string(&output_manifest)?;
    assert!(manifest_text.contains("Replacement WGS citation"));
    assert!(manifest_text.contains("trim-galore"));

    Ok(())
}

#[test]
fn study_annotate_imports_sra_ena_style_metadata_exports() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let inventory = dir.path().join("discovered_inventory.tsv");
    let canonical_manifest = dir.path().join("canonical_manifest.tsv");
    let annotations = write_accession_metadata_annotation_fixture(dir.path())?;
    let annotated_manifest = dir.path().join("annotated_manifest.tsv");

    discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: inventory.clone(),
        recursive: true,
    })?;
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: canonical_manifest,
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;

    let report = annotate_study_manifest(&StudyAnnotateOptions {
        manifest_path: dir.path().join("canonical_manifest.tsv"),
        annotations_path: annotations,
        output_path: annotated_manifest.clone(),
        overwrite_existing: false,
    })?;

    assert_eq!(report.summary.matched_rows, 2);
    assert_eq!(report.summary.after.datasets_with_citation, 2);
    assert_eq!(report.summary.after.datasets_with_expected_platform, 2);
    assert_eq!(report.summary.after.datasets_with_expected_experiment, 2);

    let manifest_text = fs::read_to_string(&annotated_manifest)?;
    assert!(manifest_text.contains("Human WGS benchmarking study | ERP300001 | PRJEB300001"));
    assert!(manifest_text.contains("PBMC single-cell atlas | SRP200001 | PRJNA200001"));
    assert!(manifest_text.contains("Illumina\tWGS"));
    assert!(manifest_text.contains("MGI/DNBSEQ\t10x Genomics v3"));

    Ok(())
}

#[test]
fn study_ingest_merges_structured_results_bundle() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let inventory = dir.path().join("discovered_inventory.tsv");
    let canonical_manifest = dir.path().join("canonical_manifest.tsv");
    let results_dir = write_study_results_bundle_fixture(dir.path())?;
    let ingested_manifest = dir.path().join("ingested_manifest.tsv");

    discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: inventory.clone(),
        recursive: true,
    })?;
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: canonical_manifest.clone(),
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;

    let report = ingest_study_results(&StudyIngestOptions {
        manifest_path: canonical_manifest,
        input_dir: results_dir,
        output_path: ingested_manifest.clone(),
        recursive: true,
        overwrite_existing: false,
    })?;

    assert_eq!(report.summary.structured_files, 2);
    assert_eq!(report.summary.json_files, 1);
    assert_eq!(report.summary.delimited_files, 1);
    assert_eq!(report.summary.records_ingested, 5);
    assert_eq!(report.summary.matched_records, 4);
    assert_eq!(report.summary.unmatched_records, 1);
    assert_eq!(report.summary.generated_annotation_rows, 2);
    assert_eq!(report.summary.after.datasets_with_baseline_name, 2);
    assert_eq!(report.summary.after.datasets_with_downstream_metrics, 2);
    assert!(ingested_manifest.exists());
    assert!(
        dir.path()
            .join("ingested_manifest.ingested_annotations.tsv")
            .exists()
    );
    assert!(
        dir.path()
            .join("ingested_manifest.ingest_summary.csv")
            .exists()
    );

    let study_output = dir.path().join("study-from-ingest");
    let study_report = generate_study_artifacts(&StudyArtifactsOptions {
        manifest_path: ingested_manifest,
        output_dir: study_output.clone(),
        sample_size: 64,
        batch_reads: 4,
        benchmark_rounds: 2,
        backend_preference: BackendPreference::Cpu,
        min_quality_override: Some(20),
    })?;
    assert_eq!(study_report.aggregate.datasets, 2);
    assert_eq!(study_report.detection.datasets_with_expected_platform, 2);
    assert_eq!(study_report.comparison.datasets_with_baseline, 2);
    assert_eq!(study_report.comparison.datasets_with_alignment_metrics, 2);
    assert!(study_output.join("study_artifacts.json").exists());

    Ok(())
}

#[test]
fn study_ingest_merges_accession_metadata_with_native_tool_outputs() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let inventory = dir.path().join("discovered_inventory.tsv");
    let canonical_manifest = dir.path().join("canonical_manifest.tsv");
    let results_dir = write_metadata_native_study_results_bundle_fixture(dir.path())?;
    let ingested_manifest = dir.path().join("metadata_native_ingested_manifest.tsv");

    discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: inventory.clone(),
        recursive: true,
    })?;
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: canonical_manifest.clone(),
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;

    let report = ingest_study_results(&StudyIngestOptions {
        manifest_path: canonical_manifest,
        input_dir: results_dir,
        output_path: ingested_manifest.clone(),
        recursive: true,
        overwrite_existing: false,
    })?;

    assert_eq!(report.summary.structured_files, 5);
    assert_eq!(report.summary.json_files, 3);
    assert_eq!(report.summary.delimited_files, 2);
    assert_eq!(report.summary.records_ingested, 6);
    assert_eq!(report.summary.matched_records, 6);
    assert_eq!(report.summary.unmatched_records, 0);
    assert_eq!(report.summary.generated_annotation_rows, 2);
    assert_eq!(report.summary.after.datasets_with_citation, 2);
    assert_eq!(report.summary.after.datasets_with_expected_platform, 2);
    assert_eq!(report.summary.after.datasets_with_expected_experiment, 2);
    assert_eq!(report.summary.after.datasets_with_baseline_name, 2);
    assert_eq!(report.summary.after.datasets_with_downstream_metrics, 2);

    let manifest_text = fs::read_to_string(&ingested_manifest)?;
    assert!(manifest_text.contains("ENA WGS benchmark study | ERP400001 | PRJEB400001"));
    assert!(
        manifest_text.contains("PBMC atlas | Single-cell experiment | SRP400001 | PRJNA400001")
    );
    assert!(manifest_text.contains("fastp"));
    assert!(manifest_text.contains("45000"));
    assert!(manifest_text.contains("38000"));

    let study_output = dir.path().join("study-from-metadata-native-ingest");
    let study_report = generate_study_artifacts(&StudyArtifactsOptions {
        manifest_path: ingested_manifest,
        output_dir: study_output.clone(),
        sample_size: 64,
        batch_reads: 4,
        benchmark_rounds: 2,
        backend_preference: BackendPreference::Cpu,
        min_quality_override: Some(20),
    })?;
    assert_eq!(study_report.aggregate.datasets, 2);
    assert_eq!(study_report.detection.datasets_with_expected_platform, 2);
    assert_eq!(study_report.detection.datasets_with_expected_experiment, 2);
    assert_eq!(study_report.comparison.datasets_with_baseline, 2);
    assert!(study_output.join("study_artifacts.json").exists());

    Ok(())
}

#[test]
fn study_ingest_cli_reports_json_bundle() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let inventory = dir.path().join("discovered_inventory.tsv");
    let canonical_manifest = dir.path().join("canonical_manifest.tsv");
    let results_dir = write_study_results_bundle_fixture(dir.path())?;
    let ingested_manifest = dir.path().join("ingested_manifest.tsv");

    discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: inventory.clone(),
        recursive: true,
    })?;
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: canonical_manifest.clone(),
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;

    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("study-ingest")
        .arg(&canonical_manifest)
        .arg(&results_dir)
        .arg(&ingested_manifest)
        .arg("--json")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["summary"]["structured_files"], 2);
    assert_eq!(report["summary"]["matched_records"], 4);
    assert_eq!(report["summary"]["unmatched_records"], 1);
    assert_eq!(report["summary"]["after"]["datasets_with_baseline_name"], 2);
    assert!(ingested_manifest.exists());

    Ok(())
}

#[test]
fn study_ingest_rejects_conflicting_structured_metrics() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let inventory = dir.path().join("discovered_inventory.tsv");
    let canonical_manifest = dir.path().join("canonical_manifest.tsv");
    let results_dir = write_conflicting_study_results_bundle_fixture(dir.path())?;
    let ingested_manifest = dir.path().join("ingested_manifest.tsv");

    discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: inventory.clone(),
        recursive: true,
    })?;
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: canonical_manifest.clone(),
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;

    let error = ingest_study_results(&StudyIngestOptions {
        manifest_path: canonical_manifest,
        input_dir: results_dir,
        output_path: ingested_manifest,
        recursive: true,
        overwrite_existing: false,
    })
    .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("conflicting values for baseline_input_bases_per_sec")
    );

    Ok(())
}

#[test]
fn study_ingest_parses_native_fastp_and_quast_outputs() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let inventory = dir.path().join("discovered_inventory.tsv");
    let canonical_manifest = dir.path().join("canonical_manifest.tsv");
    let results_dir = write_native_study_results_bundle_fixture(dir.path())?;
    let ingested_manifest = dir.path().join("native_ingested_manifest.tsv");

    discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: inventory.clone(),
        recursive: true,
    })?;
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: canonical_manifest.clone(),
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;

    let report = ingest_study_results(&StudyIngestOptions {
        manifest_path: canonical_manifest,
        input_dir: results_dir,
        output_path: ingested_manifest.clone(),
        recursive: true,
        overwrite_existing: false,
    })?;

    assert_eq!(report.summary.structured_files, 4);
    assert_eq!(report.summary.json_files, 2);
    assert_eq!(report.summary.delimited_files, 2);
    assert_eq!(report.summary.records_ingested, 4);
    assert_eq!(report.summary.matched_records, 4);
    assert_eq!(report.summary.unmatched_records, 0);
    assert_eq!(report.summary.after.datasets_with_baseline_name, 2);
    assert_eq!(report.summary.after.datasets_with_downstream_metrics, 2);

    let manifest_text = fs::read_to_string(&ingested_manifest)?;
    assert!(manifest_text.contains("fastp"));
    assert!(manifest_text.contains("45000"));
    assert!(manifest_text.contains("38000"));

    Ok(())
}

#[test]
fn study_ingest_cli_parses_native_tool_outputs() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let inventory = dir.path().join("discovered_inventory.tsv");
    let canonical_manifest = dir.path().join("canonical_manifest.tsv");
    let results_dir = write_native_study_results_bundle_fixture(dir.path())?;
    let ingested_manifest = dir.path().join("native_ingested_manifest.tsv");

    discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: inventory.clone(),
        recursive: true,
    })?;
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: canonical_manifest.clone(),
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;

    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("study-ingest")
        .arg(&canonical_manifest)
        .arg(&results_dir)
        .arg(&ingested_manifest)
        .arg("--json")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["summary"]["json_files"], 2);
    assert_eq!(report["summary"]["delimited_files"], 2);
    assert_eq!(report["summary"]["matched_records"], 4);
    assert_eq!(
        report["summary"]["after"]["datasets_with_downstream_metrics"],
        2
    );

    Ok(())
}

#[test]
fn study_ingest_parses_native_alignment_reports() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let inventory = dir.path().join("discovered_inventory.tsv");
    let canonical_manifest = dir.path().join("canonical_manifest.tsv");
    let results_dir = write_native_alignment_results_bundle_fixture(dir.path())?;
    let ingested_manifest = dir.path().join("native_alignment_ingested_manifest.tsv");

    discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: inventory.clone(),
        recursive: true,
    })?;
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: canonical_manifest.clone(),
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;

    let report = ingest_study_results(&StudyIngestOptions {
        manifest_path: canonical_manifest,
        input_dir: results_dir,
        output_path: ingested_manifest.clone(),
        recursive: true,
        overwrite_existing: false,
    })?;

    assert_eq!(report.summary.structured_files, 8);
    assert_eq!(report.summary.json_files, 0);
    assert_eq!(report.summary.delimited_files, 8);
    assert_eq!(report.summary.records_ingested, 8);
    assert_eq!(report.summary.matched_records, 8);
    assert_eq!(report.summary.unmatched_records, 0);
    assert_eq!(report.summary.generated_annotation_rows, 2);
    assert_eq!(report.summary.after.datasets_with_baseline_name, 2);
    assert_eq!(report.summary.after.datasets_with_downstream_metrics, 2);

    let manifest_text = fs::read_to_string(&ingested_manifest)?;
    assert!(manifest_text.contains("fastp"));
    assert!(manifest_text.contains("cutadapt"));
    assert!(manifest_text.contains("0.980000"));
    assert!(manifest_text.contains("0.940000"));
    assert!(manifest_text.contains("0.120000"));
    assert!(manifest_text.contains("0.200000"));

    let study_output = dir.path().join("study-from-native-alignment-ingest");
    let study_report = generate_study_artifacts(&StudyArtifactsOptions {
        manifest_path: ingested_manifest,
        output_dir: study_output.clone(),
        sample_size: 64,
        batch_reads: 4,
        benchmark_rounds: 2,
        backend_preference: BackendPreference::Cpu,
        min_quality_override: Some(20),
    })?;
    assert_eq!(study_report.aggregate.datasets, 2);
    assert_eq!(study_report.comparison.datasets_with_baseline, 2);
    assert_eq!(study_report.comparison.datasets_with_alignment_metrics, 2);
    assert_eq!(study_report.comparison.datasets_with_duplicate_metrics, 2);
    assert!(study_output.join("study_artifacts.json").exists());

    Ok(())
}

#[test]
fn study_ingest_cli_parses_native_alignment_reports() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let inventory = dir.path().join("discovered_inventory.tsv");
    let canonical_manifest = dir.path().join("canonical_manifest.tsv");
    let results_dir = write_native_alignment_results_bundle_fixture(dir.path())?;
    let ingested_manifest = dir.path().join("native_alignment_ingested_manifest.tsv");

    discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: inventory.clone(),
        recursive: true,
    })?;
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: canonical_manifest.clone(),
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;

    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("study-ingest")
        .arg(&canonical_manifest)
        .arg(&results_dir)
        .arg(&ingested_manifest)
        .arg("--json")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["summary"]["json_files"], 0);
    assert_eq!(report["summary"]["delimited_files"], 8);
    assert_eq!(report["summary"]["matched_records"], 8);
    assert_eq!(report["summary"]["after"]["datasets_with_baseline_name"], 2);
    assert_eq!(
        report["summary"]["after"]["datasets_with_downstream_metrics"],
        2
    );

    Ok(())
}

#[test]
fn study_ingest_parses_native_variant_benchmark_reports() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let inventory = dir.path().join("discovered_inventory.tsv");
    let canonical_manifest = dir.path().join("canonical_manifest.tsv");
    let results_dir = write_native_variant_results_bundle_fixture(dir.path())?;
    let ingested_manifest = dir.path().join("native_variant_ingested_manifest.tsv");

    discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: inventory.clone(),
        recursive: true,
    })?;
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: canonical_manifest.clone(),
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;

    let report = ingest_study_results(&StudyIngestOptions {
        manifest_path: canonical_manifest,
        input_dir: results_dir,
        output_path: ingested_manifest.clone(),
        recursive: true,
        overwrite_existing: false,
    })?;

    assert_eq!(report.summary.structured_files, 4);
    assert_eq!(report.summary.json_files, 0);
    assert_eq!(report.summary.delimited_files, 4);
    assert_eq!(report.summary.records_ingested, 4);
    assert_eq!(report.summary.matched_records, 4);
    assert_eq!(report.summary.unmatched_records, 0);
    assert_eq!(report.summary.generated_annotation_rows, 2);
    assert_eq!(report.summary.after.datasets_with_baseline_name, 2);
    assert_eq!(report.summary.after.datasets_with_downstream_metrics, 2);

    let manifest_text = fs::read_to_string(&ingested_manifest)?;
    assert!(manifest_text.contains("fastp"));
    assert!(manifest_text.contains("cutadapt"));
    assert!(manifest_text.contains("0.992000"));
    assert!(manifest_text.contains("0.985000"));
    assert!(manifest_text.contains("0.978000"));
    assert!(manifest_text.contains("0.970000"));

    let study_output = dir.path().join("study-from-native-variant-ingest");
    let study_report = generate_study_artifacts(&StudyArtifactsOptions {
        manifest_path: ingested_manifest,
        output_dir: study_output.clone(),
        sample_size: 64,
        batch_reads: 4,
        benchmark_rounds: 2,
        backend_preference: BackendPreference::Cpu,
        min_quality_override: Some(20),
    })?;
    assert_eq!(study_report.aggregate.datasets, 2);
    assert_eq!(study_report.comparison.datasets_with_baseline, 2);
    assert_eq!(study_report.comparison.datasets_with_variant_metrics, 2);
    assert!(study_output.join("study_artifacts.json").exists());

    Ok(())
}

#[test]
fn study_ingest_cli_parses_native_variant_benchmark_reports() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let inventory = dir.path().join("discovered_inventory.tsv");
    let canonical_manifest = dir.path().join("canonical_manifest.tsv");
    let results_dir = write_native_variant_results_bundle_fixture(dir.path())?;
    let ingested_manifest = dir.path().join("native_variant_ingested_manifest.tsv");

    discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: inventory.clone(),
        recursive: true,
    })?;
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: canonical_manifest.clone(),
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;

    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("study-ingest")
        .arg(&canonical_manifest)
        .arg(&results_dir)
        .arg(&ingested_manifest)
        .arg("--json")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["summary"]["json_files"], 0);
    assert_eq!(report["summary"]["delimited_files"], 4);
    assert_eq!(report["summary"]["matched_records"], 4);
    assert_eq!(report["summary"]["after"]["datasets_with_baseline_name"], 2);
    assert_eq!(
        report["summary"]["after"]["datasets_with_downstream_metrics"],
        2
    );

    Ok(())
}

#[test]
fn study_ingest_parses_native_coverage_reports() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let inventory = dir.path().join("discovered_inventory.tsv");
    let canonical_manifest = dir.path().join("canonical_manifest.tsv");
    let results_dir = write_native_coverage_results_bundle_fixture(dir.path())?;
    let ingested_manifest = dir.path().join("native_coverage_ingested_manifest.tsv");

    discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: inventory.clone(),
        recursive: true,
    })?;
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: canonical_manifest.clone(),
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;

    let report = ingest_study_results(&StudyIngestOptions {
        manifest_path: canonical_manifest,
        input_dir: results_dir,
        output_path: ingested_manifest.clone(),
        recursive: true,
        overwrite_existing: false,
    })?;

    assert_eq!(report.summary.structured_files, 4);
    assert_eq!(report.summary.json_files, 0);
    assert_eq!(report.summary.delimited_files, 4);
    assert_eq!(report.summary.records_ingested, 4);
    assert_eq!(report.summary.matched_records, 4);
    assert_eq!(report.summary.unmatched_records, 0);
    assert_eq!(report.summary.generated_annotation_rows, 2);
    assert_eq!(report.summary.after.datasets_with_baseline_name, 2);
    assert_eq!(report.summary.after.datasets_with_downstream_metrics, 2);

    let manifest_text = fs::read_to_string(&ingested_manifest)?;
    assert!(manifest_text.contains("31.500000"));
    assert!(manifest_text.contains("28.400000"));
    assert!(manifest_text.contains("18.500000"));
    assert!(manifest_text.contains("16.200000"));
    assert!(manifest_text.contains("0.920000"));
    assert!(manifest_text.contains("0.880000"));

    let study_output = dir.path().join("study-from-native-coverage-ingest");
    let study_report = generate_study_artifacts(&StudyArtifactsOptions {
        manifest_path: ingested_manifest,
        output_dir: study_output.clone(),
        sample_size: 64,
        batch_reads: 4,
        benchmark_rounds: 2,
        backend_preference: BackendPreference::Cpu,
        min_quality_override: Some(20),
    })?;
    assert_eq!(study_report.aggregate.datasets, 2);
    assert_eq!(study_report.comparison.datasets_with_baseline, 2);
    assert_eq!(
        study_report.comparison.datasets_with_mean_coverage_metrics,
        2
    );
    assert_eq!(
        study_report
            .comparison
            .datasets_with_coverage_breadth_metrics,
        1
    );
    assert!(
        study_output
            .join("figures")
            .join("study_mean_coverage.svg")
            .exists()
    );
    assert!(
        study_output
            .join("figures")
            .join("study_coverage_breadth.svg")
            .exists()
    );

    Ok(())
}

#[test]
fn study_ingest_cli_parses_native_coverage_reports() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let inventory = dir.path().join("discovered_inventory.tsv");
    let canonical_manifest = dir.path().join("canonical_manifest.tsv");
    let results_dir = write_native_coverage_results_bundle_fixture(dir.path())?;
    let ingested_manifest = dir.path().join("native_coverage_ingested_manifest.tsv");

    discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: inventory.clone(),
        recursive: true,
    })?;
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: canonical_manifest.clone(),
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;

    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("study-ingest")
        .arg(&canonical_manifest)
        .arg(&results_dir)
        .arg(&ingested_manifest)
        .arg("--json")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["summary"]["json_files"], 0);
    assert_eq!(report["summary"]["delimited_files"], 4);
    assert_eq!(report["summary"]["matched_records"], 4);
    assert_eq!(report["summary"]["after"]["datasets_with_baseline_name"], 2);
    assert_eq!(
        report["summary"]["after"]["datasets_with_downstream_metrics"],
        2
    );

    Ok(())
}

#[test]
fn study_ingest_rejects_conflicting_native_coverage_reports() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let inventory = dir.path().join("discovered_inventory.tsv");
    let canonical_manifest = dir.path().join("canonical_manifest.tsv");
    let results_dir = write_conflicting_native_coverage_bundle_fixture(dir.path())?;
    let ingested_manifest = dir.path().join("native_coverage_ingested_manifest.tsv");

    discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: inventory.clone(),
        recursive: true,
    })?;
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: canonical_manifest.clone(),
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;

    let error = ingest_study_results(&StudyIngestOptions {
        manifest_path: canonical_manifest,
        input_dir: results_dir,
        output_path: ingested_manifest,
        recursive: true,
        overwrite_existing: false,
    })
    .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("conflicting values for baseline_mean_coverage")
    );

    Ok(())
}

#[test]
fn study_ingest_rejects_conflicting_native_variant_reports() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let inventory = dir.path().join("discovered_inventory.tsv");
    let canonical_manifest = dir.path().join("canonical_manifest.tsv");
    let results_dir = write_conflicting_native_variant_bundle_fixture(dir.path())?;
    let ingested_manifest = dir.path().join("native_variant_ingested_manifest.tsv");

    discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: inventory.clone(),
        recursive: true,
    })?;
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: canonical_manifest.clone(),
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;

    let error = ingest_study_results(&StudyIngestOptions {
        manifest_path: canonical_manifest,
        input_dir: results_dir,
        output_path: ingested_manifest,
        recursive: true,
        overwrite_existing: false,
    })
    .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("conflicting values for baseline_variant_f1")
    );

    Ok(())
}

#[test]
fn study_ingest_rejects_conflicting_native_alignment_reports() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let inventory = dir.path().join("discovered_inventory.tsv");
    let canonical_manifest = dir.path().join("canonical_manifest.tsv");
    let results_dir = write_conflicting_native_alignment_bundle_fixture(dir.path())?;
    let ingested_manifest = dir.path().join("native_alignment_ingested_manifest.tsv");

    discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: inventory.clone(),
        recursive: true,
    })?;
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: canonical_manifest.clone(),
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;

    let error = ingest_study_results(&StudyIngestOptions {
        manifest_path: canonical_manifest,
        input_dir: results_dir,
        output_path: ingested_manifest,
        recursive: true,
        overwrite_existing: false,
    })
    .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("conflicting values for baseline_alignment_rate")
    );

    Ok(())
}

#[test]
fn study_ingest_rejects_conflicting_native_quast_outputs() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let inventory = dir.path().join("discovered_inventory.tsv");
    let canonical_manifest = dir.path().join("canonical_manifest.tsv");
    let results_dir = write_conflicting_native_quast_bundle_fixture(dir.path())?;
    let ingested_manifest = dir.path().join("native_ingested_manifest.tsv");

    discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: inventory.clone(),
        recursive: true,
    })?;
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: canonical_manifest.clone(),
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;

    let error = ingest_study_results(&StudyIngestOptions {
        manifest_path: canonical_manifest,
        input_dir: results_dir,
        output_path: ingested_manifest,
        recursive: true,
        overwrite_existing: false,
    })
    .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("conflicting values for baseline_assembly_n50")
    );

    Ok(())
}

#[test]
fn study_annotate_chains_from_discovery_into_study_artifacts() -> Result<()> {
    let dir = tempdir()?;
    let input_dir = write_study_discovery_fixture(dir.path())?;
    let inventory = dir.path().join("discovered_inventory.tsv");
    let canonical_manifest = dir.path().join("canonical_manifest.tsv");
    let annotations = write_discovery_annotation_fixture(dir.path())?;
    let annotated_manifest = dir.path().join("annotated_manifest.tsv");

    discover_study_inventory(&StudyDiscoverOptions {
        input_dir,
        output_path: inventory.clone(),
        recursive: true,
    })?;
    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: inventory,
        output_path: canonical_manifest.clone(),
        default_baseline_name: None,
        download_root: dir.path().join("downloads"),
    })?;
    annotate_study_manifest(&StudyAnnotateOptions {
        manifest_path: canonical_manifest,
        annotations_path: annotations,
        output_path: annotated_manifest.clone(),
        overwrite_existing: false,
    })?;

    let study_output = dir.path().join("study-from-annotate");
    let report = generate_study_artifacts(&StudyArtifactsOptions {
        manifest_path: annotated_manifest,
        output_dir: study_output.clone(),
        sample_size: 64,
        batch_reads: 4,
        benchmark_rounds: 2,
        backend_preference: BackendPreference::Cpu,
        min_quality_override: Some(20),
    })?;

    assert_eq!(report.aggregate.datasets, 2);
    assert_eq!(report.detection.datasets_with_expected_platform, 2);
    assert_eq!(report.detection.datasets_with_expected_experiment, 2);
    assert_eq!(report.comparison.datasets_with_baseline, 2);
    assert_eq!(report.comparison.datasets_with_alignment_metrics, 2);
    assert!(study_output.join("study_artifacts.json").exists());

    Ok(())
}

#[test]
fn study_manifest_bootstrap_preserves_relative_paths() -> Result<()> {
    let current_dir = std::env::current_dir()?;
    let target_dir = current_dir.join("target");
    let dir = tempfile::tempdir_in(&target_dir)?;
    let inventory = write_study_inventory_fixture(dir.path())?;
    let output_manifest = dir.path().join("generated_manifest.tsv");

    let relative_inventory = inventory.strip_prefix(&current_dir)?.to_path_buf();
    let relative_manifest = output_manifest.strip_prefix(&current_dir)?.to_path_buf();

    bootstrap_study_manifest(&StudyManifestOptions {
        inventory_path: relative_inventory,
        output_path: relative_manifest.clone(),
        default_baseline_name: Some("fastp".to_string()),
        download_root: dir.path().join("downloads"),
    })?;

    let study_output = dir.path().join("study-relative");
    let relative_study_output = study_output.strip_prefix(&current_dir)?.to_path_buf();
    let report = generate_study_artifacts(&StudyArtifactsOptions {
        manifest_path: relative_manifest,
        output_dir: relative_study_output,
        sample_size: 64,
        batch_reads: 4,
        benchmark_rounds: 2,
        backend_preference: BackendPreference::Cpu,
        min_quality_override: Some(20),
    })?;

    assert_eq!(report.aggregate.datasets, 2);
    assert_eq!(report.comparison.datasets_with_baseline, 2);
    assert!(study_output.join("study_artifacts.json").exists());

    Ok(())
}

#[test]
fn study_artifacts_generate_case_study_bundle() -> Result<()> {
    let dir = tempdir()?;
    let manifest = write_study_artifact_fixture(dir.path())?;
    let output_dir = dir.path().join("study-artifacts");

    let report = generate_study_artifacts(&StudyArtifactsOptions {
        manifest_path: manifest,
        output_dir: output_dir.clone(),
        sample_size: 64,
        batch_reads: 4,
        benchmark_rounds: 2,
        backend_preference: BackendPreference::Cpu,
        min_quality_override: Some(20),
    })?;

    assert_eq!(report.aggregate.datasets, 2);
    assert_eq!(report.aggregate.paired_datasets, 1);
    assert_eq!(report.detection.platform_accuracy, Some(1.0));
    assert_eq!(report.detection.experiment_accuracy, Some(1.0));
    assert_eq!(report.comparison.datasets_with_baseline, 2);
    assert_eq!(report.comparison.datasets_with_alignment_metrics, 2);
    assert_eq!(report.comparison.datasets_with_variant_metrics, 1);
    assert!(
        report
            .comparison
            .average_input_speedup_vs_baseline
            .is_some()
    );
    assert!(
        report
            .datasets
            .iter()
            .all(|dataset| dataset.baseline.is_some())
    );
    assert!(report.artifacts.len() >= 12);
    assert!(output_dir.join("study_artifacts.json").exists());
    assert!(output_dir.join("data").join("dataset_summary.csv").exists());
    assert!(
        output_dir
            .join("data")
            .join("detection_summary.csv")
            .exists()
    );
    assert!(
        output_dir
            .join("data")
            .join("baseline_comparison.csv")
            .exists()
    );
    assert!(
        output_dir
            .join("data")
            .join("downstream_metrics.csv")
            .exists()
    );
    assert!(
        output_dir
            .join("data")
            .join("datasets")
            .join("wgs_case.json")
            .exists()
    );
    assert!(
        output_dir
            .join("figures")
            .join("study_throughput.svg")
            .exists()
    );
    assert!(
        output_dir
            .join("figures")
            .join("study_cleanup.svg")
            .exists()
    );
    assert!(
        output_dir
            .join("figures")
            .join("study_correction.svg")
            .exists()
    );
    assert!(
        output_dir
            .join("figures")
            .join("study_detection_accuracy.svg")
            .exists()
    );
    assert!(
        output_dir
            .join("figures")
            .join("study_baseline_throughput.svg")
            .exists()
    );
    assert!(
        output_dir
            .join("figures")
            .join("study_alignment_rate.svg")
            .exists()
    );
    assert!(
        output_dir
            .join("figures")
            .join("study_duplicate_rate.svg")
            .exists()
    );
    assert!(
        output_dir
            .join("figures")
            .join("study_variant_f1.svg")
            .exists()
    );
    assert!(
        output_dir
            .join("figures")
            .join("study_assembly_n50.svg")
            .exists()
    );

    Ok(())
}

#[test]
fn study_artifacts_cli_reports_json_bundle() -> Result<()> {
    let dir = tempdir()?;
    let manifest = write_study_artifact_fixture(dir.path())?;
    let output_dir = dir.path().join("study-cli");

    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("study-artifacts")
        .arg(&manifest)
        .arg(&output_dir)
        .arg("--backend")
        .arg("cpu")
        .arg("--benchmark-rounds")
        .arg("2")
        .arg("--json")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["aggregate"]["datasets"], 2);
    assert_eq!(report["aggregate"]["paired_datasets"], 1);
    assert_eq!(report["detection"]["platform_accuracy"], 1.0);
    assert_eq!(report["detection"]["experiment_accuracy"], 1.0);
    assert_eq!(report["comparison"]["datasets_with_baseline"], 2);
    assert_eq!(report["comparison"]["datasets_with_variant_metrics"], 1);
    assert!(report["comparison"]["average_input_speedup_vs_baseline"].is_number());
    assert!(
        report["artifacts"]
            .as_array()
            .is_some_and(|items| !items.is_empty())
    );
    assert!(output_dir.join("data").join("dataset_summary.csv").exists());
    assert!(
        output_dir
            .join("data")
            .join("baseline_comparison.csv")
            .exists()
    );
    assert!(
        output_dir
            .join("data")
            .join("downstream_metrics.csv")
            .exists()
    );
    assert!(
        output_dir
            .join("figures")
            .join("study_throughput.svg")
            .exists()
    );
    assert!(
        output_dir
            .join("figures")
            .join("study_baseline_throughput.svg")
            .exists()
    );

    Ok(())
}

fn write_history_compare_fixture(input: &Path) -> Result<()> {
    let core = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let adapter = "AGATCGGAAGAGC";
    let mut fastq = String::new();

    for index in 0..12 {
        fastq.push_str(&format!(
            "@history-{index}\n{core}\n+\n{}\n",
            "I".repeat(core.len())
        ));
    }
    for index in 12..18 {
        let qualities = format!(
            "{}!{}{}",
            "I".repeat(18),
            "I".repeat(core.len() - 19),
            "!".repeat(adapter.len())
        );
        let mut corrupted = core.as_bytes().to_vec();
        corrupted[18] = b'T';
        let corrupted = String::from_utf8(corrupted)?;
        fastq.push_str(&format!(
            "@history-{index}\n{}{}\n+\n{}\n",
            corrupted, adapter, qualities
        ));
    }

    fs::write(input, fastq)?;
    Ok(())
}

fn write_study_artifact_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let core = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let mate2 = "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";
    let barcode_read = "ACGTACGTACGTACGTTTTTTTTTACGTACGTACGT";
    let manifest = base.join("study_manifest.tsv");
    let wgs = base.join("wgs.fastq");
    let sc_r1 = base.join("sc_r1.fastq");
    let sc_r2 = base.join("sc_r2.fastq");

    let mut wgs_fastq = String::new();
    for index in 0..8 {
        let wgs_read = random_dna(core.len(), (index + 1200) as u64);
        let wgs_read_str = String::from_utf8(wgs_read).unwrap();
        wgs_fastq.push_str(&format!(
            "@wgs-{index}\n{wgs_read_str}\n+\n{}\n",
            "I".repeat(wgs_read_str.len())
        ));
    }
    fs::write(&wgs, wgs_fastq)?;

    let mut sc_r1_fastq = String::new();
    let mut sc_r2_fastq = String::new();
    for index in 0..8 {
        sc_r1_fastq.push_str(&format!(
            "@sc-{index}/1\n{barcode_read}\n+\n{}\n",
            "I".repeat(barcode_read.len())
        ));
        sc_r2_fastq.push_str(&format!(
            "@sc-{index}/2\n{mate2}\n+\n{}\n",
            "I".repeat(mate2.len())
        ));
    }
    fs::write(&sc_r1, sc_r1_fastq)?;
    fs::write(&sc_r2, sc_r2_fastq)?;

    let manifest_body = "\
dataset_id\taccession\tcitation\tinput1\tinput2\texpected_platform\texpected_experiment\tbaseline_name\tbaseline_input_bases_per_sec\tbaseline_trimmed_read_fraction\tbaseline_discarded_read_fraction\tbaseline_corrected_bases_per_mbase\techo_alignment_rate\tbaseline_alignment_rate\techo_duplicate_rate\tbaseline_duplicate_rate\techo_variant_f1\tbaseline_variant_f1\techo_assembly_n50\tbaseline_assembly_n50\tnotes\n\
wgs_case\tSRR000001\tExample WGS benchmark\twgs.fastq\t\tmgi\twgs\tfastp\t250000\t0.020\t0.005\t0\t0.980\t0.970\t0.120\t0.140\t0.992\t0.985\t45000\t40000\tShort-read genomic case\n\
sc_case\tSRR000002\tExample scRNA benchmark\tsc_r1.fastq\tsc_r2.fastq\tmgi\t10xv3\tcutadapt\t400000\t0.050\t0.010\t0\t0.960\t0.940\t0.180\t0.200\t\t\t\t\tPaired single-cell case\n";
    fs::write(&manifest, manifest_body)?;
    Ok(manifest)
}

fn write_study_inventory_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let core = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let mate2 = "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";
    let barcode_read = "ACGTACGTACGTACGTTTTTTTTTACGTACGTACGT";
    let inventory = base.join("study_inventory.tsv");
    let wgs = base.join("wgs.fastq");
    let sc_r1 = base.join("sc_r1.fastq");
    let sc_r2 = base.join("sc_r2.fastq");

    let mut wgs_fastq = String::new();
    for index in 0..8 {
        let wgs_read = random_dna(core.len(), (index + 2200) as u64);
        let wgs_read_str = String::from_utf8(wgs_read).unwrap();
        wgs_fastq.push_str(&format!(
            "@wgs-{index}\n{wgs_read_str}\n+\n{}\n",
            "I".repeat(wgs_read_str.len())
        ));
    }
    fs::write(&wgs, wgs_fastq)?;

    let mut sc_r1_fastq = String::new();
    let mut sc_r2_fastq = String::new();
    for index in 0..8 {
        sc_r1_fastq.push_str(&format!(
            "@sc-{index}/1\n{barcode_read}\n+\n{}\n",
            "I".repeat(barcode_read.len())
        ));
        sc_r2_fastq.push_str(&format!(
            "@sc-{index}/2\n{mate2}\n+\n{}\n",
            "I".repeat(mate2.len())
        ));
    }
    fs::write(&sc_r1, sc_r1_fastq)?;
    fs::write(&sc_r2, sc_r2_fastq)?;

    let inventory_body = "\
dataset_id\taccession\tcitation\tinput1\tinput2\texpected_platform\texpected_experiment\tbaseline_name\techo_alignment_rate\tbaseline_alignment_rate\techo_duplicate_rate\tbaseline_duplicate_rate\techo_variant_f1\tbaseline_variant_f1\techo_assembly_n50\tbaseline_assembly_n50\tnotes\n\
\tSRR100001\tInventory WGS benchmark\twgs.fastq\t\tmgi\twgs\t\t0.980\t0.970\t0.120\t0.140\t0.992\t0.985\t45000\t40000\tDerived id from accession\n\
sc_case\tSRR100002\tInventory scRNA benchmark\tsc_r1.fastq\tsc_r2.fastq\tmgi\t10xv3\tcutadapt\t0.960\t0.940\t0.180\t0.200\t\t\t\t\tExplicit dataset id\n";
    fs::write(&inventory, inventory_body)?;
    Ok(inventory)
}

fn write_public_accession_metadata_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let metadata = base.join("public_accessions.tsv");
    let body = "\
run_accession\tstudy_title\texperiment_title\tstudy_accession\tbioproject\tinstrument_platform\tinstrument_model\tlibrary_strategy\tlibrary_layout\tfastq_ftp\tsample_title\n\
ERR300001\tHuman WGS benchmarking study\tWGS replicate 1\tERP300001\tPRJEB300001\tILLUMINA\tNovaSeq 6000\tWGS\tSINGLE\tftp.sra.ebi.ac.uk/vol1/ERR300001/ERR300001.fastq.gz\tBulk DNA reference\n\
SRR200001\tPBMC single-cell atlas\tSingle-cell experiment\tSRP200001\tPRJNA200001\t\tDNBSEQ-G400\tRNA-Seq\tPAIRED\tftp.sra.ebi.ac.uk/vol1/SRR200001/SRR200001_1.fastq.gz;ftp.sra.ebi.ac.uk/vol1/SRR200001/SRR200001_2.fastq.gz\t10x Genomics v3 PBMC\n";
    fs::write(&metadata, body)?;
    Ok(metadata)
}

fn write_public_accession_layout_only_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let metadata = base.join("public_layout_only.tsv");
    let body = "\
run_accession\tstudy_title\tinstrument_platform\tlibrary_strategy\tlibrary_layout\tnotes\n\
SRR400001\tLayout-only paired accession\tILLUMINA\tRNA-Seq\tPAIRED\tNo FASTQ URLs yet\n";
    fs::write(&metadata, body)?;
    Ok(metadata)
}

fn write_public_download_metadata_fixture(
    remote_base_url: &str,
    base: &Path,
) -> Result<std::path::PathBuf> {
    let metadata = base.join("public_downloads.tsv");
    let body = format!(
        "\
run_accession\tstudy_title\texperiment_title\tstudy_accession\tbioproject\tinstrument_platform\tinstrument_model\tlibrary_strategy\tlibrary_layout\tfastq_ftp\tsample_title\n\
ERR300001\tHuman WGS benchmarking study\tWGS replicate 1\tERP300001\tPRJEB300001\tILLUMINA\tNovaSeq 6000\tWGS\tSINGLE\t{remote_base_url}/ERR300001.fastq\tBulk DNA reference\n\
SRR200001\tPBMC single-cell atlas\tSingle-cell experiment\tSRP200001\tPRJNA200001\tMGI\tDNBSEQ-G400\tRNA-Seq\tPAIRED\t{remote_base_url}/SRR200001_1.fastq;{remote_base_url}/SRR200001_2.fastq\t10x Genomics v3 PBMC\n"
    );
    fs::write(&metadata, body)?;
    Ok(metadata)
}

fn rewrite_public_download_metadata_urls(path: &Path, remote_base_url: &str) -> Result<()> {
    let raw = fs::read_to_string(path)?;
    let rewritten = raw.replace("http://127.0.0.1:9", remote_base_url);
    fs::write(path, rewritten)?;
    Ok(())
}

fn write_public_accession_download_fixture(download_root: &Path) -> Result<()> {
    let wgs_dir = download_root.join("ERR300001");
    let sc_dir = download_root.join("SRR200001");
    fs::create_dir_all(&wgs_dir)?;
    fs::create_dir_all(&sc_dir)?;

    let wgs_a = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let wgs_b = "TTTTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let sc_r1 = "ACGTACGTACGTACGTTTTTTTTTACGTACGTACGT";
    let sc_r2 = "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";
    write_gzip_fastq(
        &wgs_dir.join("ERR300001.fastq.gz"),
        &format!(
            "@wgs-0\n{wgs_a}\n+\n{}\n@wgs-1\n{wgs_b}\n+\n{}\n",
            "I".repeat(wgs_a.len()),
            "I".repeat(wgs_b.len())
        ),
    )?;
    write_gzip_fastq(
        &sc_dir.join("SRR200001_1.fastq.gz"),
        &format!(
            "@sc-0/1\n{sc_r1}\n+\n{}\n@sc-1/1\n{sc_r1}\n+\n{}\n",
            "I".repeat(sc_r1.len()),
            "I".repeat(sc_r1.len())
        ),
    )?;
    write_gzip_fastq(
        &sc_dir.join("SRR200001_2.fastq.gz"),
        &format!(
            "@sc-0/2\n{sc_r2}\n+\n{}\n@sc-1/2\n{sc_r2}\n+\n{}\n",
            "I".repeat(sc_r2.len()),
            "I".repeat(sc_r2.len())
        ),
    )?;

    Ok(())
}

fn write_gzip_fastq(path: &Path, body: &str) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut encoder = GzEncoder::new(file, Compression::default());
    encoder.write_all(body.as_bytes())?;
    encoder.finish()?;
    Ok(())
}

fn sample_public_metadata_response() -> String {
    "\
run_accession\tstudy_accession\tsecondary_study_accession\tstudy_title\texperiment_accession\texperiment_title\tsample_accession\tinstrument_platform\tinstrument_model\tlibrary_strategy\tlibrary_layout\tlibrary_source\tlibrary_selection\tsample_title\tfastq_ftp\tfastq_aspera\tsubmitted_ftp\tsubmitted_aspera\n\
SRR200001\tSRP200001\tPRJNA200001\tPBMC single-cell atlas\tSRX200001\tSingle-cell experiment\tSRS200001\tILLUMINA\tNovaSeq 6000\tRNA-Seq\tPAIRED\tTRANSCRIPTOMIC\tcDNA\t10x Genomics v3 PBMC\tftp.sra.ebi.ac.uk/vol1/SRR200001/SRR200001_1.fastq.gz;ftp.sra.ebi.ac.uk/vol1/SRR200001/SRR200001_2.fastq.gz\tfasp.sra.ebi.ac.uk:/vol1/SRR200001/SRR200001_1.fastq.gz;fasp.sra.ebi.ac.uk:/vol1/SRR200001/SRR200001_2.fastq.gz\t\t\n\
SRR200002\tSRP200002\tPRJNA200002\tTumor WGS benchmark\tSRX200002\tWhole-genome replicate\tSRS200002\tMGI\tDNBSEQ-G400\tWGS\tSINGLE\tGENOMIC\tRANDOM\tTumor reference\tftp.sra.ebi.ac.uk/vol1/SRR200002/SRR200002.fastq.gz\t\t\t\n"
        .to_string()
}

fn sample_public_download_fetch_response(base_url: &str) -> String {
    format!(
        "\
run_accession\tstudy_accession\tsecondary_study_accession\tstudy_title\texperiment_accession\texperiment_title\tsample_accession\tinstrument_platform\tinstrument_model\tlibrary_strategy\tlibrary_layout\tlibrary_source\tlibrary_selection\tsample_title\tfastq_ftp\tfastq_aspera\tsubmitted_ftp\tsubmitted_aspera\n\
ERR300001\tERP300001\tPRJEB300001\tHuman WGS benchmarking study\tERX300001\tWGS replicate 1\tERS300001\tILLUMINA\tNovaSeq 6000\tWGS\tSINGLE\tGENOMIC\tRANDOM\tBulk DNA reference\t{base_url}/files/ERR300001.fastq\t\t\t\n\
SRR200001\tSRP200001\tPRJNA200001\tPBMC single-cell atlas\tSRX200001\tSingle-cell experiment\tSRS200001\tMGI\tDNBSEQ-G400\tRNA-Seq\tPAIRED\tTRANSCRIPTOMIC\tcDNA\t10x Genomics v3 PBMC\t{base_url}/files/SRR200001_1.fastq;{base_url}/files/SRR200001_2.fastq\t\t\t\n"
    )
}

fn sample_download_single_fastq_response() -> String {
    let sequence = "ACGTACGTACGTACGTACGTACGTACGTACGT";
    format!(
        "@wgs-0\n{sequence}\n+\n{}\n@wgs-1\n{sequence}\n+\n{}\n",
        "I".repeat(sequence.len()),
        "I".repeat(sequence.len())
    )
}

fn sample_download_paired_r1_fastq_response() -> String {
    let sequence = "ACGTACGTACGTACGTTTTTTTTTACGTACGTACGT";
    format!(
        "@sc-0/1\n{sequence}\n+\n{}\n@sc-1/1\n{sequence}\n+\n{}\n",
        "I".repeat(sequence.len()),
        "I".repeat(sequence.len())
    )
}

fn sample_download_paired_r2_fastq_response() -> String {
    let sequence = "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";
    format!(
        "@sc-0/2\n{sequence}\n+\n{}\n@sc-1/2\n{sequence}\n+\n{}\n",
        "I".repeat(sequence.len()),
        "I".repeat(sequence.len())
    )
}

fn sample_public_study_expansion_response() -> String {
    "\
run_accession\tstudy_accession\tsecondary_study_accession\tstudy_title\texperiment_accession\texperiment_title\tsample_accession\tinstrument_platform\tinstrument_model\tlibrary_strategy\tlibrary_layout\tlibrary_source\tlibrary_selection\tsample_title\tfastq_ftp\tfastq_aspera\tsubmitted_ftp\tsubmitted_aspera\n\
SRR300001\tSRP300001\tPRJNA300001\tExpanded public study\tSRX300001\tReplicate 1\tSRS300001\tILLUMINA\tNovaSeq 6000\tRNA-Seq\tPAIRED\tTRANSCRIPTOMIC\tcDNA\tSample A\tftp.sra.ebi.ac.uk/vol1/SRR300001/SRR300001_1.fastq.gz;ftp.sra.ebi.ac.uk/vol1/SRR300001/SRR300001_2.fastq.gz\t\t\t\n\
SRR300002\tSRP300001\tPRJNA300001\tExpanded public study\tSRX300002\tReplicate 2\tSRS300002\tILLUMINA\tNovaSeq 6000\tRNA-Seq\tPAIRED\tTRANSCRIPTOMIC\tcDNA\tSample B\tftp.sra.ebi.ac.uk/vol1/SRR300002/SRR300002_1.fastq.gz;ftp.sra.ebi.ac.uk/vol1/SRR300002/SRR300002_2.fastq.gz\t\t\t\n"
        .to_string()
}

fn sample_geo_series_bridge_response() -> String {
    "\
^SERIES = GSE400001\n\
!Series_title = GEO bridge validation series\n\
!Series_geo_accession = GSE400001\n\
!Series_relation = SRA: https://www.ncbi.nlm.nih.gov/sra?term=SRP300001\n\
!Series_relation = BioProject: https://www.ncbi.nlm.nih.gov/bioproject/PRJNA300001\n"
        .to_string()
}

fn sample_geo_series_quick_without_public_accessions_response() -> String {
    "\
^SERIES = GSE400001\n\
!Series_title = GEO bridge validation series\n\
!Series_geo_accession = GSE400001\n\
!Series_sample_id = GSM400001\n\
!Series_summary = Quick GEO bridge view omits public archive relations in this fixture.\n"
        .to_string()
}

fn sample_geo_sample_bridge_response() -> String {
    "\
^SAMPLE = GSM400001\n\
!Sample_title = GEO bridge validation sample\n\
!Sample_geo_accession = GSM400001\n\
!Sample_relation = SRA: https://www.ncbi.nlm.nih.gov/sra?term=SRR200001\n"
        .to_string()
}

#[derive(Debug, Clone)]
struct HttpFixtureResponse {
    status_code: u16,
    content_type: String,
    body: String,
}

impl HttpFixtureResponse {
    fn tsv(body: String) -> Self {
        Self {
            status_code: 200,
            content_type: "text/tab-separated-values; charset=utf-8".to_string(),
            body,
        }
    }

    fn text(body: String) -> Self {
        Self {
            status_code: 200,
            content_type: "text/plain; charset=utf-8".to_string(),
            body,
        }
    }

    fn status(status_code: u16, body: &str) -> Self {
        Self {
            status_code,
            content_type: "text/plain; charset=utf-8".to_string(),
            body: body.to_string(),
        }
    }
}

fn spawn_tsv_http_server(
    responses: Vec<String>,
) -> Result<(String, thread::JoinHandle<Result<()>>)> {
    spawn_scripted_http_server(
        responses
            .into_iter()
            .map(HttpFixtureResponse::tsv)
            .collect(),
    )
}

fn spawn_scripted_http_server(
    responses: Vec<HttpFixtureResponse>,
) -> Result<(String, thread::JoinHandle<Result<()>>)> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let address = listener.local_addr()?;
    let handle = thread::spawn(move || -> Result<()> {
        for response in responses {
            let (mut stream, _) = listener.accept()?;
            let mut buffer = [0u8; 4096];
            let _ = stream.read(&mut buffer)?;
            let status_line = if response.status_code == 200 {
                "200 OK".to_string()
            } else {
                format!("{} ERROR", response.status_code)
            };
            let reply = format!(
                "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                status_line,
                response.content_type,
                response.body.len(),
                response.body
            );
            stream.write_all(reply.as_bytes())?;
            stream.flush()?;
        }
        Ok(())
    });
    Ok((format!("http://{address}/filereport"), handle))
}

fn spawn_path_http_server<F>(build_responses: F) -> Result<(String, thread::JoinHandle<Result<()>>)>
where
    F: FnOnce(&str) -> Vec<(String, HttpFixtureResponse)> + Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let address = listener.local_addr()?;
    let base_url = format!("http://{address}");
    let responses = build_responses(&base_url);
    let response_count = responses.len();
    let routes: HashMap<String, HttpFixtureResponse> = responses.into_iter().collect();
    let handle = thread::spawn(move || -> Result<()> {
        for _ in 0..response_count {
            let (mut stream, _) = listener.accept()?;
            let mut buffer = [0u8; 4096];
            let bytes = stream.read(&mut buffer)?;
            let request = String::from_utf8_lossy(&buffer[..bytes]);
            let path = request
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|value| value.split('?').next())
                .unwrap_or("/");
            let response = routes
                .get(path)
                .cloned()
                .unwrap_or_else(|| HttpFixtureResponse::status(404, "not found"));
            let status_line = if response.status_code == 200 {
                "200 OK".to_string()
            } else {
                format!("{} ERROR", response.status_code)
            };
            let reply = format!(
                "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                status_line,
                response.content_type,
                response.body.len(),
                response.body
            );
            stream.write_all(reply.as_bytes())?;
            stream.flush()?;
        }
        Ok(())
    });
    Ok((base_url, handle))
}

fn spawn_public_download_http_server() -> Result<(String, thread::JoinHandle<Result<()>>)> {
    spawn_path_http_server(|base_url| {
        vec![
            (
                "/ena".to_string(),
                HttpFixtureResponse::tsv(sample_public_download_fetch_response(base_url)),
            ),
            (
                "/files/ERR300001.fastq".to_string(),
                HttpFixtureResponse::text(sample_download_single_fastq_response()),
            ),
            (
                "/files/SRR200001_1.fastq".to_string(),
                HttpFixtureResponse::text(sample_download_paired_r1_fastq_response()),
            ),
            (
                "/files/SRR200001_2.fastq".to_string(),
                HttpFixtureResponse::text(sample_download_paired_r2_fastq_response()),
            ),
        ]
    })
}

fn join_http_server(handle: thread::JoinHandle<Result<()>>) -> Result<()> {
    match handle.join() {
        Ok(result) => result,
        Err(_) => anyhow::bail!("mock HTTP server thread panicked"),
    }
}

fn write_study_discovery_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let root = base.join("discovery_input");
    let wgs_dir = root.join("wgs");
    let sc_dir = root.join("scrna");
    fs::create_dir_all(&wgs_dir)?;
    fs::create_dir_all(&sc_dir)?;

    let core = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let mate2 = "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";
    let barcode_read = "ACGTACGTACGTACGTTTTTTTTTACGTACGTACGT";

    let mut wgs_fastq = String::new();
    for index in 0..8 {
        wgs_fastq.push_str(&format!(
            "@wgs-{index}\n{core}\n+\n{}\n",
            "I".repeat(core.len())
        ));
    }
    fs::write(wgs_dir.join("ERR300001.fastq"), wgs_fastq)?;

    let mut sc_r1_fastq = String::new();
    let mut sc_r2_fastq = String::new();
    for index in 0..8 {
        sc_r1_fastq.push_str(&format!(
            "@sc-{index}/1\n{barcode_read}\n+\n{}\n",
            "I".repeat(barcode_read.len())
        ));
        sc_r2_fastq.push_str(&format!(
            "@sc-{index}/2\n{mate2}\n+\n{}\n",
            "I".repeat(mate2.len())
        ));
    }
    fs::write(sc_dir.join("SRR200001_R1.fastq"), sc_r1_fastq)?;
    fs::write(sc_dir.join("SRR200001_R2.fastq"), sc_r2_fastq)?;
    fs::write(root.join("README.txt"), "ignore me")?;

    Ok(root)
}

fn write_study_annotation_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let annotations = base.join("study_annotations.tsv");
    let body = "\
dataset_id\taccession\tcitation\texpected_platform\texpected_experiment\tbaseline_name\tbaseline_input_bases_per_sec\techo_alignment_rate\tbaseline_alignment_rate\techo_duplicate_rate\tbaseline_duplicate_rate\techo_variant_f1\tbaseline_variant_f1\tnotes\n\
wgs_case\tSRR000001\tReplacement WGS citation\tillumina\twgs\ttrim-galore\t275000\t0.981\t0.972\t0.118\t0.132\t\t\tCurated WGS note\n\
sc_case\tSRR000002\tCurated scRNA citation\tmgi\t10xv3\tfastp\t410000\t0.965\t0.945\t0.175\t0.195\t0.978\t0.970\tCurated single-cell note\n\
missing_case\tSRR999999\tMissing dataset\tillumina\twgs\tfastp\t100000\t0.900\t0.880\t0.200\t0.220\t0.950\t0.930\tShould stay unmatched\n";
    fs::write(&annotations, body)?;
    Ok(annotations)
}

fn write_discovery_annotation_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let annotations = base.join("discovery_annotations.tsv");
    let body = "\
accession\tcitation\texpected_platform\texpected_experiment\tbaseline_name\tbaseline_input_bases_per_sec\tbaseline_trimmed_read_fraction\tbaseline_discarded_read_fraction\techo_alignment_rate\tbaseline_alignment_rate\techo_duplicate_rate\tbaseline_duplicate_rate\techo_variant_f1\tbaseline_variant_f1\techo_assembly_n50\tbaseline_assembly_n50\tnotes\n\
ERR300001\tPublic WGS accession\tmgi\twgs\tfastp\t250000\t0.020\t0.005\t0.980\t0.970\t0.120\t0.140\t0.992\t0.985\t45000\t40000\tCurated public WGS case\n\
SRR200001\tPublic single-cell accession\tmgi\t10xv3\tcutadapt\t400000\t0.050\t0.010\t0.960\t0.940\t0.180\t0.200\t0.978\t0.970\t38000\t34000\tCurated public single-cell case\n";
    fs::write(&annotations, body)?;
    Ok(annotations)
}

fn write_accession_metadata_annotation_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let annotations = base.join("accession_metadata.tsv");
    let body = "\
run_accession\tstudy_title\tstudy_accession\tbioproject\tinstrument_platform\tinstrument_model\tlibrary_strategy\tsample_title\tnotes\n\
ERR300001\tHuman WGS benchmarking study\tERP300001\tPRJEB300001\tILLUMINA\tNovaSeq 6000\tWGS\tBulk DNA reference\tENA WGS metadata row\n\
SRR200001\tPBMC single-cell atlas\tSRP200001\tPRJNA200001\t\tDNBSEQ-G400\tRNA-Seq\t10x Genomics v3 PBMC\tSRA single-cell metadata row\n";
    fs::write(&annotations, body)?;
    Ok(annotations)
}

fn write_study_results_bundle_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let results_dir = base.join("structured_results");
    fs::create_dir_all(&results_dir)?;

    let baseline_json = results_dir.join("baseline_metrics.json");
    let baseline_body = r#"[
  {
    "dataset_id": "ERR300001",
    "baseline_name": "fastp",
    "baseline_input_bases_per_sec": 250000,
    "baseline_trimmed_read_fraction": 0.020,
    "baseline_discarded_read_fraction": 0.005,
    "notes": "JSON WGS baseline"
  },
  {
    "accession": "SRR200001",
    "baseline_name": "cutadapt",
    "baseline_input_bases_per_sec": 400000,
    "baseline_trimmed_read_fraction": 0.050,
    "baseline_discarded_read_fraction": 0.010,
    "notes": "JSON single-cell baseline"
  },
  {
    "dataset_id": "MISSING_CASE",
    "baseline_name": "fastp",
    "baseline_input_bases_per_sec": 123456
  }
]"#;
    fs::write(&baseline_json, baseline_body)?;

    let downstream_tsv = results_dir.join("downstream_metrics.tsv");
    let downstream_body = "\
dataset_id\tcitation\texpected_platform\texpected_experiment\techo_alignment_rate\tbaseline_alignment_rate\techo_duplicate_rate\tbaseline_duplicate_rate\techo_variant_f1\tbaseline_variant_f1\techo_assembly_n50\tbaseline_assembly_n50\tnotes\n\
ERR300001\tPublic WGS accession\tmgi\twgs\t0.980\t0.970\t0.120\t0.140\t0.992\t0.985\t45000\t40000\tTSV WGS downstream\n\
SRR200001\tPublic single-cell accession\tmgi\t10xv3\t0.960\t0.940\t0.180\t0.200\t0.978\t0.970\t38000\t34000\tTSV single-cell downstream\n";
    fs::write(&downstream_tsv, downstream_body)?;
    fs::write(results_dir.join("README.txt"), "ignore me")?;

    Ok(results_dir)
}

fn write_conflicting_study_results_bundle_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let results_dir = base.join("conflicting_structured_results");
    fs::create_dir_all(&results_dir)?;

    let baseline_json = results_dir.join("baseline_metrics.json");
    let baseline_body = r#"[
  {
    "dataset_id": "ERR300001",
    "baseline_name": "fastp",
    "baseline_input_bases_per_sec": 250000
  }
]"#;
    fs::write(&baseline_json, baseline_body)?;

    let conflict_tsv = results_dir.join("baseline_conflict.tsv");
    let conflict_body = "\
dataset_id\tbaseline_name\tbaseline_input_bases_per_sec\n\
ERR300001\tfastp\t260000\n";
    fs::write(&conflict_tsv, conflict_body)?;

    Ok(results_dir)
}

fn write_native_study_results_bundle_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let results_dir = base.join("native_structured_results");
    let fastp_dir = results_dir.join("fastp").join("SRR200001");
    let quast_wgs_dir = results_dir.join("ERR300001_quast");
    let quast_sc_dir = results_dir.join("SRR200001_quast");
    fs::create_dir_all(&fastp_dir)?;
    fs::create_dir_all(&quast_wgs_dir)?;
    fs::create_dir_all(&quast_sc_dir)?;

    let wgs_fastp = results_dir.join("ERR300001.fastp.json");
    let wgs_fastp_body = r#"{
  "summary": {
    "before_filtering": { "total_reads": 1000 },
    "after_filtering": { "total_reads": 920 }
  },
  "filtering_result": { "passed_filter_reads": 920 },
  "adapter_cutting": { "adapter_trimmed_reads": 180 }
}"#;
    fs::write(&wgs_fastp, wgs_fastp_body)?;

    let sc_fastp = fastp_dir.join("fastp.json");
    let sc_fastp_body = r#"{
  "summary": {
    "before_filtering": { "total_reads": 1200 },
    "after_filtering": { "total_reads": 1080 }
  },
  "filtering_result": { "passed_filter_reads": 1080 },
  "adapter_cutting": { "adapter_trimmed_reads": 240 }
}"#;
    fs::write(&sc_fastp, sc_fastp_body)?;

    let wgs_quast = quast_wgs_dir.join("report.tsv");
    let wgs_quast_body = "\
Assembly\tassembly\n\
N50\t45000\n";
    fs::write(&wgs_quast, wgs_quast_body)?;

    let sc_quast = quast_sc_dir.join("report.tsv");
    let sc_quast_body = "\
Assembly\tassembly\n\
N50\t38000\n";
    fs::write(&sc_quast, sc_quast_body)?;

    Ok(results_dir)
}

fn write_native_alignment_results_bundle_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let results_dir = base.join("native_alignment_results");
    let echo_wgs_dir = results_dir.join("echo").join("ERR300001");
    let baseline_wgs_dir = results_dir.join("fastp").join("ERR300001");
    let echo_sc_dir = results_dir.join("japalityecho").join("SRR200001");
    let baseline_sc_dir = results_dir.join("cutadapt").join("SRR200001");
    fs::create_dir_all(&echo_wgs_dir)?;
    fs::create_dir_all(&baseline_wgs_dir)?;
    fs::create_dir_all(&echo_sc_dir)?;
    fs::create_dir_all(&baseline_sc_dir)?;

    fs::write(
        echo_wgs_dir.join("flagstat.txt"),
        "\
1000 + 0 in total (QC-passed reads + QC-failed reads)\n\
980 + 0 mapped (98.00% : N/A)\n",
    )?;
    fs::write(
        echo_wgs_dir.join("picard_markduplicates.txt"),
        "\
## METRICS CLASS\tpicard.sam.DuplicationMetrics\n\
LIBRARY\tUNPAIRED_READS_EXAMINED\tREAD_PAIRS_EXAMINED\tSECONDARY_OR_SUPPLEMENTARY_RDS\tUNMAPPED_READS\tUNPAIRED_READ_DUPLICATES\tREAD_PAIR_DUPLICATES\tREAD_PAIR_OPTICAL_DUPLICATES\tPERCENT_DUPLICATION\tESTIMATED_LIBRARY_SIZE\n\
lib1\t0\t500\t0\t0\t0\t60\t0\t0.120000\t100000\n",
    )?;
    fs::write(
        baseline_wgs_dir.join("flagstat.txt"),
        "\
1000 + 0 in total (QC-passed reads + QC-failed reads)\n\
970 + 0 mapped (97.00% : N/A)\n",
    )?;
    fs::write(
        baseline_wgs_dir.join("duplication_metrics.txt"),
        "\
## METRICS CLASS\tpicard.sam.DuplicationMetrics\n\
LIBRARY\tUNPAIRED_READS_EXAMINED\tREAD_PAIRS_EXAMINED\tSECONDARY_OR_SUPPLEMENTARY_RDS\tUNMAPPED_READS\tUNPAIRED_READ_DUPLICATES\tREAD_PAIR_DUPLICATES\tREAD_PAIR_OPTICAL_DUPLICATES\tPERCENT_DUPLICATION\tESTIMATED_LIBRARY_SIZE\n\
lib1\t0\t500\t0\t0\t0\t70\t0\t0.140000\t100000\n",
    )?;
    fs::write(
        echo_sc_dir.join("flagstat.txt"),
        "\
1200 + 0 in total (QC-passed reads + QC-failed reads)\n\
1152 + 0 mapped (96.00% : N/A)\n",
    )?;
    fs::write(
        echo_sc_dir.join("markduplicates.txt"),
        "\
## METRICS CLASS\tpicard.sam.DuplicationMetrics\n\
LIBRARY\tUNPAIRED_READS_EXAMINED\tREAD_PAIRS_EXAMINED\tSECONDARY_OR_SUPPLEMENTARY_RDS\tUNMAPPED_READS\tUNPAIRED_READ_DUPLICATES\tREAD_PAIR_DUPLICATES\tREAD_PAIR_OPTICAL_DUPLICATES\tPERCENT_DUPLICATION\tESTIMATED_LIBRARY_SIZE\n\
lib1\t0\t600\t0\t0\t0\t108\t0\t0.180000\t100000\n",
    )?;
    fs::write(
        baseline_sc_dir.join("flagstat.txt"),
        "\
1200 + 0 in total (QC-passed reads + QC-failed reads)\n\
1128 + 0 mapped (94.00% : N/A)\n",
    )?;
    fs::write(
        baseline_sc_dir.join("duplication_metrics.txt"),
        "\
## METRICS CLASS\tpicard.sam.DuplicationMetrics\n\
LIBRARY\tUNPAIRED_READS_EXAMINED\tREAD_PAIRS_EXAMINED\tSECONDARY_OR_SUPPLEMENTARY_RDS\tUNMAPPED_READS\tUNPAIRED_READ_DUPLICATES\tREAD_PAIR_DUPLICATES\tREAD_PAIR_OPTICAL_DUPLICATES\tPERCENT_DUPLICATION\tESTIMATED_LIBRARY_SIZE\n\
lib1\t0\t600\t0\t0\t0\t120\t0\t0.200000\t100000\n",
    )?;

    Ok(results_dir)
}

fn write_conflicting_native_alignment_bundle_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let results_dir = base.join("conflicting_native_alignment");
    let flagstat_a = results_dir.join("fastp").join("ERR300001");
    let flagstat_b = results_dir.join("baseline").join("ERR300001_alt");
    fs::create_dir_all(&flagstat_a)?;
    fs::create_dir_all(&flagstat_b)?;

    fs::write(
        flagstat_a.join("flagstat.txt"),
        "\
1000 + 0 in total (QC-passed reads + QC-failed reads)\n\
970 + 0 mapped (97.00% : N/A)\n",
    )?;
    fs::write(
        flagstat_b.join("ERR300001.flagstat.txt"),
        "\
1000 + 0 in total (QC-passed reads + QC-failed reads)\n\
960 + 0 mapped (96.00% : N/A)\n",
    )?;

    Ok(results_dir)
}

fn write_native_variant_results_bundle_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let results_dir = base.join("native_variant_results");
    let echo_wgs_dir = results_dir.join("echo").join("ERR300001");
    let baseline_wgs_dir = results_dir.join("fastp").join("ERR300001");
    let echo_sc_dir = results_dir
        .join("japalityecho")
        .join("SRR200001")
        .join("vcfeval");
    let baseline_sc_dir = results_dir
        .join("cutadapt")
        .join("SRR200001")
        .join("vcfeval");
    fs::create_dir_all(&echo_wgs_dir)?;
    fs::create_dir_all(&baseline_wgs_dir)?;
    fs::create_dir_all(&echo_sc_dir)?;
    fs::create_dir_all(&baseline_sc_dir)?;

    fs::write(
        echo_wgs_dir.join("happy.summary.csv"),
        "\
Type,Filter,TRUTH.TP,TRUTH.FN,QUERY.TP,QUERY.FP,METRIC.Recall,METRIC.Precision,METRIC.F1_Score\n\
SNP,PASS,990,5,990,8,0.994975,0.992000,0.993486\n\
ALL,PASS,992,4,992,12,0.996000,0.988048,0.992000\n",
    )?;
    fs::write(
        baseline_wgs_dir.join("happy.summary.csv"),
        "\
Type,Filter,TRUTH.TP,TRUTH.FN,QUERY.TP,QUERY.FP,METRIC.Recall,METRIC.Precision,METRIC.F1_Score\n\
ALL,PASS,985,7,985,23,0.992944,0.977183,0.985000\n",
    )?;
    fs::write(
        echo_sc_dir.join("summary.txt"),
        "\
Threshold True-pos False-pos False-neg Precision Sensitivity F-measure\n\
None 978 22 22 0.978000 0.978000 0.978000\n",
    )?;
    fs::write(
        baseline_sc_dir.join("summary.txt"),
        "\
Threshold True-pos False-pos False-neg Precision Sensitivity F-measure\n\
None 970 30 30 0.970000 0.970000 0.970000\n",
    )?;

    Ok(results_dir)
}

fn write_conflicting_native_variant_bundle_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let results_dir = base.join("conflicting_native_variant");
    let happy_dir = results_dir.join("fastp").join("ERR300001");
    let vcfeval_dir = results_dir
        .join("baseline")
        .join("ERR300001_alt")
        .join("vcfeval");
    fs::create_dir_all(&happy_dir)?;
    fs::create_dir_all(&vcfeval_dir)?;

    fs::write(
        happy_dir.join("happy.summary.csv"),
        "\
Type,Filter,TRUTH.TP,TRUTH.FN,QUERY.TP,QUERY.FP,METRIC.Recall,METRIC.Precision,METRIC.F1_Score\n\
ALL,PASS,985,7,985,23,0.992944,0.977183,0.985000\n",
    )?;
    fs::write(
        vcfeval_dir.join("ERR300001.summary.txt"),
        "\
Threshold True-pos False-pos False-neg Precision Sensitivity F-measure\n\
None 980 20 20 0.980000 0.980000 0.980000\n",
    )?;

    Ok(results_dir)
}

fn write_native_coverage_results_bundle_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let results_dir = base.join("native_coverage_results");
    let echo_wgs_dir = results_dir.join("echo").join("ERR300001");
    let baseline_wgs_dir = results_dir.join("fastp").join("ERR300001");
    let echo_sc_dir = results_dir.join("japalityecho").join("SRR200001");
    let baseline_sc_dir = results_dir.join("cutadapt").join("SRR200001");
    fs::create_dir_all(&echo_wgs_dir)?;
    fs::create_dir_all(&baseline_wgs_dir)?;
    fs::create_dir_all(&echo_sc_dir)?;
    fs::create_dir_all(&baseline_sc_dir)?;

    fs::write(
        echo_wgs_dir.join("ERR300001.mosdepth.summary.txt"),
        "\
chrom\tlength\tbases\tmean\tmin\tmax\n\
chr1\t1000\t1000\t31.000000\t0\t80\n\
chr2\t500\t500\t32.500000\t0\t82\n\
total\t1500\t1500\t31.500000\t0\t82\n",
    )?;
    fs::write(
        baseline_wgs_dir.join("ERR300001.mosdepth.summary.txt"),
        "\
chrom\tlength\tbases\tmean\tmin\tmax\n\
chr1\t1000\t1000\t28.000000\t0\t76\n\
chr2\t500\t500\t29.200000\t0\t77\n\
total\t1500\t1500\t28.400000\t0\t77\n",
    )?;
    fs::write(
        echo_sc_dir.join("coverage.tsv"),
        "\
#rname\tstartpos\tendpos\tnumreads\tcovbases\tcoverage\tmeandepth\tmeanbaseq\tmeanmapq\n\
chr1\t1\t1000\t800\t920\t92.0\t18.500000\t35.0\t60.0\n\
total\t1\t1000\t800\t920\t92.0\t18.500000\t35.0\t60.0\n",
    )?;
    fs::write(
        baseline_sc_dir.join("coverage.tsv"),
        "\
#rname\tstartpos\tendpos\tnumreads\tcovbases\tcoverage\tmeandepth\tmeanbaseq\tmeanmapq\n\
chr1\t1\t1000\t780\t880\t88.0\t16.200000\t35.0\t60.0\n\
total\t1\t1000\t780\t880\t88.0\t16.200000\t35.0\t60.0\n",
    )?;

    Ok(results_dir)
}

fn write_conflicting_native_coverage_bundle_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let results_dir = base.join("conflicting_native_coverage");
    let coverage_a = results_dir.join("fastp").join("ERR300001");
    let coverage_b = results_dir.join("fastp").join("ERR300001_alt");
    fs::create_dir_all(&coverage_a)?;
    fs::create_dir_all(&coverage_b)?;

    fs::write(
        coverage_a.join("ERR300001.mosdepth.summary.txt"),
        "\
chrom\tlength\tbases\tmean\tmin\tmax\n\
chr1\t1000\t1000\t28.000000\t0\t76\n\
total\t1000\t1000\t28.400000\t0\t76\n",
    )?;
    fs::write(
        coverage_b.join("ERR300001.coverage.tsv"),
        "\
#rname\tstartpos\tendpos\tnumreads\tcovbases\tcoverage\tmeandepth\tmeanbaseq\tmeanmapq\n\
total\t1\t1000\t790\t910\t91.0\t30.100000\t35.0\t60.0\n",
    )?;

    Ok(results_dir)
}

fn write_metadata_native_study_results_bundle_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let results_dir = write_native_study_results_bundle_fixture(base)?;
    let metadata_json = results_dir.join("public_metadata.json");
    let metadata_body = r#"[
  {
    "run_accession": "ERR300001",
    "study_title": "ENA WGS benchmark study",
    "study_accession": "ERP400001",
    "bioproject": "PRJEB400001",
    "instrument_platform": "ILLUMINA",
    "instrument_model": "NovaSeq 6000",
    "library_strategy": "WGS",
    "notes": "Imported WGS metadata"
  },
  {
    "run_accession": "SRR200001",
    "study_title": "PBMC atlas",
    "experiment_title": "Single-cell experiment",
    "study_accession": "SRP400001",
    "bioproject": "PRJNA400001",
    "instrument_model": "DNBSEQ-G400",
    "library_strategy": "RNA-Seq",
    "sample_title": "10x Genomics v3 PBMC",
    "notes": "Imported single-cell metadata"
  }
]"#;
    fs::write(&metadata_json, metadata_body)?;
    Ok(results_dir)
}

fn write_conflicting_native_quast_bundle_fixture(base: &Path) -> Result<std::path::PathBuf> {
    let results_dir = base.join("conflicting_native_quast");
    let quast_a = results_dir.join("ERR300001_quast_a");
    let quast_b = results_dir.join("ERR300001_quast_b");
    fs::create_dir_all(&quast_a)?;
    fs::create_dir_all(&quast_b)?;

    fs::write(
        quast_a.join("report.tsv"),
        "Assembly\tassembly\nN50\t45000\n",
    )?;
    fs::write(
        quast_b.join("report.tsv"),
        "Assembly\tassembly\nN50\t46000\n",
    )?;

    Ok(results_dir)
}

fn cpu_history_compare_options() -> BenchmarkComparisonOptions {
    BenchmarkComparisonOptions {
        process: ProcessOptions {
            sample_size: 32,
            batch_reads: 6,
            backend_preference: BackendPreference::Cpu,
            forced_adapter: None,
            min_quality_override: Some(20),
            kmer_size_override: None,
                    forced_platform: None,
                    forced_experiment: None,
        },
        rounds: 2,
    }
}

#[test]
fn history_report_flags_threshold_breaches_against_baseline() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("history-alert.fastq");
    let history = dir.path().join("history").join("benchmarks.jsonl");
    write_history_compare_fixture(&input)?;
    let options = cpu_history_compare_options();
    let mut baseline = benchmark_compare_file(&input, &options)?;
    append_benchmark_comparison_history(
        &history,
        Some("baseline"),
        &input,
        None,
        &options,
        &mut baseline,
    )?;
    let mut latest = benchmark_compare_file(&input, &options)?;
    append_benchmark_comparison_history(
        &history,
        Some("latest"),
        &input,
        None,
        &options,
        &mut latest,
    )?;

    let report = read_history_report(
        &history,
        &HistoryReportOptions {
            limit: 10,
            baseline_label: Some("baseline".to_string()),
            baseline_label_prefix: None,
            baseline_latest_pass: false,
            max_wall_clock_regression_pct: None,
            min_raw_speedup: Some(1000.0),
            min_transfer_savings_pct: None,
        },
    )?;
    let regression = report.regression.expect("regression analysis should exist");
    assert_eq!(regression.status.to_string(), "alert");
    assert_eq!(regression.baseline_source, "label:baseline");
    assert_eq!(
        regression
            .baseline_entry
            .as_ref()
            .and_then(|entry| entry.label.as_deref()),
        Some("baseline")
    );
    assert!(
        regression
            .alerts
            .iter()
            .any(|alert| alert.contains("Raw speedup"))
    );

    Ok(())
}

#[test]
fn history_report_selects_latest_comparable_prefix_baseline() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("history-prefix.fastq");
    let history = dir.path().join("history").join("benchmarks.jsonl");
    write_history_compare_fixture(&input)?;

    let compare_options = cpu_history_compare_options();
    let benchmark_options = BenchmarkOptions {
        process: compare_options.process.clone(),
        rounds: 2,
        session_mode: BenchmarkSessionMode::ColdStart,
    };

    let mut comparable_baseline = benchmark_compare_file(&input, &compare_options)?;
    append_benchmark_comparison_history(
        &history,
        Some("nightly-golden-compare"),
        &input,
        None,
        &compare_options,
        &mut comparable_baseline,
    )?;

    let mut newer_non_comparable = benchmark_file(&input, &benchmark_options)?;
    append_benchmark_history(
        &history,
        Some("nightly-benchmark-noise"),
        &input,
        None,
        &benchmark_options,
        &mut newer_non_comparable,
    )?;

    let mut latest = benchmark_compare_file(&input, &compare_options)?;
    append_benchmark_comparison_history(
        &history,
        Some("candidate-latest"),
        &input,
        None,
        &compare_options,
        &mut latest,
    )?;

    let report = read_history_report(
        &history,
        &HistoryReportOptions {
            limit: 10,
            baseline_label: None,
            baseline_label_prefix: Some("nightly".to_string()),
            baseline_latest_pass: false,
            max_wall_clock_regression_pct: None,
            min_raw_speedup: None,
            min_transfer_savings_pct: None,
        },
    )?;
    let regression = report.regression.expect("regression analysis should exist");
    assert_eq!(regression.baseline_source, "label_prefix:nightly");
    assert!(regression.comparable);
    assert_eq!(regression.status.to_string(), "pass");
    assert_eq!(
        regression
            .baseline_entry
            .as_ref()
            .and_then(|entry| entry.label.as_deref()),
        Some("nightly-golden-compare")
    );
    assert_eq!(
        regression
            .baseline_entry
            .as_ref()
            .map(|entry| entry.kind.as_str()),
        Some("benchmark_compare")
    );

    Ok(())
}

#[test]
fn history_report_selects_latest_pass_baseline() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("history-latest-pass.fastq");
    let history = dir.path().join("history").join("benchmarks.jsonl");
    write_history_compare_fixture(&input)?;

    let options = cpu_history_compare_options();

    let mut base0 = benchmark_compare_file(&input, &options)?;
    base0.summary.raw_speedup = Some(2.0);
    append_benchmark_comparison_history(
        &history,
        Some("pass-0"),
        &input,
        None,
        &options,
        &mut base0,
    )?;

    let mut base1 = benchmark_compare_file(&input, &options)?;
    base1.summary.raw_speedup = Some(2.0);
    append_benchmark_comparison_history(
        &history,
        Some("pass-1"),
        &input,
        None,
        &options,
        &mut base1,
    )?;

    let mut alert_candidate = benchmark_compare_file(&input, &options)?;
    alert_candidate.summary.raw_speedup = Some(0.5);
    append_benchmark_comparison_history(
        &history,
        Some("alert-2"),
        &input,
        None,
        &options,
        &mut alert_candidate,
    )?;

    let mut latest = benchmark_compare_file(&input, &options)?;
    latest.summary.raw_speedup = Some(2.0);
    append_benchmark_comparison_history(
        &history,
        Some("latest"),
        &input,
        None,
        &options,
        &mut latest,
    )?;

    let report = read_history_report(
        &history,
        &HistoryReportOptions {
            limit: 10,
            baseline_label: None,
            baseline_label_prefix: None,
            baseline_latest_pass: true,
            max_wall_clock_regression_pct: None,
            min_raw_speedup: Some(1.0),
            min_transfer_savings_pct: None,
        },
    )?;
    let regression = report.regression.expect("regression analysis should exist");
    assert_eq!(regression.baseline_source, "latest_pass");
    assert_eq!(regression.status.to_string(), "pass");
    assert_eq!(
        regression
            .baseline_entry
            .as_ref()
            .and_then(|entry| entry.label.as_deref()),
        Some("pass-1")
    );
    assert!(regression.alerts.is_empty());

    Ok(())
}

#[test]
fn history_report_selects_latest_pass_baseline_with_prefix_filter() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("history-latest-pass-prefix.fastq");
    let history = dir.path().join("history").join("benchmarks.jsonl");
    write_history_compare_fixture(&input)?;

    let options = cpu_history_compare_options();

    let mut family_a_old = benchmark_compare_file(&input, &options)?;
    family_a_old.summary.raw_speedup = Some(2.0);
    append_benchmark_comparison_history(
        &history,
        Some("nightly-a-pass-0"),
        &input,
        None,
        &options,
        &mut family_a_old,
    )?;

    let mut family_b_newer = benchmark_compare_file(&input, &options)?;
    family_b_newer.summary.raw_speedup = Some(2.0);
    append_benchmark_comparison_history(
        &history,
        Some("nightly-b-pass-1"),
        &input,
        None,
        &options,
        &mut family_b_newer,
    )?;

    let mut family_a_latest_pass = benchmark_compare_file(&input, &options)?;
    family_a_latest_pass.summary.raw_speedup = Some(2.0);
    append_benchmark_comparison_history(
        &history,
        Some("nightly-a-pass-2"),
        &input,
        None,
        &options,
        &mut family_a_latest_pass,
    )?;

    let mut family_a_alert = benchmark_compare_file(&input, &options)?;
    family_a_alert.summary.raw_speedup = Some(0.5);
    append_benchmark_comparison_history(
        &history,
        Some("nightly-a-alert-3"),
        &input,
        None,
        &options,
        &mut family_a_alert,
    )?;

    let mut latest = benchmark_compare_file(&input, &options)?;
    latest.summary.raw_speedup = Some(2.0);
    append_benchmark_comparison_history(
        &history,
        Some("nightly-a-latest"),
        &input,
        None,
        &options,
        &mut latest,
    )?;

    let report = read_history_report(
        &history,
        &HistoryReportOptions {
            limit: 10,
            baseline_label: None,
            baseline_label_prefix: Some("nightly-a".to_string()),
            baseline_latest_pass: true,
            max_wall_clock_regression_pct: None,
            min_raw_speedup: Some(1.0),
            min_transfer_savings_pct: None,
        },
    )?;
    let regression = report.regression.expect("regression analysis should exist");
    assert_eq!(
        regression.baseline_source,
        "latest_pass:label_prefix:nightly-a"
    );
    assert_eq!(regression.status.to_string(), "pass");
    assert_eq!(
        regression
            .baseline_entry
            .as_ref()
            .and_then(|entry| entry.label.as_deref()),
        Some("nightly-a-pass-2")
    );

    Ok(())
}

#[test]
fn history_report_cli_exits_non_zero_for_configured_alert_status() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("history-alert-cli.fastq");
    let history = dir.path().join("history").join("benchmarks.jsonl");
    write_history_compare_fixture(&input)?;
    let options = cpu_history_compare_options();
    let mut baseline = benchmark_compare_file(&input, &options)?;
    append_benchmark_comparison_history(
        &history,
        Some("baseline"),
        &input,
        None,
        &options,
        &mut baseline,
    )?;
    let mut latest = benchmark_compare_file(&input, &options)?;
    append_benchmark_comparison_history(
        &history,
        Some("latest"),
        &input,
        None,
        &options,
        &mut latest,
    )?;

    let output = Command::new(env!("CARGO_BIN_EXE_japalityecho"))
        .arg("history-report")
        .arg(&history)
        .arg("--baseline-label")
        .arg("baseline")
        .arg("--min-raw-speedup")
        .arg("1000")
        .arg("--fail-on-status")
        .arg("alert")
        .arg("--json")
        .output()?;

    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8(output.stdout)?;
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    assert_eq!(report["regression"]["status"], "alert");
    assert_eq!(report["regression"]["baseline_source"], "label:baseline");

    Ok(())
}

#[test]
fn inspect_reports_cuda_scaffold_metadata() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("input.fastq");
    let core = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let mut fastq = String::new();

    for index in 0..8 {
        fastq.push_str(&format!(
            "@cuda-{index}\n{core}\n+\n{}\n",
            "I".repeat(core.len())
        ));
    }

    fs::write(&input, fastq)?;
    let inspect = inspect_inputs(&input, None, 16, BackendPreference::Cuda)?;

    assert!(!inspect.paired_end);
    let scaffold = inspect
        .accelerator_scaffold
        .expect("cuda scaffold should exist");
    let runtime = inspect
        .accelerator_runtime
        .expect("cuda runtime probe should exist");
    assert_eq!(scaffold.backend, BackendPreference::Cuda);
    assert_eq!(scaffold.source_path, CUDA_KERNEL_PATH);
    assert_eq!(runtime.backend, BackendPreference::Cuda);
    assert!(!runtime.notes.is_empty());
    assert!(
        scaffold
            .entrypoints
            .iter()
            .any(|entrypoint| entrypoint == "japalityecho_trim_correct")
    );

    Ok(())
}

#[test]
fn inspect_paired_inputs_preserves_single_cell_r1_signal() -> Result<()> {
    let dir = tempdir()?;
    let input1 = dir.path().join("tenx_r1.fastq.gz");
    let input2 = dir.path().join("tenx_r2.fastq.gz");

    let r1_sequence = "ACGTACGTACGTACGTTTTTTTTTACGTACGTACGT";
    let r2_sequence = "GCGTACGATCGATCGATCGATCGATCGATCGATCGATCGA";
    let mut r1 = String::new();
    let mut r2 = String::new();
    for index in 0..32 {
        r1.push_str(&format!(
            "@sc-{index}/1\n{r1_sequence}\n+\n{}\n",
            "I".repeat(r1_sequence.len())
        ));
        r2.push_str(&format!(
            "@sc-{index}/2\n{r2_sequence}\n+\n{}\n",
            "I".repeat(r2_sequence.len())
        ));
    }
    write_gzip_fastq(&input1, &r1)?;
    write_gzip_fastq(&input2, &r2)?;

    let inspect = inspect_inputs(&input1, Some(&input2), 32, BackendPreference::Cpu)?;

    assert!(inspect.paired_end);
    assert_eq!(
        inspect.auto_profile.experiment,
        ExperimentType::SingleCell10xV3
    );

    Ok(())
}

#[test]
fn cuda_request_reports_acceleration_preview() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("input.fastq");
    let output = dir.path().join("output.fastq");
    let core = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let mut fastq = String::new();

    for index in 0..8 {
        fastq.push_str(&format!(
            "@preview-{index}\n{core}\n+\n{}\n",
            "I".repeat(core.len())
        ));
    }

    fs::write(&input, fastq)?;
    let report = process_file(
        &input,
        &output,
        &ProcessOptions {
            sample_size: 32,
            batch_reads: 8,
            backend_preference: BackendPreference::Cuda,
            forced_adapter: None,
            min_quality_override: None,
            kmer_size_override: None,
                    forced_platform: None,
                    forced_experiment: None,
        },
    )?;

    let runtime = report
        .accelerator_runtime
        .as_ref()
        .expect("cuda runtime probe should exist");
    if runtime.status == AcceleratorRuntimeStatus::Available {
        assert_eq!(report.backend_used, "cuda-full");
        assert!(!report.acceleration_executions.is_empty());
        assert!(report.acceleration_executions[0].successful);
        assert_eq!(report.acceleration_executions[0].batch_index, 0);
        assert_eq!(report.acceleration_executions[0].stage, "full_trim_correct");
    } else {
        assert_eq!(report.backend_used, "cpu");
    }
    assert!(report.throughput.wall_clock_us > 0);
    assert_eq!(report.acceleration_previews.len(), 1);
    let preview = &report.acceleration_previews[0];
    assert_eq!(preview.label, "single");
    assert_eq!(preview.reads, 8);
    assert_eq!(preview.total_bases, 8 * core.len());
    assert!(preview.packed_bases_bytes > 0);
    assert!(preview.packed_bases_bytes * 4 >= preview.total_bases);
    assert_eq!(preview.block_size, 256);
    assert!(preview.grid_size >= 1);

    Ok(())
}

#[test]
fn cuda_full_backend_matches_cpu_behavior_when_available() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("input.fastq");
    let output_cpu = dir.path().join("output.cpu.fastq");
    let output_cuda = dir.path().join("output.cuda.fastq");
    let core = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let adapter = "AGATCGGAAGAGC";
    let mut fastq = String::new();

    for index in 0..12 {
        fastq.push_str(&format!(
            "@mix-{index}\n{core}\n+\n{}\n",
            "I".repeat(core.len())
        ));
    }

    for index in 12..20 {
        let qualities = format!("{}{}", "I".repeat(core.len() - 6), "!".repeat(6));
        fastq.push_str(&format!("@mix-{index}\n{core}\n+\n{qualities}\n"));
    }

    for index in 20..24 {
        let qualities = format!(
            "{}{}{}",
            "I".repeat(core.len() - 4),
            "!".repeat(4),
            "!".repeat(adapter.len())
        );
        fastq.push_str(&format!(
            "@mix-{index}\n{}{}\n+\n{}\n",
            core, adapter, qualities
        ));
    }

    let mut corrupted = core.as_bytes().to_vec();
    corrupted[18] = b'T';
    let corrupted = String::from_utf8(corrupted)?;
    for index in 24..30 {
        let qualities = format!(
            "{}!{}{}",
            "I".repeat(18),
            "I".repeat(core.len() - 19),
            "!".repeat(adapter.len())
        );
        fastq.push_str(&format!(
            "@mix-{index}\n{}{}\n+\n{}\n",
            corrupted, adapter, qualities
        ));
    }

    let mut ambiguous = core.as_bytes().to_vec();
    ambiguous[10] = b'N';
    let ambiguous = String::from_utf8(ambiguous)?;
    for index in 30..34 {
        fastq.push_str(&format!(
            "@mix-{index}\n{ambiguous}\n+\n{}\n",
            "I".repeat(core.len())
        ));
    }

    fs::write(&input, fastq)?;

    let cpu_report = process_file(
        &input,
        &output_cpu,
        &ProcessOptions {
            sample_size: 64,
            batch_reads: 8,
            backend_preference: BackendPreference::Cpu,
            forced_adapter: None,
            min_quality_override: Some(20),
            kmer_size_override: None,
                    forced_platform: None,
                    forced_experiment: None,
        },
    )?;

    let cuda_report = process_file(
        &input,
        &output_cuda,
        &ProcessOptions {
            sample_size: 64,
            batch_reads: 8,
            backend_preference: BackendPreference::Cuda,
            forced_adapter: None,
            min_quality_override: Some(20),
            kmer_size_override: None,
                    forced_platform: None,
                    forced_experiment: None,
        },
    )?;

    let runtime = cuda_report
        .accelerator_runtime
        .as_ref()
        .expect("cuda runtime probe should exist");
    if runtime.status != AcceleratorRuntimeStatus::Available {
        return Ok(());
    }

    assert_eq!(cuda_report.backend_used, "cuda-full");
    assert!(!cuda_report.acceleration_executions.is_empty());
    assert!(
        cuda_report
            .acceleration_executions
            .iter()
            .all(|execution| execution.successful)
    );
    assert!(cuda_report.corrected_bases >= 6);
    assert!(
        cuda_report
            .notes
            .iter()
            .any(|note| note.contains("in-flight stream slots"))
    );
    assert!(cuda_report.throughput.wall_clock_us > 0);
    assert!(cuda_report.throughput.input_bases_per_sec > 0.0);
    if cuda_report.acceleration_executions.len() > 1 {
        assert!(
            cuda_report.acceleration_executions[0].transfer_bytes
                > cuda_report.acceleration_executions[1].transfer_bytes
        );
        assert!(
            cuda_report.acceleration_executions[1]
                .notes
                .iter()
                .any(|note| note.contains("device-resident"))
        );
        assert!(
            cuda_report
                .acceleration_executions
                .iter()
                .flat_map(|execution| execution.notes.iter())
                .any(|note| note.contains("slot=0"))
        );
        assert!(
            cuda_report
                .acceleration_executions
                .iter()
                .flat_map(|execution| execution.notes.iter())
                .any(|note| note.contains("slot=1"))
        );
        assert!(cuda_report.throughput.cumulative_overlap_us > 0);
        assert!(
            cuda_report
                .acceleration_executions
                .iter()
                .all(|execution| execution.end_to_end_us >= execution.submit_us)
        );
    }

    let cpu_records = sample_records(&output_cpu, 128)?;
    let cuda_records = sample_records(&output_cuda, 128)?;
    assert_eq!(cpu_report.output_reads, cuda_report.output_reads);
    assert_eq!(cpu_records.len(), cuda_records.len());
    assert_eq!(cpu_records, cuda_records);

    Ok(())
}
