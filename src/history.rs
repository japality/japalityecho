use std::collections::BTreeSet;
use std::fs::{OpenOptions, create_dir_all};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::model::{
    BackendPreference, BenchmarkComparisonReport, BenchmarkComparisonSummary, BenchmarkReport,
    BenchmarkSessionMode, BenchmarkSummary, HistoryComparisonStats, HistoryEntrySummary,
    HistoryRegressionAnalysis, HistoryRegressionStatus, HistoryRegressionThresholds, HistoryReport,
};
use crate::process::{BenchmarkComparisonOptions, BenchmarkOptions};

const HISTORY_VERSION: u32 = 1;
const TOOL_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone)]
pub struct HistoryReportOptions {
    pub limit: usize,
    pub baseline_label: Option<String>,
    pub baseline_label_prefix: Option<String>,
    pub baseline_latest_pass: bool,
    pub max_wall_clock_regression_pct: Option<f64>,
    pub min_raw_speedup: Option<f64>,
    pub min_transfer_savings_pct: Option<f64>,
}

impl Default for HistoryReportOptions {
    fn default() -> Self {
        Self {
            limit: 10,
            baseline_label: None,
            baseline_label_prefix: None,
            baseline_latest_pass: false,
            max_wall_clock_regression_pct: None,
            min_raw_speedup: None,
            min_transfer_savings_pct: None,
        }
    }
}

#[derive(Serialize)]
struct BenchmarkHistoryMetadata<'a> {
    history_version: u32,
    tool_version: &'static str,
    recorded_at_unix_ms: u64,
    label: Option<&'a str>,
    input1: String,
    input2: Option<String>,
    paired_end: bool,
    requested_backend: BackendPreference,
    sample_size: usize,
    batch_reads: usize,
    forced_adapter: Option<&'a str>,
    min_quality_override: Option<u8>,
    rounds: usize,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BenchmarkHistoryEntry<'a> {
    Benchmark {
        #[serde(flatten)]
        metadata: BenchmarkHistoryMetadata<'a>,
        session_mode: BenchmarkSessionMode,
        summary: &'a BenchmarkSummary,
    },
    BenchmarkCompare {
        #[serde(flatten)]
        metadata: BenchmarkHistoryMetadata<'a>,
        cold_start_mode: BenchmarkSessionMode,
        reuse_session_mode: BenchmarkSessionMode,
        summary: &'a BenchmarkComparisonSummary,
    },
}

pub fn append_benchmark_history(
    path: &Path,
    label: Option<&str>,
    input1: &Path,
    input2: Option<&Path>,
    options: &BenchmarkOptions,
    report: &mut BenchmarkReport,
) -> Result<()> {
    let metadata = history_metadata(
        label,
        input1,
        input2,
        options.process.backend_preference,
        &options.process,
        options.rounds,
        report.paired_end,
    )?;
    append_history_entry(
        path,
        &BenchmarkHistoryEntry::Benchmark {
            metadata,
            session_mode: report.summary.session_mode,
            summary: &report.summary,
        },
    )?;
    report.notes.push(history_note(path, label, "benchmark"));
    Ok(())
}

pub fn append_benchmark_comparison_history(
    path: &Path,
    label: Option<&str>,
    input1: &Path,
    input2: Option<&Path>,
    options: &BenchmarkComparisonOptions,
    report: &mut BenchmarkComparisonReport,
) -> Result<()> {
    let metadata = history_metadata(
        label,
        input1,
        input2,
        options.process.backend_preference,
        &options.process,
        options.rounds,
        report.paired_end,
    )?;
    append_history_entry(
        path,
        &BenchmarkHistoryEntry::BenchmarkCompare {
            metadata,
            cold_start_mode: report.cold_start.summary.session_mode,
            reuse_session_mode: report.reuse_session.summary.session_mode,
            summary: &report.summary,
        },
    )?;
    report
        .notes
        .push(history_note(path, label, "benchmark comparison"));
    Ok(())
}

fn history_metadata<'a>(
    label: Option<&'a str>,
    input1: &Path,
    input2: Option<&Path>,
    requested_backend: BackendPreference,
    options: &'a crate::process::ProcessOptions,
    rounds: usize,
    paired_end: bool,
) -> Result<BenchmarkHistoryMetadata<'a>> {
    Ok(BenchmarkHistoryMetadata {
        history_version: HISTORY_VERSION,
        tool_version: TOOL_VERSION,
        recorded_at_unix_ms: unix_epoch_ms()?,
        label,
        input1: input1.display().to_string(),
        input2: input2.map(|path| path.display().to_string()),
        paired_end,
        requested_backend,
        sample_size: options.sample_size,
        batch_reads: options.batch_reads,
        forced_adapter: options.forced_adapter.as_deref(),
        min_quality_override: options.min_quality_override,
        rounds: rounds.max(1),
    })
}

fn append_history_entry<T: Serialize>(path: &Path, entry: &T) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        create_dir_all(parent)
            .with_context(|| format!("failed to create history directory {}", parent.display()))?;
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open benchmark history {}", path.display()))?;
    serde_json::to_writer(&mut file, entry)
        .with_context(|| format!("failed to serialize benchmark history {}", path.display()))?;
    writeln!(file)
        .with_context(|| format!("failed to finalize benchmark history {}", path.display()))?;
    Ok(())
}

fn unix_epoch_ms() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before unix epoch")?
        .as_millis()
        .min(u128::from(u64::MAX)) as u64)
}

fn history_note(path: &Path, label: Option<&str>, kind: &str) -> String {
    match label {
        Some(label) => format!(
            "Wrote {kind} history entry '{}' to {}",
            label,
            path.display()
        ),
        None => format!("Wrote {kind} history entry to {}", path.display()),
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum StoredBenchmarkHistoryEntry {
    Benchmark(StoredBenchmarkRecord),
    BenchmarkCompare(StoredBenchmarkCompareRecord),
}

#[derive(Debug, Clone, Deserialize)]
struct StoredBenchmarkRecord {
    history_version: u32,
    tool_version: String,
    recorded_at_unix_ms: u64,
    label: Option<String>,
    input1: String,
    input2: Option<String>,
    paired_end: bool,
    requested_backend: BackendPreference,
    sample_size: usize,
    batch_reads: usize,
    forced_adapter: Option<String>,
    min_quality_override: Option<u8>,
    rounds: usize,
    session_mode: BenchmarkSessionMode,
    summary: BenchmarkSummary,
}

#[derive(Debug, Clone, Deserialize)]
struct StoredBenchmarkCompareRecord {
    history_version: u32,
    tool_version: String,
    recorded_at_unix_ms: u64,
    label: Option<String>,
    input1: String,
    input2: Option<String>,
    paired_end: bool,
    requested_backend: BackendPreference,
    sample_size: usize,
    batch_reads: usize,
    forced_adapter: Option<String>,
    min_quality_override: Option<u8>,
    rounds: usize,
    cold_start_mode: BenchmarkSessionMode,
    reuse_session_mode: BenchmarkSessionMode,
    summary: BenchmarkComparisonSummary,
}

pub fn read_history_report(path: &Path, options: &HistoryReportOptions) -> Result<HistoryReport> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open benchmark history {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut entries = Vec::new();

    for (line_number, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "failed to read line {} from benchmark history {}",
                line_number + 1,
                path.display()
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: StoredBenchmarkHistoryEntry =
            serde_json::from_str(&line).with_context(|| {
                format!(
                    "failed to parse line {} from benchmark history {}",
                    line_number + 1,
                    path.display()
                )
            })?;
        entries.push(entry);
    }

    let total_entries = entries.len();
    let benchmark_entries = entries
        .iter()
        .filter(|entry| matches!(entry, StoredBenchmarkHistoryEntry::Benchmark(_)))
        .count();
    let benchmark_compare_entries = total_entries - benchmark_entries;

    let mut labels = BTreeSet::new();
    let mut requested_backends = BTreeSet::new();
    let mut latest_recorded_at_unix_ms = None;
    let mut comparison_stats = HistoryComparisonStats::default();
    let mut raw_speedups = Vec::new();
    let mut amortized_speedups = Vec::new();
    let mut transfer_savings = Vec::new();
    let summaries: Vec<_> = entries
        .iter()
        .cloned()
        .map(summarize_history_entry)
        .collect();

    for entry in &entries {
        match entry {
            StoredBenchmarkHistoryEntry::Benchmark(record) => {
                touch_record_metadata(
                    record.label.as_deref(),
                    record.requested_backend,
                    record.recorded_at_unix_ms,
                    &mut labels,
                    &mut requested_backends,
                    &mut latest_recorded_at_unix_ms,
                );
            }
            StoredBenchmarkHistoryEntry::BenchmarkCompare(record) => {
                touch_record_metadata(
                    record.label.as_deref(),
                    record.requested_backend,
                    record.recorded_at_unix_ms,
                    &mut labels,
                    &mut requested_backends,
                    &mut latest_recorded_at_unix_ms,
                );
                if let Some(raw) = record.summary.raw_speedup {
                    raw_speedups.push(raw);
                }
                if let Some(amortized) = record.summary.amortized_speedup {
                    amortized_speedups.push(amortized);
                }
                if let Some(savings) = record.summary.steady_state_transfer_savings_pct {
                    transfer_savings.push(savings);
                }
            }
        }
    }

    if let Some(last_compare) = entries.iter().rev().find_map(|entry| match entry {
        StoredBenchmarkHistoryEntry::BenchmarkCompare(record) => Some(record),
        _ => None,
    }) {
        comparison_stats.latest_average_wall_clock_delta_us =
            Some(last_compare.summary.average_wall_clock_delta_us);
        comparison_stats.latest_raw_speedup = last_compare.summary.raw_speedup;
        comparison_stats.latest_amortized_speedup = last_compare.summary.amortized_speedup;
    }
    comparison_stats.average_raw_speedup = average_f64(&raw_speedups);
    comparison_stats.average_amortized_speedup = average_f64(&amortized_speedups);
    comparison_stats.average_transfer_savings_pct = average_f64(&transfer_savings);
    comparison_stats.best_raw_speedup = raw_speedups.iter().copied().reduce(f64::max);

    let mut recent: Vec<_> = summaries
        .iter()
        .cloned()
        .rev()
        .take(options.limit.max(1))
        .collect();
    recent.reverse();
    let latest_entry = summaries.last().cloned();
    let regression = analyze_regression(&summaries, options);

    let mut notes = vec![format!(
        "Loaded {} history entries from {}",
        total_entries,
        path.display()
    )];
    if let Some(latest_entry) = &latest_entry {
        notes.push(format!(
            "Latest entry: kind={} backend={} label={}",
            latest_entry.kind,
            latest_entry.requested_backend,
            latest_entry.label.as_deref().unwrap_or("unlabeled")
        ));
    }
    if let Some(best_raw_speedup) = comparison_stats.best_raw_speedup {
        notes.push(format!(
            "Best recorded raw reuse-session speedup among compare entries: {best_raw_speedup:.3}x"
        ));
    }
    if let Some(regression) = &regression {
        notes.push(format!(
            "Regression analysis status={} baseline_source={}",
            regression.status, regression.baseline_source
        ));
        notes.extend(regression.alerts.iter().cloned());
    }

    Ok(HistoryReport {
        source_path: path.display().to_string(),
        total_entries,
        benchmark_entries,
        benchmark_compare_entries,
        labels: labels.into_iter().collect(),
        requested_backends: requested_backends.into_iter().collect(),
        latest_recorded_at_unix_ms,
        latest_entry,
        recent_entries: recent,
        comparison_stats,
        regression,
        notes,
    })
}

fn analyze_regression(
    summaries: &[HistoryEntrySummary],
    options: &HistoryReportOptions,
) -> Option<HistoryRegressionAnalysis> {
    let target_entry = summaries.last()?.clone();
    let thresholds = HistoryRegressionThresholds {
        max_wall_clock_regression_pct: options.max_wall_clock_regression_pct,
        min_raw_speedup: options.min_raw_speedup,
        min_transfer_savings_pct: options.min_transfer_savings_pct,
    };

    let (baseline_source, baseline_entry) =
        select_baseline_entry(summaries, &target_entry, options, &thresholds);
    Some(build_regression_analysis(
        target_entry,
        baseline_source,
        baseline_entry,
        thresholds,
    ))
}

fn build_regression_analysis(
    target_entry: HistoryEntrySummary,
    baseline_source: String,
    baseline_entry: Option<HistoryEntrySummary>,
    thresholds: HistoryRegressionThresholds,
) -> HistoryRegressionAnalysis {
    let Some(baseline_entry) = baseline_entry else {
        return HistoryRegressionAnalysis {
            baseline_source: baseline_source.clone(),
            target_entry,
            baseline_entry: None,
            comparable: false,
            status: HistoryRegressionStatus::NoBaseline,
            thresholds,
            wall_clock_delta_us: None,
            wall_clock_delta_pct: None,
            raw_speedup_delta: None,
            amortized_speedup_delta: None,
            transfer_savings_pct_delta: None,
            alerts: vec![missing_baseline_alert(&baseline_source)],
        };
    };

    let comparable = entries_are_comparable(&target_entry, &baseline_entry);
    if !comparable {
        return HistoryRegressionAnalysis {
            baseline_source,
            target_entry,
            baseline_entry: Some(baseline_entry),
            comparable,
            status: HistoryRegressionStatus::NotComparable,
            thresholds,
            wall_clock_delta_us: None,
            wall_clock_delta_pct: None,
            raw_speedup_delta: None,
            amortized_speedup_delta: None,
            transfer_savings_pct_delta: None,
            alerts: vec![
                "Baseline entry exists but is not comparable with the latest entry".to_string(),
            ],
        };
    }

    let wall_clock_delta_us =
        target_entry.average_wall_clock_us - baseline_entry.average_wall_clock_us;
    let wall_clock_delta_pct = (baseline_entry.average_wall_clock_us > 0.0)
        .then(|| (wall_clock_delta_us / baseline_entry.average_wall_clock_us) * 100.0);
    let raw_speedup_delta = delta_option(target_entry.raw_speedup, baseline_entry.raw_speedup);
    let amortized_speedup_delta = delta_option(
        target_entry.amortized_speedup,
        baseline_entry.amortized_speedup,
    );
    let transfer_savings_pct_delta = delta_option(
        target_entry.steady_state_transfer_savings_pct,
        baseline_entry.steady_state_transfer_savings_pct,
    );

    let mut alerts = Vec::new();
    if let (Some(max_regression_pct), Some(actual_regression_pct)) = (
        thresholds.max_wall_clock_regression_pct,
        wall_clock_delta_pct,
    ) {
        if actual_regression_pct > max_regression_pct {
            alerts.push(format!(
                "Wall-clock regression {:.1}% exceeded threshold {:.1}%",
                actual_regression_pct, max_regression_pct
            ));
        }
    }
    if let Some(min_raw_speedup) = thresholds.min_raw_speedup {
        if let Some(raw_speedup) = target_entry.raw_speedup {
            if raw_speedup < min_raw_speedup {
                alerts.push(format!(
                    "Raw speedup {:.3}x is below threshold {:.3}x",
                    raw_speedup, min_raw_speedup
                ));
            }
        }
    }
    if let Some(min_transfer_savings_pct) = thresholds.min_transfer_savings_pct {
        if let Some(transfer_savings_pct) = target_entry.steady_state_transfer_savings_pct {
            if transfer_savings_pct < min_transfer_savings_pct {
                alerts.push(format!(
                    "Transfer savings {:.1}% is below threshold {:.1}%",
                    transfer_savings_pct, min_transfer_savings_pct
                ));
            }
        }
    }

    HistoryRegressionAnalysis {
        baseline_source,
        target_entry,
        baseline_entry: Some(baseline_entry),
        comparable,
        status: if alerts.is_empty() {
            HistoryRegressionStatus::Pass
        } else {
            HistoryRegressionStatus::Alert
        },
        thresholds,
        wall_clock_delta_us: Some(wall_clock_delta_us),
        wall_clock_delta_pct,
        raw_speedup_delta,
        amortized_speedup_delta,
        transfer_savings_pct_delta,
        alerts,
    }
}

fn select_baseline_entry(
    summaries: &[HistoryEntrySummary],
    target_entry: &HistoryEntrySummary,
    options: &HistoryReportOptions,
    thresholds: &HistoryRegressionThresholds,
) -> (String, Option<HistoryEntrySummary>) {
    if options.baseline_latest_pass {
        return select_latest_pass_baseline_entry(
            summaries,
            target_entry,
            thresholds,
            options.baseline_label_prefix.as_deref(),
        );
    }

    if let Some(label) = options.baseline_label.as_deref() {
        return select_matching_baseline_entry(
            summaries,
            target_entry,
            format!("label:{label}"),
            |entry| entry.label.as_deref() == Some(label),
        );
    }

    if let Some(prefix) = options.baseline_label_prefix.as_deref() {
        return select_matching_baseline_entry(
            summaries,
            target_entry,
            format!("label_prefix:{prefix}"),
            |entry| {
                entry
                    .label
                    .as_deref()
                    .is_some_and(|label| label.starts_with(prefix))
            },
        );
    }

    (
        "previous_comparable".to_string(),
        summaries
            .iter()
            .rev()
            .skip(1)
            .find(|entry| entries_are_comparable(target_entry, entry))
            .cloned(),
    )
}

fn select_matching_baseline_entry<F>(
    summaries: &[HistoryEntrySummary],
    target_entry: &HistoryEntrySummary,
    baseline_source: String,
    mut matches: F,
) -> (String, Option<HistoryEntrySummary>)
where
    F: FnMut(&HistoryEntrySummary) -> bool,
{
    let mut latest_matching_entry = None;

    for entry in summaries.iter().rev().skip(1) {
        if !matches(entry) {
            continue;
        }

        if latest_matching_entry.is_none() {
            latest_matching_entry = Some(entry.clone());
        }

        if entries_are_comparable(target_entry, entry) {
            return (baseline_source, Some(entry.clone()));
        }
    }

    (baseline_source, latest_matching_entry)
}

fn select_latest_pass_baseline_entry(
    summaries: &[HistoryEntrySummary],
    target_entry: &HistoryEntrySummary,
    thresholds: &HistoryRegressionThresholds,
    label_prefix: Option<&str>,
) -> (String, Option<HistoryEntrySummary>) {
    let baseline_source = match label_prefix {
        Some(prefix) => format!("latest_pass:label_prefix:{prefix}"),
        None => "latest_pass".to_string(),
    };

    for candidate_index in (0..summaries.len().saturating_sub(1)).rev() {
        let candidate_entry = &summaries[candidate_index];
        if let Some(prefix) = label_prefix {
            let matches_prefix = candidate_entry
                .label
                .as_deref()
                .is_some_and(|label| label.starts_with(prefix));
            if !matches_prefix {
                continue;
            }
        }
        if !entries_are_comparable(target_entry, candidate_entry) {
            continue;
        }

        if entry_has_passing_regression(&summaries[..=candidate_index], thresholds) {
            return (baseline_source, Some(candidate_entry.clone()));
        }
    }

    (baseline_source, None)
}

fn entry_has_passing_regression(
    summaries: &[HistoryEntrySummary],
    thresholds: &HistoryRegressionThresholds,
) -> bool {
    let Some(target_entry) = summaries.last().cloned() else {
        return false;
    };
    let (baseline_source, baseline_entry) =
        select_previous_comparable_baseline_entry(summaries, &target_entry);
    let analysis = build_regression_analysis(
        target_entry,
        baseline_source,
        baseline_entry,
        thresholds.clone(),
    );
    analysis.status == HistoryRegressionStatus::Pass
}

fn select_previous_comparable_baseline_entry(
    summaries: &[HistoryEntrySummary],
    target_entry: &HistoryEntrySummary,
) -> (String, Option<HistoryEntrySummary>) {
    (
        "previous_comparable".to_string(),
        summaries
            .iter()
            .rev()
            .skip(1)
            .find(|entry| entries_are_comparable(target_entry, entry))
            .cloned(),
    )
}

fn missing_baseline_alert(baseline_source: &str) -> String {
    if baseline_source.starts_with("latest_pass") {
        "No prior passing comparable baseline matched the requested selection".to_string()
    } else {
        "No baseline entry matched the requested selection".to_string()
    }
}

fn summarize_history_entry(entry: StoredBenchmarkHistoryEntry) -> HistoryEntrySummary {
    match entry {
        StoredBenchmarkHistoryEntry::Benchmark(record) => {
            touch_unused_benchmark_fields(&record);
            HistoryEntrySummary {
                kind: "benchmark".to_string(),
                label: record.label,
                recorded_at_unix_ms: record.recorded_at_unix_ms,
                requested_backend: record.requested_backend,
                rounds: record.rounds,
                paired_end: record.paired_end,
                average_wall_clock_us: record.summary.average_wall_clock_us,
                setup_us: Some(record.summary.setup_us),
                session_mode: Some(record.session_mode),
                raw_speedup: None,
                amortized_speedup: None,
                steady_state_transfer_savings_pct: None,
            }
        }
        StoredBenchmarkHistoryEntry::BenchmarkCompare(record) => {
            touch_unused_compare_fields(&record);
            HistoryEntrySummary {
                kind: "benchmark_compare".to_string(),
                label: record.label,
                recorded_at_unix_ms: record.recorded_at_unix_ms,
                requested_backend: record.requested_backend,
                rounds: record.rounds,
                paired_end: record.paired_end,
                average_wall_clock_us: record.summary.reuse_session_average_wall_clock_us,
                setup_us: Some(record.summary.reuse_session_setup_us),
                session_mode: None,
                raw_speedup: record.summary.raw_speedup,
                amortized_speedup: record.summary.amortized_speedup,
                steady_state_transfer_savings_pct: record.summary.steady_state_transfer_savings_pct,
            }
        }
    }
}

fn touch_record_metadata(
    label: Option<&str>,
    requested_backend: BackendPreference,
    recorded_at_unix_ms: u64,
    labels: &mut BTreeSet<String>,
    requested_backends: &mut BTreeSet<BackendPreference>,
    latest_recorded_at_unix_ms: &mut Option<u64>,
) {
    if let Some(label) = label {
        labels.insert(label.to_string());
    }
    requested_backends.insert(requested_backend);
    *latest_recorded_at_unix_ms = Some(
        latest_recorded_at_unix_ms.map_or(recorded_at_unix_ms, |current| {
            current.max(recorded_at_unix_ms)
        }),
    );
}

fn average_f64(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn entries_are_comparable(target: &HistoryEntrySummary, baseline: &HistoryEntrySummary) -> bool {
    target.kind == baseline.kind
        && target.requested_backend == baseline.requested_backend
        && target.paired_end == baseline.paired_end
        && (target.kind != "benchmark" || target.session_mode == baseline.session_mode)
}

fn delta_option(current: Option<f64>, baseline: Option<f64>) -> Option<f64> {
    match (current, baseline) {
        (Some(current), Some(baseline)) => Some(current - baseline),
        _ => None,
    }
}

fn touch_unused_benchmark_fields(record: &StoredBenchmarkRecord) {
    let _ = (
        record.history_version,
        &record.tool_version,
        &record.input1,
        &record.input2,
        record.sample_size,
        record.batch_reads,
        &record.forced_adapter,
        record.min_quality_override,
    );
}

fn touch_unused_compare_fields(record: &StoredBenchmarkCompareRecord) {
    let _ = (
        record.history_version,
        &record.tool_version,
        &record.input1,
        &record.input2,
        record.sample_size,
        record.batch_reads,
        &record.forced_adapter,
        record.min_quality_override,
        record.cold_start_mode,
        record.reuse_session_mode,
    );
}
