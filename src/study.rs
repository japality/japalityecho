use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::model::{
    BackendPreference, BenchmarkSessionMode, ExperimentType, PaperArtifactFile, Platform,
    StudyAggregateSummary, StudyAnnotateReport, StudyAnnotateSummary, StudyArtifactsReport,
    StudyBaselineMetrics, StudyComparisonMetrics, StudyComparisonSummary, StudyDatasetReport,
    StudyDetectionSummary, StudyDiscoverDataset, StudyDiscoverReport, StudyDiscoverSummary,
    StudyDownloadReport, StudyDownloadSummary, StudyDownstreamMetrics, StudyFetchMetadataReport,
    StudyFetchMetadataSummary, StudyIngestReport, StudyIngestSummary, StudyManifestBootstrapReport,
    StudyManifestBootstrapSummary,
};
use crate::process::{BenchmarkOptions, ProcessOptions, benchmark_files, inspect_inputs};

#[derive(Debug, Clone)]
pub struct StudyArtifactsOptions {
    pub manifest_path: PathBuf,
    pub output_dir: PathBuf,
    pub sample_size: usize,
    pub batch_reads: usize,
    pub benchmark_rounds: usize,
    pub backend_preference: BackendPreference,
    pub min_quality_override: Option<u8>,
}

#[derive(Debug, Clone)]
pub struct StudyManifestOptions {
    pub inventory_path: PathBuf,
    pub output_path: PathBuf,
    pub default_baseline_name: Option<String>,
    pub download_root: PathBuf,
}

#[derive(Debug, Clone)]
pub struct StudyDiscoverOptions {
    pub input_dir: PathBuf,
    pub output_path: PathBuf,
    pub recursive: bool,
}

#[derive(Debug, Clone)]
pub struct StudyFetchMetadataOptions {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub base_url: String,
    pub geo_base_url: String,
    pub chunk_size: usize,
    pub cache_dir: Option<PathBuf>,
    pub retries: usize,
    pub resume_existing: bool,
}

#[derive(Debug, Clone)]
pub struct StudyDownloadOptions {
    pub input_path: PathBuf,
    pub download_root: PathBuf,
    pub retries: usize,
    pub overwrite_existing: bool,
}

#[derive(Debug, Clone)]
pub struct StudyAnnotateOptions {
    pub manifest_path: PathBuf,
    pub annotations_path: PathBuf,
    pub output_path: PathBuf,
    pub overwrite_existing: bool,
}

#[derive(Debug, Clone)]
pub struct StudyIngestOptions {
    pub manifest_path: PathBuf,
    pub input_dir: PathBuf,
    pub output_path: PathBuf,
    pub recursive: bool,
    pub overwrite_existing: bool,
}

#[derive(Debug, Clone)]
struct ManifestSourceEntry {
    dataset_id: Option<String>,
    accession: Option<String>,
    citation: Option<String>,
    input1: PathBuf,
    input2: Option<PathBuf>,
    remote_locations: Vec<String>,
    expected_platform: Option<Platform>,
    expected_experiment: Option<ExperimentType>,
    downstream: StudyDownstreamMetrics,
    baseline: Option<StudyBaselineMetrics>,
    notes: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct ManifestAnnotationEntry {
    dataset_id: Option<String>,
    accession: Option<String>,
    citation: Option<String>,
    expected_platform: Option<Platform>,
    expected_experiment: Option<ExperimentType>,
    downstream: StudyDownstreamMetrics,
    baseline: Option<StudyBaselineMetrics>,
    notes: Option<String>,
}

#[derive(Debug, Clone)]
struct ManifestEntry {
    dataset_id: String,
    dataset_id_generated: bool,
    accession: Option<String>,
    citation: Option<String>,
    input1: PathBuf,
    input2: Option<PathBuf>,
    expected_platform: Option<Platform>,
    expected_experiment: Option<ExperimentType>,
    downstream: StudyDownstreamMetrics,
    baseline: Option<StudyBaselineMetrics>,
    notes: Option<String>,
}

const CANONICAL_STUDY_MANIFEST_HEADERS: &[&str] = &[
    "dataset_id",
    "accession",
    "citation",
    "input1",
    "input2",
    "expected_platform",
    "expected_experiment",
    "baseline_name",
    "baseline_input_bases_per_sec",
    "baseline_trimmed_read_fraction",
    "baseline_discarded_read_fraction",
    "baseline_corrected_bases_per_mbase",
    "echo_alignment_rate",
    "baseline_alignment_rate",
    "echo_duplicate_rate",
    "baseline_duplicate_rate",
    "echo_variant_f1",
    "baseline_variant_f1",
    "echo_mean_coverage",
    "baseline_mean_coverage",
    "echo_coverage_breadth",
    "baseline_coverage_breadth",
    "echo_assembly_n50",
    "baseline_assembly_n50",
    "notes",
];

const CANONICAL_STUDY_ANNOTATION_HEADERS: &[&str] = &[
    "dataset_id",
    "accession",
    "citation",
    "expected_platform",
    "expected_experiment",
    "baseline_name",
    "baseline_input_bases_per_sec",
    "baseline_trimmed_read_fraction",
    "baseline_discarded_read_fraction",
    "baseline_corrected_bases_per_mbase",
    "echo_alignment_rate",
    "baseline_alignment_rate",
    "echo_duplicate_rate",
    "baseline_duplicate_rate",
    "echo_variant_f1",
    "baseline_variant_f1",
    "echo_mean_coverage",
    "baseline_mean_coverage",
    "echo_coverage_breadth",
    "baseline_coverage_breadth",
    "echo_assembly_n50",
    "baseline_assembly_n50",
    "notes",
];

const FETCHED_PUBLIC_METADATA_HEADERS: &[&str] = &[
    "run_accession",
    "study_accession",
    "study_title",
    "experiment_title",
    "instrument_platform",
    "instrument_model",
    "library_strategy",
    "library_layout",
    "library_source",
    "library_selection",
    "sample_title",
    "experiment_accession",
    "sample_accession",
    "bioproject",
    "fastq_ftp",
    "fastq_aspera",
    "submitted_ftp",
    "submitted_aspera",
    "query_accessions",
];

const ENA_FILEREPORT_FIELDS: &[&str] = &[
    "run_accession",
    "study_accession",
    "secondary_study_accession",
    "study_title",
    "experiment_accession",
    "experiment_title",
    "sample_accession",
    "instrument_platform",
    "instrument_model",
    "library_strategy",
    "library_layout",
    "library_source",
    "library_selection",
    "sample_title",
    "sample_alias",
    "fastq_ftp",
    "fastq_aspera",
    "submitted_ftp",
    "submitted_aspera",
];

const FETCHED_PUBLIC_METADATA_MATCH_FIELDS: &[&str] = &[
    "run_accession",
    "study_accession",
    "experiment_accession",
    "sample_accession",
    "bioproject",
];

const PUBLIC_ACCESSION_PREFIXES: &[&str] = &[
    "SRR", "ERR", "DRR", "SRP", "ERP", "DRP", "SRX", "ERX", "DRX", "SRS", "ERS", "DRS", "PRJNA",
    "PRJEB", "PRJDB", "GSE", "GSM",
];

const NON_GEO_PUBLIC_ACCESSION_PREFIXES: &[&str] = &[
    "SRR", "ERR", "DRR", "SRP", "ERP", "DRP", "SRX", "ERX", "DRX", "SRS", "ERS", "DRS", "PRJNA",
    "PRJEB", "PRJDB",
];

#[derive(Debug, Clone)]
struct DiscoveredFile {
    path: PathBuf,
    grouping_key: String,
    dataset_id: String,
    accession: Option<String>,
    mate: Option<u8>,
    note: Option<String>,
}

#[derive(Debug, Default)]
struct AnnotationMergeStats {
    fields_filled: usize,
    fields_overwritten: usize,
    notes_appended: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergeOutcome {
    Unchanged,
    Filled,
    Overwritten,
    Appended,
}

#[derive(Debug, Default)]
struct StructuredIngestStats {
    files_scanned: usize,
    structured_files: usize,
    json_files: usize,
    delimited_files: usize,
    ignored_files: usize,
    records_ingested: usize,
    matched_records: usize,
    unmatched_records: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeMetricSide {
    Echo,
    Baseline,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NativeMetricTarget {
    side: NativeMetricSide,
    baseline_name: Option<String>,
}

impl AnnotationMergeStats {
    fn record(&mut self, outcome: MergeOutcome) -> bool {
        match outcome {
            MergeOutcome::Unchanged => false,
            MergeOutcome::Filled => {
                self.fields_filled += 1;
                true
            }
            MergeOutcome::Overwritten => {
                self.fields_overwritten += 1;
                true
            }
            MergeOutcome::Appended => {
                self.notes_appended += 1;
                true
            }
        }
    }
}

pub fn discover_study_inventory(options: &StudyDiscoverOptions) -> Result<StudyDiscoverReport> {
    let mut files = Vec::new();
    let mut files_scanned = 0usize;
    collect_fastq_files(
        &options.input_dir,
        options.recursive,
        &mut files,
        &mut files_scanned,
    )?;
    files.sort();

    if let Some(parent) = options.output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let output_parent = options
        .output_path
        .parent()
        .unwrap_or_else(|| Path::new("."));

    let discovered = files
        .iter()
        .map(|path| discover_file_entry(path, &options.input_dir))
        .collect::<Result<Vec<_>>>()?;

    let mut grouped: Vec<(String, Vec<DiscoveredFile>)> = Vec::new();
    for file in discovered {
        if let Some((_, group)) = grouped
            .iter_mut()
            .find(|(key, _)| *key == file.grouping_key)
        {
            group.push(file);
        } else {
            grouped.push((file.grouping_key.clone(), vec![file]));
        }
    }
    grouped.sort_by(|a, b| a.0.cmp(&b.0));

    let mut manifest_entries = Vec::new();
    let mut dataset_reports = Vec::new();
    let mut seen_dataset_ids = HashSet::new();
    for (_, mut group) in grouped {
        group.sort_by_key(|file| file.path.clone());
        let mate1 = group.iter().find(|file| file.mate == Some(1)).cloned();
        let mate2 = group.iter().find(|file| file.mate == Some(2)).cloned();
        let unpaired = group.iter().find(|file| file.mate.is_none()).cloned();
        let primary = mate1
            .clone()
            .or_else(|| unpaired.clone())
            .or_else(|| mate2.clone())
            .context("discovered group unexpectedly had no primary FASTQ")?;
        let paired_end = mate1.is_some() && mate2.is_some();
        let (dataset_id, _) = allocate_dataset_id(
            Some(primary.dataset_id.as_str()),
            primary.accession.as_deref(),
            &primary.path,
            &mut seen_dataset_ids,
        );

        let mut notes = Vec::new();
        for file in &group {
            if let Some(note) = &file.note {
                if !notes.iter().any(|existing| existing == note) {
                    notes.push(note.clone());
                }
            }
        }
        if paired_end {
            notes.push("Paired mates were auto-grouped from local FASTQ filenames".to_string());
        } else if mate1.is_some() && mate2.is_none() {
            notes.push("Mate 1 was discovered without mate 2; inventory row remains single-input until curated".to_string());
        } else if mate1.is_none() && mate2.is_some() {
            notes.push("Only mate 2 matched a pairing pattern; inventory row uses that file as input1 until curated".to_string());
        }

        let entry = ManifestEntry {
            dataset_id: dataset_id.clone(),
            dataset_id_generated: false,
            accession: primary.accession.clone(),
            citation: None,
            input1: primary.path.clone(),
            input2: if paired_end {
                mate2.as_ref().map(|file| file.path.clone())
            } else {
                None
            },
            expected_platform: None,
            expected_experiment: None,
            downstream: default_study_downstream_metrics(),
            baseline: None,
            notes: if notes.is_empty() {
                None
            } else {
                Some(notes.join(" | "))
            },
        };
        manifest_entries.push(entry.clone());
        dataset_reports.push(StudyDiscoverDataset {
            dataset_id,
            accession: entry.accession,
            paired_end,
            input1: render_manifest_path_for_output(&entry.input1, output_parent),
            input2: entry
                .input2
                .as_ref()
                .map(|path| render_manifest_path_for_output(path, output_parent)),
            notes,
        });
    }

    write_canonical_manifest(
        &options.output_path,
        manifest_delimiter_for_output(&options.output_path),
        &manifest_entries,
    )?;

    let summary = StudyDiscoverSummary {
        files_scanned,
        fastq_files: files.len(),
        datasets: dataset_reports.len(),
        paired_datasets: dataset_reports
            .iter()
            .filter(|dataset| dataset.paired_end)
            .count(),
        single_end_datasets: dataset_reports
            .iter()
            .filter(|dataset| !dataset.paired_end)
            .count(),
        accession_labeled_datasets: dataset_reports
            .iter()
            .filter(|dataset| dataset.accession.is_some())
            .count(),
    };

    let mut notes = vec![
        "Discovered inventory is a pre-populated template; citation, expected labels, baseline metrics, and downstream metrics should be curated before publication use".to_string(),
        "Output inventory already uses the canonical phase-25 column set, so it can be edited in place or passed through study-manifest for normalization".to_string(),
    ];
    notes.push(format!(
        "Scanned {} files under {} and identified {} FASTQ inputs",
        summary.files_scanned,
        options.input_dir.display(),
        summary.fastq_files
    ));

    Ok(StudyDiscoverReport {
        input_dir: options.input_dir.display().to_string(),
        output_path: options.output_path.display().to_string(),
        recursive: options.recursive,
        summary,
        datasets: dataset_reports,
        notes,
    })
}

pub fn fetch_public_metadata(
    options: &StudyFetchMetadataOptions,
) -> Result<StudyFetchMetadataReport> {
    if options.chunk_size == 0 {
        bail!("metadata fetch chunk size must be greater than zero");
    }

    let requested_accessions = read_requested_accessions(&options.input_path)?;
    if requested_accessions.is_empty() {
        bail!(
            "metadata input {} did not contain any accession-like identifiers",
            options.input_path.display()
        );
    }

    let unique_accessions = dedupe_preserving_order(requested_accessions.iter().cloned());
    let requested_accession_set: HashSet<String> = unique_accessions.iter().cloned().collect();
    let cache_dir = options
        .cache_dir
        .clone()
        .unwrap_or_else(|| default_study_fetch_cache_dir(&options.output_path));
    let status_path = sibling_artifact_path(&options.output_path, "fetch_status.csv");
    let accession_positions: HashMap<String, usize> = unique_accessions
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, accession)| (accession, index))
        .collect();
    let mut accession_statuses: HashMap<String, FetchAccessionStatusRow> = unique_accessions
        .iter()
        .cloned()
        .map(|accession| {
            (
                accession.clone(),
                FetchAccessionStatusRow {
                    accession,
                    status: FetchAccessionStatusKind::Unmatched,
                    resolved_accessions: Vec::new(),
                    matched_runs: Vec::new(),
                    geo_surface: None,
                    source: None,
                    attempts: 0,
                    error: None,
                },
            )
        })
        .collect();
    let mut fetched_by_accession = HashMap::new();
    let mut cache_hits = 0usize;
    let mut remote_fetches = 0usize;
    let mut retried_chunks = 0usize;
    let mut failed_chunk_messages = Vec::new();
    let mut resumed_accessions = 0usize;
    let mut resumed_records = 0usize;
    let mut requested_accessions_by_query: HashMap<String, Vec<String>> = HashMap::new();
    let mut ena_query_accessions = Vec::new();
    let mut completed_accessions = HashSet::new();

    if let Some(parent) = options.output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::create_dir_all(&cache_dir)
        .with_context(|| format!("failed to create {}", cache_dir.display()))?;

    let existing_rows = if options.resume_existing {
        read_existing_fetched_public_metadata_rows(&options.output_path, &requested_accession_set)?
    } else {
        Vec::new()
    };
    let existing_statuses = if options.resume_existing {
        read_existing_fetch_status_rows(&status_path, &requested_accession_set)?
    } else {
        HashMap::new()
    };
    for row in existing_rows {
        if fetched_by_accession.contains_key(&row.run_accession) {
            continue;
        }
        resumed_records += 1;
        fetched_by_accession.insert(row.run_accession.clone(), row);
    }

    for accession in &unique_accessions {
        let existing_status = existing_statuses.get(accession).cloned();
        let status = accession_statuses
            .get_mut(accession)
            .with_context(|| format!("missing fetch status row for {accession}"))?;
        if let Some(existing_status) = existing_status.clone() {
            status.resolved_accessions = existing_status.resolved_accessions.clone();
            status.matched_runs = existing_status.matched_runs.clone();
            status.geo_surface = existing_status.geo_surface.clone();
            status.source = existing_status.source;
            status.attempts = existing_status.attempts;
            status.error = existing_status.error.clone();
        }
        if options.resume_existing {
            if let Some(existing_status) = existing_status.as_ref() {
                let can_resume = match existing_status.status {
                    FetchAccessionStatusKind::Matched => fetched_by_accession.values().any(|row| {
                        row.query_accessions
                            .iter()
                            .any(|query_accession| query_accession == accession)
                    }),
                    FetchAccessionStatusKind::Unmatched => true,
                    FetchAccessionStatusKind::FetchFailed => false,
                };
                if can_resume {
                    status.status = existing_status.status;
                    completed_accessions.insert(accession.clone());
                    resumed_accessions += 1;
                    continue;
                }
            }
        }
        if is_geo_accession(accession) {
            let reused_geo_resolution =
                options.resume_existing && !status.resolved_accessions.is_empty();
            let geo_resolution = if reused_geo_resolution {
                Ok(GeoBridgeFetchResult {
                    resolved_accessions: status.resolved_accessions.clone(),
                    geo_surface: status.geo_surface.clone(),
                    source: status.source.unwrap_or(MetadataChunkSource::Remote),
                    attempts: status.attempts,
                })
            } else {
                resolve_geo_accession(
                    accession,
                    &options.geo_base_url,
                    &cache_dir,
                    options.retries,
                )
            };
            match geo_resolution {
                Ok(result) => {
                    if !reused_geo_resolution {
                        match result.source {
                            MetadataChunkSource::Cache => cache_hits += 1,
                            MetadataChunkSource::Remote => {
                                remote_fetches += 1;
                                if result.attempts > 1 {
                                    retried_chunks += 1;
                                }
                            }
                        }
                    }
                    let resolved_accessions = dedupe_preserving_order(
                        result
                            .resolved_accessions
                            .into_iter()
                            .filter(|resolved| !is_geo_accession(resolved)),
                    );
                    status.resolved_accessions = resolved_accessions.clone();
                    status.geo_surface = result.geo_surface.clone();
                    status.status = FetchAccessionStatusKind::Unmatched;
                    status.source = Some(result.source);
                    status.attempts = result.attempts;
                    status.error = None;
                    if resolved_accessions.is_empty() {
                        continue;
                    }
                    for resolved in resolved_accessions {
                        requested_accessions_by_query
                            .entry(resolved.clone())
                            .or_default()
                            .push(accession.clone());
                        ena_query_accessions.push(resolved);
                    }
                }
                Err(error) => {
                    failed_chunk_messages.push(format!("[{}]: {}", accession, error.message));
                    status.status = FetchAccessionStatusKind::FetchFailed;
                    status.resolved_accessions.clear();
                    status.geo_surface = None;
                    status.source = Some(MetadataChunkSource::Remote);
                    status.attempts = error.attempts;
                    status.error = Some(error.message);
                }
            }
        } else {
            status.resolved_accessions = vec![accession.clone()];
            status.geo_surface = None;
            status.status = FetchAccessionStatusKind::Unmatched;
            status.error = None;
            requested_accessions_by_query
                .entry(accession.clone())
                .or_default()
                .push(accession.clone());
            ena_query_accessions.push(accession.clone());
        }
    }

    let ena_query_accessions =
        dedupe_preserving_order(ena_query_accessions.into_iter().filter(|query| {
            requested_accessions_by_query
                .get(query)
                .is_some_and(|requesters| {
                    requesters
                        .iter()
                        .any(|accession| !completed_accessions.contains(accession))
                })
        }));
    for chunk in ena_query_accessions.chunks(options.chunk_size) {
        let chunk_accessions = chunk.to_vec();
        let original_requests_in_chunk = dedupe_preserving_order(
            chunk_accessions
                .iter()
                .flat_map(|query| {
                    requested_accessions_by_query
                        .get(query)
                        .into_iter()
                        .flatten()
                        .cloned()
                })
                .filter(|accession| !completed_accessions.contains(accession)),
        );
        if original_requests_in_chunk.is_empty() {
            continue;
        }
        match fetch_metadata_chunk(
            &chunk_accessions,
            &options.base_url,
            &cache_dir,
            options.retries,
        ) {
            Ok(result) => {
                match result.source {
                    MetadataChunkSource::Cache => cache_hits += 1,
                    MetadataChunkSource::Remote => {
                        remote_fetches += 1;
                        if result.attempts > 1 {
                            retried_chunks += 1;
                        }
                    }
                }

                let mut matched_runs_by_accession: HashMap<String, Vec<String>> =
                    original_requests_in_chunk
                        .iter()
                        .cloned()
                        .map(|accession| (accession, Vec::new()))
                        .collect();
                for mut row in result.rows {
                    row.query_accessions =
                        dedupe_preserving_order(row.query_accessions.into_iter().flat_map(
                            |query_accession| {
                                requested_accessions_by_query
                                    .get(&query_accession)
                                    .cloned()
                                    .unwrap_or_else(|| vec![query_accession])
                            },
                        ));
                    synchronize_fetched_public_metadata_query_accessions(&mut row);

                    for accession in &row.query_accessions {
                        if let Some(matched_runs) = matched_runs_by_accession.get_mut(accession) {
                            if !matched_runs.iter().any(|run| run == &row.run_accession) {
                                matched_runs.push(row.run_accession.clone());
                            }
                        }
                    }

                    if let Some(existing) = fetched_by_accession.get_mut(&row.run_accession) {
                        merge_fetched_public_metadata_row(existing, row)?;
                    } else {
                        fetched_by_accession.insert(row.run_accession.clone(), row);
                    }
                }

                for accession in original_requests_in_chunk {
                    let chunk_matched_runs = matched_runs_by_accession
                        .remove(&accession)
                        .unwrap_or_default();
                    let status = accession_statuses
                        .get_mut(&accession)
                        .with_context(|| format!("missing fetch status row for {accession}"))?;
                    if !chunk_matched_runs.is_empty() {
                        status.matched_runs.extend(chunk_matched_runs);
                        status.matched_runs =
                            dedupe_preserving_order(status.matched_runs.drain(..));
                        status.status = FetchAccessionStatusKind::Matched;
                    } else if status.status != FetchAccessionStatusKind::Matched
                        && status.status != FetchAccessionStatusKind::FetchFailed
                    {
                        status.status = FetchAccessionStatusKind::Unmatched;
                    }
                    status.source = Some(result.source);
                    status.attempts = status.attempts.max(result.attempts);
                    if status.status == FetchAccessionStatusKind::Matched {
                        status.error = None;
                    }
                }
            }
            Err(error) => {
                failed_chunk_messages.push(format!(
                    "[{}]: {}",
                    chunk_accessions.join(","),
                    error.message
                ));
                for accession in original_requests_in_chunk {
                    let status = accession_statuses
                        .get_mut(&accession)
                        .with_context(|| format!("missing fetch status row for {accession}"))?;
                    if status.status != FetchAccessionStatusKind::Matched {
                        status.status = FetchAccessionStatusKind::FetchFailed;
                        status.matched_runs.clear();
                        status.source = Some(MetadataChunkSource::Remote);
                        status.attempts = status.attempts.max(error.attempts);
                        status.error = Some(error.message.clone());
                    }
                }
            }
        }
    }

    let geo_bridge_accessions = unique_accessions
        .iter()
        .filter(|accession| is_geo_accession(accession))
        .count();
    let geo_bridge_resolved_accessions = unique_accessions
        .iter()
        .filter(|accession| is_geo_accession(accession))
        .map(|accession| {
            accession_statuses
                .get(accession)
                .map(|status| status.resolved_accessions.len())
                .unwrap_or(0)
        })
        .sum();
    let geo_bridge_fallback_accessions = unique_accessions
        .iter()
        .filter(|accession| is_geo_accession(accession))
        .filter(|accession| {
            accession_statuses.get(*accession).is_some_and(|status| {
                !status.resolved_accessions.is_empty()
                    && status.geo_surface.as_deref() == Some(GeoBridgeSurface::FullTextSelf.label())
            })
        })
        .count();

    let matched_accession_set: HashSet<String> = accession_statuses
        .iter()
        .filter_map(|(accession, status)| {
            (status.status == FetchAccessionStatusKind::Matched).then_some(accession.clone())
        })
        .collect();
    let unmatched_accessions: Vec<String> = unique_accessions
        .iter()
        .filter_map(|accession| {
            accession_statuses.get(accession).and_then(|status| {
                (status.status == FetchAccessionStatusKind::Unmatched).then_some(accession.clone())
            })
        })
        .collect();
    let failed_accessions: Vec<String> = unique_accessions
        .iter()
        .filter_map(|accession| {
            accession_statuses.get(accession).and_then(|status| {
                (status.status == FetchAccessionStatusKind::FetchFailed)
                    .then_some(accession.clone())
            })
        })
        .collect();

    let mut ordered_rows: Vec<FetchedPublicMetadataRow> =
        fetched_by_accession.into_values().collect();
    for row in &mut ordered_rows {
        row.query_accessions.sort_by_key(|accession| {
            accession_positions
                .get(accession)
                .copied()
                .unwrap_or(usize::MAX)
        });
        synchronize_fetched_public_metadata_query_accessions(row);
    }
    ordered_rows.sort_by(|left, right| {
        let left_query_index = left
            .query_accessions
            .iter()
            .filter_map(|accession| accession_positions.get(accession).copied())
            .min()
            .unwrap_or(usize::MAX);
        let right_query_index = right
            .query_accessions
            .iter()
            .filter_map(|accession| accession_positions.get(accession).copied())
            .min()
            .unwrap_or(usize::MAX);
        left_query_index
            .cmp(&right_query_index)
            .then_with(|| left.run_accession.cmp(&right.run_accession))
    });
    let fetched_records = ordered_rows.len();

    write_fetch_status_table(&status_path, &unique_accessions, &accession_statuses)?;

    if ordered_rows.is_empty() {
        bail!(
            "metadata fetch for {} returned no matching records; see {} for fetch status details",
            options.input_path.display(),
            status_path.display()
        );
    }

    write_fetched_public_metadata_table(
        &options.output_path,
        manifest_delimiter_for_output(&options.output_path),
        &ordered_rows,
    )?;

    let summary = StudyFetchMetadataSummary {
        requested_accessions: requested_accessions.len(),
        unique_accessions: unique_accessions.len(),
        resumed_accessions,
        resumed_records,
        geo_bridge_accessions,
        geo_bridge_resolved_accessions,
        geo_bridge_fallback_accessions,
        fetched_records,
        matched_accessions: matched_accession_set.len(),
        unmatched_accessions: unmatched_accessions.len(),
        failed_accessions: failed_accessions.len(),
        cache_hits,
        remote_fetches,
        retried_chunks,
    };

    let mut notes = vec![
        "Fetched ENA-style run metadata is written in a phase-35-compatible schema, so the output can be passed directly to study-manifest".to_string(),
        format!(
            "Queried {} unique accession(s) from {} in chunk(s) of up to {}",
            summary.unique_accessions,
            options.input_path.display(),
            options.chunk_size
        ),
        if options.resume_existing {
            format!(
                "Resume mode reused {} accession status row(s) and {} existing run-level metadata row(s) before fetching missing work",
                summary.resumed_accessions,
                summary.resumed_records
            )
        } else {
            "Resume mode was disabled, so existing metadata/status outputs were not reused".to_string()
        },
        "Supported accession families currently include run, study, experiment, sample, and project-style SRA/ENA identifiers".to_string(),
        format!(
            "GEO bridge requests are resolved through {} before ENA metadata lookup when GSE/GSM identifiers are supplied",
            options.geo_base_url
        ),
        format!(
            "Fetch bookkeeping was written to {} so matched, unmatched, and failed accession states remain auditable",
            status_path.display()
        ),
    ];
    let expanded_accession_count = ordered_rows
        .iter()
        .flat_map(|row| {
            row.query_accessions
                .iter()
                .filter(move |accession| *accession != &row.run_accession)
                .cloned()
        })
        .collect::<HashSet<_>>()
        .len();
    if expanded_accession_count > 0 {
        notes.push(format!(
            "{} requested accession(s) expanded into {} run-level metadata row(s)",
            expanded_accession_count, summary.fetched_records
        ));
    }
    if summary.geo_bridge_fallback_accessions > 0 {
        notes.push(format!(
            "{} GEO accession(s) required richer full-text fallback after quick-text bridge output contained no usable public accession tokens",
            summary.geo_bridge_fallback_accessions
        ));
    }
    if summary.cache_hits > 0 {
        notes.push(format!(
            "{} fetch operation(s) were satisfied from local metadata cache {}",
            summary.cache_hits,
            cache_dir.display()
        ));
    }
    if summary.remote_fetches > 0 {
        notes.push(format!(
            "{} fetch operation(s) were performed remotely after cache lookup",
            summary.remote_fetches
        ));
    }
    if summary.retried_chunks > 0 {
        notes.push(format!(
            "{} fetch operation(s) required retry before succeeding",
            summary.retried_chunks
        ));
    }
    if !unmatched_accessions.is_empty() {
        notes.push(format!(
            "{} accession(s) were requested but not returned by the remote metadata source",
            unmatched_accessions.len()
        ));
    }
    if !failed_accessions.is_empty() {
        notes.push(format!(
            "{} accession(s) could not be fetched after retry; see the fetch status table for details",
            failed_accessions.len()
        ));
    }
    if !failed_chunk_messages.is_empty() {
        notes.push(format!(
            "{} fetch chunk(s) failed after retry exhaustion",
            failed_chunk_messages.len()
        ));
    }

    Ok(StudyFetchMetadataReport {
        input_path: options.input_path.display().to_string(),
        output_path: options.output_path.display().to_string(),
        source: "ena_filereport".to_string(),
        base_url: options.base_url.clone(),
        geo_base_url: options.geo_base_url.clone(),
        chunk_size: options.chunk_size,
        retries: options.retries,
        resume_existing: options.resume_existing,
        cache_dir: cache_dir.display().to_string(),
        status_path: status_path.display().to_string(),
        summary,
        unmatched_accessions,
        failed_accessions,
        notes,
    })
}

pub fn download_public_fastqs(options: &StudyDownloadOptions) -> Result<StudyDownloadReport> {
    fs::create_dir_all(&options.download_root)
        .with_context(|| format!("failed to create {}", options.download_root.display()))?;
    let entries = read_manifest_source(&options.input_path, false, Some(&options.download_root))?;
    if entries.is_empty() {
        bail!(
            "study download input {} did not contain any datasets",
            options.input_path.display()
        );
    }

    let status_path = options.download_root.join("download_status.csv");
    let mut status_rows = Vec::new();
    let mut requested_files = 0usize;
    let mut downloaded_files = 0usize;
    let mut resumed_files = 0usize;
    let mut skipped_existing_files = 0usize;
    let mut failed_files = 0usize;
    let mut available_bytes = 0u64;
    let mut failed_destinations = Vec::new();

    for entry in &entries {
        let expected_destinations = expected_public_download_destinations(entry);
        requested_files += expected_destinations.len();

        match select_public_download_targets(entry) {
            Ok(targets) => {
                for target in targets {
                    match execute_public_download_target(
                        &target,
                        options.retries,
                        options.overwrite_existing,
                    ) {
                        Ok(outcome) => {
                            match outcome.status {
                                StudyDownloadStatusKind::Downloaded => downloaded_files += 1,
                                StudyDownloadStatusKind::Resumed => resumed_files += 1,
                                StudyDownloadStatusKind::SkippedExisting => {
                                    skipped_existing_files += 1
                                }
                                StudyDownloadStatusKind::Failed => failed_files += 1,
                            }
                            available_bytes += outcome.bytes;
                            status_rows.push(StudyDownloadStatusRow {
                                dataset_id: target.dataset_id,
                                accession: target.accession,
                                destination: target.destination,
                                source_url: Some(target.source_url),
                                status: outcome.status,
                                bytes: outcome.bytes,
                                attempts: outcome.attempts,
                                error: None,
                            });
                        }
                        Err(error) => {
                            failed_files += 1;
                            failed_destinations.push(target.destination.display().to_string());
                            status_rows.push(StudyDownloadStatusRow {
                                dataset_id: target.dataset_id,
                                accession: target.accession,
                                destination: target.destination,
                                source_url: Some(target.source_url),
                                status: StudyDownloadStatusKind::Failed,
                                bytes: 0,
                                attempts: error.attempts,
                                error: Some(error.message),
                            });
                        }
                    }
                }
            }
            Err(error) => {
                for destination in expected_destinations {
                    failed_files += 1;
                    failed_destinations.push(destination.display().to_string());
                    status_rows.push(StudyDownloadStatusRow {
                        dataset_id: manifest_source_entry_label(entry),
                        accession: entry.accession.clone(),
                        destination,
                        source_url: None,
                        status: StudyDownloadStatusKind::Failed,
                        bytes: 0,
                        attempts: 0,
                        error: Some(error.to_string()),
                    });
                }
            }
        }
    }

    write_study_download_status_table(&status_path, &status_rows)?;
    if downloaded_files + resumed_files + skipped_existing_files == 0 {
        bail!(
            "study download for {} did not materialize any FASTQ files; see {}",
            options.input_path.display(),
            status_path.display()
        );
    }

    let summary = StudyDownloadSummary {
        datasets: entries.len(),
        requested_files,
        downloaded_files,
        resumed_files,
        skipped_existing_files,
        failed_files,
        available_bytes,
    };
    let mut notes = vec![
        "Public FASTQ downloads prefer directly fetchable HTTP/HTTPS/FTP locations and intentionally ignore Aspera-only paths when no curl-compatible URL is available".to_string(),
        format!(
            "Resolved {} dataset(s) from {} into local FASTQ targets rooted at {}",
            summary.datasets,
            options.input_path.display(),
            options.download_root.display()
        ),
    ];
    if summary.downloaded_files > 0 {
        notes.push(format!(
            "{} FASTQ file(s) were downloaded from remote public locations",
            summary.downloaded_files
        ));
    }
    if summary.resumed_files > 0 {
        notes.push(format!(
            "{} FASTQ file(s) resumed from partial .part downloads",
            summary.resumed_files
        ));
    }
    if summary.skipped_existing_files > 0 {
        notes.push(format!(
            "{} FASTQ file(s) were already present locally and were skipped",
            summary.skipped_existing_files
        ));
    }
    if summary.failed_files > 0 {
        notes.push(format!(
            "{} FASTQ file(s) failed to materialize; inspect {} for per-file errors",
            summary.failed_files,
            status_path.display()
        ));
    }

    Ok(StudyDownloadReport {
        input_path: options.input_path.display().to_string(),
        download_root: options.download_root.display().to_string(),
        status_path: status_path.display().to_string(),
        retries: options.retries,
        overwrite_existing: options.overwrite_existing,
        summary,
        failed_destinations,
        notes,
    })
}

pub fn bootstrap_study_manifest(
    options: &StudyManifestOptions,
) -> Result<StudyManifestBootstrapReport> {
    let accession_bootstrap =
        source_manifest_requires_accession_bootstrap(&options.inventory_path).unwrap_or(false);
    let source_entries =
        read_manifest_source(&options.inventory_path, false, Some(&options.download_root))?;
    if source_entries.is_empty() {
        bail!(
            "study inventory {} did not contain any datasets",
            options.inventory_path.display()
        );
    }

    let entries =
        finalize_manifest_entries(source_entries, options.default_baseline_name.as_deref());
    let output_delimiter = manifest_delimiter_for_output(&options.output_path);
    if let Some(parent) = options.output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    write_canonical_manifest(&options.output_path, output_delimiter, &entries)?;

    let provenance_summary_path =
        sibling_artifact_path(&options.output_path, "provenance_summary.csv");
    write_manifest_bootstrap_summary_csv(&provenance_summary_path, &entries)?;
    let summary = build_manifest_bootstrap_summary(&entries);

    let mut notes = vec![
        "Bootstrapped manifest is written in canonical phase-25 column order so it can be consumed directly by study-artifacts".to_string(),
        "Missing dataset_id values are deterministically derived from accession or input FASTQ filenames".to_string(),
    ];
    if accession_bootstrap {
        notes.push(format!(
            "Rows without local FASTQ paths were synthesized from public accession metadata under download root {}",
            options.download_root.display()
        ));
    }
    if let Some(default_baseline_name) = &options.default_baseline_name {
        notes.push(format!(
            "Rows without a baseline_name were filled with the default baseline '{default_baseline_name}'"
        ));
    }
    notes.push(format!(
        "Inventory {} was normalized into {} datasets",
        options.inventory_path.display(),
        summary.datasets
    ));

    Ok(StudyManifestBootstrapReport {
        input_path: options.inventory_path.display().to_string(),
        output_path: options.output_path.display().to_string(),
        provenance_summary_path: provenance_summary_path.display().to_string(),
        default_baseline_name: options.default_baseline_name.clone(),
        summary,
        notes,
    })
}

pub fn annotate_study_manifest(options: &StudyAnnotateOptions) -> Result<StudyAnnotateReport> {
    let mut entries = read_manifest(&options.manifest_path)?;
    if entries.is_empty() {
        bail!(
            "study manifest {} did not contain any datasets",
            options.manifest_path.display()
        );
    }

    let annotations = read_manifest_annotations(&options.annotations_path)?;
    if annotations.is_empty() {
        bail!(
            "annotation table {} did not contain any datasets",
            options.annotations_path.display()
        );
    }

    if let Some(parent) = options.output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let before = build_manifest_bootstrap_summary(&entries);
    let mut index_by_dataset_id = HashMap::new();
    let mut index_by_accession = HashMap::new();
    for (index, entry) in entries.iter().enumerate() {
        index_by_dataset_id.insert(normalize_lookup_key(&entry.dataset_id), index);
        if let Some(accession) = entry.accession.as_deref() {
            index_by_accession.insert(normalize_lookup_key(accession), index);
        }
    }

    let annotation_rows = annotations.len();
    let mut matched_rows = 0usize;
    let mut unmatched_rows = 0usize;
    let mut unmatched_examples = Vec::new();
    let mut touched_datasets = HashSet::new();
    let mut matched_targets = HashSet::new();
    let mut merge_stats = AnnotationMergeStats::default();

    for (annotation_index, annotation) in annotations.into_iter().enumerate() {
        let line_number = annotation_index + 2;
        let Some(target_index) = resolve_annotation_target(
            &annotation,
            &index_by_dataset_id,
            &index_by_accession,
            &options.annotations_path,
            line_number,
        )?
        else {
            unmatched_rows += 1;
            if unmatched_examples.len() < 5 {
                unmatched_examples.push(annotation_key_label(&annotation));
            }
            continue;
        };

        if !matched_targets.insert(target_index) {
            bail!(
                "annotation {} line {line_number} targets dataset '{}' more than once",
                options.annotations_path.display(),
                entries[target_index].dataset_id
            );
        }

        matched_rows += 1;
        if apply_annotation_entry(
            &mut entries[target_index],
            annotation,
            options.overwrite_existing,
            &mut merge_stats,
        ) {
            touched_datasets.insert(target_index);
        }
    }

    let output_delimiter = manifest_delimiter_for_output(&options.output_path);
    write_canonical_manifest(&options.output_path, output_delimiter, &entries)?;

    let after = build_manifest_bootstrap_summary(&entries);
    let summary = StudyAnnotateSummary {
        datasets: entries.len(),
        annotation_rows,
        matched_rows,
        unmatched_rows,
        datasets_changed: touched_datasets.len(),
        fields_filled: merge_stats.fields_filled,
        fields_overwritten: merge_stats.fields_overwritten,
        notes_appended: merge_stats.notes_appended,
        before,
        after,
    };
    let summary_path = sibling_artifact_path(&options.output_path, "annotation_summary.csv");
    write_study_annotate_summary_csv(&summary_path, &summary)?;

    let mut notes = vec![
        "Annotated manifest is written in canonical phase-25 column order so it can be consumed directly by study-artifacts".to_string(),
        "Annotation rows can match datasets by dataset_id and/or accession without requiring FASTQ paths to be repeated".to_string(),
    ];
    if options.overwrite_existing {
        notes.push(
            "Existing populated fields were eligible for overwrite when the annotation table supplied replacement values".to_string(),
        );
    } else {
        notes.push(
            "Existing populated fields were preserved by default; only empty fields were filled and note text was appended".to_string(),
        );
    }
    notes.push(format!(
        "Merged {matched_rows} of {annotation_rows} annotation rows into {} manifest datasets",
        summary.datasets
    ));
    if unmatched_rows > 0 {
        notes.push(format!(
            "{unmatched_rows} annotation rows did not match any manifest dataset"
        ));
        if !unmatched_examples.is_empty() {
            notes.push(format!(
                "Unmatched examples: {}",
                unmatched_examples.join(", ")
            ));
        }
    }

    Ok(StudyAnnotateReport {
        manifest_path: options.manifest_path.display().to_string(),
        annotations_path: options.annotations_path.display().to_string(),
        output_path: options.output_path.display().to_string(),
        summary_path: summary_path.display().to_string(),
        overwrite_existing: options.overwrite_existing,
        summary,
        notes,
    })
}

pub fn ingest_study_results(options: &StudyIngestOptions) -> Result<StudyIngestReport> {
    let entries = read_manifest(&options.manifest_path)?;
    if entries.is_empty() {
        bail!(
            "study manifest {} did not contain any datasets",
            options.manifest_path.display()
        );
    }

    if let Some(parent) = options.output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let before = build_manifest_bootstrap_summary(&entries);
    let mut index_by_dataset_id = HashMap::new();
    let mut index_by_accession = HashMap::new();
    for (index, entry) in entries.iter().enumerate() {
        index_by_dataset_id.insert(normalize_lookup_key(&entry.dataset_id), index);
        if let Some(accession) = entry.accession.as_deref() {
            index_by_accession.insert(normalize_lookup_key(accession), index);
        }
    }

    let mut files = Vec::new();
    let mut stats = StructuredIngestStats::default();
    collect_structured_files(
        &options.input_dir,
        options.recursive,
        &mut files,
        &mut stats.files_scanned,
        &mut stats.ignored_files,
    )?;
    files.sort();

    let mut aggregated = HashMap::<usize, ManifestAnnotationEntry>::new();
    let mut unmatched_examples = Vec::new();
    for path in &files {
        let ingested_rows = read_ingested_annotation_file(path)?;
        if matches_structured_json_path(path) {
            stats.json_files += 1;
        } else {
            stats.delimited_files += 1;
        }
        stats.structured_files += 1;
        stats.records_ingested += ingested_rows.len();

        for (row_offset, row) in ingested_rows.into_iter().enumerate() {
            let line_number = row_offset + 1;
            let Some(target_index) = resolve_annotation_target(
                &row,
                &index_by_dataset_id,
                &index_by_accession,
                path,
                line_number,
            )?
            else {
                stats.unmatched_records += 1;
                if unmatched_examples.len() < 5 {
                    unmatched_examples.push(format!(
                        "{}:{}:{}",
                        path.display(),
                        line_number,
                        annotation_key_label(&row)
                    ));
                }
                continue;
            };
            stats.matched_records += 1;
            merge_ingested_annotation(
                aggregated.entry(target_index).or_default(),
                row,
                path,
                line_number,
            )?;
        }
    }

    if stats.structured_files == 0 {
        bail!(
            "structured results directory {} did not contain any supported .json/.csv/.tsv files or recognized native text reports",
            options.input_dir.display()
        );
    }

    let generated_annotations_path =
        sibling_artifact_path(&options.output_path, "ingested_annotations.tsv");
    let generated_annotations = build_ingested_annotation_rows(&entries, &aggregated);
    write_canonical_annotation_table(&generated_annotations_path, &generated_annotations)?;

    let annotate_report = annotate_study_manifest(&StudyAnnotateOptions {
        manifest_path: options.manifest_path.clone(),
        annotations_path: generated_annotations_path.clone(),
        output_path: options.output_path.clone(),
        overwrite_existing: options.overwrite_existing,
    })?;

    let summary = StudyIngestSummary {
        files_scanned: stats.files_scanned,
        structured_files: stats.structured_files,
        json_files: stats.json_files,
        delimited_files: stats.delimited_files,
        ignored_files: stats.ignored_files,
        records_ingested: stats.records_ingested,
        generated_annotation_rows: generated_annotations.len(),
        matched_records: stats.matched_records,
        unmatched_records: stats.unmatched_records,
        datasets_changed: annotate_report.summary.datasets_changed,
        fields_filled: annotate_report.summary.fields_filled,
        fields_overwritten: annotate_report.summary.fields_overwritten,
        notes_appended: annotate_report.summary.notes_appended,
        before,
        after: annotate_report.summary.after.clone(),
    };

    let summary_path = sibling_artifact_path(&options.output_path, "ingest_summary.csv");
    write_study_ingest_summary_csv(&summary_path, &summary)?;

    let mut notes = vec![
        "Structured ingest scanned JSON/CSV/TSV files plus recognized native text reports and collapsed them into one canonical annotation table before updating the manifest".to_string(),
        "Rows can match datasets by dataset_id and/or accession, which lets competitor and downstream summaries live in separate files".to_string(),
    ];
    if options.overwrite_existing {
        notes.push(
            "Existing populated manifest fields were eligible for overwrite when ingested records supplied replacement values".to_string(),
        );
    } else {
        notes.push(
            "Existing populated manifest fields were preserved by default; ingest only filled empty values and appended note text".to_string(),
        );
    }
    notes.push(format!(
        "Scanned {} files and ingested {} structured rows into {} generated annotation rows",
        summary.files_scanned, summary.records_ingested, summary.generated_annotation_rows
    ));
    if summary.unmatched_records > 0 {
        notes.push(format!(
            "{} structured rows did not match any manifest dataset",
            summary.unmatched_records
        ));
        if !unmatched_examples.is_empty() {
            notes.push(format!(
                "Unmatched examples: {}",
                unmatched_examples.join(", ")
            ));
        }
    }

    Ok(StudyIngestReport {
        manifest_path: options.manifest_path.display().to_string(),
        input_dir: options.input_dir.display().to_string(),
        output_path: options.output_path.display().to_string(),
        generated_annotations_path: generated_annotations_path.display().to_string(),
        summary_path: summary_path.display().to_string(),
        recursive: options.recursive,
        overwrite_existing: options.overwrite_existing,
        summary,
        notes,
    })
}

pub fn generate_study_artifacts(options: &StudyArtifactsOptions) -> Result<StudyArtifactsReport> {
    let output_dir = options.output_dir.clone();
    let data_dir = output_dir.join("data");
    let datasets_dir = data_dir.join("datasets");
    let figures_dir = output_dir.join("figures");
    fs::create_dir_all(&datasets_dir)
        .with_context(|| format!("failed to create {}", datasets_dir.display()))?;
    fs::create_dir_all(&figures_dir)
        .with_context(|| format!("failed to create {}", figures_dir.display()))?;

    let entries = read_manifest(&options.manifest_path)?;
    if entries.is_empty() {
        bail!(
            "study manifest {} did not contain any datasets",
            options.manifest_path.display()
        );
    }

    let benchmark_options = BenchmarkOptions {
        process: ProcessOptions {
            sample_size: options.sample_size,
            batch_reads: options.batch_reads,
            backend_preference: options.backend_preference,
            forced_adapter: None,
            min_quality_override: options.min_quality_override,
            kmer_size_override: None,
                    forced_platform: None,
                    forced_experiment: None,
        },
        rounds: options.benchmark_rounds.max(1),
        session_mode: BenchmarkSessionMode::ColdStart,
    };

    let mut dataset_reports = Vec::with_capacity(entries.len());
    let mut dataset_artifacts = Vec::new();
    for entry in entries {
        let inspect = inspect_inputs(
            &entry.input1,
            entry.input2.as_deref(),
            options.sample_size,
            options.backend_preference,
        )?;
        let benchmark =
            benchmark_files(&entry.input1, entry.input2.as_deref(), &benchmark_options)?;
        let process = benchmark
            .rounds
            .first()
            .map(|round| round.process.clone())
            .context("benchmark report did not contain any rounds")?;

        let trimmed_read_fraction = ratio(process.trimmed_reads, process.input_reads);
        let discarded_read_fraction = ratio(process.discarded_reads, process.input_reads);
        let corrected_bases_per_mbase =
            density_per_mbase(process.corrected_bases, process.throughput.input_bases);
        let detected_platform = inspect.auto_profile.platform;
        let detected_experiment = inspect.auto_profile.experiment;
        let platform_match = entry
            .expected_platform
            .map(|expected| expected == detected_platform);
        let experiment_match = entry
            .expected_experiment
            .map(|expected| expected == detected_experiment);
        let comparison = build_dataset_comparison(
            process.throughput.input_bases_per_sec,
            trimmed_read_fraction,
            discarded_read_fraction,
            corrected_bases_per_mbase,
            &entry.downstream,
            entry.baseline.as_ref(),
        );

        let mut notes = vec![
            "Cleanup counts come from the first cold-start benchmark round so throughput and processing telemetry stay aligned".to_string(),
            "Any baseline or downstream biological metrics are manifest-supplied annotations rather than measurements recomputed by JapalityECHO".to_string(),
        ];
        if let Some(note) = &entry.notes {
            notes.push(note.clone());
        }
        if entry.input2.is_some() {
            notes.push("Manifest row declared paired-end inputs".to_string());
        }
        if entry.baseline.is_some() {
            notes.push(
                "Manifest row includes competitor/baseline metrics for external validation"
                    .to_string(),
            );
        }
        if downstream_has_any_metrics(&entry.downstream) {
            notes.push(
                "Manifest row includes downstream biological metrics for JapalityECHO outputs"
                    .to_string(),
            );
        }
        if platform_match == Some(false) {
            notes.push("Detected platform did not match the manifest expectation".to_string());
        }
        if experiment_match == Some(false) {
            notes.push("Detected experiment did not match the manifest expectation".to_string());
        }
        if comparison.duplicate_rate_delta.is_some() {
            notes.push(
                "Duplicate-rate deltas are JapalityECHO minus baseline, so negative values indicate fewer duplicates after preprocessing".to_string(),
            );
        }

        let dataset_report = StudyDatasetReport {
            dataset_id: entry.dataset_id.clone(),
            accession: entry.accession.clone(),
            citation: entry.citation.clone(),
            paired_end: entry.input2.is_some(),
            input1: entry.input1.display().to_string(),
            input2: entry.input2.as_ref().map(|path| path.display().to_string()),
            expected_platform: entry.expected_platform,
            detected_platform,
            platform_match,
            expected_experiment: entry.expected_experiment,
            detected_experiment,
            experiment_match,
            trimmed_read_fraction,
            discarded_read_fraction,
            corrected_bases_per_mbase,
            downstream: entry.downstream.clone(),
            baseline: entry.baseline.clone(),
            comparison,
            inspect,
            process,
            benchmark: benchmark.summary,
            notes,
        };

        let dataset_json =
            datasets_dir.join(format!("{}.json", sanitize_filename(&entry.dataset_id)));
        write_json_pretty(&dataset_json, &dataset_report)?;
        dataset_artifacts.push(artifact(
            "json",
            &dataset_json,
            &format!("Per-dataset study report for {}", entry.dataset_id),
        ));
        dataset_reports.push(dataset_report);
    }

    let aggregate = build_aggregate_summary(&dataset_reports);
    let detection = build_detection_summary(&dataset_reports);
    let comparison = build_comparison_summary(&dataset_reports);

    let summary_json = output_dir.join("study_artifacts.json");
    let dataset_summary_csv = data_dir.join("dataset_summary.csv");
    let detection_summary_csv = data_dir.join("detection_summary.csv");
    let baseline_comparison_csv = data_dir.join("baseline_comparison.csv");
    let downstream_metrics_csv = data_dir.join("downstream_metrics.csv");

    let throughput_svg = figures_dir.join("study_throughput.svg");
    let cleanup_svg = figures_dir.join("study_cleanup.svg");
    let correction_svg = figures_dir.join("study_correction.svg");
    let detection_svg = figures_dir.join("study_detection_accuracy.svg");
    let baseline_throughput_svg = figures_dir.join("study_baseline_throughput.svg");
    let alignment_svg = figures_dir.join("study_alignment_rate.svg");
    let duplicate_svg = figures_dir.join("study_duplicate_rate.svg");
    let variant_svg = figures_dir.join("study_variant_f1.svg");
    let mean_coverage_svg = figures_dir.join("study_mean_coverage.svg");
    let coverage_breadth_svg = figures_dir.join("study_coverage_breadth.svg");
    let assembly_svg = figures_dir.join("study_assembly_n50.svg");

    write_dataset_summary_csv(&dataset_summary_csv, &dataset_reports)?;
    write_detection_summary_csv(&detection_summary_csv, &detection)?;
    write_baseline_comparison_csv(&baseline_comparison_csv, &dataset_reports)?;
    write_downstream_metrics_csv(&downstream_metrics_csv, &dataset_reports)?;

    write_svg(
        &throughput_svg,
        &single_series_bar_chart_svg(
            "Case-study throughput by dataset",
            "Input throughput (Mbases/s)",
            &dataset_reports
                .iter()
                .map(|dataset| dataset.dataset_id.clone())
                .collect::<Vec<_>>(),
            &dataset_reports
                .iter()
                .map(|dataset| dataset.benchmark.average_input_bases_per_sec / 1_000_000.0)
                .collect::<Vec<_>>(),
            "#59A14F",
        ),
    )?;
    write_svg(
        &cleanup_svg,
        &grouped_bar_chart_svg(
            "Case-study cleanup rates by dataset",
            "Rate (%)",
            &dataset_reports
                .iter()
                .map(|dataset| dataset.dataset_id.clone())
                .collect::<Vec<_>>(),
            &[
                ChartSeries {
                    name: "trimmed",
                    color: "#F28E2B",
                    values: dataset_reports
                        .iter()
                        .map(|dataset| dataset.trimmed_read_fraction * 100.0)
                        .collect(),
                },
                ChartSeries {
                    name: "discarded",
                    color: "#E15759",
                    values: dataset_reports
                        .iter()
                        .map(|dataset| dataset.discarded_read_fraction * 100.0)
                        .collect(),
                },
            ],
        ),
    )?;
    write_svg(
        &correction_svg,
        &single_series_bar_chart_svg(
            "Case-study correction density by dataset",
            "Corrected bases / input Mbases",
            &dataset_reports
                .iter()
                .map(|dataset| dataset.dataset_id.clone())
                .collect::<Vec<_>>(),
            &dataset_reports
                .iter()
                .map(|dataset| dataset.corrected_bases_per_mbase)
                .collect::<Vec<_>>(),
            "#4E79A7",
        ),
    )?;

    let mut artifacts = vec![
        artifact(
            "json",
            &summary_json,
            "Top-level multi-dataset study artifact bundle summary",
        ),
        artifact(
            "csv",
            &dataset_summary_csv,
            "Per-dataset case-study summary table",
        ),
        artifact(
            "csv",
            &detection_summary_csv,
            "Aggregate zero-config detection accuracy summary",
        ),
        artifact(
            "csv",
            &baseline_comparison_csv,
            "Competitor/baseline comparison summary table",
        ),
        artifact(
            "csv",
            &downstream_metrics_csv,
            "Downstream biological metrics summary table",
        ),
        artifact("svg", &throughput_svg, "Case-study throughput figure"),
        artifact("svg", &cleanup_svg, "Case-study cleanup-rate figure"),
        artifact(
            "svg",
            &correction_svg,
            "Case-study correction-density figure",
        ),
    ];

    if let Some((categories, values)) = detection_accuracy_categories(&detection) {
        write_svg(
            &detection_svg,
            &single_series_bar_chart_svg(
                "Zero-config metadata accuracy across case-study datasets",
                "Accuracy (%)",
                &categories,
                &values,
                "#76B7B2",
            ),
        )?;
        artifacts.push(artifact(
            "svg",
            &detection_svg,
            "Zero-config detection-accuracy figure",
        ));
    }

    if let Some(artifact_file) = write_optional_comparison_figure(
        &baseline_throughput_svg,
        "JapalityECHO versus baseline throughput by dataset",
        "Input throughput (Mbases/s)",
        "Baseline",
        &dataset_reports,
        |dataset| Some(dataset.benchmark.average_input_bases_per_sec / 1_000_000.0),
        |dataset| {
            dataset
                .baseline
                .as_ref()
                .and_then(|baseline| baseline.input_bases_per_sec)
                .map(|value| value / 1_000_000.0)
        },
        "Baseline throughput comparison figure",
    )? {
        artifacts.push(artifact_file);
    }

    if let Some(artifact_file) = write_optional_comparison_figure(
        &alignment_svg,
        "JapalityECHO versus baseline alignment rate by dataset",
        "Alignment rate (%)",
        "Baseline",
        &dataset_reports,
        |dataset| dataset.downstream.alignment_rate.map(|value| value * 100.0),
        |dataset| {
            dataset
                .baseline
                .as_ref()
                .and_then(|baseline| baseline.downstream.alignment_rate)
                .map(|value| value * 100.0)
        },
        "Alignment-rate comparison figure",
    )? {
        artifacts.push(artifact_file);
    }

    if let Some(artifact_file) = write_optional_comparison_figure(
        &duplicate_svg,
        "JapalityECHO versus baseline duplicate rate by dataset",
        "Duplicate rate (%)",
        "Baseline",
        &dataset_reports,
        |dataset| dataset.downstream.duplicate_rate.map(|value| value * 100.0),
        |dataset| {
            dataset
                .baseline
                .as_ref()
                .and_then(|baseline| baseline.downstream.duplicate_rate)
                .map(|value| value * 100.0)
        },
        "Duplicate-rate comparison figure",
    )? {
        artifacts.push(artifact_file);
    }

    if let Some(artifact_file) = write_optional_comparison_figure(
        &variant_svg,
        "JapalityECHO versus baseline variant F1 by dataset",
        "Variant F1 (%)",
        "Baseline",
        &dataset_reports,
        |dataset| dataset.downstream.variant_f1.map(|value| value * 100.0),
        |dataset| {
            dataset
                .baseline
                .as_ref()
                .and_then(|baseline| baseline.downstream.variant_f1)
                .map(|value| value * 100.0)
        },
        "Variant-F1 comparison figure",
    )? {
        artifacts.push(artifact_file);
    }

    if let Some(artifact_file) = write_optional_comparison_figure(
        &mean_coverage_svg,
        "JapalityECHO versus baseline mean coverage by dataset",
        "Mean coverage",
        "Baseline",
        &dataset_reports,
        |dataset| dataset.downstream.mean_coverage,
        |dataset| {
            dataset
                .baseline
                .as_ref()
                .and_then(|baseline| baseline.downstream.mean_coverage)
        },
        "Mean-coverage comparison figure",
    )? {
        artifacts.push(artifact_file);
    }

    if let Some(artifact_file) = write_optional_comparison_figure(
        &coverage_breadth_svg,
        "JapalityECHO versus baseline coverage breadth by dataset",
        "Coverage breadth (%)",
        "Baseline",
        &dataset_reports,
        |dataset| {
            dataset
                .downstream
                .coverage_breadth
                .map(|value| value * 100.0)
        },
        |dataset| {
            dataset
                .baseline
                .as_ref()
                .and_then(|baseline| baseline.downstream.coverage_breadth)
                .map(|value| value * 100.0)
        },
        "Coverage-breadth comparison figure",
    )? {
        artifacts.push(artifact_file);
    }

    if let Some(artifact_file) = write_optional_comparison_figure(
        &assembly_svg,
        "JapalityECHO versus baseline assembly continuity by dataset",
        "Assembly N50",
        "Baseline",
        &dataset_reports,
        |dataset| dataset.downstream.assembly_n50,
        |dataset| {
            dataset
                .baseline
                .as_ref()
                .and_then(|baseline| baseline.downstream.assembly_n50)
        },
        "Assembly-N50 comparison figure",
    )? {
        artifacts.push(artifact_file);
    }

    artifacts.extend(dataset_artifacts);

    let mut notes = vec![
        "Manifest-driven study bundles are designed for local public datasets once FASTQs have been downloaded and curated".to_string(),
        "Each dataset row can optionally supply expected platform and experiment labels to quantify zero-config detection accuracy".to_string(),
        "Per-dataset cleanup telemetry comes from benchmark round 0 so publication figures and throughput numbers stay tied to the same run".to_string(),
        "Competitor and downstream biological metrics are manifest-supplied annotations, which keeps this phase compatible with external alignment/variant/assembly workflows".to_string(),
    ];
    notes.push(format!(
        "Processed {} datasets ({} paired-end) from manifest {}",
        aggregate.datasets,
        aggregate.paired_datasets,
        options.manifest_path.display()
    ));
    if let Some(platform_accuracy) = detection.platform_accuracy {
        notes.push(format!(
            "Manifest-supplied platform labels matched auto-detection at {:.1}%",
            platform_accuracy * 100.0
        ));
    }
    if let Some(experiment_accuracy) = detection.experiment_accuracy {
        notes.push(format!(
            "Manifest-supplied experiment labels matched auto-detection at {:.1}%",
            experiment_accuracy * 100.0
        ));
    }
    if let Some(speedup) = comparison.average_input_speedup_vs_baseline {
        notes.push(format!(
            "Average JapalityECHO throughput speedup over the manifest baseline was {speedup:.3}x"
        ));
    }
    if let Some(alignment_delta) = comparison.average_alignment_rate_delta {
        notes.push(format!(
            "Average alignment-rate delta versus baseline was {alignment_delta:+.3} (fraction scale)"
        ));
    }
    if let Some(duplicate_delta) = comparison.average_duplicate_rate_delta {
        notes.push(format!(
            "Average duplicate-rate delta versus baseline was {duplicate_delta:+.3} (negative is favorable)"
        ));
    }
    if let Some(variant_delta) = comparison.average_variant_f1_delta {
        notes.push(format!(
            "Average variant-F1 delta versus baseline was {variant_delta:+.3}"
        ));
    }
    if let Some(mean_coverage_ratio) = comparison.average_mean_coverage_ratio {
        notes.push(format!(
            "Average mean-coverage ratio versus baseline was {mean_coverage_ratio:.3}x"
        ));
    }
    if let Some(coverage_breadth_delta) = comparison.average_coverage_breadth_delta {
        notes.push(format!(
            "Average coverage-breadth delta versus baseline was {coverage_breadth_delta:+.3} (fraction scale)"
        ));
    }
    if let Some(assembly_ratio) = comparison.average_assembly_n50_ratio {
        notes.push(format!(
            "Average assembly N50 ratio versus baseline was {assembly_ratio:.3}x"
        ));
    }

    let report = StudyArtifactsReport {
        output_dir: output_dir.display().to_string(),
        manifest_path: options.manifest_path.display().to_string(),
        requested_backend: options.backend_preference,
        sample_size: options.sample_size,
        batch_reads: options.batch_reads,
        benchmark_rounds: options.benchmark_rounds.max(1),
        aggregate,
        detection,
        comparison,
        datasets: dataset_reports,
        artifacts,
        notes,
    };
    write_json_pretty(&summary_json, &report)?;
    Ok(report)
}

fn read_manifest(path: &Path) -> Result<Vec<ManifestEntry>> {
    read_manifest_source(path, true, None)?
        .into_iter()
        .map(|entry| {
            let dataset_id = entry.dataset_id.context("dataset_id should exist")?;
            Ok(ManifestEntry {
                dataset_id,
                dataset_id_generated: false,
                accession: entry.accession,
                citation: entry.citation,
                input1: entry.input1,
                input2: entry.input2,
                expected_platform: entry.expected_platform,
                expected_experiment: entry.expected_experiment,
                downstream: entry.downstream,
                baseline: entry.baseline,
                notes: entry.notes,
            })
        })
        .collect()
}

fn collect_fastq_files(
    path: &Path,
    recursive: bool,
    output: &mut Vec<PathBuf>,
    files_scanned: &mut usize,
) -> Result<()> {
    for entry in fs::read_dir(path)
        .with_context(|| format!("failed to read directory {}", path.display()))?
    {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let entry_path = entry.path();
        if file_type.is_dir() {
            if recursive {
                collect_fastq_files(&entry_path, recursive, output, files_scanned)?;
            }
        } else if file_type.is_file() {
            *files_scanned += 1;
            if is_fastq_path(&entry_path) {
                output.push(entry_path);
            }
        }
    }
    Ok(())
}

fn collect_structured_files(
    path: &Path,
    recursive: bool,
    output: &mut Vec<PathBuf>,
    files_scanned: &mut usize,
    ignored_files: &mut usize,
) -> Result<()> {
    for entry in fs::read_dir(path)
        .with_context(|| format!("failed to read directory {}", path.display()))?
    {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let entry_path = entry.path();
        if file_type.is_dir() {
            if recursive {
                collect_structured_files(
                    &entry_path,
                    recursive,
                    output,
                    files_scanned,
                    ignored_files,
                )?;
            }
        } else if file_type.is_file() {
            *files_scanned += 1;
            if is_structured_results_path(&entry_path) {
                output.push(entry_path);
            } else {
                *ignored_files += 1;
            }
        }
    }
    Ok(())
}

fn is_fastq_path(path: &Path) -> bool {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    file_name.ends_with(".fastq")
        || file_name.ends_with(".fq")
        || file_name.ends_with(".fastq.gz")
        || file_name.ends_with(".fq.gz")
}

fn is_structured_results_path(path: &Path) -> bool {
    matches_structured_json_path(path)
        || matches_structured_delimited_path(path)
        || matches_native_alignment_text_path(path)
        || matches_native_coverage_text_path(path)
        || matches_native_variant_text_path(path)
}

fn matches_structured_json_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
}

fn matches_structured_delimited_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("csv") || extension.eq_ignore_ascii_case("tsv")
        })
}

fn matches_native_alignment_text_path(path: &Path) -> bool {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let normalized = normalize_header(file_name);
    normalized.contains("flagstat")
        || normalized.contains("markdup")
        || normalized.contains("markduplicates")
        || normalized.contains("duplicationmetrics")
}

fn matches_native_variant_text_path(path: &Path) -> bool {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .unwrap_or_default();
    if extension != "txt" && extension != "log" {
        return false;
    }
    collect_path_hint_labels(path).into_iter().any(|label| {
        let normalized = normalize_header(&label);
        normalized.contains("vcfeval")
            || normalized.contains("happy")
            || normalized.contains("happysummary")
    })
}

fn matches_native_coverage_text_path(path: &Path) -> bool {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .unwrap_or_default();
    if extension != "txt" && extension != "log" {
        return false;
    }
    collect_path_hint_labels(path).into_iter().any(|label| {
        let normalized = normalize_header(&label);
        normalized.contains("mosdepth")
            || normalized.contains("samtoolscoverage")
            || normalized.contains("coverage")
                && (normalized.contains("samtools") || normalized.contains("depth"))
    })
}

fn read_ingested_annotation_file(path: &Path) -> Result<Vec<ManifestAnnotationEntry>> {
    if matches_structured_json_path(path) {
        read_json_annotation_rows(path)
    } else if matches_structured_delimited_path(path) {
        read_delimited_annotation_rows(path)
    } else if matches_native_alignment_text_path(path) {
        read_native_alignment_text_rows(path)
    } else if matches_native_coverage_text_path(path) {
        read_native_coverage_text_rows(path)
    } else if matches_native_variant_text_path(path) {
        read_native_variant_text_rows(path)
    } else {
        bail!("unsupported structured results file {}", path.display());
    }
}

fn read_json_annotation_rows(path: &Path) -> Result<Vec<ManifestAnnotationEntry>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read structured json {}", path.display()))?;
    let value: Value = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse json {}", path.display()))?;
    if looks_like_fastp_json(&value) {
        return read_fastp_json_annotation_rows(path, &value);
    }
    let records = match value {
        Value::Array(items) => items,
        Value::Object(object) => {
            if let Some(Value::Array(items)) = object
                .get("records")
                .or_else(|| object.get("datasets"))
                .or_else(|| object.get("items"))
            {
                items.clone()
            } else {
                vec![Value::Object(object)]
            }
        }
        _ => bail!(
            "structured json {} must be an object, an array of objects, or an object with a records/datasets/items array",
            path.display()
        ),
    };

    records
        .iter()
        .enumerate()
        .map(|(index, value)| parse_json_annotation_row(value, path, index + 1))
        .collect()
}

fn read_delimited_annotation_rows(path: &Path) -> Result<Vec<ManifestAnnotationEntry>> {
    let raw = fs::read_to_string(path).with_context(|| {
        format!(
            "failed to read structured delimited file {}",
            path.display()
        )
    })?;
    if looks_like_quast_report(&raw) {
        read_quast_report_annotation_rows(path, &raw)
    } else if looks_like_samtools_coverage_table(&raw) {
        read_samtools_coverage_annotation_rows(path, &raw)
    } else if looks_like_mosdepth_summary_table(&raw) {
        read_mosdepth_summary_annotation_rows(path, &raw)
    } else if looks_like_picard_markduplicates_report(&raw) {
        read_picard_markduplicates_annotation_rows(path, &raw)
    } else if looks_like_happy_summary_table(&raw) {
        read_happy_summary_annotation_rows(path, &raw)
    } else if looks_like_vcfeval_summary_table(&raw) {
        read_vcfeval_summary_annotation_rows(path, &raw)
    } else {
        read_manifest_annotations(path)
    }
}

fn read_native_alignment_text_rows(path: &Path) -> Result<Vec<ManifestAnnotationEntry>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read native text report {}", path.display()))?;
    if looks_like_samtools_flagstat(&raw) {
        read_samtools_flagstat_annotation_rows(path, &raw)
    } else if looks_like_picard_markduplicates_report(&raw) {
        read_picard_markduplicates_annotation_rows(path, &raw)
    } else {
        bail!(
            "native text report {} did not match a supported samtools flagstat or Picard MarkDuplicates format",
            path.display()
        );
    }
}

fn read_native_coverage_text_rows(path: &Path) -> Result<Vec<ManifestAnnotationEntry>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read native text report {}", path.display()))?;
    if looks_like_samtools_coverage_table(&raw) {
        read_samtools_coverage_annotation_rows(path, &raw)
    } else if looks_like_mosdepth_summary_table(&raw) {
        read_mosdepth_summary_annotation_rows(path, &raw)
    } else {
        bail!(
            "native text report {} did not match a supported samtools coverage or mosdepth summary format",
            path.display()
        );
    }
}

fn read_native_variant_text_rows(path: &Path) -> Result<Vec<ManifestAnnotationEntry>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read native variant report {}", path.display()))?;
    if looks_like_vcfeval_summary_table(&raw) {
        read_vcfeval_summary_annotation_rows(path, &raw)
    } else {
        bail!(
            "native variant report {} did not match a supported vcfeval summary format",
            path.display()
        );
    }
}

fn looks_like_fastp_json(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    let Some(summary) = object.get("summary").and_then(Value::as_object) else {
        return false;
    };
    summary.contains_key("before_filtering")
        && summary.contains_key("after_filtering")
        && object.contains_key("filtering_result")
}

fn looks_like_quast_report(raw: &str) -> bool {
    let Some(header_line) = raw.lines().find(|line| {
        let trimmed = line.trim();
        !trimmed.is_empty() && !trimmed.starts_with('#')
    }) else {
        return false;
    };
    let delimiter = if header_line.contains('\t') {
        '\t'
    } else {
        ','
    };
    let headers = split_manifest_row(header_line, delimiter);
    headers
        .first()
        .is_some_and(|header| normalize_header(header) == "assembly")
        && headers.len() > 1
}

fn looks_like_samtools_flagstat(raw: &str) -> bool {
    let mut saw_total = false;
    let mut saw_mapped = false;
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some(label) = flagstat_label_after_counts(trimmed) else {
            continue;
        };
        if label.starts_with("in total") {
            saw_total = true;
        } else if label.starts_with("mapped ") || label == "mapped" || label.starts_with("mapped (")
        {
            saw_mapped = true;
        }
    }
    saw_total && saw_mapped
}

fn looks_like_samtools_coverage_table(raw: &str) -> bool {
    raw.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .any(|line| {
            let normalized: Vec<String> = split_flexible_table_row(line)
                .iter()
                .map(|cell| normalize_header(cell))
                .collect();
            normalized.iter().any(|header| header == "rname")
                && normalized.iter().any(|header| header == "coverage")
                && normalized.iter().any(|header| header == "meandepth")
        })
}

fn looks_like_mosdepth_summary_table(raw: &str) -> bool {
    let Some(header_line) = raw.lines().find(|line| {
        let trimmed = line.trim();
        !trimmed.is_empty() && !trimmed.starts_with('#')
    }) else {
        return false;
    };
    let headers = split_flexible_table_row(header_line);
    optional_column(&headers, &["chrom"]).is_some()
        && optional_column(&headers, &["length"]).is_some()
        && optional_column(&headers, &["mean"]).is_some()
        && optional_column(&headers, &["bases"]).is_some()
}

fn looks_like_picard_markduplicates_report(raw: &str) -> bool {
    raw.lines().any(|line| {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            return false;
        }
        let delimiter = if trimmed.contains('\t') { '\t' } else { ',' };
        let cells = split_manifest_row(trimmed, delimiter);
        let normalized: Vec<String> = cells.iter().map(|cell| normalize_header(cell)).collect();
        normalized
            .iter()
            .any(|header| header == "percentduplication")
            && normalized.iter().any(|header| {
                header == "readpairduplicates"
                    || header == "unpairedreadduplicates"
                    || header == "readpairsexamined"
                    || header == "unpairedreadsexamined"
            })
    })
}

fn looks_like_happy_summary_table(raw: &str) -> bool {
    let Some(header_line) = raw.lines().find(|line| {
        let trimmed = line.trim();
        !trimmed.is_empty() && !trimmed.starts_with('#')
    }) else {
        return false;
    };
    let delimiter = if header_line.contains('\t') {
        '\t'
    } else if header_line.contains(',') {
        ','
    } else {
        return false;
    };
    let headers = split_manifest_row(header_line, delimiter);
    let normalized: Vec<String> = headers
        .iter()
        .map(|header| normalize_header(header))
        .collect();
    normalized
        .iter()
        .any(|header| header == "metricf1score" || header == "f1score")
        && normalized.iter().any(|header| {
            header == "metricprecision"
                || header == "metricrecall"
                || header == "truthtp"
                || header == "queryfp"
        })
}

fn looks_like_vcfeval_summary_table(raw: &str) -> bool {
    let Some(header_line) = raw.lines().find(|line| {
        let trimmed = line.trim();
        !trimmed.is_empty() && !trimmed.starts_with('#')
    }) else {
        return false;
    };
    let headers = split_flexible_table_row(header_line);
    let normalized: Vec<String> = headers
        .iter()
        .map(|header| normalize_header(header))
        .collect();
    normalized
        .iter()
        .any(|header| header == "fmeasure" || header == "fscore" || header == "f1score")
        && normalized.iter().any(|header| {
            header == "precision"
                || header == "sensitivity"
                || header == "recall"
                || header == "truepos"
        })
}

fn read_fastp_json_annotation_rows(
    path: &Path,
    value: &Value,
) -> Result<Vec<ManifestAnnotationEntry>> {
    let (dataset_id, accession) = infer_dataset_hint_from_path(path).ok_or_else(|| {
        anyhow::anyhow!(
            "native fastp json {} could not infer dataset_id/accession from the file path; rename the file or place it under a dataset-specific directory",
            path.display()
        )
    })?;
    let before_reads = json_value_at_path(value, &["summary", "before_filtering", "total_reads"])
        .and_then(json_value_as_f64)
        .context(format!(
            "native fastp json {} is missing summary.before_filtering.total_reads",
            path.display()
        ))?;
    if before_reads <= 0.0 {
        bail!(
            "native fastp json {} reported non-positive before_filtering.total_reads",
            path.display()
        );
    }
    let after_reads = json_value_at_path(value, &["summary", "after_filtering", "total_reads"])
        .and_then(json_value_as_f64)
        .or_else(|| {
            json_value_at_path(value, &["filtering_result", "passed_filter_reads"])
                .and_then(json_value_as_f64)
        })
        .context(format!(
            "native fastp json {} is missing summary.after_filtering.total_reads",
            path.display()
        ))?;
    let adapter_trimmed_reads =
        json_value_at_path(value, &["adapter_cutting", "adapter_trimmed_reads"])
            .and_then(json_value_as_f64);

    let baseline = StudyBaselineMetrics {
        name: "fastp".to_string(),
        input_bases_per_sec: None,
        trimmed_read_fraction: adapter_trimmed_reads
            .map(|reads| (reads / before_reads).clamp(0.0, 1.0)),
        discarded_read_fraction: Some(
            ((before_reads - after_reads) / before_reads).clamp(0.0, 1.0),
        ),
        corrected_bases_per_mbase: None,
        downstream: default_study_downstream_metrics(),
    };

    Ok(vec![ManifestAnnotationEntry {
        dataset_id: Some(dataset_id),
        accession,
        citation: None,
        expected_platform: None,
        expected_experiment: None,
        downstream: default_study_downstream_metrics(),
        baseline: Some(baseline),
        notes: None,
    }])
}

fn read_samtools_flagstat_annotation_rows(
    path: &Path,
    raw: &str,
) -> Result<Vec<ManifestAnnotationEntry>> {
    let mut total_reads = None;
    let mut mapped_reads = None;
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some(label) = flagstat_label_after_counts(trimmed) else {
            continue;
        };
        if total_reads.is_none() && label.starts_with("in total") {
            total_reads = Some(parse_flagstat_leading_counts(trimmed, path, "in total")?);
        } else if mapped_reads.is_none()
            && (label.starts_with("mapped ") || label == "mapped" || label.starts_with("mapped ("))
        {
            mapped_reads = Some(parse_flagstat_leading_counts(trimmed, path, "mapped")?);
        }
    }

    let total_reads = total_reads.with_context(|| {
        format!(
            "native samtools flagstat {} did not contain an 'in total' line",
            path.display()
        )
    })?;
    if total_reads <= 0.0 {
        bail!(
            "native samtools flagstat {} reported non-positive total read count",
            path.display()
        );
    }
    let mapped_reads = mapped_reads.with_context(|| {
        format!(
            "native samtools flagstat {} did not contain a top-level 'mapped' line",
            path.display()
        )
    })?;

    build_native_metric_annotation(
        path,
        "samtools flagstat",
        StudyDownstreamMetrics {
            alignment_rate: Some((mapped_reads / total_reads).clamp(0.0, 1.0)),
            ..default_study_downstream_metrics()
        },
    )
}

fn read_quast_report_annotation_rows(
    path: &Path,
    raw: &str,
) -> Result<Vec<ManifestAnnotationEntry>> {
    let mut lines = raw.lines().filter(|line| {
        let trimmed = line.trim();
        !trimmed.is_empty() && !trimmed.starts_with('#')
    });
    let header_line = lines
        .next()
        .with_context(|| format!("native quast report {} is empty", path.display()))?;
    let delimiter = if header_line.contains('\t') {
        '\t'
    } else {
        ','
    };
    let headers = split_manifest_row(header_line, delimiter);
    if headers.len() < 2 || normalize_header(&headers[0]) != "assembly" {
        bail!(
            "native quast report {} must start with an Assembly header row",
            path.display()
        );
    }

    let path_hint = infer_dataset_hint_from_path(path);
    let mut column_hints = Vec::new();
    for column_index in 1..headers.len() {
        let label = headers[column_index].trim();
        let hint = infer_dataset_hint_from_label(label).or_else(|| {
            if headers.len() == 2 {
                path_hint.clone()
            } else {
                None
            }
        });
        let hint = hint.with_context(|| {
            format!(
                "native quast report {} could not infer a dataset key for assembly column '{}'",
                path.display(),
                label
            )
        })?;
        column_hints.push((column_index, label.to_string(), hint));
    }

    let mut n50_by_column = HashMap::new();
    for line in lines {
        let cells = split_manifest_row(line, delimiter);
        if cells.is_empty() {
            continue;
        }
        if normalize_header(&cells[0]) == "n50" {
            for (column_index, _, _) in &column_hints {
                let value = cells
                    .get(*column_index)
                    .map(String::as_str)
                    .unwrap_or("")
                    .trim();
                if value.is_empty() || value == "-" {
                    continue;
                }
                let parsed = value.replace(',', "").parse::<f64>().map_err(|error| {
                    anyhow::anyhow!(
                        "native quast report {} failed to parse N50 value '{}' for column {}: {error}",
                        path.display(),
                        value,
                        column_index
                    )
                })?;
                n50_by_column.insert(*column_index, parsed);
            }
            break;
        }
    }

    if n50_by_column.is_empty() {
        bail!(
            "native quast report {} did not contain an N50 row",
            path.display()
        );
    }

    let mut entries = Vec::new();
    for (column_index, _label, (dataset_id, accession)) in column_hints {
        if let Some(assembly_n50) = n50_by_column.get(&column_index).copied() {
            entries.push(ManifestAnnotationEntry {
                dataset_id: Some(dataset_id),
                accession,
                citation: None,
                expected_platform: None,
                expected_experiment: None,
                downstream: default_study_downstream_metrics(),
                baseline: Some(StudyBaselineMetrics {
                    name: "baseline".to_string(),
                    input_bases_per_sec: None,
                    trimmed_read_fraction: None,
                    discarded_read_fraction: None,
                    corrected_bases_per_mbase: None,
                    downstream: StudyDownstreamMetrics {
                        alignment_rate: None,
                        duplicate_rate: None,
                        variant_f1: None,
                        mean_coverage: None,
                        coverage_breadth: None,
                        assembly_n50: Some(assembly_n50),
                    },
                }),
                notes: None,
            });
        }
    }
    Ok(entries)
}

fn read_picard_markduplicates_annotation_rows(
    path: &Path,
    raw: &str,
) -> Result<Vec<ManifestAnnotationEntry>> {
    let filtered_lines: Vec<&str> = raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    let mut header_cells = None;
    let mut data_cells = None;
    for index in 0..filtered_lines.len() {
        let line = filtered_lines[index];
        let delimiter = if line.contains('\t') { '\t' } else { ',' };
        let cells = split_manifest_row(line, delimiter);
        let normalized: Vec<String> = cells.iter().map(|cell| normalize_header(cell)).collect();
        if normalized
            .iter()
            .any(|header| header == "percentduplication")
        {
            header_cells = Some(cells);
            data_cells = filtered_lines
                .get(index + 1)
                .map(|next| split_manifest_row(next, delimiter));
            break;
        }
    }

    let headers = header_cells.with_context(|| {
        format!(
            "native Picard MarkDuplicates report {} did not contain a metrics header row",
            path.display()
        )
    })?;
    let cells = data_cells.with_context(|| {
        format!(
            "native Picard MarkDuplicates report {} did not contain a metrics data row",
            path.display()
        )
    })?;

    let header_index = |aliases: &[&str]| optional_column(&headers, aliases);
    let duplicate_rate = if let Some(index) = header_index(&["PERCENT_DUPLICATION"]) {
        parse_optional_native_report_f64(
            cells.get(index).map(String::as_str).unwrap_or(""),
            path,
            "PERCENT_DUPLICATION",
        )?
    } else {
        None
    }
    .or_else(|| {
        let read_pairs_examined = header_index(&["READ_PAIRS_EXAMINED"])
            .and_then(|index| cells.get(index))
            .and_then(|value| parse_native_report_f64_quiet(value));
        let read_pair_duplicates = header_index(&["READ_PAIR_DUPLICATES"])
            .and_then(|index| cells.get(index))
            .and_then(|value| parse_native_report_f64_quiet(value));
        let unpaired_reads_examined = header_index(&["UNPAIRED_READS_EXAMINED"])
            .and_then(|index| cells.get(index))
            .and_then(|value| parse_native_report_f64_quiet(value));
        let unpaired_read_duplicates = header_index(&["UNPAIRED_READ_DUPLICATES"])
            .and_then(|index| cells.get(index))
            .and_then(|value| parse_native_report_f64_quiet(value));
        let total_examined =
            unpaired_reads_examined.unwrap_or(0.0) + read_pairs_examined.unwrap_or(0.0) * 2.0;
        let total_duplicates =
            unpaired_read_duplicates.unwrap_or(0.0) + read_pair_duplicates.unwrap_or(0.0) * 2.0;
        (total_examined > 0.0).then(|| (total_duplicates / total_examined).clamp(0.0, 1.0))
    })
    .with_context(|| {
        format!(
            "native Picard MarkDuplicates report {} did not expose a duplicate rate",
            path.display()
        )
    })?;

    build_native_metric_annotation(
        path,
        "Picard MarkDuplicates",
        StudyDownstreamMetrics {
            duplicate_rate: Some(duplicate_rate),
            ..default_study_downstream_metrics()
        },
    )
}

fn read_mosdepth_summary_annotation_rows(
    path: &Path,
    raw: &str,
) -> Result<Vec<ManifestAnnotationEntry>> {
    let filtered_lines: Vec<&str> = raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    let header_line = filtered_lines
        .first()
        .copied()
        .with_context(|| format!("native mosdepth summary {} is empty", path.display()))?;
    let headers = split_flexible_table_row(header_line);
    let chrom_index = required_report_column(&headers, &["chrom"], path)?;
    let length_index = required_report_column(&headers, &["length"], path)?;
    let mean_index = required_report_column(&headers, &["mean"], path)?;

    let mut preferred: Option<(u8, f64, f64)> = None;
    let mut weighted_sum = 0.0;
    let mut total_length = 0.0;

    for (offset, line) in filtered_lines.iter().enumerate().skip(1) {
        let line_number = offset + 1;
        let cells = split_flexible_table_row(line);
        if cells.is_empty() {
            continue;
        }
        let Some(mean_coverage) =
            parse_optional_native_report_f64(cell(&cells, mean_index), path, "mean")?
        else {
            continue;
        };
        let length = parse_optional_native_report_f64(cell(&cells, length_index), path, "length")?
            .unwrap_or(0.0);
        let priority = match normalize_header(cell(&cells, chrom_index)).as_str() {
            "total" | "all" | "overall" => 0,
            "region" => 1,
            _ => 2,
        };
        if priority < 2 {
            let candidate = (priority, mean_coverage, length);
            match preferred {
                None => preferred = Some(candidate),
                Some(existing) if candidate.0 < existing.0 => preferred = Some(candidate),
                Some(existing) if candidate.0 == existing.0 && candidate.2 > existing.2 => {
                    preferred = Some(candidate)
                }
                _ => {}
            }
        }
        if length > 0.0 {
            weighted_sum += mean_coverage * length;
            total_length += length;
        } else if line_number == 0 {
            continue;
        }
    }

    let mean_coverage = preferred
        .map(|(_, mean_coverage, _)| mean_coverage)
        .or_else(|| (total_length > 0.0).then(|| weighted_sum / total_length))
        .with_context(|| {
            format!(
                "native mosdepth summary {} did not expose a usable mean coverage value",
                path.display()
            )
        })?;
    build_native_metric_annotation(
        path,
        "mosdepth summary",
        StudyDownstreamMetrics {
            mean_coverage: Some(mean_coverage),
            ..default_study_downstream_metrics()
        },
    )
}

fn read_samtools_coverage_annotation_rows(
    path: &Path,
    raw: &str,
) -> Result<Vec<ManifestAnnotationEntry>> {
    let filtered_lines: Vec<&str> = raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    let header_index = filtered_lines
        .iter()
        .position(|line| {
            let normalized: Vec<String> = split_flexible_table_row(line)
                .iter()
                .map(|cell| normalize_header(cell))
                .collect();
            normalized.iter().any(|header| header == "rname")
                && normalized.iter().any(|header| header == "coverage")
                && normalized.iter().any(|header| header == "meandepth")
        })
        .with_context(|| {
            format!(
                "native samtools coverage report {} did not contain a supported header row",
                path.display()
            )
        })?;
    let headers = split_flexible_table_row(filtered_lines[header_index]);
    let rname_index = required_report_column(&headers, &["#rname", "rname"], path)?;
    let start_index = optional_column(&headers, &["startpos", "start"]);
    let end_index = optional_column(&headers, &["endpos", "end"]);
    let covbases_index = optional_column(&headers, &["covbases", "coveredbases"]);
    let coverage_index = required_report_column(&headers, &["coverage"], path)?;
    let mean_depth_index = required_report_column(&headers, &["meandepth", "meancoverage"], path)?;

    let mut preferred: Option<(f64, Option<f64>, f64)> = None;
    let mut weighted_depth_sum = 0.0;
    let mut total_length = 0.0;
    let mut covered_bases_sum = 0.0;
    let mut saw_covbases = false;
    let mut weighted_breadth_sum = 0.0;
    let mut saw_weighted_breadth = false;

    for (offset, line) in filtered_lines.iter().enumerate().skip(header_index + 1) {
        let line_number = offset + 1;
        let cells = split_flexible_table_row(line);
        if cells.is_empty() {
            continue;
        }
        let Some(mean_coverage) =
            parse_optional_native_report_f64(cell(&cells, mean_depth_index), path, "meandepth")?
        else {
            continue;
        };
        let coverage_breadth =
            parse_optional_native_report_f64(cell(&cells, coverage_index), path, "coverage")?
                .map(normalize_fraction_metric);
        let length = match (start_index, end_index) {
            (Some(start_index), Some(end_index)) => {
                let start =
                    parse_optional_native_report_f64(cell(&cells, start_index), path, "startpos")?;
                let end =
                    parse_optional_native_report_f64(cell(&cells, end_index), path, "endpos")?;
                match (start, end) {
                    (Some(start), Some(end)) if end >= start => Some(end - start + 1.0),
                    _ => None,
                }
            }
            _ => None,
        };
        let covbases = covbases_index
            .map(|index| parse_optional_native_report_f64(cell(&cells, index), path, "covbases"))
            .transpose()?
            .flatten();
        let normalized_label = normalize_header(cell(&cells, rname_index));
        if matches!(
            normalized_label.as_str(),
            "total" | "all" | "overall" | "genome"
        ) {
            let candidate = (mean_coverage, coverage_breadth, length.unwrap_or(0.0));
            match preferred {
                None => preferred = Some(candidate),
                Some(existing) if candidate.2 > existing.2 => preferred = Some(candidate),
                _ => {}
            }
            continue;
        }

        let Some(length) = length.filter(|length| *length > 0.0) else {
            continue;
        };
        weighted_depth_sum += mean_coverage * length;
        total_length += length;
        if let Some(covbases) = covbases {
            covered_bases_sum += covbases.clamp(0.0, length);
            saw_covbases = true;
        } else if let Some(coverage_breadth) = coverage_breadth {
            weighted_breadth_sum += coverage_breadth * length;
            saw_weighted_breadth = true;
        } else if line_number == 0 {
            continue;
        }
    }

    let (mean_coverage, coverage_breadth) =
        if let Some((mean_coverage, coverage_breadth, _)) = preferred {
            (Some(mean_coverage), coverage_breadth)
        } else {
            let mean_coverage = (total_length > 0.0).then(|| weighted_depth_sum / total_length);
            let coverage_breadth = if saw_covbases && total_length > 0.0 {
                Some((covered_bases_sum / total_length).clamp(0.0, 1.0))
            } else if saw_weighted_breadth && total_length > 0.0 {
                Some((weighted_breadth_sum / total_length).clamp(0.0, 1.0))
            } else {
                None
            };
            (mean_coverage, coverage_breadth)
        };
    let mean_coverage = mean_coverage.with_context(|| {
        format!(
            "native samtools coverage report {} did not expose a usable mean coverage value",
            path.display()
        )
    })?;
    build_native_metric_annotation(
        path,
        "samtools coverage",
        StudyDownstreamMetrics {
            mean_coverage: Some(mean_coverage),
            coverage_breadth,
            ..default_study_downstream_metrics()
        },
    )
}

fn read_happy_summary_annotation_rows(
    path: &Path,
    raw: &str,
) -> Result<Vec<ManifestAnnotationEntry>> {
    let mut lines = raw.lines().filter(|line| {
        let trimmed = line.trim();
        !trimmed.is_empty() && !trimmed.starts_with('#')
    });
    let header_line = lines
        .next()
        .with_context(|| format!("native hap.py summary {} is empty", path.display()))?;
    let delimiter = if header_line.contains('\t') {
        '\t'
    } else {
        ','
    };
    let headers = split_manifest_row(header_line, delimiter);
    let type_index = optional_column(&headers, &["Type", "VAR_TYPE"]);
    let filter_index = optional_column(&headers, &["Filter", "Subset"]);
    let f1_index = optional_column(&headers, &["METRIC.F1_Score", "F1_Score", "F1Score", "F1"]);
    let precision_index =
        optional_column(&headers, &["METRIC.Precision", "Precision", "PRECISION"]);
    let recall_index = optional_column(&headers, &["METRIC.Recall", "Recall", "Sensitivity"]);
    let truth_tp_index = optional_column(&headers, &["TRUTH.TP"]);
    let truth_fn_index = optional_column(&headers, &["TRUTH.FN"]);
    let query_tp_index = optional_column(&headers, &["QUERY.TP"]);
    let query_fp_index = optional_column(&headers, &["QUERY.FP"]);

    let mut best: Option<(u8, f64)> = None;
    for (offset, line) in lines.enumerate() {
        let line_number = offset + 2;
        let cells = split_manifest_row(line, delimiter);
        if cells.is_empty() {
            continue;
        }
        let Some(f1) = parse_variant_metric_from_cells(
            &cells,
            f1_index,
            precision_index,
            recall_index,
            truth_tp_index,
            truth_fn_index,
            query_tp_index,
            query_fp_index,
            path,
            line_number,
        )?
        else {
            continue;
        };
        let row_priority = happy_summary_row_priority(
            type_index.map(|index| cell(&cells, index)),
            filter_index.map(|index| cell(&cells, index)),
        );
        let candidate = (row_priority, f1);
        match best {
            None => best = Some(candidate),
            Some(existing) if candidate.0 < existing.0 => best = Some(candidate),
            Some(existing) if candidate.0 == existing.0 && candidate.1 > existing.1 => {
                best = Some(candidate)
            }
            _ => {}
        }
    }

    let variant_f1 = best.map(|(_, f1)| f1).with_context(|| {
        format!(
            "native hap.py summary {} did not expose a usable F1 row",
            path.display()
        )
    })?;
    build_native_metric_annotation(
        path,
        "hap.py summary",
        StudyDownstreamMetrics {
            variant_f1: Some(variant_f1),
            ..default_study_downstream_metrics()
        },
    )
}

fn read_vcfeval_summary_annotation_rows(
    path: &Path,
    raw: &str,
) -> Result<Vec<ManifestAnnotationEntry>> {
    let filtered_lines: Vec<&str> = raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    let header_index = filtered_lines
        .iter()
        .position(|line| {
            let normalized: Vec<String> = split_flexible_table_row(line)
                .iter()
                .map(|value| normalize_header(value))
                .collect();
            normalized
                .iter()
                .any(|header| header == "fmeasure" || header == "fscore" || header == "f1score")
                && normalized.iter().any(|header| {
                    header == "precision"
                        || header == "sensitivity"
                        || header == "recall"
                        || header == "truepos"
                })
        })
        .with_context(|| {
            format!(
                "native vcfeval summary {} did not contain a supported header row",
                path.display()
            )
        })?;
    let headers = split_flexible_table_row(filtered_lines[header_index]);
    let f1_index = optional_column(&headers, &["F-measure", "F_score", "F-score", "F1"]);
    let precision_index = optional_column(&headers, &["Precision"]);
    let recall_index = optional_column(&headers, &["Sensitivity", "Recall"]);
    let true_pos_index = optional_column(&headers, &["True-pos", "TruePos", "TP"]);
    let false_pos_index = optional_column(&headers, &["False-pos", "FalsePos", "FP"]);
    let false_neg_index = optional_column(&headers, &["False-neg", "FalseNeg", "FN"]);

    let mut best_f1: Option<f64> = None;
    for (offset, line) in filtered_lines.iter().enumerate().skip(header_index + 1) {
        let line_number = offset + 1;
        let cells = split_flexible_table_row(line);
        if cells.is_empty() {
            continue;
        }
        let Some(f1) = parse_variant_metric_from_cells(
            &cells,
            f1_index,
            precision_index,
            recall_index,
            true_pos_index,
            false_neg_index,
            true_pos_index,
            false_pos_index,
            path,
            line_number,
        )?
        else {
            continue;
        };
        best_f1 = Some(best_f1.map_or(f1, |existing| existing.max(f1)));
    }

    let variant_f1 = best_f1.with_context(|| {
        format!(
            "native vcfeval summary {} did not expose a usable F-measure row",
            path.display()
        )
    })?;
    build_native_metric_annotation(
        path,
        "vcfeval summary",
        StudyDownstreamMetrics {
            variant_f1: Some(variant_f1),
            ..default_study_downstream_metrics()
        },
    )
}

fn split_flexible_table_row(line: &str) -> Vec<String> {
    if line.contains('\t') {
        split_manifest_row(line, '\t')
    } else if line.contains(',') {
        split_manifest_row(line, ',')
    } else {
        line.split_whitespace()
            .map(|cell| cell.to_string())
            .collect()
    }
}

fn happy_summary_row_priority(type_label: Option<&str>, filter_label: Option<&str>) -> u8 {
    let normalized_type = type_label.map(normalize_header).unwrap_or_default();
    let normalized_filter = filter_label.map(normalize_header).unwrap_or_default();
    match (normalized_type.as_str(), normalized_filter.as_str()) {
        ("all", "pass" | "all" | "") => 0,
        ("total", "pass" | "all" | "") => 0,
        ("all" | "total", _) => 1,
        (_, "pass" | "all" | "") => 2,
        _ => 3,
    }
}

fn parse_variant_metric_from_cells(
    cells: &[String],
    f1_index: Option<usize>,
    precision_index: Option<usize>,
    recall_index: Option<usize>,
    truth_tp_index: Option<usize>,
    truth_fn_index: Option<usize>,
    query_tp_index: Option<usize>,
    query_fp_index: Option<usize>,
    path: &Path,
    line_number: usize,
) -> Result<Option<f64>> {
    if let Some(index) = f1_index {
        let f1 = parse_optional_native_report_f64(cell(cells, index), path, "variant_f1")?
            .map(normalize_fraction_metric);
        if f1.is_some() {
            return Ok(f1);
        }
    }

    let precision = precision_index
        .map(|index| parse_optional_native_report_f64(cell(cells, index), path, "precision"))
        .transpose()?
        .flatten()
        .map(normalize_fraction_metric);
    let recall = recall_index
        .map(|index| parse_optional_native_report_f64(cell(cells, index), path, "recall"))
        .transpose()?
        .flatten()
        .map(normalize_fraction_metric);
    if let Some((precision, recall)) = precision.zip(recall) {
        let denominator = precision + recall;
        if denominator > 0.0 {
            return Ok(Some(
                (2.0 * precision * recall / denominator).clamp(0.0, 1.0),
            ));
        }
    }

    let truth_tp = truth_tp_index
        .map(|index| parse_optional_native_report_f64(cell(cells, index), path, "truth_tp"))
        .transpose()?
        .flatten();
    let truth_fn = truth_fn_index
        .map(|index| parse_optional_native_report_f64(cell(cells, index), path, "truth_fn"))
        .transpose()?
        .flatten();
    let query_tp = query_tp_index
        .map(|index| parse_optional_native_report_f64(cell(cells, index), path, "query_tp"))
        .transpose()?
        .flatten();
    let query_fp = query_fp_index
        .map(|index| parse_optional_native_report_f64(cell(cells, index), path, "query_fp"))
        .transpose()?
        .flatten();

    let tp = query_tp.or(truth_tp);
    if let Some(tp) = tp {
        let fn_count = truth_fn.unwrap_or(0.0);
        let fp_count = query_fp.unwrap_or(0.0);
        let denominator = 2.0 * tp + fp_count + fn_count;
        if denominator > 0.0 {
            return Ok(Some((2.0 * tp / denominator).clamp(0.0, 1.0)));
        }
    }

    if line_number == 0 { Ok(None) } else { Ok(None) }
}

fn normalize_fraction_metric(value: f64) -> f64 {
    if value > 1.0 && value <= 100.0 {
        (value / 100.0).clamp(0.0, 1.0)
    } else {
        value.clamp(0.0, 1.0)
    }
}

fn parse_json_annotation_row(
    value: &Value,
    path: &Path,
    record_number: usize,
) -> Result<ManifestAnnotationEntry> {
    let object = value.as_object().with_context(|| {
        format!(
            "structured json {} record {} is not a JSON object",
            path.display(),
            record_number
        )
    })?;

    let dataset_id = json_optional_text_field(object, &["dataset_id"], path, record_number)?;
    let accession = json_optional_text_field(
        object,
        &["accession", "run_accession", "run", "run_acc"],
        path,
        record_number,
    )?;
    if dataset_id.is_none() && accession.is_none() {
        bail!(
            "structured json {} record {} must provide dataset_id and/or accession",
            path.display(),
            record_number
        );
    }

    let citation_candidates = text_fields_from_json_aliases(
        object,
        &[
            &["study_title", "title"],
            &["experiment_title", "experiment", "experiment_name"],
            &["study_accession", "study", "srastudy"],
            &["bioproject", "project_accession", "project"],
            &["study_pubmed_id", "pubmed_id", "pmid"],
            &["study_doi", "doi"],
        ],
        path,
        record_number,
    )?;
    let citation =
        json_optional_text_field(object, &["citation", "reference"], path, record_number)?
            .or_else(|| compose_metadata_citation(&citation_candidates));
    let platform_texts = text_fields_from_json_aliases(
        object,
        &[
            &["expected_platform"],
            &["platform", "instrument_platform"],
            &["instrument_model", "model", "instrument"],
        ],
        path,
        record_number,
    )?;
    let expected_platform =
        json_optional_platform_field(object, &["expected_platform"], path, record_number)?
            .or_else(|| infer_platform_from_metadata(&platform_texts));
    let experiment_texts = text_fields_from_json_aliases(
        object,
        &[
            &["expected_experiment", "experiment", "experiment_title"],
            &["library_strategy", "librarystrategy", "assay_type"],
            &[
                "library_source",
                "library_selection",
                "sample_title",
                "sample_name",
            ],
        ],
        path,
        record_number,
    )?;
    let expected_experiment =
        json_optional_experiment_field(object, &["expected_experiment"], path, record_number)?
            .or_else(|| infer_experiment_from_metadata(&experiment_texts, expected_platform));
    let downstream = StudyDownstreamMetrics {
        alignment_rate: json_optional_f64_field(
            object,
            &["echo_alignment_rate", "japalityecho_alignment_rate"],
            path,
            record_number,
        )?,
        duplicate_rate: json_optional_f64_field(
            object,
            &["echo_duplicate_rate", "japalityecho_duplicate_rate"],
            path,
            record_number,
        )?,
        variant_f1: json_optional_f64_field(
            object,
            &["echo_variant_f1", "japalityecho_variant_f1"],
            path,
            record_number,
        )?,
        mean_coverage: json_optional_f64_field(
            object,
            &[
                "echo_mean_coverage",
                "japalityecho_mean_coverage",
                "echo_mean_depth",
                "japalityecho_mean_depth",
            ],
            path,
            record_number,
        )?,
        coverage_breadth: json_optional_f64_field(
            object,
            &[
                "echo_coverage_breadth",
                "japalityecho_coverage_breadth",
                "echo_coverage_fraction",
                "japalityecho_coverage_fraction",
                "echo_covered_bases_fraction",
                "japalityecho_covered_bases_fraction",
            ],
            path,
            record_number,
        )?,
        assembly_n50: json_optional_f64_field(
            object,
            &["echo_assembly_n50", "japalityecho_assembly_n50"],
            path,
            record_number,
        )?,
    };
    let baseline_name = json_optional_text_field(
        object,
        &["baseline_name", "competitor_name", "baseline_tool"],
        path,
        record_number,
    )?;
    let baseline_input_bases_per_sec = json_optional_f64_field(
        object,
        &[
            "baseline_input_bases_per_sec",
            "competitor_input_bases_per_sec",
        ],
        path,
        record_number,
    )?;
    let baseline_trimmed_read_fraction = json_optional_f64_field(
        object,
        &[
            "baseline_trimmed_read_fraction",
            "competitor_trimmed_read_fraction",
        ],
        path,
        record_number,
    )?;
    let baseline_discarded_read_fraction = json_optional_f64_field(
        object,
        &[
            "baseline_discarded_read_fraction",
            "competitor_discarded_read_fraction",
        ],
        path,
        record_number,
    )?;
    let baseline_corrected_bases_per_mbase = json_optional_f64_field(
        object,
        &[
            "baseline_corrected_bases_per_mbase",
            "competitor_corrected_bases_per_mbase",
        ],
        path,
        record_number,
    )?;
    let baseline_downstream = StudyDownstreamMetrics {
        alignment_rate: json_optional_f64_field(
            object,
            &["baseline_alignment_rate", "competitor_alignment_rate"],
            path,
            record_number,
        )?,
        duplicate_rate: json_optional_f64_field(
            object,
            &["baseline_duplicate_rate", "competitor_duplicate_rate"],
            path,
            record_number,
        )?,
        variant_f1: json_optional_f64_field(
            object,
            &["baseline_variant_f1", "competitor_variant_f1"],
            path,
            record_number,
        )?,
        mean_coverage: json_optional_f64_field(
            object,
            &[
                "baseline_mean_coverage",
                "competitor_mean_coverage",
                "baseline_mean_depth",
                "competitor_mean_depth",
            ],
            path,
            record_number,
        )?,
        coverage_breadth: json_optional_f64_field(
            object,
            &[
                "baseline_coverage_breadth",
                "competitor_coverage_breadth",
                "baseline_coverage_fraction",
                "competitor_coverage_fraction",
                "baseline_covered_bases_fraction",
                "competitor_covered_bases_fraction",
            ],
            path,
            record_number,
        )?,
        assembly_n50: json_optional_f64_field(
            object,
            &["baseline_assembly_n50", "competitor_assembly_n50"],
            path,
            record_number,
        )?,
    };
    let baseline = if baseline_name.is_some()
        || baseline_input_bases_per_sec.is_some()
        || baseline_trimmed_read_fraction.is_some()
        || baseline_discarded_read_fraction.is_some()
        || baseline_corrected_bases_per_mbase.is_some()
        || downstream_has_any_metrics(&baseline_downstream)
    {
        Some(StudyBaselineMetrics {
            name: baseline_name.unwrap_or_else(|| "baseline".to_string()),
            input_bases_per_sec: baseline_input_bases_per_sec,
            trimmed_read_fraction: baseline_trimmed_read_fraction,
            discarded_read_fraction: baseline_discarded_read_fraction,
            corrected_bases_per_mbase: baseline_corrected_bases_per_mbase,
            downstream: baseline_downstream,
        })
    } else {
        None
    };
    let notes = json_optional_text_field(object, &["notes", "note"], path, record_number)?;

    Ok(ManifestAnnotationEntry {
        dataset_id,
        accession,
        citation,
        expected_platform,
        expected_experiment,
        downstream,
        baseline,
        notes,
    })
}

fn json_optional_text_field(
    object: &serde_json::Map<String, Value>,
    aliases: &[&str],
    path: &Path,
    record_number: usize,
) -> Result<Option<String>> {
    let Some(value) = json_value_for_aliases(object, aliases) else {
        return Ok(None);
    };
    match value {
        Value::Null => Ok(None),
        Value::String(text) => Ok(Some(text.trim().to_string()).filter(|value| !value.is_empty())),
        Value::Number(number) => Ok(Some(number.to_string())),
        Value::Bool(flag) => Ok(Some(flag.to_string())),
        _ => bail!(
            "structured json {} record {} field {} must be a string, number, or boolean",
            path.display(),
            record_number,
            aliases[0]
        ),
    }
}

fn json_optional_f64_field(
    object: &serde_json::Map<String, Value>,
    aliases: &[&str],
    path: &Path,
    record_number: usize,
) -> Result<Option<f64>> {
    let Some(value) = json_value_for_aliases(object, aliases) else {
        return Ok(None);
    };
    match value {
        Value::Null => Ok(None),
        Value::Number(number) => number.as_f64().map(Some).with_context(|| {
            format!(
                "structured json {} record {} field {} is not representable as f64",
                path.display(),
                record_number,
                aliases[0]
            )
        }),
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                trimmed.parse::<f64>().map(Some).map_err(|error| {
                    anyhow::anyhow!(
                        "structured json {} record {} field {} failed to parse '{trimmed}' as f64: {error}",
                        path.display(),
                        record_number,
                        aliases[0]
                    )
                })
            }
        }
        _ => bail!(
            "structured json {} record {} field {} must be numeric or a numeric string",
            path.display(),
            record_number,
            aliases[0]
        ),
    }
}

fn json_optional_platform_field(
    object: &serde_json::Map<String, Value>,
    aliases: &[&str],
    path: &Path,
    record_number: usize,
) -> Result<Option<Platform>> {
    let Some(text) = json_optional_text_field(object, aliases, path, record_number)? else {
        return Ok(None);
    };
    parse_optional_platform(&text, path, record_number)
}

fn json_optional_experiment_field(
    object: &serde_json::Map<String, Value>,
    aliases: &[&str],
    path: &Path,
    record_number: usize,
) -> Result<Option<ExperimentType>> {
    let Some(text) = json_optional_text_field(object, aliases, path, record_number)? else {
        return Ok(None);
    };
    parse_optional_experiment(&text, path, record_number)
}

fn json_value_for_aliases<'a>(
    object: &'a serde_json::Map<String, Value>,
    aliases: &[&str],
) -> Option<&'a Value> {
    object.iter().find_map(|(key, value)| {
        aliases
            .iter()
            .any(|alias| normalize_header(key) == normalize_header(alias))
            .then_some(value)
    })
}

fn merge_ingested_annotation(
    target: &mut ManifestAnnotationEntry,
    incoming: ManifestAnnotationEntry,
    path: &Path,
    record_number: usize,
) -> Result<()> {
    merge_ingested_optional_field(
        &mut target.dataset_id,
        incoming.dataset_id,
        "dataset_id",
        path,
        record_number,
    )?;
    merge_ingested_optional_field(
        &mut target.accession,
        incoming.accession,
        "accession",
        path,
        record_number,
    )?;
    merge_ingested_optional_field(
        &mut target.citation,
        incoming.citation,
        "citation",
        path,
        record_number,
    )?;
    merge_ingested_optional_field(
        &mut target.expected_platform,
        incoming.expected_platform,
        "expected_platform",
        path,
        record_number,
    )?;
    merge_ingested_optional_field(
        &mut target.expected_experiment,
        incoming.expected_experiment,
        "expected_experiment",
        path,
        record_number,
    )?;
    merge_ingested_downstream(
        &mut target.downstream,
        incoming.downstream,
        path,
        record_number,
        "echo",
    )?;
    merge_ingested_baseline(&mut target.baseline, incoming.baseline, path, record_number)?;
    merge_ingested_notes(&mut target.notes, incoming.notes);
    Ok(())
}

fn merge_ingested_optional_field<T: PartialEq + Clone>(
    target: &mut Option<T>,
    incoming: Option<T>,
    field_name: &str,
    path: &Path,
    record_number: usize,
) -> Result<()> {
    let Some(incoming) = incoming else {
        return Ok(());
    };
    match target {
        None => {
            *target = Some(incoming);
            Ok(())
        }
        Some(existing) if *existing == incoming => Ok(()),
        Some(_) => bail!(
            "structured input {} record {} provided conflicting values for {}",
            path.display(),
            record_number,
            field_name
        ),
    }
}

fn merge_ingested_downstream(
    target: &mut StudyDownstreamMetrics,
    incoming: StudyDownstreamMetrics,
    path: &Path,
    record_number: usize,
    prefix: &str,
) -> Result<()> {
    merge_ingested_optional_field(
        &mut target.alignment_rate,
        incoming.alignment_rate,
        &format!("{prefix}_alignment_rate"),
        path,
        record_number,
    )?;
    merge_ingested_optional_field(
        &mut target.duplicate_rate,
        incoming.duplicate_rate,
        &format!("{prefix}_duplicate_rate"),
        path,
        record_number,
    )?;
    merge_ingested_optional_field(
        &mut target.variant_f1,
        incoming.variant_f1,
        &format!("{prefix}_variant_f1"),
        path,
        record_number,
    )?;
    merge_ingested_optional_field(
        &mut target.mean_coverage,
        incoming.mean_coverage,
        &format!("{prefix}_mean_coverage"),
        path,
        record_number,
    )?;
    merge_ingested_optional_field(
        &mut target.coverage_breadth,
        incoming.coverage_breadth,
        &format!("{prefix}_coverage_breadth"),
        path,
        record_number,
    )?;
    merge_ingested_optional_field(
        &mut target.assembly_n50,
        incoming.assembly_n50,
        &format!("{prefix}_assembly_n50"),
        path,
        record_number,
    )?;
    Ok(())
}

fn merge_ingested_baseline(
    target: &mut Option<StudyBaselineMetrics>,
    incoming: Option<StudyBaselineMetrics>,
    path: &Path,
    record_number: usize,
) -> Result<()> {
    let Some(incoming) = incoming else {
        return Ok(());
    };
    let target = target.get_or_insert_with(default_study_baseline_metrics);

    let incoming_name = normalize_baseline_name(Some(incoming.name.as_str()));
    let target_name = normalize_baseline_name(Some(target.name.as_str()));
    match (target_name.as_deref(), incoming_name.as_deref()) {
        (None, Some(incoming_name)) => target.name = incoming_name.to_string(),
        (Some(existing), Some(incoming_name)) if existing != incoming_name => bail!(
            "structured input {} record {} provided conflicting baseline_name values",
            path.display(),
            record_number
        ),
        _ => {}
    }

    merge_ingested_optional_field(
        &mut target.input_bases_per_sec,
        incoming.input_bases_per_sec,
        "baseline_input_bases_per_sec",
        path,
        record_number,
    )?;
    merge_ingested_optional_field(
        &mut target.trimmed_read_fraction,
        incoming.trimmed_read_fraction,
        "baseline_trimmed_read_fraction",
        path,
        record_number,
    )?;
    merge_ingested_optional_field(
        &mut target.discarded_read_fraction,
        incoming.discarded_read_fraction,
        "baseline_discarded_read_fraction",
        path,
        record_number,
    )?;
    merge_ingested_optional_field(
        &mut target.corrected_bases_per_mbase,
        incoming.corrected_bases_per_mbase,
        "baseline_corrected_bases_per_mbase",
        path,
        record_number,
    )?;
    merge_ingested_downstream(
        &mut target.downstream,
        incoming.downstream,
        path,
        record_number,
        "baseline",
    )?;
    Ok(())
}

fn merge_ingested_notes(target: &mut Option<String>, incoming: Option<String>) {
    let Some(incoming) = incoming.filter(|value| !value.trim().is_empty()) else {
        return;
    };
    match target {
        None => *target = Some(incoming),
        Some(existing) => {
            if !existing
                .split('|')
                .map(str::trim)
                .any(|segment| segment == incoming.as_str())
            {
                existing.push_str(" | ");
                existing.push_str(&incoming);
            }
        }
    }
}

fn normalize_baseline_name(name: Option<&str>) -> Option<String> {
    name.map(str::trim)
        .filter(|value| !value.is_empty() && *value != "baseline")
        .map(ToString::to_string)
}

fn infer_dataset_hint_from_path(path: &Path) -> Option<(String, Option<String>)> {
    collect_path_hint_labels(path)
        .into_iter()
        .find_map(|candidate| infer_dataset_hint_from_label(&candidate))
}

fn infer_dataset_hint_from_label(label: &str) -> Option<(String, Option<String>)> {
    let stripped = strip_known_result_suffixes(label);
    let accession = extract_accession_token(&stripped).or_else(|| extract_accession_token(label));
    if let Some(accession) = accession {
        return Some((sanitize_filename(&accession), Some(accession)));
    }

    let stripped =
        stripped.trim_matches(|character| character == '_' || character == '-' || character == '.');
    if stripped.is_empty() {
        return None;
    }
    let dataset_id = sanitize_filename(stripped);
    if is_generic_dataset_hint(&dataset_id) {
        None
    } else {
        Some((dataset_id, None))
    }
}

fn strip_known_result_suffixes(value: &str) -> String {
    let mut current = value.trim().to_string();
    loop {
        let lowered = current.to_ascii_lowercase();
        let suffixes = [
            ".txt",
            ".log",
            ".json",
            ".csv",
            ".tsv",
            ".happy",
            "_happy",
            "-happy",
            ".vcfeval",
            "_vcfeval",
            "-vcfeval",
            ".fastp",
            "_fastp",
            "-fastp",
            ".flagstat",
            "_flagstat",
            "-flagstat",
            ".markdup",
            "_markdup",
            "-markdup",
            ".markduplicates",
            "_markduplicates",
            "-markduplicates",
            ".duplication_metrics",
            "_duplication_metrics",
            "-duplication_metrics",
            ".duplicationmetrics",
            "_duplicationmetrics",
            "-duplicationmetrics",
            ".coverage",
            "_coverage",
            "-coverage",
            ".mosdepth",
            "_mosdepth",
            "-mosdepth",
            ".quast",
            "_quast",
            "-quast",
            ".report",
            "_report",
            "-report",
            ".summary",
            "_summary",
            "-summary",
            ".results",
            "_results",
            "-results",
            ".metrics",
            "_metrics",
            "-metrics",
            ".annotations",
            "_annotations",
            "-annotations",
            ".baseline_metrics",
            "_baseline_metrics",
            "-baseline_metrics",
            ".downstream_metrics",
            "_downstream_metrics",
            "-downstream_metrics",
        ];
        let mut removed = false;
        for suffix in suffixes {
            if lowered.ends_with(suffix) && current.len() > suffix.len() {
                current.truncate(current.len() - suffix.len());
                current = current
                    .trim_end_matches(|character| {
                        character == '_' || character == '-' || character == '.'
                    })
                    .to_string();
                removed = true;
                break;
            }
        }
        if !removed {
            break;
        }
    }
    current
}

fn is_generic_dataset_hint(value: &str) -> bool {
    matches!(
        normalize_header(value).as_str(),
        "fastp"
            | "quast"
            | "report"
            | "results"
            | "result"
            | "structuredresults"
            | "structuredresult"
            | "baseline"
            | "metrics"
            | "annotations"
            | "summary"
            | "txt"
            | "log"
            | "flagstat"
            | "markdup"
            | "markduplicates"
            | "duplication"
            | "picardmarkduplicates"
            | "picard"
            | "duplicationmetrics"
            | "samtools"
            | "samtoolscoverage"
            | "coverage"
            | "mosdepth"
            | "happy"
            | "happysummary"
            | "vcfeval"
            | "echo"
            | "japalityecho"
            | "dataset"
            | "assembly"
            | "assemblies"
            | "contigs"
            | "contigs1"
            | "contigs2"
            | "scaffolds"
            | "scaffold"
    )
}

fn collect_path_hint_labels(path: &Path) -> Vec<String> {
    let mut candidates = Vec::new();
    if let Some(file_name) = path.file_name().and_then(|value| value.to_str()) {
        candidates.push(file_name.to_string());
    }
    if let Some(stem) = path.file_stem().and_then(|value| value.to_str()) {
        candidates.push(stem.to_string());
    }
    if let Some(parent) = path.parent() {
        for ancestor in parent.ancestors().take(4) {
            if let Some(name) = ancestor.file_name().and_then(|value| value.to_str()) {
                candidates.push(name.to_string());
            }
        }
    }
    candidates
}

fn build_native_metric_annotation(
    path: &Path,
    source_name: &str,
    downstream_metrics: StudyDownstreamMetrics,
) -> Result<Vec<ManifestAnnotationEntry>> {
    let (dataset_id, accession) = infer_dataset_hint_from_path(path).ok_or_else(|| {
        anyhow::anyhow!(
            "native {} report {} could not infer dataset_id/accession from the file path; rename the file or place it under a dataset-specific directory",
            source_name,
            path.display()
        )
    })?;
    let target = infer_native_metric_target_from_path(path);
    let mut downstream = default_study_downstream_metrics();
    let mut baseline = None;
    match target.side {
        NativeMetricSide::Echo => {
            downstream = downstream_metrics;
        }
        NativeMetricSide::Baseline => {
            let mut baseline_metrics = default_study_baseline_metrics();
            if let Some(name) = target.baseline_name {
                baseline_metrics.name = name;
            }
            baseline_metrics.downstream = downstream_metrics;
            baseline = Some(baseline_metrics);
        }
    }

    Ok(vec![ManifestAnnotationEntry {
        dataset_id: Some(dataset_id),
        accession,
        citation: None,
        expected_platform: None,
        expected_experiment: None,
        downstream,
        baseline,
        notes: None,
    }])
}

fn infer_native_metric_target_from_path(path: &Path) -> NativeMetricTarget {
    for label in collect_path_hint_labels(path) {
        for token in label
            .split(|character: char| !character.is_ascii_alphanumeric())
            .filter(|token| !token.is_empty())
        {
            let normalized = normalize_header(token);
            match normalized.as_str() {
                "echo" | "japalityecho" => {
                    return NativeMetricTarget {
                        side: NativeMetricSide::Echo,
                        baseline_name: None,
                    };
                }
                "fastp" => {
                    return NativeMetricTarget {
                        side: NativeMetricSide::Baseline,
                        baseline_name: Some("fastp".to_string()),
                    };
                }
                "cutadapt" => {
                    return NativeMetricTarget {
                        side: NativeMetricSide::Baseline,
                        baseline_name: Some("cutadapt".to_string()),
                    };
                }
                "trimgalore" => {
                    return NativeMetricTarget {
                        side: NativeMetricSide::Baseline,
                        baseline_name: Some("trim-galore".to_string()),
                    };
                }
                "trimmomatic" => {
                    return NativeMetricTarget {
                        side: NativeMetricSide::Baseline,
                        baseline_name: Some("trimmomatic".to_string()),
                    };
                }
                "bbduk" => {
                    return NativeMetricTarget {
                        side: NativeMetricSide::Baseline,
                        baseline_name: Some("bbduk".to_string()),
                    };
                }
                "adapterremoval" => {
                    return NativeMetricTarget {
                        side: NativeMetricSide::Baseline,
                        baseline_name: Some("adapterremoval".to_string()),
                    };
                }
                "skewer" => {
                    return NativeMetricTarget {
                        side: NativeMetricSide::Baseline,
                        baseline_name: Some("skewer".to_string()),
                    };
                }
                "baseline" | "competitor" => {
                    return NativeMetricTarget {
                        side: NativeMetricSide::Baseline,
                        baseline_name: None,
                    };
                }
                _ => {}
            }
        }
    }
    NativeMetricTarget {
        side: NativeMetricSide::Baseline,
        baseline_name: None,
    }
}

fn flagstat_label_after_counts(line: &str) -> Option<String> {
    let mut parts = line.split_whitespace();
    parts.next()?;
    if parts.next()? != "+" {
        return None;
    }
    parts.next()?;
    Some(parts.collect::<Vec<_>>().join(" "))
}

fn parse_flagstat_leading_counts(line: &str, path: &Path, context_label: &str) -> Result<f64> {
    let mut parts = line.split_whitespace();
    let primary = parts.next().with_context(|| {
        format!(
            "native samtools flagstat {} had a malformed '{}' line",
            path.display(),
            context_label
        )
    })?;
    let plus = parts.next().with_context(|| {
        format!(
            "native samtools flagstat {} had a malformed '{}' line",
            path.display(),
            context_label
        )
    })?;
    let secondary = parts.next().with_context(|| {
        format!(
            "native samtools flagstat {} had a malformed '{}' line",
            path.display(),
            context_label
        )
    })?;
    if plus != "+" {
        bail!(
            "native samtools flagstat {} had a malformed '{}' line",
            path.display(),
            context_label
        );
    }
    Ok(parse_native_report_f64(primary, path, context_label)?
        + parse_native_report_f64(secondary, path, context_label)?)
}

fn parse_native_report_f64(value: &str, path: &Path, field_name: &str) -> Result<f64> {
    value
        .trim()
        .trim_matches('%')
        .replace(',', "")
        .parse::<f64>()
        .map_err(|error| {
            anyhow::anyhow!(
                "failed to parse native report field '{}' value '{}' in {}: {error}",
                field_name,
                value,
                path.display()
            )
        })
}

fn parse_optional_native_report_f64(
    value: &str,
    path: &Path,
    field_name: &str,
) -> Result<Option<f64>> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed == "-" {
        Ok(None)
    } else {
        parse_native_report_f64(trimmed, path, field_name).map(Some)
    }
}

fn parse_native_report_f64_quiet(value: &str) -> Option<f64> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed == "-" {
        None
    } else {
        trimmed
            .trim_matches('%')
            .replace(',', "")
            .parse::<f64>()
            .ok()
    }
}

fn json_value_at_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for segment in path {
        current = current
            .as_object()
            .and_then(|object| object.get(*segment))?;
    }
    Some(current)
}

fn text_fields_from_json_aliases(
    object: &serde_json::Map<String, Value>,
    aliases_groups: &[&[&str]],
    path: &Path,
    record_number: usize,
) -> Result<Vec<String>> {
    let mut values = Vec::new();
    for aliases in aliases_groups {
        if let Some(value) = json_optional_text_field(object, aliases, path, record_number)? {
            values.push(value);
        }
    }
    Ok(values)
}

fn json_value_as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn compose_metadata_citation(candidates: &[String]) -> Option<String> {
    let mut parts = Vec::new();
    for candidate in candidates {
        let trimmed = candidate.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !parts.iter().any(|existing: &String| existing == trimmed) {
            parts.push(trimmed.to_string());
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" | "))
    }
}

fn infer_platform_from_metadata(values: &[String]) -> Option<Platform> {
    for value in values {
        if let Ok(platform) = value.parse::<Platform>() {
            if platform != Platform::Unknown {
                return Some(platform);
            }
        }
        let normalized = normalize_header(value);
        let inferred = if normalized.contains("illumina")
            || normalized.contains("novaseq")
            || normalized.contains("nextseq")
            || normalized.contains("miseq")
            || normalized.contains("hiseq")
        {
            Some(Platform::Illumina)
        } else if normalized.contains("mgi")
            || normalized.contains("dnbseq")
            || normalized.contains("mgiseq")
        {
            Some(Platform::Mgi)
        } else if normalized.contains("nanopore")
            || normalized.contains("promethion")
            || normalized.contains("gridion")
            || normalized.contains("minion")
            || normalized == "ont"
        {
            Some(Platform::Ont)
        } else if normalized.contains("pacbio")
            || normalized.contains("sequel")
            || normalized.contains("revio")
            || normalized.contains("rsii")
        {
            Some(Platform::PacBio)
        } else {
            None
        };
        if inferred.is_some() {
            return inferred;
        }
    }
    None
}

fn infer_experiment_from_metadata(
    values: &[String],
    platform_hint: Option<Platform>,
) -> Option<ExperimentType> {
    let normalized_values: Vec<String> =
        values.iter().map(|value| normalize_header(value)).collect();

    if normalized_values.iter().any(|normalized| {
        normalized.contains("10x") && (normalized.contains("v2") || normalized.contains("version2"))
    }) {
        return Some(ExperimentType::SingleCell10xV2);
    }

    if normalized_values.iter().any(|normalized| {
        normalized.contains("10x")
            && (normalized.contains("v3")
                || normalized.contains("version3")
                || normalized.contains("3prime"))
    }) {
        return Some(ExperimentType::SingleCell10xV3);
    }

    if normalized_values
        .iter()
        .any(|normalized| normalized.contains("atac"))
    {
        return Some(ExperimentType::AtacSeq);
    }

    if normalized_values.iter().any(|normalized| {
        normalized.contains("rnaseq")
            || normalized.contains("transcriptome")
            || normalized.contains("mrna")
    }) {
        return Some(ExperimentType::RnaSeq);
    }

    if normalized_values.iter().any(|normalized| {
        normalized.contains("wgs")
            || normalized.contains("wholegenome")
            || normalized.contains("genome")
    }) {
        return Some(ExperimentType::Wgs);
    }

    if normalized_values
        .iter()
        .any(|normalized| normalized.contains("longread"))
    {
        return Some(ExperimentType::LongRead);
    }
    match platform_hint {
        Some(Platform::Ont | Platform::PacBio) => Some(ExperimentType::LongRead),
        _ => None,
    }
}

fn text_fields_from_cells(cells: &[String], indices: &[Option<usize>]) -> Vec<String> {
    let mut values = Vec::new();
    for index in indices.iter().flatten() {
        let value = cell(cells, *index).trim();
        if !value.is_empty() {
            values.push(value.to_string());
        }
    }
    values
}

fn discover_file_entry(path: &Path, root: &Path) -> Result<DiscoveredFile> {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .with_context(|| format!("FASTQ path {} is not valid UTF-8", path.display()))?;
    let stem = fastq_stem(file_name);
    let (pair_base, mate, note) = parse_mate_signature(&stem);
    let accession = extract_accession_token(&stem).or_else(|| extract_accession_token(&pair_base));
    let dataset_id = sanitize_filename(accession.as_deref().unwrap_or(&pair_base));
    let relative_parent = path
        .parent()
        .and_then(|parent| parent.strip_prefix(root).ok())
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    let grouping_key = format!("{relative_parent}::{pair_base}");
    Ok(DiscoveredFile {
        path: path.to_path_buf(),
        grouping_key,
        dataset_id,
        accession,
        mate,
        note,
    })
}

fn fastq_stem(file_name: &str) -> String {
    let lowered = file_name.to_ascii_lowercase();
    if lowered.ends_with(".fastq.gz") {
        file_name[..file_name.len() - ".fastq.gz".len()].to_string()
    } else if lowered.ends_with(".fq.gz") {
        file_name[..file_name.len() - ".fq.gz".len()].to_string()
    } else if lowered.ends_with(".fastq") {
        file_name[..file_name.len() - ".fastq".len()].to_string()
    } else if lowered.ends_with(".fq") {
        file_name[..file_name.len() - ".fq".len()].to_string()
    } else {
        file_name.to_string()
    }
}

fn parse_mate_signature(stem: &str) -> (String, Option<u8>, Option<String>) {
    const PAIR_SUFFIXES: &[(&str, u8)] = &[
        ("_r1_001", 1),
        ("_r2_001", 2),
        ("_r1", 1),
        ("_r2", 2),
        (".r1", 1),
        (".r2", 2),
        ("-r1", 1),
        ("-r2", 2),
        ("_1", 1),
        ("_2", 2),
        (".1", 1),
        (".2", 2),
        ("-1", 1),
        ("-2", 2),
    ];

    let lowered = stem.to_ascii_lowercase();
    for (suffix, mate) in PAIR_SUFFIXES {
        if lowered.ends_with(suffix) && stem.len() > suffix.len() {
            let base = stem[..stem.len() - suffix.len()].to_string();
            return (
                base,
                Some(*mate),
                Some(format!(
                    "Detected mate suffix pattern {}",
                    suffix.to_ascii_uppercase()
                )),
            );
        }
    }
    (stem.to_string(), None, None)
}

fn extract_accession_token(value: &str) -> Option<String> {
    extract_accession_tokens(value).into_iter().next()
}

fn extract_accession_tokens(value: &str) -> Vec<String> {
    dedupe_preserving_order(
        value
            .split(|character: char| !character.is_ascii_alphanumeric())
            .map(|token| token.trim())
            .filter_map(|token| {
                normalize_accession_token_with_prefixes(token, PUBLIC_ACCESSION_PREFIXES)
            }),
    )
}

fn extract_non_geo_public_accession_tokens(value: &str) -> Vec<String> {
    dedupe_preserving_order(
        value
            .split(|character: char| !character.is_ascii_alphanumeric())
            .map(|token| token.trim())
            .filter_map(|token| {
                normalize_accession_token_with_prefixes(token, NON_GEO_PUBLIC_ACCESSION_PREFIXES)
            }),
    )
}

fn normalize_accession_token_with_prefixes(token: &str, prefixes: &[&str]) -> Option<String> {
    let token_upper = token.trim().to_ascii_uppercase();
    prefixes.iter().find_map(|prefix| {
        if token_upper.starts_with(prefix)
            && token_upper.len() > prefix.len()
            && token_upper[prefix.len()..]
                .chars()
                .all(|character| character.is_ascii_digit())
        {
            Some(token_upper.clone())
        } else {
            None
        }
    })
}

fn is_geo_accession(accession: &str) -> bool {
    let normalized = accession.trim().to_ascii_uppercase();
    normalized.starts_with("GSE") || normalized.starts_with("GSM")
}

#[derive(Debug, Clone)]
struct FetchedPublicMetadataRow {
    run_accession: String,
    query_accessions: Vec<String>,
    fields: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetadataChunkSource {
    Cache,
    Remote,
}

impl MetadataChunkSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cache => "cache",
            Self::Remote => "remote",
        }
    }
}

#[derive(Debug, Clone)]
struct MetadataChunkFetchResult {
    rows: Vec<FetchedPublicMetadataRow>,
    source: MetadataChunkSource,
    attempts: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FetchAccessionStatusKind {
    Matched,
    Unmatched,
    FetchFailed,
}

impl FetchAccessionStatusKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Matched => "matched",
            Self::Unmatched => "unmatched",
            Self::FetchFailed => "fetch_failed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeoBridgeSurface {
    QuickTextSelf,
    FullTextSelf,
}

impl GeoBridgeSurface {
    fn label(self) -> &'static str {
        match self {
            Self::QuickTextSelf => "quick_text_self",
            Self::FullTextSelf => "full_text_self",
        }
    }

    fn query_target(self) -> &'static str {
        "self"
    }

    fn query_form(self) -> &'static str {
        "text"
    }

    fn query_view(self) -> &'static str {
        match self {
            Self::QuickTextSelf => "quick",
            Self::FullTextSelf => "full",
        }
    }
}

const GEO_BRIDGE_SURFACES: &[GeoBridgeSurface] = &[
    GeoBridgeSurface::QuickTextSelf,
    GeoBridgeSurface::FullTextSelf,
];

#[derive(Debug, Clone)]
struct FetchAccessionStatusRow {
    accession: String,
    status: FetchAccessionStatusKind,
    resolved_accessions: Vec<String>,
    matched_runs: Vec<String>,
    geo_surface: Option<String>,
    source: Option<MetadataChunkSource>,
    attempts: usize,
    error: Option<String>,
}

#[derive(Debug, Clone)]
struct MetadataChunkFetchError {
    attempts: usize,
    message: String,
}

#[derive(Debug, Clone)]
struct GeoBridgeFetchResult {
    resolved_accessions: Vec<String>,
    geo_surface: Option<String>,
    source: MetadataChunkSource,
    attempts: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StudyDownloadStatusKind {
    Downloaded,
    Resumed,
    SkippedExisting,
    Failed,
}

impl StudyDownloadStatusKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Downloaded => "downloaded",
            Self::Resumed => "resumed",
            Self::SkippedExisting => "skipped_existing",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone)]
struct StudyDownloadStatusRow {
    dataset_id: String,
    accession: Option<String>,
    destination: PathBuf,
    source_url: Option<String>,
    status: StudyDownloadStatusKind,
    bytes: u64,
    attempts: usize,
    error: Option<String>,
}

#[derive(Debug, Clone)]
struct PublicDownloadTarget {
    dataset_id: String,
    accession: Option<String>,
    destination: PathBuf,
    source_url: String,
}

#[derive(Debug, Clone)]
struct DownloadExecutionOutcome {
    status: StudyDownloadStatusKind,
    bytes: u64,
    attempts: usize,
}

fn read_requested_accessions(path: &Path) -> Result<Vec<String>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read metadata input {}", path.display()))?;
    let filtered_lines: Vec<(usize, String)> = raw
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                None
            } else {
                Some((index + 1, line.to_string()))
            }
        })
        .collect();
    if filtered_lines.is_empty() {
        return Ok(Vec::new());
    }

    let first_line = filtered_lines[0].1.as_str();
    if first_line.contains('\t') || first_line.contains(',') {
        let delimiter = if first_line.contains('\t') { '\t' } else { ',' };
        let headers = split_manifest_row(first_line, delimiter);
        if looks_like_accession_source_header(&headers) {
            return read_requested_accessions_from_table(
                path,
                &headers,
                &filtered_lines[1..],
                delimiter,
            );
        }
    }

    Ok(filtered_lines
        .iter()
        .flat_map(|(_, line)| extract_accession_tokens(line))
        .collect())
}

fn looks_like_accession_source_header(headers: &[String]) -> bool {
    optional_column(
        headers,
        &[
            "run_accession",
            "accession",
            "run",
            "run_id",
            "study_accession",
            "experiment_accession",
            "sample_accession",
            "geo_accession",
            "series_accession",
            "sample_geo_accession",
            "bioproject",
            "project_accession",
            "project",
            "dataset_id",
            "input1",
            "input2",
            "fastq_ftp",
            "submitted_ftp",
            "fastq_aspera",
            "submitted_aspera",
            "fastq_1",
            "fastq_2",
        ],
    )
    .is_some()
}

fn read_requested_accessions_from_table(
    path: &Path,
    headers: &[String],
    lines: &[(usize, String)],
    delimiter: char,
) -> Result<Vec<String>> {
    let run_accession_index =
        optional_column(headers, &["run_accession", "accession", "run", "run_id"]);
    let experiment_accession_index = optional_column(headers, &["experiment_accession"]);
    let sample_accession_index = optional_column(headers, &["sample_accession"]);
    let study_accession_index = optional_column(headers, &["study_accession", "study", "srastudy"]);
    let geo_accession_index = optional_column(
        headers,
        &[
            "geo_accession",
            "series_accession",
            "sample_geo_accession",
            "geo_series_accession",
            "geo_sample_accession",
            "gse",
            "gsm",
        ],
    );
    let project_accession_index = optional_column(
        headers,
        &[
            "bioproject",
            "project_accession",
            "project",
            "secondary_study_accession",
        ],
    );
    let dataset_id_index = optional_column(headers, &["dataset_id"]);
    let input1_index = optional_column(headers, &["input1"]);
    let input2_index = optional_column(headers, &["input2"]);
    let fastq_ftp_index = optional_column(headers, &["fastq_ftp", "ftp"]);
    let submitted_ftp_index = optional_column(headers, &["submitted_ftp"]);
    let fastq_aspera_index = optional_column(headers, &["fastq_aspera", "aspera"]);
    let submitted_aspera_index = optional_column(headers, &["submitted_aspera"]);
    let fastq_1_index = optional_column(
        headers,
        &[
            "fastq_1",
            "fastq_file_1",
            "filename_1",
            "file_1",
            "read1_ftp",
            "ftp_1",
        ],
    );
    let fastq_2_index = optional_column(
        headers,
        &[
            "fastq_2",
            "fastq_file_2",
            "filename_2",
            "file_2",
            "read2_ftp",
            "ftp_2",
        ],
    );
    let query_accessions_index =
        optional_column(headers, &["query_accessions", "requested_accessions"]);
    if run_accession_index.is_none()
        && experiment_accession_index.is_none()
        && sample_accession_index.is_none()
        && study_accession_index.is_none()
        && geo_accession_index.is_none()
        && project_accession_index.is_none()
        && dataset_id_index.is_none()
        && input1_index.is_none()
        && input2_index.is_none()
        && fastq_ftp_index.is_none()
        && submitted_ftp_index.is_none()
        && fastq_aspera_index.is_none()
        && submitted_aspera_index.is_none()
        && fastq_1_index.is_none()
        && fastq_2_index.is_none()
        && query_accessions_index.is_none()
    {
        bail!(
            "metadata input {} does not expose any accession-bearing columns",
            path.display()
        );
    }

    let mut accessions = Vec::new();
    for (_, line) in lines {
        let cells = split_manifest_row(line, delimiter);
        let mut primary_accession = None;
        for index in [
            run_accession_index,
            experiment_accession_index,
            sample_accession_index,
            study_accession_index,
            geo_accession_index,
            project_accession_index,
        ]
        .into_iter()
        .flatten()
        {
            if let Some(accession) = extract_accession_token(cell(&cells, index)) {
                primary_accession = Some(accession);
                break;
            }
        }
        if let Some(accession) = primary_accession {
            accessions.push(accession);
            continue;
        }

        accessions.extend(dedupe_preserving_order(
            text_fields_from_cells(
                &cells,
                &[
                    dataset_id_index,
                    input1_index,
                    input2_index,
                    study_accession_index,
                    experiment_accession_index,
                    sample_accession_index,
                    geo_accession_index,
                    project_accession_index,
                    fastq_1_index,
                    fastq_2_index,
                    fastq_ftp_index,
                    submitted_ftp_index,
                    fastq_aspera_index,
                    submitted_aspera_index,
                    query_accessions_index,
                ],
            )
            .into_iter()
            .flat_map(|value| extract_accession_tokens(&value)),
        ));
    }
    Ok(accessions)
}

fn dedupe_preserving_order<I>(values: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for value in values {
        if seen.insert(value.clone()) {
            deduped.push(value);
        }
    }
    deduped
}

fn default_study_fetch_cache_dir(output_path: &Path) -> PathBuf {
    let stem = output_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("study_fetch_metadata");
    output_path.with_file_name(format!("{stem}.cache"))
}

fn metadata_chunk_cache_path(cache_dir: &Path, base_url: &str, accessions: &[String]) -> PathBuf {
    let mut normalized_accessions = accessions.to_vec();
    normalized_accessions.sort();
    let cache_key = stable_metadata_chunk_cache_key(base_url, &normalized_accessions);
    cache_dir.join(format!("{cache_key}.tsv"))
}

fn geo_bridge_cache_path(cache_dir: &Path, geo_base_url: &str, accession: &str) -> PathBuf {
    let cache_key = stable_metadata_chunk_cache_key(geo_base_url, &[accession.to_string()]);
    cache_dir.join(format!("geo-{cache_key}.txt"))
}

fn stable_metadata_chunk_cache_key(base_url: &str, accessions: &[String]) -> String {
    let mut state = 0xcbf29ce484222325u64;
    stable_hash_update(&mut state, base_url.as_bytes());
    stable_hash_update(&mut state, b"\x1fread_run\x1f");
    stable_hash_update(&mut state, ENA_FILEREPORT_FIELDS.join(",").as_bytes());
    for accession in accessions {
        stable_hash_update(&mut state, b"\x1e");
        stable_hash_update(&mut state, accession.as_bytes());
    }
    format!("{state:016x}")
}

fn stable_hash_update(state: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *state ^= u64::from(*byte);
        *state = state.wrapping_mul(0x100000001b3);
    }
}

fn fetch_metadata_chunk(
    accessions: &[String],
    base_url: &str,
    cache_dir: &Path,
    retries: usize,
) -> Result<MetadataChunkFetchResult, MetadataChunkFetchError> {
    let cache_path = metadata_chunk_cache_path(cache_dir, base_url, accessions);
    if cache_path.exists() {
        if let Ok(raw) = fs::read_to_string(&cache_path) {
            if let Ok(rows) = parse_fetched_public_metadata_rows(
                &raw,
                &cache_path.display().to_string(),
                accessions,
            ) {
                return Ok(MetadataChunkFetchResult {
                    rows,
                    source: MetadataChunkSource::Cache,
                    attempts: 0,
                });
            }
        }
    }

    let mut attempts = 0usize;
    let mut last_error = None;
    for _ in 0..=retries {
        attempts += 1;
        match fetch_ena_metadata_chunk(accessions, base_url).and_then(|raw| {
            let rows = parse_fetched_public_metadata_rows(&raw, base_url, accessions)?;
            fs::write(&cache_path, raw)
                .with_context(|| format!("failed to write {}", cache_path.display()))?;
            Ok(rows)
        }) {
            Ok(rows) => {
                return Ok(MetadataChunkFetchResult {
                    rows,
                    source: MetadataChunkSource::Remote,
                    attempts,
                });
            }
            Err(error) => last_error = Some(error.to_string()),
        }
    }

    Err(MetadataChunkFetchError {
        attempts,
        message: last_error.unwrap_or_else(|| "unknown metadata fetch failure".to_string()),
    })
}

fn resolve_geo_accession(
    accession: &str,
    geo_base_url: &str,
    cache_dir: &Path,
    retries: usize,
) -> Result<GeoBridgeFetchResult, MetadataChunkFetchError> {
    let cache_path = geo_bridge_cache_path(cache_dir, geo_base_url, accession);
    if cache_path.exists() {
        if let Ok(raw) = fs::read_to_string(&cache_path) {
            let resolved = parse_geo_bridge_accessions(&raw);
            return Ok(GeoBridgeFetchResult {
                resolved_accessions: resolved,
                geo_surface: parse_geo_bridge_selected_surface(&raw),
                source: MetadataChunkSource::Cache,
                attempts: 0,
            });
        }
    }

    let mut attempts = 0usize;
    let mut last_error = None;
    for _ in 0..=retries {
        attempts += 1;
        match fetch_geo_bridge_bundle(accession, geo_base_url).and_then(
            |(raw, resolved, geo_surface)| {
                fs::write(&cache_path, raw)
                    .with_context(|| format!("failed to write {}", cache_path.display()))?;
                Ok((resolved, geo_surface))
            },
        ) {
            Ok((resolved_accessions, geo_surface)) => {
                return Ok(GeoBridgeFetchResult {
                    resolved_accessions,
                    geo_surface,
                    source: MetadataChunkSource::Remote,
                    attempts,
                });
            }
            Err(error) => last_error = Some(error.to_string()),
        }
    }

    Err(MetadataChunkFetchError {
        attempts,
        message: last_error.unwrap_or_else(|| "unknown GEO bridge failure".to_string()),
    })
}

fn fetch_geo_bridge_bundle(
    accession: &str,
    geo_base_url: &str,
) -> Result<(String, Vec<String>, Option<String>)> {
    let mut responses = Vec::new();
    for surface in GEO_BRIDGE_SURFACES {
        let raw = fetch_geo_bridge_surface_text(accession, geo_base_url, *surface)?;
        let resolved = parse_geo_bridge_accessions(&raw);
        responses.push((*surface, raw));
        if !resolved.is_empty() {
            return Ok((
                serialize_geo_bridge_cache_bundle(Some(*surface), &responses),
                resolved,
                Some(surface.label().to_string()),
            ));
        }
    }
    let last_surface = responses
        .last()
        .map(|(surface, _)| surface.label().to_string());
    Ok((
        serialize_geo_bridge_cache_bundle(None, &responses),
        Vec::new(),
        last_surface,
    ))
}

fn serialize_geo_bridge_cache_bundle(
    selected_surface: Option<GeoBridgeSurface>,
    responses: &[(GeoBridgeSurface, String)],
) -> String {
    let mut bundle = String::new();
    bundle.push_str("# geo_bridge_selected_surface=");
    bundle.push_str(
        selected_surface
            .map(GeoBridgeSurface::label)
            .unwrap_or("none"),
    );
    bundle.push('\n');
    for (index, (surface, raw)) in responses.iter().enumerate() {
        bundle.push_str("# geo_bridge_surface=");
        bundle.push_str(surface.label());
        bundle.push('\n');
        bundle.push_str(raw.trim_end_matches('\n'));
        bundle.push('\n');
        if index + 1 < responses.len() {
            bundle.push('\n');
        }
    }
    bundle
}

fn parse_geo_bridge_selected_surface(raw: &str) -> Option<String> {
    raw.lines()
        .find_map(|line| line.strip_prefix("# geo_bridge_selected_surface="))
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != "none")
        .map(str::to_string)
}

fn fetch_geo_bridge_surface_text(
    accession: &str,
    geo_base_url: &str,
    surface: GeoBridgeSurface,
) -> Result<String> {
    let output = Command::new("curl")
        .arg("--fail")
        .arg("--silent")
        .arg("--show-error")
        .arg("--location")
        .arg("--get")
        .arg(geo_base_url)
        .arg("--data-urlencode")
        .arg(format!("acc={accession}"))
        .arg("--data-urlencode")
        .arg(format!("targ={}", surface.query_target()))
        .arg("--data-urlencode")
        .arg(format!("form={}", surface.query_form()))
        .arg("--data-urlencode")
        .arg(format!("view={}", surface.query_view()))
        .output()
        .with_context(|| "failed to launch curl for GEO bridge fetch".to_string())?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        bail!(
            "curl GEO bridge fetch failed for {} [{}]: {}",
            accession,
            surface.label(),
            if stderr.is_empty() {
                format!("exit status {}", output.status)
            } else {
                stderr
            }
        );
    }
    String::from_utf8(output.stdout)
        .with_context(|| format!("GEO bridge response for {} was not valid UTF-8", accession))
}

fn parse_geo_bridge_accessions(raw: &str) -> Vec<String> {
    dedupe_preserving_order(
        raw.lines()
            .flat_map(extract_non_geo_public_accession_tokens),
    )
}

fn fetch_ena_metadata_chunk(accessions: &[String], base_url: &str) -> Result<String> {
    let joined_accessions = accessions.join(",");
    let batch_response = fetch_ena_metadata_request(&joined_accessions, base_url)?;
    if accessions.len() <= 1 || metadata_response_has_data_rows(&batch_response) {
        return Ok(batch_response);
    }

    let mut per_accession_responses = Vec::with_capacity(accessions.len());
    for accession in accessions {
        per_accession_responses.push(fetch_ena_metadata_request(accession, base_url)?);
    }
    Ok(merge_metadata_response_bodies(&per_accession_responses).unwrap_or(batch_response))
}

fn fetch_ena_metadata_request(accession_query: &str, base_url: &str) -> Result<String> {
    let output = Command::new("curl")
        .arg("--fail")
        .arg("--silent")
        .arg("--show-error")
        .arg("--location")
        .arg("--get")
        .arg(base_url)
        .arg("--data-urlencode")
        .arg(format!("accession={accession_query}"))
        .arg("--data-urlencode")
        .arg("result=read_run")
        .arg("--data-urlencode")
        .arg(format!("fields={}", ENA_FILEREPORT_FIELDS.join(",")))
        .arg("--data-urlencode")
        .arg("format=tsv")
        .arg("--data-urlencode")
        .arg("download=false")
        .output()
        .with_context(|| "failed to launch curl for public metadata fetch".to_string())?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        bail!(
            "curl metadata fetch failed for accession chunk [{}]: {}",
            accession_query,
            if stderr.is_empty() {
                format!("exit status {}", output.status)
            } else {
                stderr
            }
        );
    }
    String::from_utf8(output.stdout).with_context(|| {
        format!(
            "metadata response for [{}] was not valid UTF-8",
            accession_query
        )
    })
}

fn metadata_response_has_data_rows(raw: &str) -> bool {
    raw.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .nth(1)
        .is_some()
}

fn merge_metadata_response_bodies(response_bodies: &[String]) -> Option<String> {
    let mut header = None::<String>;
    let mut rows = Vec::new();
    for body in response_bodies {
        let filtered_lines: Vec<&str> = body
            .lines()
            .map(str::trim_end)
            .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
            .collect();
        if filtered_lines.is_empty() {
            continue;
        }
        if header.is_none() {
            header = Some(filtered_lines[0].to_string());
        }
        for line in filtered_lines.into_iter().skip(1) {
            rows.push(line.to_string());
        }
    }

    header.map(|header| {
        let mut merged = header;
        merged.push('\n');
        if !rows.is_empty() {
            merged.push_str(&rows.join("\n"));
            merged.push('\n');
        }
        merged
    })
}

fn parse_fetched_public_metadata_rows(
    raw: &str,
    source_description: &str,
    requested_accessions: &[String],
) -> Result<Vec<FetchedPublicMetadataRow>> {
    let filtered_lines: Vec<&str> = raw
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
        .collect();
    if filtered_lines.is_empty() {
        return Ok(Vec::new());
    }

    let header_line = filtered_lines[0];
    let delimiter = if header_line.contains('\t') {
        '\t'
    } else {
        ','
    };
    let headers = split_manifest_row(header_line, delimiter);
    let run_accession_index = optional_column(&headers, &["run_accession", "accession", "run"])
        .with_context(|| {
            format!(
                "metadata source {} did not return a run_accession/accession column",
                source_description
            )
        })?;
    let study_accession_index = optional_column(&headers, &["study_accession"]);
    let study_title_index = optional_column(&headers, &["study_title"]);
    let experiment_accession_index = optional_column(&headers, &["experiment_accession"]);
    let experiment_title_index = optional_column(&headers, &["experiment_title"]);
    let sample_accession_index = optional_column(&headers, &["sample_accession"]);
    let instrument_platform_index = optional_column(&headers, &["instrument_platform"]);
    let instrument_model_index = optional_column(&headers, &["instrument_model"]);
    let library_strategy_index = optional_column(&headers, &["library_strategy"]);
    let library_layout_index = optional_column(&headers, &["library_layout"]);
    let library_source_index = optional_column(&headers, &["library_source"]);
    let library_selection_index = optional_column(&headers, &["library_selection"]);
    let bioproject_index = optional_column(
        &headers,
        &[
            "secondary_study_accession",
            "bioproject",
            "project_accession",
            "project",
        ],
    );
    let sample_title_index = optional_column(&headers, &["sample_title", "sample_alias"]);
    let fastq_ftp_index = optional_column(&headers, &["fastq_ftp"]);
    let fastq_aspera_index = optional_column(&headers, &["fastq_aspera"]);
    let submitted_ftp_index = optional_column(&headers, &["submitted_ftp"]);
    let submitted_aspera_index = optional_column(&headers, &["submitted_aspera"]);
    let query_accessions_index = optional_column(&headers, &["query_accessions"]);

    let mut rows = Vec::new();
    for line in filtered_lines.iter().skip(1) {
        let cells = split_manifest_row(line, delimiter);
        let run_accession = cell(&cells, run_accession_index).trim();
        if run_accession.is_empty() {
            continue;
        }
        let mut row = FetchedPublicMetadataRow {
            run_accession: run_accession.to_string(),
            query_accessions: Vec::new(),
            fields: HashMap::new(),
        };
        insert_fetched_public_metadata_field(&mut row, "run_accession", Some(run_accession));
        insert_fetched_public_metadata_field(
            &mut row,
            "study_accession",
            optional_text_field(&cells, study_accession_index).as_deref(),
        );
        insert_fetched_public_metadata_field(
            &mut row,
            "study_title",
            optional_text_field(&cells, study_title_index).as_deref(),
        );
        insert_fetched_public_metadata_field(
            &mut row,
            "experiment_accession",
            optional_text_field(&cells, experiment_accession_index).as_deref(),
        );
        insert_fetched_public_metadata_field(
            &mut row,
            "experiment_title",
            optional_text_field(&cells, experiment_title_index).as_deref(),
        );
        insert_fetched_public_metadata_field(
            &mut row,
            "sample_accession",
            optional_text_field(&cells, sample_accession_index).as_deref(),
        );
        insert_fetched_public_metadata_field(
            &mut row,
            "instrument_platform",
            optional_text_field(&cells, instrument_platform_index).as_deref(),
        );
        insert_fetched_public_metadata_field(
            &mut row,
            "instrument_model",
            optional_text_field(&cells, instrument_model_index).as_deref(),
        );
        insert_fetched_public_metadata_field(
            &mut row,
            "library_strategy",
            optional_text_field(&cells, library_strategy_index).as_deref(),
        );
        insert_fetched_public_metadata_field(
            &mut row,
            "library_layout",
            optional_text_field(&cells, library_layout_index).as_deref(),
        );
        insert_fetched_public_metadata_field(
            &mut row,
            "library_source",
            optional_text_field(&cells, library_source_index).as_deref(),
        );
        insert_fetched_public_metadata_field(
            &mut row,
            "library_selection",
            optional_text_field(&cells, library_selection_index).as_deref(),
        );
        insert_fetched_public_metadata_field(
            &mut row,
            "bioproject",
            optional_text_field(&cells, bioproject_index).as_deref(),
        );
        insert_fetched_public_metadata_field(
            &mut row,
            "sample_title",
            optional_text_field(&cells, sample_title_index).as_deref(),
        );
        insert_fetched_public_metadata_field(
            &mut row,
            "fastq_ftp",
            optional_text_field(&cells, fastq_ftp_index).as_deref(),
        );
        insert_fetched_public_metadata_field(
            &mut row,
            "fastq_aspera",
            optional_text_field(&cells, fastq_aspera_index).as_deref(),
        );
        insert_fetched_public_metadata_field(
            &mut row,
            "submitted_ftp",
            optional_text_field(&cells, submitted_ftp_index).as_deref(),
        );
        insert_fetched_public_metadata_field(
            &mut row,
            "submitted_aspera",
            optional_text_field(&cells, submitted_aspera_index).as_deref(),
        );
        row.query_accessions = query_accessions_index
            .map(|index| split_query_accessions_field(cell(&cells, index)))
            .unwrap_or_default();
        if row.query_accessions.is_empty() {
            row.query_accessions = requested_accessions
                .iter()
                .filter(|accession| fetched_public_metadata_row_matches_accession(&row, accession))
                .cloned()
                .collect();
        }
        if row.query_accessions.is_empty() {
            continue;
        }
        synchronize_fetched_public_metadata_query_accessions(&mut row);
        rows.push(row);
    }
    Ok(rows)
}

fn insert_fetched_public_metadata_field(
    row: &mut FetchedPublicMetadataRow,
    key: &str,
    value: Option<&str>,
) {
    if let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) {
        row.fields.insert(key.to_string(), value.to_string());
    }
}

fn split_query_accessions_field(value: &str) -> Vec<String> {
    dedupe_preserving_order(
        value
            .split(';')
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(str::to_string),
    )
}

fn fetched_public_metadata_row_matches_accession(
    row: &FetchedPublicMetadataRow,
    accession: &str,
) -> bool {
    let normalized_accession = normalize_lookup_key(accession);
    FETCHED_PUBLIC_METADATA_MATCH_FIELDS.iter().any(|field| {
        row.fields.get(*field).is_some_and(|value| {
            extract_accession_tokens(value)
                .into_iter()
                .any(|token| normalize_lookup_key(&token) == normalized_accession)
        })
    })
}

fn synchronize_fetched_public_metadata_query_accessions(row: &mut FetchedPublicMetadataRow) {
    row.query_accessions = dedupe_preserving_order(row.query_accessions.drain(..));
    if row.query_accessions.is_empty() {
        row.fields.remove("query_accessions");
    } else {
        row.fields.insert(
            "query_accessions".to_string(),
            row.query_accessions.join(";"),
        );
    }
}

fn read_existing_fetched_public_metadata_rows(
    path: &Path,
    requested_accessions: &HashSet<String>,
) -> Result<Vec<FetchedPublicMetadataRow>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read existing metadata output {}", path.display()))?;
    let mut rows = parse_fetched_public_metadata_rows(&raw, &path.display().to_string(), &[])?;
    rows.retain(|row| {
        row.query_accessions
            .iter()
            .any(|accession| requested_accessions.contains(accession))
    });
    Ok(rows)
}

fn read_existing_fetch_status_rows(
    path: &Path,
    requested_accessions: &HashSet<String>,
) -> Result<HashMap<String, FetchAccessionStatusRow>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read existing fetch status {}", path.display()))?;
    let filtered_lines: Vec<&str> = raw
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
        .collect();
    if filtered_lines.is_empty() {
        return Ok(HashMap::new());
    }
    let header_line = filtered_lines[0];
    let delimiter = if header_line.contains('\t') {
        '\t'
    } else {
        ','
    };
    let headers = split_manifest_row(header_line, delimiter);
    let accession_index = required_column(&headers, &["accession"]).with_context(|| {
        format!(
            "existing fetch status {} is missing accession",
            path.display()
        )
    })?;
    let status_index = required_column(&headers, &["status"])
        .with_context(|| format!("existing fetch status {} is missing status", path.display()))?;
    let matched_runs_index = optional_column(&headers, &["matched_runs"]);
    let source_index = optional_column(&headers, &["source"]);
    let attempts_index = optional_column(&headers, &["attempts"]);
    let error_index = optional_column(&headers, &["error"]);
    let geo_surface_index = optional_column(&headers, &["geo_surface"]);
    let resolved_accessions_index = optional_column(&headers, &["resolved_accessions"]);

    let mut rows = HashMap::new();
    for line in filtered_lines.iter().skip(1) {
        let cells = split_manifest_row(line, delimiter);
        let accession = cell(&cells, accession_index).trim().to_string();
        if accession.is_empty() || !requested_accessions.contains(&accession) {
            continue;
        }
        let status = parse_fetch_accession_status_kind(cell(&cells, status_index), path)?;
        let matched_runs = matched_runs_index
            .map(|index| split_query_accessions_field(cell(&cells, index)))
            .unwrap_or_default();
        let resolved_accessions = resolved_accessions_index
            .map(|index| split_query_accessions_field(cell(&cells, index)))
            .unwrap_or_default();
        let source = source_index
            .map(|index| parse_metadata_chunk_source(cell(&cells, index), path))
            .transpose()?;
        let attempts = attempts_index
            .map(|index| parse_existing_status_attempts(cell(&cells, index), path))
            .transpose()?
            .flatten()
            .unwrap_or(0);
        let error = error_index
            .map(|index| cell(&cells, index).trim())
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        let geo_surface = geo_surface_index
            .map(|index| cell(&cells, index).trim())
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        rows.insert(
            accession.clone(),
            FetchAccessionStatusRow {
                accession,
                status,
                resolved_accessions,
                matched_runs,
                geo_surface,
                source,
                attempts,
                error,
            },
        );
    }
    Ok(rows)
}

fn parse_fetch_accession_status_kind(value: &str, path: &Path) -> Result<FetchAccessionStatusKind> {
    match value.trim() {
        "matched" => Ok(FetchAccessionStatusKind::Matched),
        "unmatched" => Ok(FetchAccessionStatusKind::Unmatched),
        "fetch_failed" => Ok(FetchAccessionStatusKind::FetchFailed),
        other => bail!(
            "existing fetch status {} contains unknown status '{}'",
            path.display(),
            other
        ),
    }
}

fn parse_metadata_chunk_source(value: &str, path: &Path) -> Result<MetadataChunkSource> {
    match value.trim() {
        "cache" => Ok(MetadataChunkSource::Cache),
        "remote" => Ok(MetadataChunkSource::Remote),
        other => bail!(
            "existing fetch status {} contains unknown source '{}'",
            path.display(),
            other
        ),
    }
}

fn parse_existing_status_attempts(value: &str, path: &Path) -> Result<Option<usize>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    trimmed.parse::<usize>().map(Some).with_context(|| {
        format!(
            "existing fetch status {} contains invalid attempts value '{}'",
            path.display(),
            trimmed
        )
    })
}

fn merge_fetched_public_metadata_row(
    existing: &mut FetchedPublicMetadataRow,
    incoming: FetchedPublicMetadataRow,
) -> Result<()> {
    if existing.run_accession != incoming.run_accession {
        bail!(
            "cannot merge fetched metadata rows for different accessions: {} vs {}",
            existing.run_accession,
            incoming.run_accession
        );
    }
    existing
        .query_accessions
        .extend(incoming.query_accessions.iter().cloned());
    synchronize_fetched_public_metadata_query_accessions(existing);
    for header in FETCHED_PUBLIC_METADATA_HEADERS {
        if *header == "query_accessions" {
            continue;
        }
        let incoming_value = incoming
            .fields
            .get(*header)
            .map(String::as_str)
            .unwrap_or("");
        if incoming_value.is_empty() {
            continue;
        }
        match existing.fields.get_mut(*header) {
            None => {
                existing
                    .fields
                    .insert((*header).to_string(), incoming_value.to_string());
            }
            Some(existing_value) if existing_value.trim().is_empty() => {
                *existing_value = incoming_value.to_string();
            }
            Some(existing_value) if existing_value == incoming_value => {}
            Some(existing_value) => {
                bail!(
                    "conflicting fetched metadata values for accession {} column {}: '{}' vs '{}'",
                    existing.run_accession,
                    header,
                    existing_value,
                    incoming_value
                );
            }
        }
    }
    Ok(())
}

fn read_manifest_source(
    path: &Path,
    require_dataset_id: bool,
    download_root: Option<&Path>,
) -> Result<Vec<ManifestSourceEntry>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read manifest {}", path.display()))?;
    let filtered_lines: Vec<(usize, String)> = raw
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                None
            } else {
                Some((index + 1, line.to_string()))
            }
        })
        .collect();
    let (header_line_number, header_line) = filtered_lines
        .first()
        .with_context(|| format!("manifest {} is empty", path.display()))?;
    let delimiter = if header_line.contains('\t') {
        '\t'
    } else {
        ','
    };
    let headers = split_manifest_row(header_line, delimiter);
    let data_lines = &filtered_lines[1..];
    if optional_column(&headers, &["input1"]).is_none() {
        let Some(download_root) = download_root else {
            bail!(
                "manifest {} line {header_line_number} is missing required manifest column input1",
                path.display()
            );
        };
        return read_accession_manifest_source(
            path,
            &headers,
            data_lines,
            delimiter,
            require_dataset_id,
            download_root,
        );
    }

    let dataset_id_index =
        if require_dataset_id {
            Some(required_column(&headers, &["dataset_id"]).with_context(|| {
                format!("manifest {} line {header_line_number}", path.display())
            })?)
        } else {
            optional_column(&headers, &["dataset_id"])
        };
    let input1_index = required_column(&headers, &["input1"])
        .with_context(|| format!("manifest {} line {header_line_number}", path.display()))?;
    let input2_index = optional_column(&headers, &["input2", "mate2"]);
    let fastq_ftp_index = optional_column(&headers, &["fastq_ftp", "ftp"]);
    let submitted_ftp_index = optional_column(&headers, &["submitted_ftp"]);
    let fastq_aspera_index = optional_column(&headers, &["fastq_aspera", "aspera"]);
    let submitted_aspera_index = optional_column(&headers, &["submitted_aspera"]);
    let fastq_1_index = optional_column(
        &headers,
        &[
            "fastq_1",
            "fastq_file_1",
            "filename_1",
            "file_1",
            "read1_ftp",
            "ftp_1",
        ],
    );
    let fastq_2_index = optional_column(
        &headers,
        &[
            "fastq_2",
            "fastq_file_2",
            "filename_2",
            "file_2",
            "read2_ftp",
            "ftp_2",
        ],
    );
    let accession_index = optional_column(&headers, &["accession", "study_accession"]);
    let citation_index = optional_column(&headers, &["citation", "reference"]);
    let expected_platform_index = optional_column(&headers, &["expected_platform", "platform"]);
    let expected_experiment_index =
        optional_column(&headers, &["expected_experiment", "experiment"]);
    let baseline_name_index = optional_column(
        &headers,
        &["baseline_name", "competitor_name", "baseline_tool"],
    );
    let baseline_input_bases_index = optional_column(
        &headers,
        &[
            "baseline_input_bases_per_sec",
            "competitor_input_bases_per_sec",
        ],
    );
    let baseline_trimmed_fraction_index = optional_column(
        &headers,
        &[
            "baseline_trimmed_read_fraction",
            "competitor_trimmed_read_fraction",
        ],
    );
    let baseline_discarded_fraction_index = optional_column(
        &headers,
        &[
            "baseline_discarded_read_fraction",
            "competitor_discarded_read_fraction",
        ],
    );
    let baseline_corrected_density_index = optional_column(
        &headers,
        &[
            "baseline_corrected_bases_per_mbase",
            "competitor_corrected_bases_per_mbase",
        ],
    );
    let echo_alignment_index = optional_column(
        &headers,
        &["echo_alignment_rate", "japalityecho_alignment_rate"],
    );
    let baseline_alignment_index = optional_column(
        &headers,
        &["baseline_alignment_rate", "competitor_alignment_rate"],
    );
    let echo_duplicate_index = optional_column(
        &headers,
        &["echo_duplicate_rate", "japalityecho_duplicate_rate"],
    );
    let baseline_duplicate_index = optional_column(
        &headers,
        &["baseline_duplicate_rate", "competitor_duplicate_rate"],
    );
    let echo_variant_index =
        optional_column(&headers, &["echo_variant_f1", "japalityecho_variant_f1"]);
    let baseline_variant_index =
        optional_column(&headers, &["baseline_variant_f1", "competitor_variant_f1"]);
    let echo_mean_coverage_index = optional_column(
        &headers,
        &[
            "echo_mean_coverage",
            "japalityecho_mean_coverage",
            "echo_mean_depth",
            "japalityecho_mean_depth",
        ],
    );
    let baseline_mean_coverage_index = optional_column(
        &headers,
        &[
            "baseline_mean_coverage",
            "competitor_mean_coverage",
            "baseline_mean_depth",
            "competitor_mean_depth",
        ],
    );
    let echo_coverage_breadth_index = optional_column(
        &headers,
        &[
            "echo_coverage_breadth",
            "japalityecho_coverage_breadth",
            "echo_coverage_fraction",
            "japalityecho_coverage_fraction",
            "echo_covered_bases_fraction",
            "japalityecho_covered_bases_fraction",
        ],
    );
    let baseline_coverage_breadth_index = optional_column(
        &headers,
        &[
            "baseline_coverage_breadth",
            "competitor_coverage_breadth",
            "baseline_coverage_fraction",
            "competitor_coverage_fraction",
            "baseline_covered_bases_fraction",
            "competitor_covered_bases_fraction",
        ],
    );
    let echo_assembly_index = optional_column(
        &headers,
        &["echo_assembly_n50", "japalityecho_assembly_n50"],
    );
    let baseline_assembly_index = optional_column(
        &headers,
        &["baseline_assembly_n50", "competitor_assembly_n50"],
    );
    let notes_index = optional_column(&headers, &["notes", "note"]);

    let manifest_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut seen_ids = HashSet::new();
    let mut entries = Vec::new();
    for (line_number, line) in data_lines
        .iter()
        .map(|(line_number, line)| (*line_number, line.as_str()))
    {
        let cells = split_manifest_row(line, delimiter);
        let dataset_id = optional_text_field(&cells, dataset_id_index);
        if require_dataset_id && dataset_id.is_none() {
            bail!(
                "manifest {} line {line_number} has an empty dataset_id",
                path.display()
            );
        }
        if let Some(dataset_id) = dataset_id.as_ref() {
            if !seen_ids.insert(dataset_id.clone()) {
                bail!(
                    "manifest {} line {line_number} repeats dataset_id '{}'",
                    path.display(),
                    dataset_id
                );
            }
        }

        let input1 = resolve_manifest_path(manifest_dir, cell(&cells, input1_index))?;
        let input2 = input2_index
            .map(|index| cell(&cells, index).trim())
            .filter(|value| !value.is_empty())
            .map(|value| resolve_manifest_path(manifest_dir, value))
            .transpose()?;
        let remote_locations = collect_public_remote_locations(
            &cells,
            &[
                fastq_1_index,
                fastq_2_index,
                fastq_ftp_index,
                submitted_ftp_index,
                fastq_aspera_index,
                submitted_aspera_index,
            ],
        );
        let accession = optional_text_field(&cells, accession_index);
        let citation = optional_text_field(&cells, citation_index);
        let expected_platform = expected_platform_index
            .map(|index| parse_optional_platform(cell(&cells, index), path, line_number))
            .transpose()?
            .flatten();
        let expected_experiment = expected_experiment_index
            .map(|index| parse_optional_experiment(cell(&cells, index), path, line_number))
            .transpose()?
            .flatten();
        let downstream = StudyDownstreamMetrics {
            alignment_rate: parse_optional_f64_field(
                &cells,
                echo_alignment_index,
                path,
                line_number,
            )?,
            duplicate_rate: parse_optional_f64_field(
                &cells,
                echo_duplicate_index,
                path,
                line_number,
            )?,
            variant_f1: parse_optional_f64_field(&cells, echo_variant_index, path, line_number)?,
            mean_coverage: parse_optional_f64_field(
                &cells,
                echo_mean_coverage_index,
                path,
                line_number,
            )?,
            coverage_breadth: parse_optional_f64_field(
                &cells,
                echo_coverage_breadth_index,
                path,
                line_number,
            )?,
            assembly_n50: parse_optional_f64_field(&cells, echo_assembly_index, path, line_number)?,
        };
        let baseline_name = optional_text_field(&cells, baseline_name_index);
        let baseline_input_bases_per_sec =
            parse_optional_f64_field(&cells, baseline_input_bases_index, path, line_number)?;
        let baseline_trimmed_read_fraction =
            parse_optional_f64_field(&cells, baseline_trimmed_fraction_index, path, line_number)?;
        let baseline_discarded_read_fraction =
            parse_optional_f64_field(&cells, baseline_discarded_fraction_index, path, line_number)?;
        let baseline_corrected_bases_per_mbase =
            parse_optional_f64_field(&cells, baseline_corrected_density_index, path, line_number)?;
        let baseline_downstream = StudyDownstreamMetrics {
            alignment_rate: parse_optional_f64_field(
                &cells,
                baseline_alignment_index,
                path,
                line_number,
            )?,
            duplicate_rate: parse_optional_f64_field(
                &cells,
                baseline_duplicate_index,
                path,
                line_number,
            )?,
            variant_f1: parse_optional_f64_field(
                &cells,
                baseline_variant_index,
                path,
                line_number,
            )?,
            mean_coverage: parse_optional_f64_field(
                &cells,
                baseline_mean_coverage_index,
                path,
                line_number,
            )?,
            coverage_breadth: parse_optional_f64_field(
                &cells,
                baseline_coverage_breadth_index,
                path,
                line_number,
            )?,
            assembly_n50: parse_optional_f64_field(
                &cells,
                baseline_assembly_index,
                path,
                line_number,
            )?,
        };
        let baseline = if baseline_name.is_some()
            || baseline_input_bases_per_sec.is_some()
            || baseline_trimmed_read_fraction.is_some()
            || baseline_discarded_read_fraction.is_some()
            || baseline_corrected_bases_per_mbase.is_some()
            || downstream_has_any_metrics(&baseline_downstream)
        {
            Some(StudyBaselineMetrics {
                name: baseline_name.unwrap_or_else(|| "baseline".to_string()),
                input_bases_per_sec: baseline_input_bases_per_sec,
                trimmed_read_fraction: baseline_trimmed_read_fraction,
                discarded_read_fraction: baseline_discarded_read_fraction,
                corrected_bases_per_mbase: baseline_corrected_bases_per_mbase,
                downstream: baseline_downstream,
            })
        } else {
            None
        };
        let notes = optional_text_field(&cells, notes_index);

        entries.push(ManifestSourceEntry {
            dataset_id,
            accession,
            citation,
            input1,
            input2,
            remote_locations,
            expected_platform,
            expected_experiment,
            downstream,
            baseline,
            notes,
        });
    }

    Ok(entries)
}

fn source_manifest_requires_accession_bootstrap(path: &Path) -> Result<bool> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read manifest {}", path.display()))?;
    let header_line = raw
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .with_context(|| format!("manifest {} is empty", path.display()))?;
    let delimiter = if header_line.contains('\t') {
        '\t'
    } else {
        ','
    };
    let headers = split_manifest_row(header_line, delimiter);
    Ok(optional_column(&headers, &["input1"]).is_none()
        && (optional_column(&headers, &["run_accession", "accession", "run"]).is_some()
            || optional_column(
                &headers,
                &[
                    "fastq_ftp",
                    "submitted_ftp",
                    "fastq_aspera",
                    "submitted_aspera",
                    "library_layout",
                    "layout",
                ],
            )
            .is_some()))
}

fn read_accession_manifest_source(
    path: &Path,
    headers: &[String],
    lines: &[(usize, String)],
    delimiter: char,
    require_dataset_id: bool,
    download_root: &Path,
) -> Result<Vec<ManifestSourceEntry>> {
    let dataset_id_index = if require_dataset_id {
        Some(required_column(headers, &["dataset_id"]).with_context(|| {
            format!(
                "manifest {} requires dataset_id for accession bootstrap",
                path.display()
            )
        })?)
    } else {
        optional_column(headers, &["dataset_id"])
    };
    let accession_index = optional_column(
        headers,
        &[
            "run_accession",
            "accession",
            "run",
            "run_id",
            "experiment_accession",
        ],
    );
    let citation_index = optional_column(headers, &["citation", "reference"]);
    let study_title_index = optional_column(headers, &["study_title", "title"]);
    let experiment_title_index = optional_column(
        headers,
        &[
            "experiment_title",
            "experiment_title_alias",
            "experiment_name",
        ],
    );
    let study_accession_index = optional_column(
        headers,
        &["study_accession", "study", "srastudy", "study acc"],
    );
    let bioproject_index =
        optional_column(headers, &["bioproject", "project_accession", "project"]);
    let pubmed_index = optional_column(headers, &["study_pubmed_id", "pubmed_id", "pmid"]);
    let doi_index = optional_column(headers, &["study_doi", "doi"]);
    let expected_platform_index = optional_column(headers, &["expected_platform"]);
    let platform_index = optional_column(headers, &["platform"]);
    let expected_experiment_index = optional_column(headers, &["expected_experiment"]);
    let experiment_index = optional_column(headers, &["experiment"]);
    let instrument_platform_index =
        optional_column(headers, &["instrument_platform", "instrument"]);
    let instrument_model_index = optional_column(headers, &["instrument_model", "model"]);
    let library_strategy_index = optional_column(headers, &["library_strategy", "librarystrategy"]);
    let assay_type_index = optional_column(headers, &["assay_type"]);
    let library_source_index = optional_column(headers, &["library_source"]);
    let library_selection_index = optional_column(headers, &["library_selection"]);
    let sample_title_index =
        optional_column(headers, &["sample_title", "sample_name", "sample_alias"]);
    let library_layout_index = optional_column(headers, &["library_layout", "layout"]);
    let paired_end_index = optional_column(headers, &["paired_end", "paired", "is_paired"]);
    let spots_with_mates_index =
        optional_column(headers, &["spots_with_mates", "mate_spots", "mates"]);
    let fastq_ftp_index = optional_column(headers, &["fastq_ftp", "ftp"]);
    let submitted_ftp_index = optional_column(headers, &["submitted_ftp"]);
    let fastq_aspera_index = optional_column(headers, &["fastq_aspera", "aspera"]);
    let submitted_aspera_index = optional_column(headers, &["submitted_aspera"]);
    let fastq_1_index = optional_column(
        headers,
        &[
            "fastq_1",
            "fastq_file_1",
            "filename_1",
            "file_1",
            "read1_ftp",
            "ftp_1",
        ],
    );
    let fastq_2_index = optional_column(
        headers,
        &[
            "fastq_2",
            "fastq_file_2",
            "filename_2",
            "file_2",
            "read2_ftp",
            "ftp_2",
        ],
    );
    let baseline_name_index = optional_column(
        headers,
        &["baseline_name", "competitor_name", "baseline_tool"],
    );
    let baseline_input_bases_index = optional_column(
        headers,
        &[
            "baseline_input_bases_per_sec",
            "competitor_input_bases_per_sec",
        ],
    );
    let baseline_trimmed_fraction_index = optional_column(
        headers,
        &[
            "baseline_trimmed_read_fraction",
            "competitor_trimmed_read_fraction",
        ],
    );
    let baseline_discarded_fraction_index = optional_column(
        headers,
        &[
            "baseline_discarded_read_fraction",
            "competitor_discarded_read_fraction",
        ],
    );
    let baseline_corrected_density_index = optional_column(
        headers,
        &[
            "baseline_corrected_bases_per_mbase",
            "competitor_corrected_bases_per_mbase",
        ],
    );
    let echo_alignment_index = optional_column(
        headers,
        &["echo_alignment_rate", "japalityecho_alignment_rate"],
    );
    let baseline_alignment_index = optional_column(
        headers,
        &["baseline_alignment_rate", "competitor_alignment_rate"],
    );
    let echo_duplicate_index = optional_column(
        headers,
        &["echo_duplicate_rate", "japalityecho_duplicate_rate"],
    );
    let baseline_duplicate_index = optional_column(
        headers,
        &["baseline_duplicate_rate", "competitor_duplicate_rate"],
    );
    let echo_variant_index =
        optional_column(headers, &["echo_variant_f1", "japalityecho_variant_f1"]);
    let baseline_variant_index =
        optional_column(headers, &["baseline_variant_f1", "competitor_variant_f1"]);
    let echo_mean_coverage_index = optional_column(
        headers,
        &[
            "echo_mean_coverage",
            "japalityecho_mean_coverage",
            "echo_mean_depth",
            "japalityecho_mean_depth",
        ],
    );
    let baseline_mean_coverage_index = optional_column(
        headers,
        &[
            "baseline_mean_coverage",
            "competitor_mean_coverage",
            "baseline_mean_depth",
            "competitor_mean_depth",
        ],
    );
    let echo_coverage_breadth_index = optional_column(
        headers,
        &[
            "echo_coverage_breadth",
            "japalityecho_coverage_breadth",
            "echo_coverage_fraction",
            "japalityecho_coverage_fraction",
            "echo_covered_bases_fraction",
            "japalityecho_covered_bases_fraction",
        ],
    );
    let baseline_coverage_breadth_index = optional_column(
        headers,
        &[
            "baseline_coverage_breadth",
            "competitor_coverage_breadth",
            "baseline_coverage_fraction",
            "competitor_coverage_fraction",
            "baseline_covered_bases_fraction",
            "competitor_covered_bases_fraction",
        ],
    );
    let echo_assembly_index =
        optional_column(headers, &["echo_assembly_n50", "japalityecho_assembly_n50"]);
    let baseline_assembly_index = optional_column(
        headers,
        &["baseline_assembly_n50", "competitor_assembly_n50"],
    );
    let notes_index = optional_column(headers, &["notes", "note"]);

    let mut seen_ids = HashSet::new();
    let mut entries = Vec::new();
    for (line_number, line) in lines
        .iter()
        .map(|(line_number, line)| (*line_number, line.as_str()))
    {
        let cells = split_manifest_row(line, delimiter);
        let dataset_id = optional_text_field(&cells, dataset_id_index);
        if require_dataset_id && dataset_id.is_none() {
            bail!(
                "manifest {} line {line_number} has an empty dataset_id",
                path.display()
            );
        }
        if let Some(dataset_id) = dataset_id.as_ref() {
            if !seen_ids.insert(dataset_id.clone()) {
                bail!(
                    "manifest {} line {line_number} repeats dataset_id '{}'",
                    path.display(),
                    dataset_id
                );
            }
        }

        let accession = optional_text_field(&cells, accession_index)
            .map(|value| extract_accession_token(&value).unwrap_or(value))
            .or_else(|| {
                text_fields_from_cells(
                    &cells,
                    &[
                        fastq_1_index,
                        fastq_2_index,
                        fastq_ftp_index,
                        submitted_ftp_index,
                    ],
                )
                .into_iter()
                .find_map(|value| extract_accession_token(&value))
            });
        if dataset_id.is_none() && accession.is_none() {
            bail!(
                "manifest {} line {line_number} must provide dataset_id and/or run_accession/accession for accession bootstrap",
                path.display()
            );
        }

        let citation = optional_text_field(&cells, citation_index).or_else(|| {
            compose_metadata_citation(&text_fields_from_cells(
                &cells,
                &[
                    study_title_index,
                    experiment_title_index,
                    study_accession_index,
                    bioproject_index,
                    pubmed_index,
                    doi_index,
                ],
            ))
        });
        let platform_texts = text_fields_from_cells(
            &cells,
            &[
                expected_platform_index,
                platform_index,
                instrument_platform_index,
                instrument_model_index,
            ],
        );
        let expected_platform = expected_platform_index
            .map(|index| parse_optional_platform(cell(&cells, index), path, line_number))
            .transpose()?
            .flatten()
            .or_else(|| infer_platform_from_metadata(&platform_texts));
        let experiment_texts = text_fields_from_cells(
            &cells,
            &[
                expected_experiment_index,
                experiment_index,
                experiment_title_index,
                library_strategy_index,
                assay_type_index,
                library_source_index,
                library_selection_index,
                sample_title_index,
            ],
        );
        let expected_experiment = expected_experiment_index
            .map(|index| parse_optional_experiment(cell(&cells, index), path, line_number))
            .transpose()?
            .flatten()
            .or_else(|| infer_experiment_from_metadata(&experiment_texts, expected_platform));

        let remote_locations = collect_public_remote_locations(
            &cells,
            &[
                fastq_1_index,
                fastq_2_index,
                fastq_ftp_index,
                submitted_ftp_index,
                fastq_aspera_index,
                submitted_aspera_index,
            ],
        );
        let paired_end = infer_public_paired_end(
            &cells,
            &[library_layout_index],
            &[paired_end_index],
            spots_with_mates_index,
            &remote_locations,
            path,
            line_number,
        )?;
        let identifier = accession
            .as_deref()
            .or(dataset_id.as_deref())
            .context("identifier should exist for accession bootstrap")?;
        let (input1, input2, synthesized_path_note) =
            synthesize_public_input_paths(identifier, &remote_locations, paired_end, download_root);

        let downstream = StudyDownstreamMetrics {
            alignment_rate: parse_optional_f64_field(
                &cells,
                echo_alignment_index,
                path,
                line_number,
            )?,
            duplicate_rate: parse_optional_f64_field(
                &cells,
                echo_duplicate_index,
                path,
                line_number,
            )?,
            variant_f1: parse_optional_f64_field(&cells, echo_variant_index, path, line_number)?,
            mean_coverage: parse_optional_f64_field(
                &cells,
                echo_mean_coverage_index,
                path,
                line_number,
            )?,
            coverage_breadth: parse_optional_f64_field(
                &cells,
                echo_coverage_breadth_index,
                path,
                line_number,
            )?,
            assembly_n50: parse_optional_f64_field(&cells, echo_assembly_index, path, line_number)?,
        };
        let baseline_name = optional_text_field(&cells, baseline_name_index);
        let baseline_input_bases_per_sec =
            parse_optional_f64_field(&cells, baseline_input_bases_index, path, line_number)?;
        let baseline_trimmed_read_fraction =
            parse_optional_f64_field(&cells, baseline_trimmed_fraction_index, path, line_number)?;
        let baseline_discarded_read_fraction =
            parse_optional_f64_field(&cells, baseline_discarded_fraction_index, path, line_number)?;
        let baseline_corrected_bases_per_mbase =
            parse_optional_f64_field(&cells, baseline_corrected_density_index, path, line_number)?;
        let baseline_downstream = StudyDownstreamMetrics {
            alignment_rate: parse_optional_f64_field(
                &cells,
                baseline_alignment_index,
                path,
                line_number,
            )?,
            duplicate_rate: parse_optional_f64_field(
                &cells,
                baseline_duplicate_index,
                path,
                line_number,
            )?,
            variant_f1: parse_optional_f64_field(
                &cells,
                baseline_variant_index,
                path,
                line_number,
            )?,
            mean_coverage: parse_optional_f64_field(
                &cells,
                baseline_mean_coverage_index,
                path,
                line_number,
            )?,
            coverage_breadth: parse_optional_f64_field(
                &cells,
                baseline_coverage_breadth_index,
                path,
                line_number,
            )?,
            assembly_n50: parse_optional_f64_field(
                &cells,
                baseline_assembly_index,
                path,
                line_number,
            )?,
        };
        let baseline = if baseline_name.is_some()
            || baseline_input_bases_per_sec.is_some()
            || baseline_trimmed_read_fraction.is_some()
            || baseline_discarded_read_fraction.is_some()
            || baseline_corrected_bases_per_mbase.is_some()
            || downstream_has_any_metrics(&baseline_downstream)
        {
            Some(StudyBaselineMetrics {
                name: baseline_name.unwrap_or_else(|| "baseline".to_string()),
                input_bases_per_sec: baseline_input_bases_per_sec,
                trimmed_read_fraction: baseline_trimmed_read_fraction,
                discarded_read_fraction: baseline_discarded_read_fraction,
                corrected_bases_per_mbase: baseline_corrected_bases_per_mbase,
                downstream: baseline_downstream,
            })
        } else {
            None
        };

        let mut notes = Vec::new();
        if let Some(note) = optional_text_field(&cells, notes_index) {
            notes.push(note);
        }
        if let Some(note) = synthesized_path_note {
            notes.push(note);
        }
        let notes = (!notes.is_empty()).then(|| notes.join(" | "));

        entries.push(ManifestSourceEntry {
            dataset_id,
            accession,
            citation,
            input1,
            input2,
            remote_locations,
            expected_platform,
            expected_experiment,
            downstream,
            baseline,
            notes,
        });
    }

    Ok(entries)
}

fn collect_public_remote_locations(cells: &[String], indices: &[Option<usize>]) -> Vec<String> {
    let mut values = Vec::new();
    for index in indices.iter().flatten() {
        for value in split_remote_location_field(cell(cells, *index)) {
            if !values.iter().any(|existing| existing == &value) {
                values.push(value);
            }
        }
    }
    values
}

fn split_remote_location_field(value: &str) -> Vec<String> {
    value
        .split([';', ','])
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn infer_public_paired_end(
    cells: &[String],
    layout_indices: &[Option<usize>],
    bool_indices: &[Option<usize>],
    spots_with_mates_index: Option<usize>,
    remote_locations: &[String],
    path: &Path,
    line_number: usize,
) -> Result<bool> {
    for index in layout_indices.iter().flatten() {
        if let Some(flag) = parse_boolish_text(cell(cells, *index)) {
            return Ok(flag);
        }
    }
    for index in bool_indices.iter().flatten() {
        if let Some(flag) = parse_boolish_text(cell(cells, *index)) {
            return Ok(flag);
        }
    }
    if let Some(index) = spots_with_mates_index {
        if let Some(spots_with_mates) =
            parse_optional_f64_field(cells, Some(index), path, line_number)?
        {
            if spots_with_mates > 0.0 {
                return Ok(true);
            }
        }
    }
    Ok(remote_locations_indicate_paired(remote_locations))
}

fn parse_boolish_text(value: &str) -> Option<bool> {
    match normalize_header(value).as_str() {
        "paired" | "pairedend" | "pe" | "true" | "yes" | "y" | "2" => Some(true),
        "single" | "singleend" | "se" | "false" | "no" | "n" | "1" => Some(false),
        _ => None,
    }
}

fn remote_locations_indicate_paired(remote_locations: &[String]) -> bool {
    let mut saw_mate1 = false;
    let mut saw_mate2 = false;
    let mut saw_any_mate = false;
    for location in remote_locations {
        let Some(file_name) = basename_from_remote_location(location) else {
            continue;
        };
        let (_, mate, _) = parse_mate_signature(&fastq_stem(&file_name));
        match mate {
            Some(1) => {
                saw_mate1 = true;
                saw_any_mate = true;
            }
            Some(2) => {
                saw_mate2 = true;
                saw_any_mate = true;
            }
            _ => {}
        }
    }
    (saw_mate1 && saw_mate2) || (!saw_any_mate && remote_locations.len() == 2)
}

fn synthesize_public_input_paths(
    identifier: &str,
    remote_locations: &[String],
    paired_end: bool,
    download_root: &Path,
) -> (PathBuf, Option<PathBuf>, Option<String>) {
    let base_name = sanitize_filename(identifier);
    let base_dir = download_root.join(&base_name);
    let file_names: Vec<String> = remote_locations
        .iter()
        .filter_map(|location| basename_from_remote_location(location))
        .collect();
    let unique_file_names: Vec<String> =
        file_names.into_iter().fold(Vec::new(), |mut acc, value| {
            if !acc.iter().any(|existing| existing == &value) {
                acc.push(value);
            }
            acc
        });

    if paired_end && unique_file_names.len() == 2 {
        let mut mate1 = None;
        let mut mate2 = None;
        for file_name in &unique_file_names {
            let (_, mate, _) = parse_mate_signature(&fastq_stem(file_name));
            match mate {
                Some(1) => mate1 = Some(file_name.clone()),
                Some(2) => mate2 = Some(file_name.clone()),
                _ => {}
            }
        }
        let ordered = if let (Some(mate1), Some(mate2)) = (mate1, mate2) {
            vec![mate1, mate2]
        } else {
            unique_file_names.clone()
        };
        return (
            base_dir.join(&ordered[0]),
            Some(base_dir.join(&ordered[1])),
            Some(
                "Placeholder inputs preserve remote FASTQ basenames under the configured download root"
                    .to_string(),
            ),
        );
    }

    if !paired_end && unique_file_names.len() == 1 {
        return (
            base_dir.join(&unique_file_names[0]),
            None,
            Some(
                "Placeholder input preserves the remote FASTQ basename under the configured download root"
                    .to_string(),
            ),
        );
    }

    if paired_end {
        (
            base_dir.join(format!("{base_name}_1.fastq.gz")),
            Some(base_dir.join(format!("{base_name}_2.fastq.gz"))),
            Some(if unique_file_names.len() > 2 {
                format!(
                    "Public metadata listed {} FASTQ files; manifest assumes merged mate FASTQs under the configured download root",
                    unique_file_names.len()
                )
            } else {
                "Public metadata did not expose unambiguous mate filenames; manifest assumes paired FASTQs under the configured download root".to_string()
            }),
        )
    } else {
        (
            base_dir.join(format!("{base_name}.fastq.gz")),
            None,
            Some(
                "Public metadata did not expose a direct FASTQ filename; manifest assumes a single FASTQ under the configured download root"
                    .to_string(),
            ),
        )
    }
}

fn basename_from_remote_location(value: &str) -> Option<String> {
    let trimmed = value.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return None;
    }
    trimmed
        .rsplit('/')
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn expected_public_download_destinations(entry: &ManifestSourceEntry) -> Vec<PathBuf> {
    let mut destinations = vec![entry.input1.clone()];
    if let Some(input2) = entry.input2.clone() {
        destinations.push(input2);
    }
    destinations
}

fn manifest_source_entry_label(entry: &ManifestSourceEntry) -> String {
    entry
        .dataset_id
        .clone()
        .or_else(|| entry.accession.clone())
        .unwrap_or_else(|| "dataset".to_string())
}

fn select_public_download_targets(
    entry: &ManifestSourceEntry,
) -> Result<Vec<PublicDownloadTarget>> {
    #[derive(Debug, Clone)]
    struct Candidate {
        source_url: String,
        file_name: String,
        mate: Option<u8>,
    }

    let mut candidates = Vec::new();
    for location in &entry.remote_locations {
        let Some(source_url) = normalize_public_download_url(location) else {
            continue;
        };
        let Some(file_name) = basename_from_remote_location(location)
            .or_else(|| basename_from_remote_location(&source_url))
        else {
            continue;
        };
        if candidates
            .iter()
            .any(|existing: &Candidate| existing.file_name == file_name)
        {
            continue;
        }
        let (_, mate, _) = parse_mate_signature(&fastq_stem(&file_name));
        candidates.push(Candidate {
            source_url,
            file_name,
            mate,
        });
    }

    if candidates.is_empty() {
        bail!(
            "dataset '{}' does not expose any curl-compatible public FASTQ locations",
            manifest_source_entry_label(entry)
        );
    }

    let dataset_id = manifest_source_entry_label(entry);
    let accession = entry.accession.clone();
    let desired_input1_name = entry
        .input1
        .file_name()
        .and_then(|value| value.to_str())
        .map(str::to_string);
    let desired_input2_name = entry
        .input2
        .as_ref()
        .and_then(|path| path.file_name())
        .and_then(|value| value.to_str())
        .map(str::to_string);

    let find_exact = |name: &str| {
        candidates
            .iter()
            .find(|candidate| candidate.file_name == name)
    };
    let find_mate = |mate: u8| {
        candidates
            .iter()
            .find(|candidate| candidate.mate == Some(mate))
    };

    if let Some(input2) = entry.input2.as_ref() {
        let left = desired_input1_name
            .as_deref()
            .and_then(find_exact)
            .or_else(|| find_mate(1));
        let right = desired_input2_name
            .as_deref()
            .and_then(find_exact)
            .or_else(|| find_mate(2));
        if let (Some(left), Some(right)) = (left, right) {
            if left.source_url != right.source_url {
                return Ok(vec![
                    PublicDownloadTarget {
                        dataset_id: dataset_id.clone(),
                        accession: accession.clone(),
                        destination: entry.input1.clone(),
                        source_url: left.source_url.clone(),
                    },
                    PublicDownloadTarget {
                        dataset_id,
                        accession,
                        destination: input2.clone(),
                        source_url: right.source_url.clone(),
                    },
                ]);
            }
        }
        if candidates.len() == 2 {
            return Ok(vec![
                PublicDownloadTarget {
                    dataset_id: dataset_id.clone(),
                    accession: accession.clone(),
                    destination: entry.input1.clone(),
                    source_url: candidates[0].source_url.clone(),
                },
                PublicDownloadTarget {
                    dataset_id,
                    accession,
                    destination: input2.clone(),
                    source_url: candidates[1].source_url.clone(),
                },
            ]);
        }
        bail!(
            "dataset '{}' exposes {} supported remote FASTQ candidates but they could not be mapped unambiguously onto paired inputs {} and {}",
            manifest_source_entry_label(entry),
            candidates.len(),
            entry.input1.display(),
            input2.display()
        );
    }

    if let Some(desired_name) = desired_input1_name.as_deref().and_then(find_exact) {
        return Ok(vec![PublicDownloadTarget {
            dataset_id,
            accession,
            destination: entry.input1.clone(),
            source_url: desired_name.source_url.clone(),
        }]);
    }
    if candidates.len() == 1 {
        return Ok(vec![PublicDownloadTarget {
            dataset_id,
            accession,
            destination: entry.input1.clone(),
            source_url: candidates[0].source_url.clone(),
        }]);
    }
    bail!(
        "dataset '{}' exposes {} supported remote FASTQ candidates but single-end destination {} could not be mapped unambiguously",
        manifest_source_entry_label(entry),
        candidates.len(),
        entry.input1.display()
    )
}

fn normalize_public_download_url(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.contains("://") {
        return Some(trimmed.to_string());
    }
    if trimmed.starts_with("fasp.") || trimmed.contains(":/") {
        return None;
    }
    if trimmed.contains('/') && trimmed.contains('.') {
        return Some(format!("ftp://{trimmed}"));
    }
    None
}

fn execute_public_download_target(
    target: &PublicDownloadTarget,
    retries: usize,
    overwrite_existing: bool,
) -> Result<DownloadExecutionOutcome, MetadataChunkFetchError> {
    if let Some(parent) = target.destination.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            return Err(MetadataChunkFetchError {
                attempts: 0,
                message: format!("failed to create {}: {error}", parent.display()),
            });
        }
    }
    if !overwrite_existing {
        if let Ok(metadata) = fs::metadata(&target.destination) {
            if metadata.is_file() && metadata.len() > 0 {
                return Ok(DownloadExecutionOutcome {
                    status: StudyDownloadStatusKind::SkippedExisting,
                    bytes: metadata.len(),
                    attempts: 0,
                });
            }
        }
    }

    let temp_path = partial_download_path(&target.destination);
    if overwrite_existing {
        let _ = fs::remove_file(&target.destination);
        let _ = fs::remove_file(&temp_path);
    }
    let resume_partial = temp_path
        .metadata()
        .map(|metadata| metadata.is_file() && metadata.len() > 0)
        .unwrap_or(false);

    let mut attempts = 0usize;
    let mut last_error = None;
    for _ in 0..=retries {
        attempts += 1;
        let mut command = Command::new("curl");
        command
            .arg("--fail")
            .arg("--silent")
            .arg("--show-error")
            .arg("--location");
        if resume_partial {
            command.arg("--continue-at").arg("-");
        }
        command
            .arg("--output")
            .arg(&temp_path)
            .arg(&target.source_url);
        match command.output() {
            Ok(output) if output.status.success() => {
                if let Err(error) = fs::rename(&temp_path, &target.destination) {
                    last_error = Some(format!(
                        "failed to finalize download {} -> {}: {error}",
                        temp_path.display(),
                        target.destination.display()
                    ));
                    continue;
                }
                let bytes = match fs::metadata(&target.destination) {
                    Ok(metadata) => metadata.len(),
                    Err(error) => {
                        return Err(MetadataChunkFetchError {
                            attempts,
                            message: format!(
                                "download completed but {} could not be stat'ed: {error}",
                                target.destination.display()
                            ),
                        });
                    }
                };
                return Ok(DownloadExecutionOutcome {
                    status: if resume_partial {
                        StudyDownloadStatusKind::Resumed
                    } else {
                        StudyDownloadStatusKind::Downloaded
                    },
                    bytes,
                    attempts,
                });
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                last_error = Some(if stderr.is_empty() {
                    format!("curl download failed with exit status {}", output.status)
                } else {
                    stderr
                });
            }
            Err(error) => {
                last_error = Some(format!(
                    "failed to launch curl for public FASTQ download: {error}"
                ));
            }
        }
    }
    Err(MetadataChunkFetchError {
        attempts,
        message: last_error.unwrap_or_else(|| "unknown public FASTQ download failure".to_string()),
    })
}

fn partial_download_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("download.fastq");
    path.with_file_name(format!("{file_name}.part"))
}

fn write_study_download_status_table(path: &Path, rows: &[StudyDownloadStatusRow]) -> Result<()> {
    let mut output = String::new();
    output.push_str("dataset_id,accession,destination,source_url,status,bytes,attempts,error\n");
    for row in rows {
        let values = vec![
            manifest_cell(&row.dataset_id, ','),
            manifest_cell_option(row.accession.as_deref(), ','),
            manifest_cell(&row.destination.display().to_string(), ','),
            manifest_cell_option(row.source_url.as_deref(), ','),
            manifest_cell(row.status.as_str(), ','),
            row.bytes.to_string(),
            row.attempts.to_string(),
            manifest_cell_option(row.error.as_deref(), ','),
        ];
        output.push_str(&values.join(","));
        output.push('\n');
    }
    fs::write(path, output).with_context(|| format!("failed to write {}", path.display()))
}

fn read_manifest_annotations(path: &Path) -> Result<Vec<ManifestAnnotationEntry>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read annotation table {}", path.display()))?;
    let mut lines = raw.lines().enumerate().filter_map(|(index, line)| {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            None
        } else {
            Some((index + 1, line))
        }
    });
    let (header_line_number, header_line) = lines
        .next()
        .with_context(|| format!("annotation table {} is empty", path.display()))?;
    let delimiter = if header_line.contains('\t') {
        '\t'
    } else {
        ','
    };
    let headers = split_manifest_row(header_line, delimiter);

    let dataset_id_index = optional_column(&headers, &["dataset_id"]);
    let accession_index = optional_column(
        &headers,
        &["accession", "run_accession", "run", "run_acc", "run acc"],
    );
    if dataset_id_index.is_none() && accession_index.is_none() {
        bail!(
            "annotation table {} line {header_line_number} must include dataset_id and/or accession",
            path.display()
        );
    }

    let citation_index = optional_column(&headers, &["citation", "reference"]);
    let study_title_index = optional_column(&headers, &["study_title", "title"]);
    let experiment_title_index = optional_column(
        &headers,
        &[
            "experiment_title",
            "experiment_title_alias",
            "experiment_name",
        ],
    );
    let study_accession_index = optional_column(
        &headers,
        &["study_accession", "study", "srastudy", "study acc"],
    );
    let bioproject_index =
        optional_column(&headers, &["bioproject", "project_accession", "project"]);
    let pubmed_index = optional_column(&headers, &["study_pubmed_id", "pubmed_id", "pmid"]);
    let doi_index = optional_column(&headers, &["study_doi", "doi"]);
    let expected_platform_index = optional_column(&headers, &["expected_platform"]);
    let platform_index = optional_column(&headers, &["platform"]);
    let expected_experiment_index = optional_column(&headers, &["expected_experiment"]);
    let experiment_index = optional_column(&headers, &["experiment"]);
    let instrument_platform_index =
        optional_column(&headers, &["instrument_platform", "instrument"]);
    let instrument_model_index = optional_column(&headers, &["instrument_model", "model"]);
    let library_strategy_index =
        optional_column(&headers, &["library_strategy", "librarystrategy"]);
    let assay_type_index = optional_column(&headers, &["assay_type"]);
    let library_source_index = optional_column(&headers, &["library_source"]);
    let library_selection_index = optional_column(&headers, &["library_selection"]);
    let sample_title_index = optional_column(&headers, &["sample_title", "sample_name"]);
    let baseline_name_index = optional_column(
        &headers,
        &["baseline_name", "competitor_name", "baseline_tool"],
    );
    let baseline_input_bases_index = optional_column(
        &headers,
        &[
            "baseline_input_bases_per_sec",
            "competitor_input_bases_per_sec",
        ],
    );
    let baseline_trimmed_fraction_index = optional_column(
        &headers,
        &[
            "baseline_trimmed_read_fraction",
            "competitor_trimmed_read_fraction",
        ],
    );
    let baseline_discarded_fraction_index = optional_column(
        &headers,
        &[
            "baseline_discarded_read_fraction",
            "competitor_discarded_read_fraction",
        ],
    );
    let baseline_corrected_density_index = optional_column(
        &headers,
        &[
            "baseline_corrected_bases_per_mbase",
            "competitor_corrected_bases_per_mbase",
        ],
    );
    let echo_alignment_index = optional_column(
        &headers,
        &["echo_alignment_rate", "japalityecho_alignment_rate"],
    );
    let baseline_alignment_index = optional_column(
        &headers,
        &["baseline_alignment_rate", "competitor_alignment_rate"],
    );
    let echo_duplicate_index = optional_column(
        &headers,
        &["echo_duplicate_rate", "japalityecho_duplicate_rate"],
    );
    let baseline_duplicate_index = optional_column(
        &headers,
        &["baseline_duplicate_rate", "competitor_duplicate_rate"],
    );
    let echo_variant_index =
        optional_column(&headers, &["echo_variant_f1", "japalityecho_variant_f1"]);
    let baseline_variant_index =
        optional_column(&headers, &["baseline_variant_f1", "competitor_variant_f1"]);
    let echo_mean_coverage_index = optional_column(
        &headers,
        &[
            "echo_mean_coverage",
            "japalityecho_mean_coverage",
            "echo_mean_depth",
            "japalityecho_mean_depth",
        ],
    );
    let baseline_mean_coverage_index = optional_column(
        &headers,
        &[
            "baseline_mean_coverage",
            "competitor_mean_coverage",
            "baseline_mean_depth",
            "competitor_mean_depth",
        ],
    );
    let echo_coverage_breadth_index = optional_column(
        &headers,
        &[
            "echo_coverage_breadth",
            "japalityecho_coverage_breadth",
            "echo_coverage_fraction",
            "japalityecho_coverage_fraction",
            "echo_covered_bases_fraction",
            "japalityecho_covered_bases_fraction",
        ],
    );
    let baseline_coverage_breadth_index = optional_column(
        &headers,
        &[
            "baseline_coverage_breadth",
            "competitor_coverage_breadth",
            "baseline_coverage_fraction",
            "competitor_coverage_fraction",
            "baseline_covered_bases_fraction",
            "competitor_covered_bases_fraction",
        ],
    );
    let echo_assembly_index = optional_column(
        &headers,
        &["echo_assembly_n50", "japalityecho_assembly_n50"],
    );
    let baseline_assembly_index = optional_column(
        &headers,
        &["baseline_assembly_n50", "competitor_assembly_n50"],
    );
    let notes_index = optional_column(&headers, &["notes", "note"]);

    let mut entries = Vec::new();
    for (line_number, line) in lines {
        let cells = split_manifest_row(line, delimiter);
        let dataset_id = optional_text_field(&cells, dataset_id_index);
        let accession = optional_text_field(&cells, accession_index);
        if dataset_id.is_none() && accession.is_none() {
            bail!(
                "annotation table {} line {line_number} must provide dataset_id and/or accession",
                path.display()
            );
        }

        let citation = optional_text_field(&cells, citation_index).or_else(|| {
            compose_metadata_citation(&text_fields_from_cells(
                &cells,
                &[
                    study_title_index,
                    experiment_title_index,
                    study_accession_index,
                    bioproject_index,
                    pubmed_index,
                    doi_index,
                ],
            ))
        });
        let platform_texts = text_fields_from_cells(
            &cells,
            &[
                expected_platform_index,
                platform_index,
                instrument_platform_index,
                instrument_model_index,
            ],
        );
        let expected_platform = expected_platform_index
            .map(|index| parse_optional_platform(cell(&cells, index), path, line_number))
            .transpose()?
            .flatten()
            .or_else(|| infer_platform_from_metadata(&platform_texts));
        let experiment_texts = text_fields_from_cells(
            &cells,
            &[
                expected_experiment_index,
                experiment_index,
                experiment_title_index,
                library_strategy_index,
                assay_type_index,
                library_source_index,
                library_selection_index,
                sample_title_index,
            ],
        );
        let expected_experiment = expected_experiment_index
            .map(|index| parse_optional_experiment(cell(&cells, index), path, line_number))
            .transpose()?
            .flatten()
            .or_else(|| infer_experiment_from_metadata(&experiment_texts, expected_platform));
        let downstream = StudyDownstreamMetrics {
            alignment_rate: parse_optional_f64_field(
                &cells,
                echo_alignment_index,
                path,
                line_number,
            )?,
            duplicate_rate: parse_optional_f64_field(
                &cells,
                echo_duplicate_index,
                path,
                line_number,
            )?,
            variant_f1: parse_optional_f64_field(&cells, echo_variant_index, path, line_number)?,
            mean_coverage: parse_optional_f64_field(
                &cells,
                echo_mean_coverage_index,
                path,
                line_number,
            )?,
            coverage_breadth: parse_optional_f64_field(
                &cells,
                echo_coverage_breadth_index,
                path,
                line_number,
            )?,
            assembly_n50: parse_optional_f64_field(&cells, echo_assembly_index, path, line_number)?,
        };
        let baseline_name = optional_text_field(&cells, baseline_name_index);
        let baseline_input_bases_per_sec =
            parse_optional_f64_field(&cells, baseline_input_bases_index, path, line_number)?;
        let baseline_trimmed_read_fraction =
            parse_optional_f64_field(&cells, baseline_trimmed_fraction_index, path, line_number)?;
        let baseline_discarded_read_fraction =
            parse_optional_f64_field(&cells, baseline_discarded_fraction_index, path, line_number)?;
        let baseline_corrected_bases_per_mbase =
            parse_optional_f64_field(&cells, baseline_corrected_density_index, path, line_number)?;
        let baseline_downstream = StudyDownstreamMetrics {
            alignment_rate: parse_optional_f64_field(
                &cells,
                baseline_alignment_index,
                path,
                line_number,
            )?,
            duplicate_rate: parse_optional_f64_field(
                &cells,
                baseline_duplicate_index,
                path,
                line_number,
            )?,
            variant_f1: parse_optional_f64_field(
                &cells,
                baseline_variant_index,
                path,
                line_number,
            )?,
            mean_coverage: parse_optional_f64_field(
                &cells,
                baseline_mean_coverage_index,
                path,
                line_number,
            )?,
            coverage_breadth: parse_optional_f64_field(
                &cells,
                baseline_coverage_breadth_index,
                path,
                line_number,
            )?,
            assembly_n50: parse_optional_f64_field(
                &cells,
                baseline_assembly_index,
                path,
                line_number,
            )?,
        };
        let baseline = if baseline_name.is_some()
            || baseline_input_bases_per_sec.is_some()
            || baseline_trimmed_read_fraction.is_some()
            || baseline_discarded_read_fraction.is_some()
            || baseline_corrected_bases_per_mbase.is_some()
            || downstream_has_any_metrics(&baseline_downstream)
        {
            Some(StudyBaselineMetrics {
                name: baseline_name.unwrap_or_else(|| "baseline".to_string()),
                input_bases_per_sec: baseline_input_bases_per_sec,
                trimmed_read_fraction: baseline_trimmed_read_fraction,
                discarded_read_fraction: baseline_discarded_read_fraction,
                corrected_bases_per_mbase: baseline_corrected_bases_per_mbase,
                downstream: baseline_downstream,
            })
        } else {
            None
        };
        let notes = optional_text_field(&cells, notes_index);

        entries.push(ManifestAnnotationEntry {
            dataset_id,
            accession,
            citation,
            expected_platform,
            expected_experiment,
            downstream,
            baseline,
            notes,
        });
    }

    Ok(entries)
}

fn resolve_annotation_target(
    annotation: &ManifestAnnotationEntry,
    index_by_dataset_id: &HashMap<String, usize>,
    index_by_accession: &HashMap<String, usize>,
    path: &Path,
    line_number: usize,
) -> Result<Option<usize>> {
    let dataset_match = annotation.dataset_id.as_deref().and_then(|dataset_id| {
        index_by_dataset_id
            .get(&normalize_lookup_key(dataset_id))
            .copied()
    });
    let accession_match = annotation.accession.as_deref().and_then(|accession| {
        index_by_accession
            .get(&normalize_lookup_key(accession))
            .copied()
    });

    match (dataset_match, accession_match) {
        (Some(dataset_match), Some(accession_match)) if dataset_match != accession_match => bail!(
            "annotation table {} line {line_number} matched different datasets for dataset_id '{}' and accession '{}'",
            path.display(),
            annotation.dataset_id.as_deref().unwrap_or(""),
            annotation.accession.as_deref().unwrap_or("")
        ),
        (Some(index), _) | (_, Some(index)) => Ok(Some(index)),
        (None, None) => Ok(None),
    }
}

fn apply_annotation_entry(
    target: &mut ManifestEntry,
    annotation: ManifestAnnotationEntry,
    overwrite_existing: bool,
    stats: &mut AnnotationMergeStats,
) -> bool {
    let mut changed = false;
    changed |= stats.record(merge_optional_value(
        &mut target.accession,
        annotation.accession,
        overwrite_existing,
    ));
    changed |= stats.record(merge_optional_value(
        &mut target.citation,
        annotation.citation,
        overwrite_existing,
    ));
    changed |= stats.record(merge_optional_value(
        &mut target.expected_platform,
        annotation.expected_platform,
        overwrite_existing,
    ));
    changed |= stats.record(merge_optional_value(
        &mut target.expected_experiment,
        annotation.expected_experiment,
        overwrite_existing,
    ));
    changed |= apply_downstream_annotation(
        &mut target.downstream,
        annotation.downstream,
        overwrite_existing,
        stats,
    );
    changed |= apply_baseline_annotation(
        &mut target.baseline,
        annotation.baseline,
        overwrite_existing,
        stats,
    );
    changed |= stats.record(merge_notes(
        &mut target.notes,
        annotation.notes,
        overwrite_existing,
    ));
    changed
}

fn apply_downstream_annotation(
    target: &mut StudyDownstreamMetrics,
    annotation: StudyDownstreamMetrics,
    overwrite_existing: bool,
    stats: &mut AnnotationMergeStats,
) -> bool {
    let mut changed = false;
    changed |= stats.record(merge_optional_value(
        &mut target.alignment_rate,
        annotation.alignment_rate,
        overwrite_existing,
    ));
    changed |= stats.record(merge_optional_value(
        &mut target.duplicate_rate,
        annotation.duplicate_rate,
        overwrite_existing,
    ));
    changed |= stats.record(merge_optional_value(
        &mut target.variant_f1,
        annotation.variant_f1,
        overwrite_existing,
    ));
    changed |= stats.record(merge_optional_value(
        &mut target.mean_coverage,
        annotation.mean_coverage,
        overwrite_existing,
    ));
    changed |= stats.record(merge_optional_value(
        &mut target.coverage_breadth,
        annotation.coverage_breadth,
        overwrite_existing,
    ));
    changed |= stats.record(merge_optional_value(
        &mut target.assembly_n50,
        annotation.assembly_n50,
        overwrite_existing,
    ));
    changed
}

fn apply_baseline_annotation(
    target: &mut Option<StudyBaselineMetrics>,
    annotation: Option<StudyBaselineMetrics>,
    overwrite_existing: bool,
    stats: &mut AnnotationMergeStats,
) -> bool {
    let Some(annotation) = annotation else {
        return false;
    };

    let target = target.get_or_insert_with(default_study_baseline_metrics);
    let mut changed = false;
    changed |= stats.record(merge_baseline_name(
        &mut target.name,
        &annotation.name,
        overwrite_existing,
    ));
    changed |= stats.record(merge_optional_value(
        &mut target.input_bases_per_sec,
        annotation.input_bases_per_sec,
        overwrite_existing,
    ));
    changed |= stats.record(merge_optional_value(
        &mut target.trimmed_read_fraction,
        annotation.trimmed_read_fraction,
        overwrite_existing,
    ));
    changed |= stats.record(merge_optional_value(
        &mut target.discarded_read_fraction,
        annotation.discarded_read_fraction,
        overwrite_existing,
    ));
    changed |= stats.record(merge_optional_value(
        &mut target.corrected_bases_per_mbase,
        annotation.corrected_bases_per_mbase,
        overwrite_existing,
    ));
    changed |= apply_downstream_annotation(
        &mut target.downstream,
        annotation.downstream,
        overwrite_existing,
        stats,
    );
    changed
}

fn merge_optional_value<T: Clone + PartialEq>(
    target: &mut Option<T>,
    source: Option<T>,
    overwrite_existing: bool,
) -> MergeOutcome {
    let Some(source) = source else {
        return MergeOutcome::Unchanged;
    };
    match target {
        None => {
            *target = Some(source);
            MergeOutcome::Filled
        }
        Some(current) if *current == source => MergeOutcome::Unchanged,
        Some(_) if overwrite_existing => {
            *target = Some(source);
            MergeOutcome::Overwritten
        }
        Some(_) => MergeOutcome::Unchanged,
    }
}

fn merge_baseline_name(
    target: &mut String,
    source: &str,
    overwrite_existing: bool,
) -> MergeOutcome {
    let source = source.trim();
    if source.is_empty() || target == source {
        return MergeOutcome::Unchanged;
    }
    if target.trim().is_empty() || target == "baseline" {
        *target = source.to_string();
        MergeOutcome::Filled
    } else if overwrite_existing {
        *target = source.to_string();
        MergeOutcome::Overwritten
    } else {
        MergeOutcome::Unchanged
    }
}

fn merge_notes(
    target: &mut Option<String>,
    source: Option<String>,
    overwrite_existing: bool,
) -> MergeOutcome {
    let Some(source) = source
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    else {
        return MergeOutcome::Unchanged;
    };

    match target {
        None => {
            *target = Some(source);
            MergeOutcome::Filled
        }
        Some(current) if current.trim().is_empty() => {
            *current = source;
            MergeOutcome::Filled
        }
        Some(current) if current.trim() == source => MergeOutcome::Unchanged,
        Some(current) if overwrite_existing => {
            *current = source;
            MergeOutcome::Overwritten
        }
        Some(current) => {
            if current
                .split('|')
                .map(str::trim)
                .any(|segment| segment == source)
            {
                MergeOutcome::Unchanged
            } else {
                current.push_str(" | ");
                current.push_str(&source);
                MergeOutcome::Appended
            }
        }
    }
}

fn default_study_baseline_metrics() -> StudyBaselineMetrics {
    StudyBaselineMetrics {
        name: "baseline".to_string(),
        input_bases_per_sec: None,
        trimmed_read_fraction: None,
        discarded_read_fraction: None,
        corrected_bases_per_mbase: None,
        downstream: default_study_downstream_metrics(),
    }
}

fn default_study_downstream_metrics() -> StudyDownstreamMetrics {
    StudyDownstreamMetrics {
        alignment_rate: None,
        duplicate_rate: None,
        variant_f1: None,
        mean_coverage: None,
        coverage_breadth: None,
        assembly_n50: None,
    }
}

fn downstream_has_any_metrics(metrics: &StudyDownstreamMetrics) -> bool {
    metrics.alignment_rate.is_some()
        || metrics.duplicate_rate.is_some()
        || metrics.variant_f1.is_some()
        || metrics.mean_coverage.is_some()
        || metrics.coverage_breadth.is_some()
        || metrics.assembly_n50.is_some()
}

fn annotation_key_label(annotation: &ManifestAnnotationEntry) -> String {
    match (
        annotation.dataset_id.as_deref(),
        annotation.accession.as_deref(),
    ) {
        (Some(dataset_id), Some(accession)) => format!("{dataset_id}/{accession}"),
        (Some(dataset_id), None) => dataset_id.to_string(),
        (None, Some(accession)) => accession.to_string(),
        (None, None) => "unknown".to_string(),
    }
}

fn normalize_lookup_key(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

fn finalize_manifest_entries(
    entries: Vec<ManifestSourceEntry>,
    default_baseline_name: Option<&str>,
) -> Vec<ManifestEntry> {
    let mut seen_ids = HashSet::new();
    entries
        .into_iter()
        .map(|entry| {
            let (dataset_id, dataset_id_generated) = allocate_dataset_id(
                entry.dataset_id.as_deref(),
                entry.accession.as_deref(),
                &entry.input1,
                &mut seen_ids,
            );
            let baseline = entry
                .baseline
                .map(|mut baseline| {
                    if baseline.name == "baseline" {
                        if let Some(default_baseline_name) = default_baseline_name {
                            baseline.name = default_baseline_name.to_string();
                        }
                    }
                    baseline
                })
                .or_else(|| {
                    default_baseline_name.map(|default_baseline_name| StudyBaselineMetrics {
                        name: default_baseline_name.to_string(),
                        input_bases_per_sec: None,
                        trimmed_read_fraction: None,
                        discarded_read_fraction: None,
                        corrected_bases_per_mbase: None,
                        downstream: default_study_downstream_metrics(),
                    })
                });

            ManifestEntry {
                dataset_id,
                dataset_id_generated,
                accession: entry.accession,
                citation: entry.citation,
                input1: entry.input1,
                input2: entry.input2,
                expected_platform: entry.expected_platform,
                expected_experiment: entry.expected_experiment,
                downstream: entry.downstream,
                baseline,
                notes: entry.notes,
            }
        })
        .collect()
}

fn allocate_dataset_id(
    requested_id: Option<&str>,
    accession: Option<&str>,
    input1: &Path,
    seen_ids: &mut HashSet<String>,
) -> (String, bool) {
    let provided = requested_id.filter(|value| !value.trim().is_empty());
    let generated = provided.is_none();
    let base = provided
        .filter(|value| !value.trim().is_empty())
        .map(sanitize_filename)
        .or_else(|| accession.map(sanitize_filename))
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            input1
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(sanitize_filename)
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "dataset".to_string())
        });

    let mut candidate = base.clone();
    let mut suffix = 2usize;
    while !seen_ids.insert(candidate.clone()) {
        candidate = format!("{base}-{suffix}");
        suffix += 1;
    }
    (candidate, generated)
}

fn manifest_delimiter_for_output(path: &Path) -> char {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .as_deref()
    {
        Some("csv") => ',',
        _ => '\t',
    }
}

fn sibling_artifact_path(path: &Path, suffix: &str) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("study_manifest");
    path.with_file_name(format!("{stem}.{suffix}"))
}

fn write_canonical_manifest(path: &Path, delimiter: char, entries: &[ManifestEntry]) -> Result<()> {
    let output_parent = path.parent().unwrap_or_else(|| Path::new("."));
    let delimiter_str = delimiter.to_string();
    let mut output = String::new();
    output.push_str(&CANONICAL_STUDY_MANIFEST_HEADERS.join(&delimiter_str));
    output.push('\n');
    for entry in entries {
        let row = vec![
            manifest_cell(&entry.dataset_id, delimiter),
            manifest_cell_option(entry.accession.as_deref(), delimiter),
            manifest_cell_option(entry.citation.as_deref(), delimiter),
            manifest_cell(
                &render_manifest_path_for_output(&entry.input1, output_parent),
                delimiter,
            ),
            manifest_cell_option(
                entry
                    .input2
                    .as_ref()
                    .map(|path| render_manifest_path_for_output(path, output_parent))
                    .as_deref(),
                delimiter,
            ),
            manifest_cell_option_label(entry.expected_platform.as_ref(), delimiter),
            manifest_cell_option_label(entry.expected_experiment.as_ref(), delimiter),
            manifest_cell_option(
                entry
                    .baseline
                    .as_ref()
                    .map(|baseline| baseline.name.as_str()),
                delimiter,
            ),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.input_bases_per_sec),
            ),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.trimmed_read_fraction),
            ),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.discarded_read_fraction),
            ),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.corrected_bases_per_mbase),
            ),
            manifest_numeric_option(entry.downstream.alignment_rate),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.downstream.alignment_rate),
            ),
            manifest_numeric_option(entry.downstream.duplicate_rate),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.downstream.duplicate_rate),
            ),
            manifest_numeric_option(entry.downstream.variant_f1),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.downstream.variant_f1),
            ),
            manifest_numeric_option(entry.downstream.mean_coverage),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.downstream.mean_coverage),
            ),
            manifest_numeric_option(entry.downstream.coverage_breadth),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.downstream.coverage_breadth),
            ),
            manifest_numeric_option(entry.downstream.assembly_n50),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.downstream.assembly_n50),
            ),
            manifest_cell_option(entry.notes.as_deref(), delimiter),
        ];
        output.push_str(&row.join(&delimiter_str));
        output.push('\n');
    }
    fs::write(path, output).with_context(|| format!("failed to write {}", path.display()))
}

fn write_fetched_public_metadata_table(
    path: &Path,
    delimiter: char,
    rows: &[FetchedPublicMetadataRow],
) -> Result<()> {
    let delimiter_str = delimiter.to_string();
    let mut output = String::new();
    output.push_str(&FETCHED_PUBLIC_METADATA_HEADERS.join(&delimiter_str));
    output.push('\n');
    for row in rows {
        let line = FETCHED_PUBLIC_METADATA_HEADERS
            .iter()
            .map(|header| {
                manifest_cell_option(row.fields.get(*header).map(String::as_str), delimiter)
            })
            .collect::<Vec<_>>()
            .join(&delimiter_str);
        output.push_str(&line);
        output.push('\n');
    }
    fs::write(path, output).with_context(|| format!("failed to write {}", path.display()))
}

fn write_fetch_status_table(
    path: &Path,
    ordered_accessions: &[String],
    statuses: &HashMap<String, FetchAccessionStatusRow>,
) -> Result<()> {
    let mut output = String::new();
    output.push_str(
        "accession,status,matched_run_rows,matched_runs,source,attempts,error,geo_surface,resolved_accessions\n",
    );
    for accession in ordered_accessions {
        let status = statuses
            .get(accession)
            .with_context(|| format!("missing fetch status row for {accession}"))?;
        let row = vec![
            manifest_cell(&status.accession, ','),
            manifest_cell(status.status.as_str(), ','),
            status.matched_runs.len().to_string(),
            manifest_cell(&status.matched_runs.join(";"), ','),
            manifest_cell_option(status.source.map(MetadataChunkSource::as_str), ','),
            status.attempts.to_string(),
            manifest_cell_option(status.error.as_deref(), ','),
            manifest_cell_option(status.geo_surface.as_deref(), ','),
            manifest_cell(&status.resolved_accessions.join(";"), ','),
        ];
        output.push_str(&row.join(","));
        output.push('\n');
    }
    fs::write(path, output).with_context(|| format!("failed to write {}", path.display()))
}

fn build_manifest_bootstrap_summary(entries: &[ManifestEntry]) -> StudyManifestBootstrapSummary {
    StudyManifestBootstrapSummary {
        datasets: entries.len(),
        paired_datasets: entries
            .iter()
            .filter(|entry| entry.input2.is_some())
            .count(),
        datasets_with_generated_id: entries
            .iter()
            .filter(|entry| entry.dataset_id_generated)
            .count(),
        datasets_with_accession: entries
            .iter()
            .filter(|entry| entry.accession.is_some())
            .count(),
        datasets_with_citation: entries
            .iter()
            .filter(|entry| entry.citation.is_some())
            .count(),
        datasets_with_expected_platform: entries
            .iter()
            .filter(|entry| entry.expected_platform.is_some())
            .count(),
        datasets_with_expected_experiment: entries
            .iter()
            .filter(|entry| entry.expected_experiment.is_some())
            .count(),
        datasets_with_baseline_name: entries
            .iter()
            .filter(|entry| entry.baseline.is_some())
            .count(),
        datasets_with_downstream_metrics: entries
            .iter()
            .filter(|entry| {
                downstream_has_any_metrics(&entry.downstream)
                    || entry
                        .baseline
                        .as_ref()
                        .is_some_and(|baseline| downstream_has_any_metrics(&baseline.downstream))
            })
            .count(),
    }
}

fn write_manifest_bootstrap_summary_csv(path: &Path, entries: &[ManifestEntry]) -> Result<()> {
    let summary = build_manifest_bootstrap_summary(entries);
    let csv = format!(
        "metric,value\n\
datasets,{}\n\
paired_datasets,{}\n\
datasets_with_generated_id,{}\n\
datasets_with_accession,{}\n\
datasets_with_citation,{}\n\
datasets_with_expected_platform,{}\n\
datasets_with_expected_experiment,{}\n\
datasets_with_baseline_name,{}\n\
datasets_with_downstream_metrics,{}\n",
        summary.datasets,
        summary.paired_datasets,
        summary.datasets_with_generated_id,
        summary.datasets_with_accession,
        summary.datasets_with_citation,
        summary.datasets_with_expected_platform,
        summary.datasets_with_expected_experiment,
        summary.datasets_with_baseline_name,
        summary.datasets_with_downstream_metrics,
    );
    fs::write(path, csv).with_context(|| format!("failed to write {}", path.display()))
}

fn write_study_annotate_summary_csv(path: &Path, summary: &StudyAnnotateSummary) -> Result<()> {
    let csv = format!(
        "metric,before,after\n\
datasets,{},{}\n\
paired_datasets,{},{}\n\
datasets_with_generated_id,{},{}\n\
datasets_with_accession,{},{}\n\
datasets_with_citation,{},{}\n\
datasets_with_expected_platform,{},{}\n\
datasets_with_expected_experiment,{},{}\n\
datasets_with_baseline_name,{},{}\n\
datasets_with_downstream_metrics,{},{}\n\
annotation_rows,,{}\n\
matched_rows,,{}\n\
unmatched_rows,,{}\n\
datasets_changed,,{}\n\
fields_filled,,{}\n\
fields_overwritten,,{}\n\
notes_appended,,{}\n",
        summary.before.datasets,
        summary.after.datasets,
        summary.before.paired_datasets,
        summary.after.paired_datasets,
        summary.before.datasets_with_generated_id,
        summary.after.datasets_with_generated_id,
        summary.before.datasets_with_accession,
        summary.after.datasets_with_accession,
        summary.before.datasets_with_citation,
        summary.after.datasets_with_citation,
        summary.before.datasets_with_expected_platform,
        summary.after.datasets_with_expected_platform,
        summary.before.datasets_with_expected_experiment,
        summary.after.datasets_with_expected_experiment,
        summary.before.datasets_with_baseline_name,
        summary.after.datasets_with_baseline_name,
        summary.before.datasets_with_downstream_metrics,
        summary.after.datasets_with_downstream_metrics,
        summary.annotation_rows,
        summary.matched_rows,
        summary.unmatched_rows,
        summary.datasets_changed,
        summary.fields_filled,
        summary.fields_overwritten,
        summary.notes_appended,
    );
    fs::write(path, csv).with_context(|| format!("failed to write {}", path.display()))
}

fn build_ingested_annotation_rows(
    manifest_entries: &[ManifestEntry],
    aggregated: &HashMap<usize, ManifestAnnotationEntry>,
) -> Vec<ManifestAnnotationEntry> {
    let mut rows = Vec::new();
    for (index, manifest_entry) in manifest_entries.iter().enumerate() {
        let Some(aggregated_entry) = aggregated.get(&index) else {
            continue;
        };
        rows.push(ManifestAnnotationEntry {
            dataset_id: Some(manifest_entry.dataset_id.clone()),
            accession: aggregated_entry
                .accession
                .clone()
                .or_else(|| manifest_entry.accession.clone()),
            citation: aggregated_entry.citation.clone(),
            expected_platform: aggregated_entry.expected_platform,
            expected_experiment: aggregated_entry.expected_experiment,
            downstream: aggregated_entry.downstream.clone(),
            baseline: aggregated_entry.baseline.clone(),
            notes: aggregated_entry.notes.clone(),
        });
    }
    rows
}

fn write_canonical_annotation_table(
    path: &Path,
    entries: &[ManifestAnnotationEntry],
) -> Result<()> {
    let delimiter = manifest_delimiter_for_output(path);
    let delimiter_str = delimiter.to_string();
    let mut output = String::new();
    output.push_str(&CANONICAL_STUDY_ANNOTATION_HEADERS.join(&delimiter_str));
    output.push('\n');
    for entry in entries {
        let normalized_baseline_name = entry
            .baseline
            .as_ref()
            .and_then(|baseline| normalize_baseline_name(Some(baseline.name.as_str())));
        let row = vec![
            manifest_cell_option(entry.dataset_id.as_deref(), delimiter),
            manifest_cell_option(entry.accession.as_deref(), delimiter),
            manifest_cell_option(entry.citation.as_deref(), delimiter),
            manifest_cell_option_label(entry.expected_platform.as_ref(), delimiter),
            manifest_cell_option_label(entry.expected_experiment.as_ref(), delimiter),
            manifest_cell_option(normalized_baseline_name.as_deref(), delimiter),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.input_bases_per_sec),
            ),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.trimmed_read_fraction),
            ),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.discarded_read_fraction),
            ),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.corrected_bases_per_mbase),
            ),
            manifest_numeric_option(entry.downstream.alignment_rate),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.downstream.alignment_rate),
            ),
            manifest_numeric_option(entry.downstream.duplicate_rate),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.downstream.duplicate_rate),
            ),
            manifest_numeric_option(entry.downstream.variant_f1),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.downstream.variant_f1),
            ),
            manifest_numeric_option(entry.downstream.mean_coverage),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.downstream.mean_coverage),
            ),
            manifest_numeric_option(entry.downstream.coverage_breadth),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.downstream.coverage_breadth),
            ),
            manifest_numeric_option(entry.downstream.assembly_n50),
            manifest_numeric_option(
                entry
                    .baseline
                    .as_ref()
                    .and_then(|baseline| baseline.downstream.assembly_n50),
            ),
            manifest_cell_option(entry.notes.as_deref(), delimiter),
        ];
        output.push_str(&row.join(&delimiter_str));
        output.push('\n');
    }
    fs::write(path, output).with_context(|| format!("failed to write {}", path.display()))
}

fn write_study_ingest_summary_csv(path: &Path, summary: &StudyIngestSummary) -> Result<()> {
    let csv = format!(
        "metric,before,after\n\
files_scanned,,{}\n\
structured_files,,{}\n\
json_files,,{}\n\
delimited_files,,{}\n\
ignored_files,,{}\n\
records_ingested,,{}\n\
generated_annotation_rows,,{}\n\
matched_records,,{}\n\
unmatched_records,,{}\n\
datasets_changed,,{}\n\
fields_filled,,{}\n\
fields_overwritten,,{}\n\
notes_appended,,{}\n\
datasets,{},{}\n\
paired_datasets,{},{}\n\
datasets_with_generated_id,{},{}\n\
datasets_with_accession,{},{}\n\
datasets_with_citation,{},{}\n\
datasets_with_expected_platform,{},{}\n\
datasets_with_expected_experiment,{},{}\n\
datasets_with_baseline_name,{},{}\n\
datasets_with_downstream_metrics,{},{}\n",
        summary.files_scanned,
        summary.structured_files,
        summary.json_files,
        summary.delimited_files,
        summary.ignored_files,
        summary.records_ingested,
        summary.generated_annotation_rows,
        summary.matched_records,
        summary.unmatched_records,
        summary.datasets_changed,
        summary.fields_filled,
        summary.fields_overwritten,
        summary.notes_appended,
        summary.before.datasets,
        summary.after.datasets,
        summary.before.paired_datasets,
        summary.after.paired_datasets,
        summary.before.datasets_with_generated_id,
        summary.after.datasets_with_generated_id,
        summary.before.datasets_with_accession,
        summary.after.datasets_with_accession,
        summary.before.datasets_with_citation,
        summary.after.datasets_with_citation,
        summary.before.datasets_with_expected_platform,
        summary.after.datasets_with_expected_platform,
        summary.before.datasets_with_expected_experiment,
        summary.after.datasets_with_expected_experiment,
        summary.before.datasets_with_baseline_name,
        summary.after.datasets_with_baseline_name,
        summary.before.datasets_with_downstream_metrics,
        summary.after.datasets_with_downstream_metrics,
    );
    fs::write(path, csv).with_context(|| format!("failed to write {}", path.display()))
}

fn split_manifest_row(line: &str, delimiter: char) -> Vec<String> {
    line.split(delimiter)
        .map(|cell| cell.trim().to_string())
        .collect()
}

fn required_column(headers: &[String], aliases: &[&str]) -> Result<usize> {
    optional_column(headers, aliases)
        .with_context(|| format!("missing required manifest column {}", aliases.join("/")))
}

fn required_report_column(headers: &[String], aliases: &[&str], path: &Path) -> Result<usize> {
    optional_column(headers, aliases).with_context(|| {
        format!(
            "missing required native report column {} in {}",
            aliases.join("/"),
            path.display()
        )
    })
}

fn optional_column(headers: &[String], aliases: &[&str]) -> Option<usize> {
    headers.iter().position(|header| {
        aliases
            .iter()
            .any(|alias| normalize_header(header) == normalize_header(alias))
    })
}

fn normalize_header(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn cell(cells: &[String], index: usize) -> &str {
    cells.get(index).map(String::as_str).unwrap_or("")
}

fn resolve_manifest_path(base_dir: &Path, value: &str) -> Result<PathBuf> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        bail!("manifest path field must not be empty");
    }
    let candidate = PathBuf::from(trimmed);
    let resolved = if candidate.is_absolute() {
        candidate
    } else {
        base_dir.join(candidate)
    };
    if !resolved.exists() {
        bail!("manifest input {} does not exist", resolved.display());
    }
    Ok(resolved)
}

fn optional_text_field(cells: &[String], index: Option<usize>) -> Option<String> {
    index
        .map(|index| cell(cells, index).trim().to_string())
        .filter(|value| !value.is_empty())
}

fn parse_optional_platform(
    value: &str,
    path: &Path,
    line_number: usize,
) -> Result<Option<Platform>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    trimmed
        .parse::<Platform>()
        .map(Some)
        .map_err(|error| anyhow::anyhow!("manifest {} line {line_number}: {error}", path.display()))
}

fn parse_optional_experiment(
    value: &str,
    path: &Path,
    line_number: usize,
) -> Result<Option<ExperimentType>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    trimmed
        .parse::<ExperimentType>()
        .map(Some)
        .map_err(|error| anyhow::anyhow!("manifest {} line {line_number}: {error}", path.display()))
}

fn parse_optional_f64_field(
    cells: &[String],
    index: Option<usize>,
    path: &Path,
    line_number: usize,
) -> Result<Option<f64>> {
    let Some(index) = index else {
        return Ok(None);
    };
    let trimmed = cell(cells, index).trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    trimmed.parse::<f64>().map(Some).map_err(|error| {
        anyhow::anyhow!(
            "manifest {} line {line_number}: failed to parse '{trimmed}' as a numeric metric: {error}",
            path.display()
        )
    })
}

fn build_dataset_comparison(
    echo_input_bases_per_sec: f64,
    echo_trimmed_read_fraction: f64,
    echo_discarded_read_fraction: f64,
    echo_corrected_bases_per_mbase: f64,
    downstream: &StudyDownstreamMetrics,
    baseline: Option<&StudyBaselineMetrics>,
) -> StudyComparisonMetrics {
    let input_speedup_vs_baseline = baseline
        .and_then(|baseline| baseline.input_bases_per_sec)
        .and_then(|baseline| ratio_option_value(echo_input_bases_per_sec, baseline));
    let trimmed_read_fraction_delta = baseline
        .and_then(|baseline| baseline.trimmed_read_fraction)
        .map(|baseline| echo_trimmed_read_fraction - baseline);
    let discarded_read_fraction_delta = baseline
        .and_then(|baseline| baseline.discarded_read_fraction)
        .map(|baseline| echo_discarded_read_fraction - baseline);
    let corrected_bases_per_mbase_delta = baseline
        .and_then(|baseline| baseline.corrected_bases_per_mbase)
        .map(|baseline| echo_corrected_bases_per_mbase - baseline);
    let alignment_rate_delta = baseline
        .and_then(|baseline| baseline.downstream.alignment_rate)
        .zip(downstream.alignment_rate)
        .map(|(baseline, echo)| echo - baseline);
    let duplicate_rate_delta = baseline
        .and_then(|baseline| baseline.downstream.duplicate_rate)
        .zip(downstream.duplicate_rate)
        .map(|(baseline, echo)| echo - baseline);
    let variant_f1_delta = baseline
        .and_then(|baseline| baseline.downstream.variant_f1)
        .zip(downstream.variant_f1)
        .map(|(baseline, echo)| echo - baseline);
    let mean_coverage_ratio = baseline
        .and_then(|baseline| baseline.downstream.mean_coverage)
        .zip(downstream.mean_coverage)
        .and_then(|(baseline, echo)| ratio_option_value(echo, baseline));
    let coverage_breadth_delta = baseline
        .and_then(|baseline| baseline.downstream.coverage_breadth)
        .zip(downstream.coverage_breadth)
        .map(|(baseline, echo)| echo - baseline);
    let assembly_n50_ratio = baseline
        .and_then(|baseline| baseline.downstream.assembly_n50)
        .zip(downstream.assembly_n50)
        .and_then(|(baseline, echo)| ratio_option_value(echo, baseline));

    StudyComparisonMetrics {
        input_speedup_vs_baseline,
        trimmed_read_fraction_delta,
        discarded_read_fraction_delta,
        corrected_bases_per_mbase_delta,
        alignment_rate_delta,
        duplicate_rate_delta,
        variant_f1_delta,
        mean_coverage_ratio,
        coverage_breadth_delta,
        assembly_n50_ratio,
    }
}

fn build_aggregate_summary(datasets: &[StudyDatasetReport]) -> StudyAggregateSummary {
    let dataset_count = datasets.len().max(1) as f64;
    StudyAggregateSummary {
        datasets: datasets.len(),
        paired_datasets: datasets.iter().filter(|dataset| dataset.paired_end).count(),
        average_input_bases_per_sec: datasets
            .iter()
            .map(|dataset| dataset.benchmark.average_input_bases_per_sec)
            .sum::<f64>()
            / dataset_count,
        average_trimmed_read_fraction: datasets
            .iter()
            .map(|dataset| dataset.trimmed_read_fraction)
            .sum::<f64>()
            / dataset_count,
        average_discarded_read_fraction: datasets
            .iter()
            .map(|dataset| dataset.discarded_read_fraction)
            .sum::<f64>()
            / dataset_count,
        average_corrected_bases_per_mbase: datasets
            .iter()
            .map(|dataset| dataset.corrected_bases_per_mbase)
            .sum::<f64>()
            / dataset_count,
    }
}

fn build_detection_summary(datasets: &[StudyDatasetReport]) -> StudyDetectionSummary {
    let datasets_with_expected_platform = datasets
        .iter()
        .filter(|dataset| dataset.expected_platform.is_some())
        .count();
    let matched_platform_datasets = datasets
        .iter()
        .filter(|dataset| dataset.platform_match == Some(true))
        .count();
    let datasets_with_expected_experiment = datasets
        .iter()
        .filter(|dataset| dataset.expected_experiment.is_some())
        .count();
    let matched_experiment_datasets = datasets
        .iter()
        .filter(|dataset| dataset.experiment_match == Some(true))
        .count();

    StudyDetectionSummary {
        datasets_with_expected_platform,
        matched_platform_datasets,
        platform_accuracy: ratio_option(matched_platform_datasets, datasets_with_expected_platform),
        datasets_with_expected_experiment,
        matched_experiment_datasets,
        experiment_accuracy: ratio_option(
            matched_experiment_datasets,
            datasets_with_expected_experiment,
        ),
    }
}

fn build_comparison_summary(datasets: &[StudyDatasetReport]) -> StudyComparisonSummary {
    let datasets_with_baseline = datasets
        .iter()
        .filter(|dataset| dataset.baseline.is_some())
        .count();
    let datasets_with_baseline_throughput = datasets
        .iter()
        .filter(|dataset| dataset.comparison.input_speedup_vs_baseline.is_some())
        .count();
    let datasets_with_alignment_metrics = datasets
        .iter()
        .filter(|dataset| dataset.comparison.alignment_rate_delta.is_some())
        .count();
    let datasets_with_duplicate_metrics = datasets
        .iter()
        .filter(|dataset| dataset.comparison.duplicate_rate_delta.is_some())
        .count();
    let datasets_with_variant_metrics = datasets
        .iter()
        .filter(|dataset| dataset.comparison.variant_f1_delta.is_some())
        .count();
    let datasets_with_mean_coverage_metrics = datasets
        .iter()
        .filter(|dataset| dataset.comparison.mean_coverage_ratio.is_some())
        .count();
    let datasets_with_coverage_breadth_metrics = datasets
        .iter()
        .filter(|dataset| dataset.comparison.coverage_breadth_delta.is_some())
        .count();
    let datasets_with_assembly_metrics = datasets
        .iter()
        .filter(|dataset| dataset.comparison.assembly_n50_ratio.is_some())
        .count();

    StudyComparisonSummary {
        datasets_with_baseline,
        datasets_with_baseline_throughput,
        datasets_with_alignment_metrics,
        datasets_with_duplicate_metrics,
        datasets_with_variant_metrics,
        datasets_with_mean_coverage_metrics,
        datasets_with_coverage_breadth_metrics,
        datasets_with_assembly_metrics,
        average_input_speedup_vs_baseline: average_option(
            datasets
                .iter()
                .filter_map(|dataset| dataset.comparison.input_speedup_vs_baseline),
        ),
        average_trimmed_read_fraction_delta: average_option(
            datasets
                .iter()
                .filter_map(|dataset| dataset.comparison.trimmed_read_fraction_delta),
        ),
        average_discarded_read_fraction_delta: average_option(
            datasets
                .iter()
                .filter_map(|dataset| dataset.comparison.discarded_read_fraction_delta),
        ),
        average_corrected_bases_per_mbase_delta: average_option(
            datasets
                .iter()
                .filter_map(|dataset| dataset.comparison.corrected_bases_per_mbase_delta),
        ),
        average_alignment_rate_delta: average_option(
            datasets
                .iter()
                .filter_map(|dataset| dataset.comparison.alignment_rate_delta),
        ),
        average_duplicate_rate_delta: average_option(
            datasets
                .iter()
                .filter_map(|dataset| dataset.comparison.duplicate_rate_delta),
        ),
        average_variant_f1_delta: average_option(
            datasets
                .iter()
                .filter_map(|dataset| dataset.comparison.variant_f1_delta),
        ),
        average_mean_coverage_ratio: average_option(
            datasets
                .iter()
                .filter_map(|dataset| dataset.comparison.mean_coverage_ratio),
        ),
        average_coverage_breadth_delta: average_option(
            datasets
                .iter()
                .filter_map(|dataset| dataset.comparison.coverage_breadth_delta),
        ),
        average_assembly_n50_ratio: average_option(
            datasets
                .iter()
                .filter_map(|dataset| dataset.comparison.assembly_n50_ratio),
        ),
    }
}

fn write_dataset_summary_csv(path: &Path, datasets: &[StudyDatasetReport]) -> Result<()> {
    let mut csv = String::from(
        "dataset_id,accession,citation,paired_end,expected_platform,detected_platform,platform_match,expected_experiment,detected_experiment,experiment_match,backend_used,zero_copy_candidate,overlap_depth,input_reads,output_reads,trimmed_reads,trimmed_fraction,discarded_reads,discarded_fraction,corrected_bases,corrected_bases_per_mbase,average_input_bases_per_sec,average_output_bases_per_sec,baseline_name,baseline_input_bases_per_sec,input_speedup_vs_baseline,echo_alignment_rate,baseline_alignment_rate,alignment_rate_delta,echo_duplicate_rate,baseline_duplicate_rate,duplicate_rate_delta,echo_variant_f1,baseline_variant_f1,variant_f1_delta,echo_mean_coverage,baseline_mean_coverage,mean_coverage_ratio,echo_coverage_breadth,baseline_coverage_breadth,coverage_breadth_delta,echo_assembly_n50,baseline_assembly_n50,assembly_n50_ratio\n",
    );
    for dataset in datasets {
        let baseline = dataset.baseline.as_ref();
        let row = vec![
            csv_escape(&dataset.dataset_id),
            csv_escape_option(dataset.accession.as_deref()),
            csv_escape_option(dataset.citation.as_deref()),
            dataset.paired_end.to_string(),
            csv_escape_option_label(dataset.expected_platform.as_ref()),
            csv_escape(&dataset.detected_platform.to_string()),
            option_bool_csv(dataset.platform_match),
            csv_escape_option_label(dataset.expected_experiment.as_ref()),
            csv_escape(&dataset.detected_experiment.to_string()),
            option_bool_csv(dataset.experiment_match),
            csv_escape(&dataset.process.backend_used),
            dataset.process.zero_copy_candidate.to_string(),
            dataset.process.execution_plan.overlap_depth.to_string(),
            dataset.process.input_reads.to_string(),
            dataset.process.output_reads.to_string(),
            dataset.process.trimmed_reads.to_string(),
            format!("{:.6}", dataset.trimmed_read_fraction),
            dataset.process.discarded_reads.to_string(),
            format!("{:.6}", dataset.discarded_read_fraction),
            dataset.process.corrected_bases.to_string(),
            format!("{:.6}", dataset.corrected_bases_per_mbase),
            format!("{:.6}", dataset.benchmark.average_input_bases_per_sec),
            format!("{:.6}", dataset.benchmark.average_output_bases_per_sec),
            csv_escape_option(baseline.map(|baseline| baseline.name.as_str())),
            option_csv(baseline.and_then(|baseline| baseline.input_bases_per_sec)),
            option_csv(dataset.comparison.input_speedup_vs_baseline),
            option_csv(dataset.downstream.alignment_rate),
            option_csv(baseline.and_then(|baseline| baseline.downstream.alignment_rate)),
            option_csv(dataset.comparison.alignment_rate_delta),
            option_csv(dataset.downstream.duplicate_rate),
            option_csv(baseline.and_then(|baseline| baseline.downstream.duplicate_rate)),
            option_csv(dataset.comparison.duplicate_rate_delta),
            option_csv(dataset.downstream.variant_f1),
            option_csv(baseline.and_then(|baseline| baseline.downstream.variant_f1)),
            option_csv(dataset.comparison.variant_f1_delta),
            option_csv(dataset.downstream.mean_coverage),
            option_csv(baseline.and_then(|baseline| baseline.downstream.mean_coverage)),
            option_csv(dataset.comparison.mean_coverage_ratio),
            option_csv(dataset.downstream.coverage_breadth),
            option_csv(baseline.and_then(|baseline| baseline.downstream.coverage_breadth)),
            option_csv(dataset.comparison.coverage_breadth_delta),
            option_csv(dataset.downstream.assembly_n50),
            option_csv(baseline.and_then(|baseline| baseline.downstream.assembly_n50)),
            option_csv(dataset.comparison.assembly_n50_ratio),
        ];
        csv.push_str(&row.join(","));
        csv.push('\n');
    }
    fs::write(path, csv).with_context(|| format!("failed to write {}", path.display()))
}

fn write_detection_summary_csv(path: &Path, detection: &StudyDetectionSummary) -> Result<()> {
    let csv = format!(
        "metric,value\n\
datasets_with_expected_platform,{}\n\
matched_platform_datasets,{}\n\
platform_accuracy,{}\n\
datasets_with_expected_experiment,{}\n\
matched_experiment_datasets,{}\n\
experiment_accuracy,{}\n",
        detection.datasets_with_expected_platform,
        detection.matched_platform_datasets,
        option_csv(detection.platform_accuracy),
        detection.datasets_with_expected_experiment,
        detection.matched_experiment_datasets,
        option_csv(detection.experiment_accuracy),
    );
    fs::write(path, csv).with_context(|| format!("failed to write {}", path.display()))
}

fn write_baseline_comparison_csv(path: &Path, datasets: &[StudyDatasetReport]) -> Result<()> {
    let mut csv = String::from(
        "dataset_id,baseline_name,echo_input_bases_per_sec,baseline_input_bases_per_sec,input_speedup_vs_baseline,echo_trimmed_read_fraction,baseline_trimmed_read_fraction,trimmed_read_fraction_delta,echo_discarded_read_fraction,baseline_discarded_read_fraction,discarded_read_fraction_delta,echo_corrected_bases_per_mbase,baseline_corrected_bases_per_mbase,corrected_bases_per_mbase_delta\n",
    );
    for dataset in datasets {
        let Some(baseline) = &dataset.baseline else {
            continue;
        };
        csv.push_str(&format!(
            "{},{},{:.6},{},{},{:.6},{},{},{:.6},{},{},{:.6},{},{}\n",
            csv_escape(&dataset.dataset_id),
            csv_escape(&baseline.name),
            dataset.benchmark.average_input_bases_per_sec,
            option_csv(baseline.input_bases_per_sec),
            option_csv(dataset.comparison.input_speedup_vs_baseline),
            dataset.trimmed_read_fraction,
            option_csv(baseline.trimmed_read_fraction),
            option_csv(dataset.comparison.trimmed_read_fraction_delta),
            dataset.discarded_read_fraction,
            option_csv(baseline.discarded_read_fraction),
            option_csv(dataset.comparison.discarded_read_fraction_delta),
            dataset.corrected_bases_per_mbase,
            option_csv(baseline.corrected_bases_per_mbase),
            option_csv(dataset.comparison.corrected_bases_per_mbase_delta)
        ));
    }
    fs::write(path, csv).with_context(|| format!("failed to write {}", path.display()))
}

fn write_downstream_metrics_csv(path: &Path, datasets: &[StudyDatasetReport]) -> Result<()> {
    let mut csv = String::from(
        "dataset_id,baseline_name,echo_alignment_rate,baseline_alignment_rate,alignment_rate_delta,echo_duplicate_rate,baseline_duplicate_rate,duplicate_rate_delta,echo_variant_f1,baseline_variant_f1,variant_f1_delta,echo_mean_coverage,baseline_mean_coverage,mean_coverage_ratio,echo_coverage_breadth,baseline_coverage_breadth,coverage_breadth_delta,echo_assembly_n50,baseline_assembly_n50,assembly_n50_ratio\n",
    );
    for dataset in datasets {
        let baseline = dataset.baseline.as_ref();
        let row = vec![
            csv_escape(&dataset.dataset_id),
            csv_escape_option(baseline.map(|baseline| baseline.name.as_str())),
            option_csv(dataset.downstream.alignment_rate),
            option_csv(baseline.and_then(|baseline| baseline.downstream.alignment_rate)),
            option_csv(dataset.comparison.alignment_rate_delta),
            option_csv(dataset.downstream.duplicate_rate),
            option_csv(baseline.and_then(|baseline| baseline.downstream.duplicate_rate)),
            option_csv(dataset.comparison.duplicate_rate_delta),
            option_csv(dataset.downstream.variant_f1),
            option_csv(baseline.and_then(|baseline| baseline.downstream.variant_f1)),
            option_csv(dataset.comparison.variant_f1_delta),
            option_csv(dataset.downstream.mean_coverage),
            option_csv(baseline.and_then(|baseline| baseline.downstream.mean_coverage)),
            option_csv(dataset.comparison.mean_coverage_ratio),
            option_csv(dataset.downstream.coverage_breadth),
            option_csv(baseline.and_then(|baseline| baseline.downstream.coverage_breadth)),
            option_csv(dataset.comparison.coverage_breadth_delta),
            option_csv(dataset.downstream.assembly_n50),
            option_csv(baseline.and_then(|baseline| baseline.downstream.assembly_n50)),
            option_csv(dataset.comparison.assembly_n50_ratio),
        ];
        csv.push_str(&row.join(","));
        csv.push('\n');
    }
    fs::write(path, csv).with_context(|| format!("failed to write {}", path.display()))
}

fn write_optional_comparison_figure<F, G>(
    path: &Path,
    title: &str,
    y_label: &str,
    baseline_name: &str,
    datasets: &[StudyDatasetReport],
    echo_value: F,
    baseline_value: G,
    description: &str,
) -> Result<Option<PaperArtifactFile>>
where
    F: Fn(&StudyDatasetReport) -> Option<f64>,
    G: Fn(&StudyDatasetReport) -> Option<f64>,
{
    let mut categories = Vec::new();
    let mut echo_values = Vec::new();
    let mut baseline_values = Vec::new();
    for dataset in datasets {
        if let (Some(echo), Some(baseline)) = (echo_value(dataset), baseline_value(dataset)) {
            categories.push(dataset.dataset_id.clone());
            echo_values.push(echo);
            baseline_values.push(baseline);
        }
    }
    if categories.is_empty() {
        return Ok(None);
    }

    write_svg(
        path,
        &grouped_bar_chart_svg(
            title,
            y_label,
            &categories,
            &[
                ChartSeries {
                    name: "JapalityECHO",
                    color: "#4E79A7",
                    values: echo_values,
                },
                ChartSeries {
                    name: baseline_name,
                    color: "#F28E2B",
                    values: baseline_values,
                },
            ],
        ),
    )?;
    Ok(Some(artifact("svg", path, description)))
}

fn detection_accuracy_categories(
    detection: &StudyDetectionSummary,
) -> Option<(Vec<String>, Vec<f64>)> {
    let mut categories = Vec::new();
    let mut values = Vec::new();
    if let Some(platform_accuracy) = detection.platform_accuracy {
        categories.push("platform".to_string());
        values.push(platform_accuracy * 100.0);
    }
    if let Some(experiment_accuracy) = detection.experiment_accuracy {
        categories.push("experiment".to_string());
        values.push(experiment_accuracy * 100.0);
    }
    if categories.is_empty() {
        None
    } else {
        Some((categories, values))
    }
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

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn density_per_mbase(numerator: usize, bases: usize) -> f64 {
    if bases == 0 {
        0.0
    } else {
        numerator as f64 / bases as f64 * 1_000_000.0
    }
}

fn ratio_option(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

fn ratio_option_value(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator == 0.0 {
        None
    } else {
        Some(numerator / denominator)
    }
}

fn average_option<I>(values: I) -> Option<f64>
where
    I: IntoIterator<Item = f64>,
{
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value;
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

fn option_csv(value: Option<f64>) -> String {
    value.map(|value| format!("{value:.6}")).unwrap_or_default()
}

fn option_bool_csv(value: Option<bool>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn manifest_numeric_option(value: Option<f64>) -> String {
    value.map(|value| format!("{value:.6}")).unwrap_or_default()
}

fn manifest_cell(value: &str, delimiter: char) -> String {
    if value.contains(delimiter)
        || value.contains('"')
        || value.contains('\n')
        || value.contains('\r')
    {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn manifest_cell_option(value: Option<&str>, delimiter: char) -> String {
    value
        .map(|value| manifest_cell(value, delimiter))
        .unwrap_or_default()
}

fn manifest_cell_option_label<T: std::fmt::Display>(value: Option<&T>, delimiter: char) -> String {
    value
        .map(ToString::to_string)
        .map(|value| manifest_cell(&value, delimiter))
        .unwrap_or_default()
}

fn render_manifest_path_for_output(path: &Path, output_parent: &Path) -> String {
    let canonical_path = path.canonicalize().ok();
    let canonical_parent = output_parent.canonicalize().ok();
    if let (Some(canonical_path), Some(canonical_parent)) =
        (canonical_path.as_ref(), canonical_parent.as_ref())
    {
        if let Ok(relative) = canonical_path.strip_prefix(canonical_parent) {
            return relative.display().to_string();
        }
    }
    canonical_path
        .unwrap_or_else(|| path.to_path_buf())
        .display()
        .to_string()
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn csv_escape_option(value: Option<&str>) -> String {
    value.map(csv_escape).unwrap_or_default()
}

fn csv_escape_option_label<T: std::fmt::Display>(value: Option<&T>) -> String {
    value
        .map(ToString::to_string)
        .map(|value| csv_escape(&value))
        .unwrap_or_default()
}

fn sanitize_filename(value: &str) -> String {
    let sanitized: String = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "dataset".to_string()
    } else {
        sanitized
    }
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
