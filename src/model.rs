use std::str::FromStr;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Platform {
    Illumina,
    Mgi,
    Ont,
    PacBio,
    IonTorrent,
    Unknown,
}

impl Platform {
    pub fn label(self) -> &'static str {
        match self {
            Self::Illumina => "Illumina",
            Self::Mgi => "MGI/DNBSEQ",
            Self::Ont => "Oxford Nanopore",
            Self::PacBio => "PacBio",
            Self::IonTorrent => "Ion Torrent",
            Self::Unknown => "Unknown",
        }
    }
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl FromStr for Platform {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let normalized = normalize_enum_label(value);
        match normalized.as_str() {
            "illumina" => Ok(Self::Illumina),
            "mgi" | "dnbseq" | "mgidnbseq" => Ok(Self::Mgi),
            "ont" | "nanopore" | "oxfordnanopore" => Ok(Self::Ont),
            "pacbio" => Ok(Self::PacBio),
            "iontorrent" | "ion_torrent" | "torrent" => Ok(Self::IonTorrent),
            "unknown" | "" => Ok(Self::Unknown),
            _ => Err(format!("unsupported platform '{value}'")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentType {
    Wgs,
    RnaSeq,
    SingleCell10xV2,
    SingleCell10xV3,
    AtacSeq,
    LongRead,
    Unknown,
}

impl ExperimentType {
    pub fn label(self) -> &'static str {
        match self {
            Self::Wgs => "WGS",
            Self::RnaSeq => "RNA-seq",
            Self::SingleCell10xV2 => "10x Genomics v2",
            Self::SingleCell10xV3 => "10x Genomics v3",
            Self::AtacSeq => "ATAC-seq",
            Self::LongRead => "Long-read",
            Self::Unknown => "Unknown",
        }
    }
}

impl std::fmt::Display for ExperimentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl FromStr for ExperimentType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let normalized = normalize_enum_label(value);
        match normalized.as_str() {
            "wgs" => Ok(Self::Wgs),
            "rnaseq" => Ok(Self::RnaSeq),
            "singlecell10xv2" | "10xv2" | "10xgenomicsv2" => Ok(Self::SingleCell10xV2),
            "singlecell10xv3" | "10xv3" | "10xgenomicsv3" => Ok(Self::SingleCell10xV3),
            "atacseq" | "atac" => Ok(Self::AtacSeq),
            "longread" => Ok(Self::LongRead),
            "unknown" | "" => Ok(Self::Unknown),
            _ => Err(format!("unsupported experiment '{value}'")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendPreference {
    Auto,
    Cpu,
    Cuda,
    Hip,
}

impl BackendPreference {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Hip => "hip",
        }
    }
}

impl std::fmt::Display for BackendPreference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AdapterCandidate {
    pub name: String,
    pub sequence: String,
    pub support: usize,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BarcodeHint {
    pub prefix_bases: usize,
    pub umi_bases: usize,
    pub poly_t_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct QualityProfile {
    pub mean_phred: f64,
    pub tail_drop: f64,
    pub cycle_means: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AutoProfile {
    pub platform: Platform,
    pub experiment: ExperimentType,
    pub read_count: usize,
    pub mean_read_length: f64,
    pub median_read_length: usize,
    pub length_stddev: f64,
    pub quality_profile: QualityProfile,
    pub adapter_candidates: Vec<AdapterCandidate>,
    pub barcode_hint: Option<BarcodeHint>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExecutionPlan {
    pub requested_backend: BackendPreference,
    pub trim_min_quality: u8,
    pub kmer_size: usize,
    pub trusted_kmer_min_count: u32,
    pub minimum_output_length: usize,
    pub zero_copy_candidate: bool,
    pub overlap_depth: usize,
    pub adapter_candidates: Vec<AdapterCandidate>,
    pub barcode_hint: Option<BarcodeHint>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AcceleratorScaffold {
    pub backend: BackendPreference,
    pub language: String,
    pub module_name: String,
    pub source_path: String,
    pub entrypoints: Vec<String>,
    pub threads_per_block: u32,
    pub vector_width_bases: usize,
    pub overlapped_streams: usize,
    pub zero_copy_candidate: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorRuntimeStatus {
    Available,
    Unavailable,
    NotRequested,
}

impl AcceleratorRuntimeStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Available => "available",
            Self::Unavailable => "unavailable",
            Self::NotRequested => "not_requested",
        }
    }
}

impl std::fmt::Display for AcceleratorRuntimeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AcceleratorRuntime {
    pub backend: BackendPreference,
    pub status: AcceleratorRuntimeStatus,
    pub driver_hint: Option<String>,
    pub compiler_hint: Option<String>,
    pub device_name: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AccelerationPreview {
    pub label: String,
    pub batch_index: usize,
    pub reads: usize,
    pub total_bases: usize,
    pub packed_bases_bytes: usize,
    pub quality_bytes: usize,
    pub offset_bytes: usize,
    pub read_pitch: usize,
    pub block_size: u32,
    pub grid_size: u32,
    pub shared_mem_bytes: usize,
    pub stream_count: usize,
    pub ambiguous_bases: usize,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AccelerationExecution {
    pub backend: BackendPreference,
    pub stage: String,
    pub successful: bool,
    pub kernel_name: String,
    pub batch_index: usize,
    pub host_pinned_bytes: usize,
    pub device_bytes: usize,
    pub transfer_bytes: usize,
    pub reads: usize,
    pub returned_trim_offsets: usize,
    pub submit_us: u64,
    pub wait_us: u64,
    pub end_to_end_us: u64,
    pub overlap_us: u64,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ThroughputSummary {
    pub wall_clock_us: u64,
    pub input_bases: usize,
    pub output_bases: usize,
    pub input_reads_per_sec: f64,
    pub output_reads_per_sec: f64,
    pub input_bases_per_sec: f64,
    pub output_bases_per_sec: f64,
    pub batches_per_sec: f64,
    pub cumulative_submit_us: u64,
    pub cumulative_wait_us: u64,
    pub cumulative_end_to_end_us: u64,
    pub cumulative_overlap_us: u64,
    pub max_batch_end_to_end_us: u64,
    pub average_submit_us: f64,
    pub average_wait_us: f64,
    pub average_end_to_end_us: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkSessionMode {
    ColdStart,
    ReuseSession,
}

impl BenchmarkSessionMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::ColdStart => "cold_start",
            Self::ReuseSession => "reuse_session",
        }
    }
}

impl std::fmt::Display for BenchmarkSessionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastqRecord {
    pub header: String,
    pub sequence: Vec<u8>,
    pub qualities: Vec<u8>,
}

impl FastqRecord {
    pub fn new(header: impl Into<String>, sequence: Vec<u8>, qualities: Vec<u8>) -> Self {
        assert_eq!(
            sequence.len(),
            qualities.len(),
            "FASTQ records must have matching sequence and quality lengths"
        );
        Self {
            header: header.into(),
            sequence,
            qualities,
        }
    }

    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    pub fn phred_scores(&self) -> impl Iterator<Item = u8> + '_ {
        self.qualities.iter().map(|q| q.saturating_sub(33))
    }

    pub fn mean_phred(&self) -> f64 {
        let len = self.qualities.len();
        if len == 0 {
            return 0.0;
        }
        let total: u64 = self.phred_scores().map(u64::from).sum();
        total as f64 / len as f64
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadPair {
    pub left: FastqRecord,
    pub right: FastqRecord,
}

impl ReadPair {
    pub fn new(left: FastqRecord, right: FastqRecord) -> Self {
        Self { left, right }
    }

    pub fn total_bases(&self) -> usize {
        self.left.len() + self.right.len()
    }
}

#[derive(Debug, Clone)]
pub struct ReadBatch {
    pub batch_index: usize,
    pub records: Vec<FastqRecord>,
    pub total_bases: usize,
}

impl ReadBatch {
    pub fn new(batch_index: usize, records: Vec<FastqRecord>) -> Self {
        let total_bases = records.iter().map(FastqRecord::len).sum();
        Self {
            batch_index,
            records,
            total_bases,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReadPairBatch {
    pub batch_index: usize,
    pub pairs: Vec<ReadPair>,
    pub total_bases: usize,
}

impl ReadPairBatch {
    pub fn new(batch_index: usize, pairs: Vec<ReadPair>) -> Self {
        let total_bases = pairs.iter().map(ReadPair::total_bases).sum();
        Self {
            batch_index,
            pairs,
            total_bases,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum IndelType {
    Deletion,
    Insertion,
}

#[derive(Debug, Clone, Serialize)]
pub struct IndelCorrection {
    pub position: usize,
    pub correction_type: IndelType,
    /// For single-base: backward-compatible Option.
    /// For multi-base: use `bases` instead.
    pub base: Option<u8>,
    /// Bases involved: inserted bases for Insertion, removed bases for Deletion.
    /// Length 0 means legacy single-base mode (check `base` field).
    #[serde(default)]
    pub bases: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ProcessedRecord {
    pub header: String,
    pub sequence: Vec<u8>,
    pub qualities: Vec<u8>,
    pub corrected_positions: Vec<usize>,
    pub indel_corrections: Vec<IndelCorrection>,
    pub trimmed_bases: usize,
    pub trimmed_adapter: Option<String>,
}

impl ProcessedRecord {
    pub fn len(&self) -> usize {
        self.sequence.len()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct InspectReport {
    pub paired_end: bool,
    pub auto_profile: AutoProfile,
    pub execution_plan: ExecutionPlan,
    pub accelerator_scaffold: Option<AcceleratorScaffold>,
    pub accelerator_runtime: Option<AcceleratorRuntime>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcessReport {
    pub paired_end: bool,
    pub input_pairs: Option<usize>,
    pub output_pairs: Option<usize>,
    pub discarded_pairs: Option<usize>,
    pub input_reads: usize,
    pub output_reads: usize,
    pub discarded_reads: usize,
    pub corrected_bases: usize,
    pub indel_corrections: usize,
    pub trimmed_reads: usize,
    pub trimmed_bases: usize,
    pub batches_processed: usize,
    pub backend_used: String,
    pub zero_copy_candidate: bool,
    pub accelerator_scaffold: Option<AcceleratorScaffold>,
    pub accelerator_runtime: Option<AcceleratorRuntime>,
    pub acceleration_previews: Vec<AccelerationPreview>,
    pub acceleration_executions: Vec<AccelerationExecution>,
    pub auto_profile: AutoProfile,
    pub execution_plan: ExecutionPlan,
    pub throughput: ThroughputSummary,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvaluationMetricSummary {
    pub true_positive: usize,
    pub false_positive: usize,
    pub false_negative: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvaluationScenarioSummary {
    pub name: String,
    pub description: String,
    pub reads: usize,
    pub expected_retained_reads: usize,
    pub observed_retained_reads: usize,
    pub exact_match_reads: usize,
    pub exact_match_rate: f64,
    pub retention: EvaluationMetricSummary,
    pub trimming: EvaluationMetricSummary,
    pub correction: EvaluationMetricSummary,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvaluationAggregateSummary {
    pub total_reads: usize,
    pub expected_retained_reads: usize,
    pub observed_retained_reads: usize,
    pub exact_match_reads: usize,
    pub exact_match_rate: f64,
    pub retention: EvaluationMetricSummary,
    pub trimming: EvaluationMetricSummary,
    pub correction: EvaluationMetricSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvaluationReport {
    pub suite_name: String,
    pub requested_backend: BackendPreference,
    pub reads_per_scenario: usize,
    pub total_scenarios: usize,
    pub aggregate: EvaluationAggregateSummary,
    pub scenarios: Vec<EvaluationScenarioSummary>,
    pub process: ProcessReport,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PairedEvaluationScenarioSummary {
    pub name: String,
    pub description: String,
    pub pairs: usize,
    pub expected_retained_pairs: usize,
    pub observed_retained_pairs: usize,
    pub exact_match_pairs: usize,
    pub exact_match_rate: f64,
    pub retention: EvaluationMetricSummary,
    pub trimming: EvaluationMetricSummary,
    pub correction: EvaluationMetricSummary,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PairedEvaluationAggregateSummary {
    pub total_pairs: usize,
    pub expected_retained_pairs: usize,
    pub observed_retained_pairs: usize,
    pub exact_match_pairs: usize,
    pub exact_match_rate: f64,
    pub retention: EvaluationMetricSummary,
    pub trimming: EvaluationMetricSummary,
    pub correction: EvaluationMetricSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct PairedEvaluationReport {
    pub suite_name: String,
    pub requested_backend: BackendPreference,
    pub pairs_per_scenario: usize,
    pub total_scenarios: usize,
    pub aggregate: PairedEvaluationAggregateSummary,
    pub scenarios: Vec<PairedEvaluationScenarioSummary>,
    pub process: ProcessReport,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackendParitySummary {
    pub single_end_exact_match_delta: f64,
    pub single_end_retention_f1_delta: f64,
    pub single_end_trimming_f1_delta: f64,
    pub single_end_correction_f1_delta: f64,
    pub paired_end_exact_match_delta: f64,
    pub paired_end_retention_f1_delta: f64,
    pub paired_end_trimming_f1_delta: f64,
    pub paired_end_correction_f1_delta: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PaperArtifactFile {
    pub kind: String,
    pub path: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PaperArtifactsReport {
    pub output_dir: String,
    pub requested_accelerator_backend: BackendPreference,
    pub reads_per_scenario: usize,
    pub benchmark_rounds: usize,
    pub cpu_evaluation: EvaluationReport,
    pub accelerator_evaluation: EvaluationReport,
    pub paired_cpu_evaluation: PairedEvaluationReport,
    pub paired_accelerator_evaluation: PairedEvaluationReport,
    pub parity: BackendParitySummary,
    pub cpu_benchmark: BenchmarkSummary,
    pub accelerator_cold_benchmark: BenchmarkSummary,
    pub accelerator_reuse_benchmark: BenchmarkSummary,
    pub accelerator_compare: BenchmarkComparisonSummary,
    pub artifacts: Vec<PaperArtifactFile>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PublicationBundleReport {
    pub input_path: String,
    pub output_dir: String,
    pub metadata_path: String,
    pub download_root: String,
    pub final_manifest_path: String,
    pub manuscript_summary_path: String,
    pub fetch: StudyFetchMetadataReport,
    pub download: StudyDownloadReport,
    pub manifest: StudyManifestBootstrapReport,
    pub annotate: Option<StudyAnnotateReport>,
    pub ingest: Option<StudyIngestReport>,
    pub study_artifacts: StudyArtifactsReport,
    pub paper_artifacts: PaperArtifactsReport,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct StudyDownstreamMetrics {
    pub alignment_rate: Option<f64>,
    pub duplicate_rate: Option<f64>,
    pub variant_f1: Option<f64>,
    pub mean_coverage: Option<f64>,
    pub coverage_breadth: Option<f64>,
    pub assembly_n50: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyBaselineMetrics {
    pub name: String,
    pub input_bases_per_sec: Option<f64>,
    pub trimmed_read_fraction: Option<f64>,
    pub discarded_read_fraction: Option<f64>,
    pub corrected_bases_per_mbase: Option<f64>,
    pub downstream: StudyDownstreamMetrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyComparisonMetrics {
    pub input_speedup_vs_baseline: Option<f64>,
    pub trimmed_read_fraction_delta: Option<f64>,
    pub discarded_read_fraction_delta: Option<f64>,
    pub corrected_bases_per_mbase_delta: Option<f64>,
    pub alignment_rate_delta: Option<f64>,
    pub duplicate_rate_delta: Option<f64>,
    pub variant_f1_delta: Option<f64>,
    pub mean_coverage_ratio: Option<f64>,
    pub coverage_breadth_delta: Option<f64>,
    pub assembly_n50_ratio: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyDatasetReport {
    pub dataset_id: String,
    pub accession: Option<String>,
    pub citation: Option<String>,
    pub paired_end: bool,
    pub input1: String,
    pub input2: Option<String>,
    pub expected_platform: Option<Platform>,
    pub detected_platform: Platform,
    pub platform_match: Option<bool>,
    pub expected_experiment: Option<ExperimentType>,
    pub detected_experiment: ExperimentType,
    pub experiment_match: Option<bool>,
    pub trimmed_read_fraction: f64,
    pub discarded_read_fraction: f64,
    pub corrected_bases_per_mbase: f64,
    pub downstream: StudyDownstreamMetrics,
    pub baseline: Option<StudyBaselineMetrics>,
    pub comparison: StudyComparisonMetrics,
    pub inspect: InspectReport,
    pub process: ProcessReport,
    pub benchmark: BenchmarkSummary,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyAggregateSummary {
    pub datasets: usize,
    pub paired_datasets: usize,
    pub average_input_bases_per_sec: f64,
    pub average_trimmed_read_fraction: f64,
    pub average_discarded_read_fraction: f64,
    pub average_corrected_bases_per_mbase: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyDetectionSummary {
    pub datasets_with_expected_platform: usize,
    pub matched_platform_datasets: usize,
    pub platform_accuracy: Option<f64>,
    pub datasets_with_expected_experiment: usize,
    pub matched_experiment_datasets: usize,
    pub experiment_accuracy: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyComparisonSummary {
    pub datasets_with_baseline: usize,
    pub datasets_with_baseline_throughput: usize,
    pub datasets_with_alignment_metrics: usize,
    pub datasets_with_duplicate_metrics: usize,
    pub datasets_with_variant_metrics: usize,
    pub datasets_with_mean_coverage_metrics: usize,
    pub datasets_with_coverage_breadth_metrics: usize,
    pub datasets_with_assembly_metrics: usize,
    pub average_input_speedup_vs_baseline: Option<f64>,
    pub average_trimmed_read_fraction_delta: Option<f64>,
    pub average_discarded_read_fraction_delta: Option<f64>,
    pub average_corrected_bases_per_mbase_delta: Option<f64>,
    pub average_alignment_rate_delta: Option<f64>,
    pub average_duplicate_rate_delta: Option<f64>,
    pub average_variant_f1_delta: Option<f64>,
    pub average_mean_coverage_ratio: Option<f64>,
    pub average_coverage_breadth_delta: Option<f64>,
    pub average_assembly_n50_ratio: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyManifestBootstrapSummary {
    pub datasets: usize,
    pub paired_datasets: usize,
    pub datasets_with_generated_id: usize,
    pub datasets_with_accession: usize,
    pub datasets_with_citation: usize,
    pub datasets_with_expected_platform: usize,
    pub datasets_with_expected_experiment: usize,
    pub datasets_with_baseline_name: usize,
    pub datasets_with_downstream_metrics: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyManifestBootstrapReport {
    pub input_path: String,
    pub output_path: String,
    pub provenance_summary_path: String,
    pub default_baseline_name: Option<String>,
    pub summary: StudyManifestBootstrapSummary,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyDiscoverDataset {
    pub dataset_id: String,
    pub accession: Option<String>,
    pub paired_end: bool,
    pub input1: String,
    pub input2: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyDiscoverSummary {
    pub files_scanned: usize,
    pub fastq_files: usize,
    pub datasets: usize,
    pub paired_datasets: usize,
    pub single_end_datasets: usize,
    pub accession_labeled_datasets: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyDiscoverReport {
    pub input_dir: String,
    pub output_path: String,
    pub recursive: bool,
    pub summary: StudyDiscoverSummary,
    pub datasets: Vec<StudyDiscoverDataset>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyFetchMetadataSummary {
    pub requested_accessions: usize,
    pub unique_accessions: usize,
    pub resumed_accessions: usize,
    pub resumed_records: usize,
    pub geo_bridge_accessions: usize,
    pub geo_bridge_resolved_accessions: usize,
    pub geo_bridge_fallback_accessions: usize,
    pub fetched_records: usize,
    pub matched_accessions: usize,
    pub unmatched_accessions: usize,
    pub failed_accessions: usize,
    pub cache_hits: usize,
    pub remote_fetches: usize,
    pub retried_chunks: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyFetchMetadataReport {
    pub input_path: String,
    pub output_path: String,
    pub source: String,
    pub base_url: String,
    pub geo_base_url: String,
    pub chunk_size: usize,
    pub retries: usize,
    pub resume_existing: bool,
    pub cache_dir: String,
    pub status_path: String,
    pub summary: StudyFetchMetadataSummary,
    pub unmatched_accessions: Vec<String>,
    pub failed_accessions: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyDownloadSummary {
    pub datasets: usize,
    pub requested_files: usize,
    pub downloaded_files: usize,
    pub resumed_files: usize,
    pub skipped_existing_files: usize,
    pub failed_files: usize,
    pub available_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyDownloadReport {
    pub input_path: String,
    pub download_root: String,
    pub status_path: String,
    pub retries: usize,
    pub overwrite_existing: bool,
    pub summary: StudyDownloadSummary,
    pub failed_destinations: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyAnnotateSummary {
    pub datasets: usize,
    pub annotation_rows: usize,
    pub matched_rows: usize,
    pub unmatched_rows: usize,
    pub datasets_changed: usize,
    pub fields_filled: usize,
    pub fields_overwritten: usize,
    pub notes_appended: usize,
    pub before: StudyManifestBootstrapSummary,
    pub after: StudyManifestBootstrapSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyAnnotateReport {
    pub manifest_path: String,
    pub annotations_path: String,
    pub output_path: String,
    pub summary_path: String,
    pub overwrite_existing: bool,
    pub summary: StudyAnnotateSummary,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyIngestSummary {
    pub files_scanned: usize,
    pub structured_files: usize,
    pub json_files: usize,
    pub delimited_files: usize,
    pub ignored_files: usize,
    pub records_ingested: usize,
    pub generated_annotation_rows: usize,
    pub matched_records: usize,
    pub unmatched_records: usize,
    pub datasets_changed: usize,
    pub fields_filled: usize,
    pub fields_overwritten: usize,
    pub notes_appended: usize,
    pub before: StudyManifestBootstrapSummary,
    pub after: StudyManifestBootstrapSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyIngestReport {
    pub manifest_path: String,
    pub input_dir: String,
    pub output_path: String,
    pub generated_annotations_path: String,
    pub summary_path: String,
    pub recursive: bool,
    pub overwrite_existing: bool,
    pub summary: StudyIngestSummary,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudyArtifactsReport {
    pub output_dir: String,
    pub manifest_path: String,
    pub requested_backend: BackendPreference,
    pub sample_size: usize,
    pub batch_reads: usize,
    pub benchmark_rounds: usize,
    pub aggregate: StudyAggregateSummary,
    pub detection: StudyDetectionSummary,
    pub comparison: StudyComparisonSummary,
    pub datasets: Vec<StudyDatasetReport>,
    pub artifacts: Vec<PaperArtifactFile>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkRound {
    pub round_index: usize,
    pub process: ProcessReport,
}

fn normalize_enum_label(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub session_mode: BenchmarkSessionMode,
    pub setup_us: u64,
    pub rounds: usize,
    pub backends: Vec<String>,
    pub backend_consistent: bool,
    pub average_wall_clock_us: f64,
    pub best_wall_clock_us: u64,
    pub worst_wall_clock_us: u64,
    pub average_input_reads_per_sec: f64,
    pub average_output_reads_per_sec: f64,
    pub average_input_bases_per_sec: f64,
    pub average_output_bases_per_sec: f64,
    pub average_cumulative_overlap_us: f64,
    pub best_cumulative_overlap_us: u64,
    pub warmup_wall_clock_us: u64,
    pub steady_state_average_wall_clock_us: Option<f64>,
    pub warmup_penalty_pct: Option<f64>,
    pub warmup_transfer_bytes: usize,
    pub steady_state_average_transfer_bytes: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkReport {
    pub paired_end: bool,
    pub summary: BenchmarkSummary,
    pub rounds: Vec<BenchmarkRound>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparisonSummary {
    pub rounds: usize,
    pub cold_start_backends: Vec<String>,
    pub reuse_session_backends: Vec<String>,
    pub backend_match: bool,
    pub cold_start_average_wall_clock_us: f64,
    pub reuse_session_average_wall_clock_us: f64,
    pub reuse_session_setup_us: u64,
    pub reuse_session_amortized_average_wall_clock_us: f64,
    pub average_wall_clock_delta_us: f64,
    pub amortized_wall_clock_delta_us: f64,
    pub raw_speedup: Option<f64>,
    pub amortized_speedup: Option<f64>,
    pub input_bases_per_sec_uplift_pct: Option<f64>,
    pub output_bases_per_sec_uplift_pct: Option<f64>,
    pub average_overlap_delta_us: f64,
    pub steady_state_transfer_savings_bytes: Option<f64>,
    pub steady_state_transfer_savings_pct: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkComparisonReport {
    pub paired_end: bool,
    pub cold_start: BenchmarkReport,
    pub reuse_session: BenchmarkReport,
    pub summary: BenchmarkComparisonSummary,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HistoryReport {
    pub source_path: String,
    pub total_entries: usize,
    pub benchmark_entries: usize,
    pub benchmark_compare_entries: usize,
    pub labels: Vec<String>,
    pub requested_backends: Vec<BackendPreference>,
    pub latest_recorded_at_unix_ms: Option<u64>,
    pub latest_entry: Option<HistoryEntrySummary>,
    pub recent_entries: Vec<HistoryEntrySummary>,
    pub comparison_stats: HistoryComparisonStats,
    pub regression: Option<HistoryRegressionAnalysis>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HistoryEntrySummary {
    pub kind: String,
    pub label: Option<String>,
    pub recorded_at_unix_ms: u64,
    pub requested_backend: BackendPreference,
    pub rounds: usize,
    pub paired_end: bool,
    pub average_wall_clock_us: f64,
    pub setup_us: Option<u64>,
    pub session_mode: Option<BenchmarkSessionMode>,
    pub raw_speedup: Option<f64>,
    pub amortized_speedup: Option<f64>,
    pub steady_state_transfer_savings_pct: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct HistoryComparisonStats {
    pub latest_average_wall_clock_delta_us: Option<f64>,
    pub latest_raw_speedup: Option<f64>,
    pub latest_amortized_speedup: Option<f64>,
    pub average_raw_speedup: Option<f64>,
    pub average_amortized_speedup: Option<f64>,
    pub average_transfer_savings_pct: Option<f64>,
    pub best_raw_speedup: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct HistoryRegressionThresholds {
    pub max_wall_clock_regression_pct: Option<f64>,
    pub min_raw_speedup: Option<f64>,
    pub min_transfer_savings_pct: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryRegressionStatus {
    Pass,
    Alert,
    NoBaseline,
    NotComparable,
}

impl HistoryRegressionStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Alert => "alert",
            Self::NoBaseline => "no_baseline",
            Self::NotComparable => "not_comparable",
        }
    }
}

impl std::fmt::Display for HistoryRegressionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct HistoryRegressionAnalysis {
    pub baseline_source: String,
    pub target_entry: HistoryEntrySummary,
    pub baseline_entry: Option<HistoryEntrySummary>,
    pub comparable: bool,
    pub status: HistoryRegressionStatus,
    pub thresholds: HistoryRegressionThresholds,
    pub wall_clock_delta_us: Option<f64>,
    pub wall_clock_delta_pct: Option<f64>,
    pub raw_speedup_delta: Option<f64>,
    pub amortized_speedup_delta: Option<f64>,
    pub transfer_savings_pct_delta: Option<f64>,
    pub alerts: Vec<String>,
}
