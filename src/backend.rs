use crate::cuda_runtime::CudaSession;
use crate::gpu::{
    PreparedAdapterCandidates, PreparedTrustedKmers, decode_packed_base_code,
    packed_ambiguity_pitch, packed_base_pitch, prepare_adapter_candidates, prepare_trusted_kmers,
    probe_runtime, scaffold_for_backend,
};
use rayon::prelude::*;
use serde::Serialize;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::algorithm::{process_record, process_record_with_quality_cutoff};
use crate::model::{
    AccelerationExecution, AcceleratorRuntime, AcceleratorRuntimeStatus, AcceleratorScaffold,
    BackendPreference, ExecutionPlan, ProcessedRecord, ReadBatch,
};
use crate::spectrum::KmerSpectrum;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    PageableHost,
    PinnedHost,
    Device,
    MappedZeroCopy,
}

impl std::fmt::Display for MemoryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PageableHost => f.write_str("pageable_host"),
            Self::PinnedHost => f.write_str("pinned_host"),
            Self::Device => f.write_str("device"),
            Self::MappedZeroCopy => f.write_str("mapped_zero_copy"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BatchLayout {
    pub batch_index: usize,
    pub reads: usize,
    pub total_bases: usize,
    pub host_bytes: usize,
    pub preferred_memory: MemoryKind,
    pub zero_copy_candidate: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TransferPlan {
    pub source: MemoryKind,
    pub destination: MemoryKind,
    pub bytes: usize,
    pub asynchronous: bool,
    pub zero_copy: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct BatchStats {
    pub processed_reads: usize,
    pub corrected_bases: usize,
    pub indel_corrections: usize,
    pub trimmed_reads: usize,
    pub trimmed_bases: usize,
}

#[derive(Debug, Clone)]
pub struct ProcessedBatch {
    pub layout: BatchLayout,
    pub records: Vec<ProcessedRecord>,
    pub stats: BatchStats,
    pub transfer_plan: TransferPlan,
    pub acceleration_execution: Option<AccelerationExecution>,
    pub timing: BatchTiming,
}

#[derive(Debug, Clone, Default)]
pub struct BatchTiming {
    pub batch_index: usize,
    pub submit_us: u64,
    pub wait_us: u64,
    pub end_to_end_us: u64,
    pub overlap_us: u64,
}

pub trait PendingProcessedBatch {
    fn wait(self: Box<Self>) -> ProcessedBatch;
}

pub trait ExecutionBackend {
    fn name(&self) -> &'static str;
    fn transfer_plan(&self, batch: &ReadBatch, plan: &ExecutionPlan) -> TransferPlan;
    fn submit_batch(
        &self,
        batch: ReadBatch,
        plan: Arc<ExecutionPlan>,
        spectrum: Arc<KmerSpectrum>,
    ) -> Box<dyn PendingProcessedBatch>;
}

struct ReadyProcessedBatch(Option<ProcessedBatch>);

impl PendingProcessedBatch for ReadyProcessedBatch {
    fn wait(mut self: Box<Self>) -> ProcessedBatch {
        self.0.take().expect("ready processed batch must exist")
    }
}

pub struct CpuBackend;

impl ExecutionBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn transfer_plan(&self, batch: &ReadBatch, plan: &ExecutionPlan) -> TransferPlan {
        let layout = BatchLayout::from_batch(batch, plan);
        let mut notes = vec![
            "CPU backend keeps data in host memory for this build".to_string(),
            "Pinned and mapped-memory descriptors are preserved for future CUDA/HIP kernels"
                .to_string(),
        ];

        if layout.zero_copy_candidate {
            notes.push(
                "Batch shape is compatible with overlapped DMA + zero-copy planning once a GPU backend is linked"
                    .to_string(),
            );
        }

        TransferPlan {
            source: layout.preferred_memory,
            destination: MemoryKind::PageableHost,
            bytes: layout.host_bytes,
            asynchronous: false,
            zero_copy: false,
            notes,
        }
    }

    fn submit_batch(
        &self,
        batch: ReadBatch,
        plan: Arc<ExecutionPlan>,
        spectrum: Arc<KmerSpectrum>,
    ) -> Box<dyn PendingProcessedBatch> {
        let submitted_at = Instant::now();
        let mut processed = process_cpu_batch(&batch, &plan, &spectrum);
        let submit_us = duration_as_us(submitted_at.elapsed());
        processed.timing = BatchTiming {
            batch_index: processed.layout.batch_index,
            submit_us,
            wait_us: 0,
            end_to_end_us: submit_us,
            overlap_us: 0,
        };
        Box::new(ReadyProcessedBatch(Some(processed)))
    }
}

pub struct CudaBackend {
    scaffold: AcceleratorScaffold,
    session: Arc<CudaSession>,
    trusted_kmers: PreparedTrustedKmers,
    adapters: PreparedAdapterCandidates,
}

impl CudaBackend {
    pub fn new(
        scaffold: AcceleratorScaffold,
        session: Arc<CudaSession>,
        trusted_kmers: PreparedTrustedKmers,
        adapters: PreparedAdapterCandidates,
    ) -> Self {
        Self {
            scaffold,
            session,
            trusted_kmers,
            adapters,
        }
    }
}

impl ExecutionBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda-full"
    }

    fn transfer_plan(&self, batch: &ReadBatch, _plan: &ExecutionPlan) -> TransferPlan {
        let read_pitch = batch
            .records
            .iter()
            .map(|record| record.len())
            .max()
            .unwrap_or(0);
        let packed_read_pitch = packed_base_pitch(read_pitch);
        let ambiguity_pitch = packed_ambiguity_pitch(read_pitch);
        let length_bytes = batch.records.len() * std::mem::size_of::<u32>();
        let base_bytes = packed_read_pitch * batch.records.len();
        let ambiguity_bytes = ambiguity_pitch * batch.records.len();
        let quality_bytes = read_pitch * batch.records.len();
        let output_bytes =
            batch.records.len() * std::mem::size_of::<u32>() * 2 + base_bytes + ambiguity_bytes;
        let trusted_bytes = self.trusted_kmers.keys.len() * std::mem::size_of::<u64>()
            + self.trusted_kmers.counts.len() * std::mem::size_of::<u32>();
        let adapter_bytes = self.adapters.codes.len()
            + self.adapters.lengths.len() * std::mem::size_of::<u32>()
            + self.adapters.supports.len() * std::mem::size_of::<u32>();
        TransferPlan {
            source: MemoryKind::PinnedHost,
            destination: MemoryKind::Device,
            bytes: base_bytes
                + ambiguity_bytes
                + quality_bytes
                + length_bytes
                + output_bytes
                + trusted_bytes
                + adapter_bytes,
            asynchronous: true,
            zero_copy: false,
            notes: vec![
                format!(
                    "CUDA full backend stages batch {} through reusable pinned host memory with packed base lanes",
                    batch.batch_index
                ),
                format!(
                    "CUDA kernel '{}' will run on {} via {} async pipeline slots",
                    "japalityecho_trim_correct",
                    self.session.device_name(),
                    self.session.pipeline_slots()
                ),
                "Trusted k-mer and adapter tables stay resident on device after warm-up upload"
                    .to_string(),
                format!(
                    "Base staging uses 2-bit packing ({}B/read) plus ambiguity bitmap ({}B/read)",
                    packed_read_pitch, ambiguity_pitch
                ),
                format!("PTX artifact: {}", self.session.ptx_path().display()),
            ],
        }
    }

    fn submit_batch(
        &self,
        batch: ReadBatch,
        plan: Arc<ExecutionPlan>,
        spectrum: Arc<KmerSpectrum>,
    ) -> Box<dyn PendingProcessedBatch> {
        let submitted_at = Instant::now();
        let transfer_plan = self.transfer_plan(&batch, &plan);
        let dispatch = self.session.submit_trim_correct(
            &batch,
            &plan,
            &self.scaffold,
            &self.trusted_kmers,
            &self.adapters,
        );

        match dispatch {
            Ok(pending) => Box::new(CudaPendingProcessedBatch {
                batch,
                plan,
                spectrum,
                transfer_plan,
                session: Arc::clone(&self.session),
                pending: Some(pending),
                adapter_names: self.adapters.names.clone(),
                submitted_at,
                submit_us: duration_as_us(submitted_at.elapsed()),
            }),
            Err(error) => {
                let mut processed =
                    process_cuda_fallback_batch(batch, &plan, &spectrum, transfer_plan, error);
                let submit_us = duration_as_us(submitted_at.elapsed());
                processed.timing = BatchTiming {
                    batch_index: processed.layout.batch_index,
                    submit_us,
                    wait_us: 0,
                    end_to_end_us: submit_us,
                    overlap_us: 0,
                };
                if let Some(execution) = processed.acceleration_execution.as_mut() {
                    apply_timing_to_execution(execution, &processed.timing);
                }
                Box::new(ReadyProcessedBatch(Some(processed)))
            }
        }
    }
}

fn process_cpu_batch(
    batch: &ReadBatch,
    plan: &ExecutionPlan,
    spectrum: &KmerSpectrum,
) -> ProcessedBatch {
    let layout = BatchLayout::from_batch(batch, plan);
    let transfer_plan = CpuBackend.transfer_plan(batch, plan);
    let corrected_bases = AtomicUsize::new(0);
    let indel_corrections_count = AtomicUsize::new(0);
    let trimmed_reads = AtomicUsize::new(0);
    let trimmed_bases = AtomicUsize::new(0);
    let records: Vec<ProcessedRecord> = batch
        .records
        .par_iter()
        .map(|record| {
            let processed = process_record(record, plan, spectrum);
            corrected_bases.fetch_add(processed.corrected_positions.len(), Ordering::Relaxed);
            indel_corrections_count.fetch_add(processed.indel_corrections.len(), Ordering::Relaxed);
            if processed.trimmed_bases > 0 {
                trimmed_reads.fetch_add(1, Ordering::Relaxed);
                trimmed_bases.fetch_add(processed.trimmed_bases, Ordering::Relaxed);
            }
            processed
        })
        .collect();

    let stats = BatchStats {
        processed_reads: records.len(),
        corrected_bases: corrected_bases.load(Ordering::Relaxed),
        indel_corrections: indel_corrections_count.load(Ordering::Relaxed),
        trimmed_reads: trimmed_reads.load(Ordering::Relaxed),
        trimmed_bases: trimmed_bases.load(Ordering::Relaxed),
    };

    ProcessedBatch {
        layout,
        records,
        stats,
        transfer_plan,
        acceleration_execution: None,
        timing: BatchTiming::default(),
    }
}

fn process_cuda_fallback_batch(
    batch: ReadBatch,
    plan: &ExecutionPlan,
    spectrum: &KmerSpectrum,
    mut transfer_plan: TransferPlan,
    error: anyhow::Error,
) -> ProcessedBatch {
    let layout = BatchLayout::from_batch(&batch, plan);
    transfer_plan.source = layout.preferred_memory;
    transfer_plan.destination = MemoryKind::PageableHost;
    transfer_plan.bytes = layout.host_bytes;
    transfer_plan.notes.push(format!(
        "CUDA dispatch failed for batch {}; falling back to CPU correction/trimming: {error}",
        batch.batch_index
    ));
    let records = collect_processed_records_with_hints(&batch, plan, spectrum, None);
    let acceleration_execution = Some(AccelerationExecution {
        backend: BackendPreference::Cuda,
        stage: "full_trim_correct".to_string(),
        successful: false,
        kernel_name: "japalityecho_trim_correct".to_string(),
        batch_index: batch.batch_index,
        host_pinned_bytes: 0,
        device_bytes: 0,
        transfer_bytes: 0,
        reads: batch.records.len(),
        returned_trim_offsets: 0,
        submit_us: 0,
        wait_us: 0,
        end_to_end_us: 0,
        overlap_us: 0,
        notes: vec![format!(
            "CUDA session was available, but dispatch failed and CPU fallback processed this batch: {error}"
        )],
    });
    let stats = build_batch_stats(&records);
    ProcessedBatch {
        layout,
        records,
        stats,
        transfer_plan,
        acceleration_execution,
        timing: BatchTiming::default(),
    }
}

struct CudaPendingProcessedBatch {
    batch: ReadBatch,
    plan: Arc<ExecutionPlan>,
    spectrum: Arc<KmerSpectrum>,
    transfer_plan: TransferPlan,
    session: Arc<CudaSession>,
    pending: Option<crate::cuda_runtime::CudaPendingDispatch>,
    adapter_names: Vec<String>,
    submitted_at: Instant,
    submit_us: u64,
}

impl PendingProcessedBatch for CudaPendingProcessedBatch {
    fn wait(mut self: Box<Self>) -> ProcessedBatch {
        let layout = BatchLayout::from_batch(&self.batch, &self.plan);
        let mut transfer_plan = self.transfer_plan.clone();
        let wait_started = Instant::now();

        let (records, mut acceleration_execution) = match self.session.wait_trim_correct(
            self.pending
                .take()
                .expect("pending CUDA dispatch must exist"),
        ) {
            Ok(result) => {
                transfer_plan.bytes = result.execution.transfer_bytes;
                transfer_plan
                    .notes
                    .extend(result.execution.notes.iter().cloned());
                (
                    collect_processed_records_from_gpu(&self.batch, &result, &self.adapter_names),
                    Some(result.execution),
                )
            }
            Err(error) => {
                transfer_plan.source = layout.preferred_memory;
                transfer_plan.destination = MemoryKind::PageableHost;
                transfer_plan.bytes = layout.host_bytes;
                transfer_plan.notes.push(format!(
                    "CUDA dispatch failed for batch {}; falling back to CPU correction/trimming: {error}",
                    self.batch.batch_index
                ));
                let fallback_execution = AccelerationExecution {
                    backend: BackendPreference::Cuda,
                    stage: "full_trim_correct".to_string(),
                    successful: false,
                    kernel_name: "japalityecho_trim_correct".to_string(),
                    batch_index: self.batch.batch_index,
                    host_pinned_bytes: 0,
                    device_bytes: 0,
                    transfer_bytes: 0,
                    reads: self.batch.records.len(),
                    returned_trim_offsets: 0,
                    submit_us: 0,
                    wait_us: 0,
                    end_to_end_us: 0,
                    overlap_us: 0,
                    notes: vec![format!(
                        "CUDA session was available, but async dispatch failed and CPU fallback processed this batch: {error}"
                    )],
                };
                (
                    collect_processed_records_with_hints(
                        &self.batch,
                        &self.plan,
                        &self.spectrum,
                        None,
                    ),
                    Some(fallback_execution),
                )
            }
        };

        let wait_us = duration_as_us(wait_started.elapsed());
        let end_to_end_us = duration_as_us(self.submitted_at.elapsed());
        let timing = BatchTiming {
            batch_index: layout.batch_index,
            submit_us: self.submit_us,
            wait_us,
            end_to_end_us,
            overlap_us: end_to_end_us.saturating_sub(self.submit_us.saturating_add(wait_us)),
        };
        if let Some(execution) = acceleration_execution.as_mut() {
            apply_timing_to_execution(execution, &timing);
        }
        let stats = build_batch_stats(&records);
        ProcessedBatch {
            layout,
            records,
            stats,
            transfer_plan,
            acceleration_execution,
            timing,
        }
    }
}

fn duration_as_us(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

fn apply_timing_to_execution(execution: &mut AccelerationExecution, timing: &BatchTiming) {
    execution.batch_index = timing.batch_index;
    execution.submit_us = timing.submit_us;
    execution.wait_us = timing.wait_us;
    execution.end_to_end_us = timing.end_to_end_us;
    execution.overlap_us = timing.overlap_us;
    execution.notes.push(format!(
        "Host timing: submit={}us wait={}us end_to_end={}us overlap={}us",
        timing.submit_us, timing.wait_us, timing.end_to_end_us, timing.overlap_us
    ));
}

impl BatchLayout {
    fn from_batch(batch: &ReadBatch, plan: &ExecutionPlan) -> Self {
        let header_bytes: usize = batch.records.iter().map(|record| record.header.len()).sum();
        Self {
            batch_index: batch.batch_index,
            reads: batch.records.len(),
            total_bases: batch.total_bases,
            host_bytes: batch.total_bases * 2 + header_bytes,
            preferred_memory: if plan.zero_copy_candidate {
                MemoryKind::PinnedHost
            } else {
                MemoryKind::PageableHost
            },
            zero_copy_candidate: plan.zero_copy_candidate,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BackendResolution {
    pub backend_name: &'static str,
    pub accelerator_scaffold: Option<AcceleratorScaffold>,
    pub accelerator_runtime: Option<AcceleratorRuntime>,
    pub notes: Vec<String>,
}

pub struct BackendInstance {
    pub backend: Box<dyn ExecutionBackend>,
    pub notes: Vec<String>,
}

pub struct HeterogeneousScheduler {
    requested: BackendPreference,
}

impl HeterogeneousScheduler {
    pub fn new(requested: BackendPreference) -> Self {
        Self { requested }
    }

    pub fn resolve(&self, plan: &ExecutionPlan) -> BackendResolution {
        let mut notes = Vec::new();
        let accelerator_scaffold = scaffold_for_backend(self.requested, plan);
        let accelerator_runtime = probe_runtime(self.requested);

        match self.requested {
            BackendPreference::Cpu => {
                notes.push("CPU backend explicitly requested".to_string());
            }
            BackendPreference::Auto => {
                if plan.zero_copy_candidate {
                    notes.push(
                        "Auto mode detected a GPU-friendly transfer shape, but this build only ships the CPU executor"
                            .to_string(),
                    );
                } else {
                    notes.push(
                        "Auto mode kept execution on CPU for the profiled workload".to_string(),
                    );
                }
            }
            BackendPreference::Cuda => {
                notes.push(
                    "CUDA was requested; this build can use a full CUDA trim/correct executor when runtime setup succeeds"
                        .to_string(),
                );
            }
            BackendPreference::Hip => {
                notes.push(
                    "HIP was requested; this build exposes a kernel scaffold and falls back to CPU execution"
                        .to_string(),
                );
            }
        }

        if let Some(scaffold) = &accelerator_scaffold {
            notes.push(format!(
                "Prepared {} kernel module '{}' with entrypoints {}",
                scaffold.language,
                scaffold.module_name,
                scaffold.entrypoints.join(", ")
            ));
        }

        if let Some(runtime) = &accelerator_runtime {
            notes.push(format!(
                "{} runtime probe: {}",
                runtime.backend, runtime.status
            ));
            if let Some(device_name) = &runtime.device_name {
                notes.push(format!("Detected accelerator device: {device_name}"));
            }
        }

        let backend_name = match self.requested {
            BackendPreference::Cuda
                if accelerator_runtime.as_ref().is_some_and(|runtime| {
                    runtime.status == AcceleratorRuntimeStatus::Available
                }) =>
            {
                "cuda-full"
            }
            _ => "cpu",
        };

        BackendResolution {
            backend_name,
            accelerator_scaffold,
            accelerator_runtime,
            notes,
        }
    }

    pub fn instantiate(
        &self,
        resolution: &BackendResolution,
        plan: &ExecutionPlan,
        spectrum: &KmerSpectrum,
    ) -> BackendInstance {
        match self.requested {
            BackendPreference::Cuda
                if resolution
                    .accelerator_runtime
                    .as_ref()
                    .is_some_and(|runtime| {
                        runtime.status == AcceleratorRuntimeStatus::Available
                    }) =>
            {
                if let Some(scaffold) = resolution.accelerator_scaffold.clone() {
                    match CudaSession::try_new(&scaffold) {
                        Ok(session) => {
                            let trusted_kmers =
                                prepare_trusted_kmers(spectrum, plan.trusted_kmer_min_count);
                            let adapters = prepare_adapter_candidates(&plan.adapter_candidates, 4);
                            let session = Arc::new(session);
                            BackendInstance {
                                backend: Box::new(CudaBackend::new(
                                    scaffold,
                                    Arc::clone(&session),
                                    trusted_kmers,
                                    adapters,
                                )),
                                notes: vec![
                                    "CUDA full backend initialized successfully".to_string(),
                                    format!(
                                        "CUDA batch pipeline armed with {} in-flight stream slots",
                                        session.pipeline_slots()
                                    ),
                                ],
                            }
                        }
                        Err(error) => BackendInstance {
                            backend: Box::new(CpuBackend),
                            notes: vec![format!(
                                "CUDA runtime was available but backend initialization failed; using CPU fallback: {error}"
                            )],
                        },
                    }
                } else {
                    BackendInstance {
                        backend: Box::new(CpuBackend),
                        notes: vec![
                            "CUDA was requested but no accelerator scaffold was available; using CPU fallback"
                                .to_string(),
                        ],
                    }
                }
            }
            _ => BackendInstance {
                backend: Box::new(CpuBackend),
                notes: Vec::new(),
            },
        }
    }
}

fn collect_processed_records_with_hints(
    batch: &ReadBatch,
    plan: &ExecutionPlan,
    spectrum: &KmerSpectrum,
    trim_offsets: Option<&[u32]>,
) -> Vec<ProcessedRecord> {
    batch
        .records
        .par_iter()
        .enumerate()
        .map(|(index, record)| {
            let trim_hint = trim_offsets.and_then(|offsets| offsets.get(index).copied());
            match trim_hint {
                Some(offset) => process_record_with_quality_cutoff(
                    record,
                    plan,
                    spectrum,
                    Some(offset as usize),
                ),
                None => process_record(record, plan, spectrum),
            }
        })
        .collect()
}

fn collect_processed_records_from_gpu(
    batch: &ReadBatch,
    dispatch: &crate::cuda_runtime::CudaDispatchResult,
    adapter_names: &[String],
) -> Vec<ProcessedRecord> {
    batch
        .records
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let row_start = index * dispatch.packed_read_pitch;
            let row_end = row_start + dispatch.packed_read_pitch;
            let ambiguity_start = index * dispatch.ambiguity_pitch;
            let ambiguity_end = ambiguity_start + dispatch.ambiguity_pitch;
            let row = &dispatch.corrected_packed_codes[row_start..row_end];
            let ambiguity = &dispatch.corrected_ambiguity_bits[ambiguity_start..ambiguity_end];
            let trim_offset = dispatch
                .trim_offsets
                .get(index)
                .copied()
                .unwrap_or(record.len() as u32) as usize;
            let trim_offset = trim_offset.min(record.len());
            let mut corrected_positions = Vec::new();
            let mut sequence = Vec::with_capacity(trim_offset);

            for position in 0..record.len() {
                let corrected_base = decode_packed_base_code(row, ambiguity, position);
                if corrected_base != record.sequence[position].to_ascii_uppercase() {
                    corrected_positions.push(position);
                }
                if position < trim_offset {
                    sequence.push(corrected_base);
                }
            }

            let mut qualities = record.qualities.clone();
            qualities.truncate(trim_offset);
            let trimmed_bases = record.len().saturating_sub(trim_offset);
            let trimmed_adapter = dispatch
                .adapter_hits
                .get(index)
                .copied()
                .filter(|hit| *hit != u32::MAX)
                .and_then(|hit| adapter_names.get(hit as usize).cloned());

            ProcessedRecord {
                header: record.header.clone(),
                sequence,
                qualities,
                corrected_positions,
                indel_corrections: Vec::new(),
                trimmed_bases,
                trimmed_adapter,
            }
        })
        .collect()
}

fn build_batch_stats(records: &[ProcessedRecord]) -> BatchStats {
    let mut stats = BatchStats::default();
    for record in records {
        stats.processed_reads += 1;
        stats.corrected_bases += record.corrected_positions.len();
        stats.indel_corrections += record.indel_corrections.len();
        if record.trimmed_bases > 0 {
            stats.trimmed_reads += 1;
            stats.trimmed_bases += record.trimmed_bases;
        }
    }
    stats
}
