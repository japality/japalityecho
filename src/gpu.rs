use std::process::Command;

use crate::model::{
    AccelerationPreview, AcceleratorRuntime, AcceleratorRuntimeStatus, AcceleratorScaffold,
    AdapterCandidate, BackendPreference, ExecutionPlan, ReadBatch,
};
use crate::spectrum::KmerSpectrum;

pub const CUDA_KERNEL_PATH: &str = "kernels/cuda/japalityecho_trim_correct.cu";
pub const HIP_KERNEL_PATH: &str = "kernels/hip/japalityecho_trim_correct.hip.cpp";
pub const CUDA_KERNEL_SOURCE: &str = include_str!("../kernels/cuda/japalityecho_trim_correct.cu");
pub const HIP_KERNEL_SOURCE: &str =
    include_str!("../kernels/hip/japalityecho_trim_correct.hip.cpp");

#[derive(Debug, Clone)]
pub struct PreparedDeviceBatch {
    pub packed_bases: Vec<u8>,
    pub base_codes: Vec<u8>,
    pub qualities: Vec<u8>,
    pub read_lengths: Vec<u32>,
    pub read_pitch: usize,
    pub ambiguous_bases: usize,
}

#[derive(Debug, Clone)]
pub struct PreparedTrustedKmers {
    pub k: usize,
    pub keys: Vec<u64>,
    pub counts: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct PreparedAdapterCandidates {
    pub names: Vec<String>,
    pub codes: Vec<u8>,
    pub lengths: Vec<u32>,
    pub supports: Vec<u32>,
    pub pitch: usize,
}

pub fn scaffold_for_backend(
    backend: BackendPreference,
    plan: &ExecutionPlan,
) -> Option<AcceleratorScaffold> {
    match backend {
        BackendPreference::Cuda => Some(cuda_scaffold(plan)),
        BackendPreference::Hip => Some(hip_scaffold(plan)),
        _ => None,
    }
}

pub fn probe_runtime(backend: BackendPreference) -> Option<AcceleratorRuntime> {
    match backend {
        BackendPreference::Cuda => Some(probe_cuda_runtime()),
        BackendPreference::Hip => Some(probe_hip_runtime()),
        BackendPreference::Auto | BackendPreference::Cpu => None,
    }
}

pub fn prepare_acceleration_preview(
    label: impl Into<String>,
    batch: &ReadBatch,
    plan: &ExecutionPlan,
    scaffold: &AcceleratorScaffold,
) -> AccelerationPreview {
    let prepared = prepare_device_batch(batch);
    let reads = batch.records.len();
    let grid_size =
        ((reads.max(1) as u32) + scaffold.threads_per_block - 1) / scaffold.threads_per_block;
    let offset_bytes = reads * std::mem::size_of::<u32>() * 3;
    let shared_mem_bytes = plan.kmer_size * std::mem::size_of::<u32>() * 8;

    let mut notes = vec![
        format!(
            "Packed {} bases into {} bytes using 2-bit encoding",
            batch.total_bases,
            prepared.packed_bases.len()
        ),
        format!(
            "Launch preview: grid={} block={} streams={}",
            grid_size, scaffold.threads_per_block, scaffold.overlapped_streams
        ),
    ];
    if plan.zero_copy_candidate {
        notes.push("Batch remains eligible for pinned-host zero-copy promotion".to_string());
    }
    if prepared.ambiguous_bases > 0 {
        notes.push(format!(
            "Preview coerced {} ambiguous bases into the 2-bit packing lane",
            prepared.ambiguous_bases
        ));
    }

    AccelerationPreview {
        label: label.into(),
        batch_index: batch.batch_index,
        reads,
        total_bases: batch.total_bases,
        packed_bases_bytes: prepared.packed_bases.len(),
        quality_bytes: prepared.qualities.len(),
        offset_bytes,
        read_pitch: prepared.read_pitch,
        block_size: scaffold.threads_per_block,
        grid_size,
        shared_mem_bytes,
        stream_count: scaffold.overlapped_streams,
        ambiguous_bases: prepared.ambiguous_bases,
        notes,
    }
}

pub fn prepare_device_batch(batch: &ReadBatch) -> PreparedDeviceBatch {
    let (packed_bases, ambiguous_bases) = pack_batch_2bit(batch);
    let read_pitch = batch
        .records
        .iter()
        .map(|record| record.len())
        .max()
        .unwrap_or(0);
    let mut base_codes = vec![4u8; read_pitch * batch.records.len()];
    let mut qualities = vec![0u8; read_pitch * batch.records.len()];
    let mut read_lengths = Vec::with_capacity(batch.records.len());

    for (record_index, record) in batch.records.iter().enumerate() {
        read_lengths.push(record.len() as u32);
        for (base_index, &base) in record.sequence.iter().enumerate() {
            base_codes[record_index * read_pitch + base_index] = encode_base_code(base);
        }
        for (base_index, phred) in record.phred_scores().enumerate() {
            qualities[record_index * read_pitch + base_index] = phred;
        }
    }

    PreparedDeviceBatch {
        packed_bases,
        base_codes,
        qualities,
        read_lengths,
        read_pitch,
        ambiguous_bases,
    }
}

pub fn packed_base_pitch(read_pitch: usize) -> usize {
    read_pitch.div_ceil(4)
}

pub fn packed_ambiguity_pitch(read_pitch: usize) -> usize {
    read_pitch.div_ceil(8)
}

pub fn prepare_trusted_kmers(spectrum: &KmerSpectrum, min_count: u32) -> PreparedTrustedKmers {
    let entries = spectrum.trusted_entries(min_count);
    let mut keys = Vec::with_capacity(entries.len());
    let mut counts = Vec::with_capacity(entries.len());
    for (key, count) in entries {
        keys.push(key);
        counts.push(count);
    }

    PreparedTrustedKmers {
        k: spectrum.k(),
        keys,
        counts,
    }
}

pub fn prepare_adapter_candidates(
    candidates: &[AdapterCandidate],
    max_candidates: usize,
) -> PreparedAdapterCandidates {
    let selected: Vec<_> = candidates
        .iter()
        .filter(|candidate| candidate.sequence.len() >= 8)
        .take(max_candidates.max(1))
        .collect();
    let pitch = selected
        .iter()
        .map(|candidate| candidate.sequence.len())
        .max()
        .unwrap_or(0);

    let mut names = Vec::with_capacity(selected.len());
    let mut lengths = Vec::with_capacity(selected.len());
    let mut supports = Vec::with_capacity(selected.len());
    let mut codes = vec![4u8; pitch * selected.len()];

    for (candidate_index, candidate) in selected.iter().enumerate() {
        names.push(candidate.name.clone());
        lengths.push(candidate.sequence.len() as u32);
        supports.push(candidate.support.min(u32::MAX as usize) as u32);
        for (base_index, &base) in candidate.sequence.as_bytes().iter().enumerate() {
            codes[candidate_index * pitch + base_index] = encode_base_code(base);
        }
    }

    PreparedAdapterCandidates {
        names,
        codes,
        lengths,
        supports,
        pitch,
    }
}

pub fn decode_base_code(code: u8) -> u8 {
    match code {
        0 => b'A',
        1 => b'C',
        2 => b'G',
        3 => b'T',
        _ => b'N',
    }
}

pub fn decode_packed_base_code(packed_codes: &[u8], ambiguity_bits: &[u8], position: usize) -> u8 {
    if packed_ambiguity_bit(ambiguity_bits, position) {
        return b'N';
    }

    let byte_index = position / 4;
    let shift = 6 - ((position % 4) * 2);
    let code = packed_codes
        .get(byte_index)
        .copied()
        .map(|byte| (byte >> shift) & 0b11)
        .unwrap_or(0);
    decode_base_code(code)
}

fn packed_ambiguity_bit(bits: &[u8], position: usize) -> bool {
    let byte_index = position / 8;
    let shift = 7 - (position % 8);
    bits.get(byte_index)
        .is_some_and(|byte| ((byte >> shift) & 0b1) != 0)
}

fn cuda_scaffold(plan: &ExecutionPlan) -> AcceleratorScaffold {
    AcceleratorScaffold {
        backend: BackendPreference::Cuda,
        language: "cuda".to_string(),
        module_name: "japalityecho_cuda_trim_correct".to_string(),
        source_path: CUDA_KERNEL_PATH.to_string(),
        entrypoints: vec![
            "japalityecho_pack_reads".to_string(),
            "japalityecho_trim_correct".to_string(),
            "japalityecho_scatter_kept".to_string(),
        ],
        threads_per_block: 256,
        vector_width_bases: plan.kmer_size.max(16),
        overlapped_streams: plan.overlap_depth.max(2),
        zero_copy_candidate: plan.zero_copy_candidate,
        notes: vec![
            format!(
                "CUDA scaffold mirrors the current planner settings: k={} trusted floor={} min len={}",
                plan.kmer_size, plan.trusted_kmer_min_count, plan.minimum_output_length
            ),
            format!(
                "Pinned host batches are expected to feed {} overlapped CUDA streams",
                plan.overlap_depth.max(2)
            ),
            format!(
                "Embedded template loaded ({} bytes) for future NVRTC/offline compilation",
                CUDA_KERNEL_SOURCE.len()
            ),
        ],
    }
}

fn hip_scaffold(plan: &ExecutionPlan) -> AcceleratorScaffold {
    AcceleratorScaffold {
        backend: BackendPreference::Hip,
        language: "hip".to_string(),
        module_name: "japalityecho_hip_trim_correct".to_string(),
        source_path: HIP_KERNEL_PATH.to_string(),
        entrypoints: vec![
            "japalityecho_pack_reads".to_string(),
            "japalityecho_trim_correct".to_string(),
            "japalityecho_scatter_kept".to_string(),
        ],
        threads_per_block: 256,
        vector_width_bases: plan.kmer_size.max(16),
        overlapped_streams: plan.overlap_depth.max(2),
        zero_copy_candidate: plan.zero_copy_candidate,
        notes: vec![
            format!(
                "HIP scaffold mirrors the current planner settings: k={} trusted floor={} min len={}",
                plan.kmer_size, plan.trusted_kmer_min_count, plan.minimum_output_length
            ),
            format!(
                "Pinned host batches are expected to feed {} overlapped HIP streams",
                plan.overlap_depth.max(2)
            ),
            format!(
                "Embedded template loaded ({} bytes) for future hipRTC/offline compilation",
                HIP_KERNEL_SOURCE.len()
            ),
        ],
    }
}

fn probe_cuda_runtime() -> AcceleratorRuntime {
    let compiler_hint = command_last_nonempty_line("nvcc", &["--version"]);
    let device_name =
        command_first_nonempty_line("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"]);
    let status = if device_name.is_some() {
        AcceleratorRuntimeStatus::Available
    } else {
        AcceleratorRuntimeStatus::Unavailable
    };

    let mut notes = Vec::new();
    match &compiler_hint {
        Some(version) => notes.push(format!("CUDA toolkit probe: {version}")),
        None => notes.push("CUDA toolkit probe: nvcc not found on PATH".to_string()),
    }
    match &device_name {
        Some(name) => notes.push(format!("CUDA device probe: {name}")),
        None => notes
            .push("CUDA device probe: nvidia-smi unavailable or no device detected".to_string()),
    }

    AcceleratorRuntime {
        backend: BackendPreference::Cuda,
        status,
        driver_hint: device_name
            .as_ref()
            .map(|_| "nvidia-smi".to_string())
            .or_else(|| {
                compiler_hint
                    .as_ref()
                    .map(|_| "cuda-toolkit-only".to_string())
            }),
        compiler_hint,
        device_name,
        notes,
    }
}

fn probe_hip_runtime() -> AcceleratorRuntime {
    let compiler_hint = command_last_nonempty_line("hipcc", &["--version"]);
    let device_name = probe_rocm_device();
    let status = if device_name.is_some() {
        AcceleratorRuntimeStatus::Available
    } else {
        AcceleratorRuntimeStatus::Unavailable
    };

    let mut notes = Vec::new();
    match &compiler_hint {
        Some(version) => notes.push(format!("HIP toolkit probe: {version}")),
        None => notes.push("HIP toolkit probe: hipcc not found on PATH".to_string()),
    }
    match &device_name {
        Some(name) => notes.push(format!("ROCm device probe: {name}")),
        None => {
            notes.push("ROCm device probe: rocminfo unavailable or no device detected".to_string())
        }
    }

    AcceleratorRuntime {
        backend: BackendPreference::Hip,
        status,
        driver_hint: device_name
            .as_ref()
            .map(|_| "rocminfo".to_string())
            .or_else(|| {
                compiler_hint
                    .as_ref()
                    .map(|_| "rocm-toolkit-only".to_string())
            }),
        compiler_hint,
        device_name,
        notes,
    }
}

fn probe_rocm_device() -> Option<String> {
    let lines = command_lines("rocminfo", &[])?;
    lines
        .iter()
        .find(|line| line.contains("gfx"))
        .cloned()
        .or_else(|| {
            lines
                .iter()
                .find(|line| line.contains("Marketing Name"))
                .cloned()
        })
}

fn command_first_nonempty_line(program: &str, args: &[&str]) -> Option<String> {
    command_lines(program, args)?
        .into_iter()
        .find(|line| !line.is_empty())
}

fn command_last_nonempty_line(program: &str, args: &[&str]) -> Option<String> {
    command_lines(program, args)?
        .into_iter()
        .rev()
        .find(|line| !line.is_empty())
}

fn command_lines(program: &str, args: &[&str]) -> Option<Vec<String>> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let merged = if stdout.trim().is_empty() {
        stderr.as_ref()
    } else {
        stdout.as_ref()
    };

    let lines: Vec<String> = merged
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    (!lines.is_empty()).then_some(lines)
}

fn pack_batch_2bit(batch: &ReadBatch) -> (Vec<u8>, usize) {
    let mut packed = Vec::with_capacity((batch.total_bases * 2).div_ceil(8));
    let mut current_byte = 0u8;
    let mut filled_bits = 0u8;
    let mut ambiguous_bases = 0usize;

    for record in &batch.records {
        for &base in &record.sequence {
            let encoded = encode_base_code(base);
            if encoded > 3 {
                ambiguous_bases += 1;
            }

            current_byte = (current_byte << 2) | (encoded & 0b11);
            filled_bits += 2;
            if filled_bits == 8 {
                packed.push(current_byte);
                current_byte = 0;
                filled_bits = 0;
            }
        }
    }

    if filled_bits > 0 {
        current_byte <<= 8 - filled_bits;
        packed.push(current_byte);
    }

    (packed, ambiguous_bases)
}

pub fn encode_base_code(base: u8) -> u8 {
    match base.to_ascii_uppercase() {
        b'A' => 0u8,
        b'C' => 1u8,
        b'G' => 2u8,
        b'T' => 3u8,
        _ => 4u8,
    }
}
