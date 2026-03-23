use std::cell::{Cell, RefCell};
use std::collections::hash_map::DefaultHasher;
use std::ffi::{CString, c_char, c_int, c_uint, c_void};
use std::fs;
use std::hash::{Hash, Hasher};
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr;
use std::slice;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use libloading::Library;

use crate::gpu::{
    PreparedAdapterCandidates, PreparedTrustedKmers, encode_base_code, packed_ambiguity_pitch,
    packed_base_pitch,
};
use crate::model::{
    AccelerationExecution, AcceleratorScaffold, BackendPreference, ExecutionPlan, ReadBatch,
};

type CuResult = i32;
type CuDevice = i32;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuDevicePtr = u64;
type CuStream = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;

type CuInit = unsafe extern "C" fn(u32) -> CuResult;
type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CuResult;
type CuDeviceGet = unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult;
type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult;
type CuCtxCreate = unsafe extern "C" fn(*mut CuContext, c_uint, CuDevice) -> CuResult;
type CuCtxDestroy = unsafe extern "C" fn(CuContext) -> CuResult;
type CuMemHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CuResult;
type CuMemFreeHost = unsafe extern "C" fn(*mut c_void) -> CuResult;
type CuMemAlloc = unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult;
type CuMemFree = unsafe extern "C" fn(CuDevicePtr) -> CuResult;
type CuMemcpyHtoD = unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult;
type CuMemcpyHtoDAsync =
    unsafe extern "C" fn(CuDevicePtr, *const c_void, usize, CuStream) -> CuResult;
type CuMemcpyDtoHAsync =
    unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize, CuStream) -> CuResult;
type CuModuleLoadData = unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult;
type CuModuleUnload = unsafe extern "C" fn(CuModule) -> CuResult;
type CuModuleGetFunction =
    unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
type CuStreamCreate = unsafe extern "C" fn(*mut CuStream, c_uint) -> CuResult;
type CuStreamDestroy = unsafe extern "C" fn(CuStream) -> CuResult;
type CuStreamSynchronize = unsafe extern "C" fn(CuStream) -> CuResult;
type CuLaunchKernel = unsafe extern "C" fn(
    CuFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;

struct CudaApi {
    _library: Library,
    cu_init: CuInit,
    cu_device_get_count: CuDeviceGetCount,
    cu_device_get: CuDeviceGet,
    cu_device_get_name: CuDeviceGetName,
    cu_ctx_create: CuCtxCreate,
    cu_ctx_destroy: CuCtxDestroy,
    cu_mem_host_alloc: CuMemHostAlloc,
    cu_mem_free_host: CuMemFreeHost,
    cu_mem_alloc: CuMemAlloc,
    cu_mem_free: CuMemFree,
    cu_memcpy_htod: CuMemcpyHtoD,
    cu_memcpy_htod_async: CuMemcpyHtoDAsync,
    cu_memcpy_dtoh_async: CuMemcpyDtoHAsync,
    cu_module_load_data: CuModuleLoadData,
    cu_module_unload: CuModuleUnload,
    cu_module_get_function: CuModuleGetFunction,
    cu_stream_create: CuStreamCreate,
    cu_stream_destroy: CuStreamDestroy,
    cu_stream_synchronize: CuStreamSynchronize,
    cu_launch_kernel: CuLaunchKernel,
}

impl CudaApi {
    fn load() -> Result<Arc<Self>> {
        let library = load_cuda_library()?;
        // SAFETY: The loaded symbols are resolved from libcuda and retained alongside the library.
        unsafe {
            Ok(Arc::new(Self {
                cu_init: load_symbol(&library, b"cuInit\0")?,
                cu_device_get_count: load_symbol(&library, b"cuDeviceGetCount\0")?,
                cu_device_get: load_symbol(&library, b"cuDeviceGet\0")?,
                cu_device_get_name: load_symbol(&library, b"cuDeviceGetName\0")?,
                cu_ctx_create: load_symbol(&library, b"cuCtxCreate_v2\0")?,
                cu_ctx_destroy: load_symbol(&library, b"cuCtxDestroy_v2\0")?,
                cu_mem_host_alloc: load_symbol(&library, b"cuMemHostAlloc\0")?,
                cu_mem_free_host: load_symbol(&library, b"cuMemFreeHost\0")?,
                cu_mem_alloc: load_symbol(&library, b"cuMemAlloc_v2\0")?,
                cu_mem_free: load_symbol(&library, b"cuMemFree_v2\0")?,
                cu_memcpy_htod: load_symbol(&library, b"cuMemcpyHtoD_v2\0")?,
                cu_memcpy_htod_async: load_symbol(&library, b"cuMemcpyHtoDAsync_v2\0")?,
                cu_memcpy_dtoh_async: load_symbol(&library, b"cuMemcpyDtoHAsync_v2\0")?,
                cu_module_load_data: load_symbol(&library, b"cuModuleLoadData\0")?,
                cu_module_unload: load_symbol(&library, b"cuModuleUnload\0")?,
                cu_module_get_function: load_symbol(&library, b"cuModuleGetFunction\0")?,
                cu_stream_create: load_symbol(&library, b"cuStreamCreate\0")?,
                cu_stream_destroy: load_symbol(&library, b"cuStreamDestroy_v2\0")?,
                cu_stream_synchronize: load_symbol(&library, b"cuStreamSynchronize\0")?,
                cu_launch_kernel: load_symbol(&library, b"cuLaunchKernel\0")?,
                _library: library,
            }))
        }
    }
}

pub struct CudaDispatchResult {
    pub corrected_packed_codes: Vec<u8>,
    pub corrected_ambiguity_bits: Vec<u8>,
    pub trim_offsets: Vec<u32>,
    pub adapter_hits: Vec<u32>,
    pub read_pitch: usize,
    pub packed_read_pitch: usize,
    pub ambiguity_pitch: usize,
    pub execution: AccelerationExecution,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CudaReferenceSignature {
    k: usize,
    trusted_keys_len: usize,
    trusted_keys_hash: u64,
    trusted_counts_hash: u64,
    adapter_pitch: usize,
    adapter_codes_hash: u64,
    adapter_lengths_hash: u64,
    adapter_supports_hash: u64,
}

impl CudaReferenceSignature {
    fn from_inputs(
        trusted_kmers: &PreparedTrustedKmers,
        adapters: &PreparedAdapterCandidates,
    ) -> Self {
        Self {
            k: trusted_kmers.k,
            trusted_keys_len: trusted_kmers.keys.len(),
            trusted_keys_hash: slice_hash(&trusted_kmers.keys),
            trusted_counts_hash: slice_hash(&trusted_kmers.counts),
            adapter_pitch: adapters.pitch,
            adapter_codes_hash: slice_hash(&adapters.codes),
            adapter_lengths_hash: slice_hash(&adapters.lengths),
            adapter_supports_hash: slice_hash(&adapters.supports),
        }
    }
}

struct CudaReferenceBuffers {
    signature: CudaReferenceSignature,
    trusted_keys: CudaDeviceBuffer,
    trusted_counts: CudaDeviceBuffer,
    adapter_codes: CudaDeviceBuffer,
    adapter_lengths: CudaDeviceBuffer,
    adapter_supports: CudaDeviceBuffer,
    total_bytes: usize,
}

impl CudaReferenceBuffers {
    fn upload(
        api: &Arc<CudaApi>,
        trusted_kmers: &PreparedTrustedKmers,
        adapters: &PreparedAdapterCandidates,
    ) -> Result<Self> {
        let trusted_keys_bytes = trusted_kmers.keys.len() * size_of::<u64>();
        let trusted_counts_bytes = trusted_kmers.counts.len() * size_of::<u32>();
        let adapter_bytes = adapters.codes.len();
        let adapter_lengths_bytes = adapters.lengths.len() * size_of::<u32>();
        let adapter_supports_bytes = adapters.supports.len() * size_of::<u32>();

        let trusted_keys = CudaDeviceBuffer::allocate(api, trusted_keys_bytes)?;
        let trusted_counts = CudaDeviceBuffer::allocate(api, trusted_counts_bytes)?;
        let adapter_codes = CudaDeviceBuffer::allocate(api, adapter_bytes)?;
        let adapter_lengths = CudaDeviceBuffer::allocate(api, adapter_lengths_bytes)?;
        let adapter_supports = CudaDeviceBuffer::allocate(api, adapter_supports_bytes)?;

        // SAFETY: Host slices and device pointers are valid for the supplied byte ranges.
        unsafe {
            copy_htod_from_slice(
                api,
                &trusted_keys,
                as_bytes_of_slice(&trusted_kmers.keys),
                trusted_keys_bytes,
                "trusted_keys",
            )?;
            copy_htod_from_slice(
                api,
                &trusted_counts,
                as_bytes_of_slice(&trusted_kmers.counts),
                trusted_counts_bytes,
                "trusted_counts",
            )?;
            copy_htod_from_slice(
                api,
                &adapter_codes,
                &adapters.codes,
                adapter_bytes,
                "adapter_codes",
            )?;
            copy_htod_from_slice(
                api,
                &adapter_lengths,
                as_bytes_of_slice(&adapters.lengths),
                adapter_lengths_bytes,
                "adapter_lengths",
            )?;
            copy_htod_from_slice(
                api,
                &adapter_supports,
                as_bytes_of_slice(&adapters.supports),
                adapter_supports_bytes,
                "adapter_supports",
            )?;
        }

        Ok(Self {
            signature: CudaReferenceSignature::from_inputs(trusted_kmers, adapters),
            total_bytes: trusted_keys.len
                + trusted_counts.len
                + adapter_codes.len
                + adapter_lengths.len
                + adapter_supports.len,
            trusted_keys,
            trusted_counts,
            adapter_codes,
            adapter_lengths,
            adapter_supports,
        })
    }

    fn matches(
        &self,
        trusted_kmers: &PreparedTrustedKmers,
        adapters: &PreparedAdapterCandidates,
    ) -> bool {
        self.signature == CudaReferenceSignature::from_inputs(trusted_kmers, adapters)
    }
}

struct CudaSlot {
    stream: CuStream,
    scratch: RefCell<CudaScratch>,
    in_use: Cell<bool>,
}

#[derive(Default)]
struct CudaScratch {
    host_bases: Option<CudaPinnedHostBuffer>,
    host_base_ambiguity: Option<CudaPinnedHostBuffer>,
    host_qualities: Option<CudaPinnedHostBuffer>,
    host_lengths: Option<CudaPinnedHostBuffer>,
    host_corrected: Option<CudaPinnedHostBuffer>,
    host_corrected_ambiguity: Option<CudaPinnedHostBuffer>,
    host_trim_offsets: Option<CudaPinnedHostBuffer>,
    host_adapter_hits: Option<CudaPinnedHostBuffer>,
    device_bases: Option<CudaDeviceBuffer>,
    device_base_ambiguity: Option<CudaDeviceBuffer>,
    device_qualities: Option<CudaDeviceBuffer>,
    device_lengths: Option<CudaDeviceBuffer>,
    device_corrected: Option<CudaDeviceBuffer>,
    device_corrected_ambiguity: Option<CudaDeviceBuffer>,
    device_trim_offsets: Option<CudaDeviceBuffer>,
    device_adapter_hits: Option<CudaDeviceBuffer>,
}

impl CudaScratch {
    fn can_reuse(
        &self,
        base_bytes: usize,
        base_ambiguity_bytes: usize,
        qualities_bytes: usize,
        lengths_bytes: usize,
        corrected_bytes: usize,
        corrected_ambiguity_bytes: usize,
        trim_output_bytes: usize,
        adapter_hits_bytes: usize,
    ) -> bool {
        buffer_capacity(&self.host_bases) >= base_bytes
            && buffer_capacity(&self.host_base_ambiguity) >= base_ambiguity_bytes
            && buffer_capacity(&self.host_qualities) >= qualities_bytes
            && buffer_capacity(&self.host_lengths) >= lengths_bytes
            && buffer_capacity(&self.host_corrected) >= corrected_bytes
            && buffer_capacity(&self.host_corrected_ambiguity) >= corrected_ambiguity_bytes
            && buffer_capacity(&self.host_trim_offsets) >= trim_output_bytes
            && buffer_capacity(&self.host_adapter_hits) >= adapter_hits_bytes
            && buffer_capacity(&self.device_bases) >= base_bytes
            && buffer_capacity(&self.device_base_ambiguity) >= base_ambiguity_bytes
            && buffer_capacity(&self.device_qualities) >= qualities_bytes
            && buffer_capacity(&self.device_lengths) >= lengths_bytes
            && buffer_capacity(&self.device_corrected) >= corrected_bytes
            && buffer_capacity(&self.device_corrected_ambiguity) >= corrected_ambiguity_bytes
            && buffer_capacity(&self.device_trim_offsets) >= trim_output_bytes
            && buffer_capacity(&self.device_adapter_hits) >= adapter_hits_bytes
    }

    fn total_host_capacity(&self) -> usize {
        buffer_capacity(&self.host_bases)
            + buffer_capacity(&self.host_base_ambiguity)
            + buffer_capacity(&self.host_qualities)
            + buffer_capacity(&self.host_lengths)
            + buffer_capacity(&self.host_corrected)
            + buffer_capacity(&self.host_corrected_ambiguity)
            + buffer_capacity(&self.host_trim_offsets)
            + buffer_capacity(&self.host_adapter_hits)
    }

    fn total_device_capacity(&self) -> usize {
        buffer_capacity(&self.device_bases)
            + buffer_capacity(&self.device_base_ambiguity)
            + buffer_capacity(&self.device_qualities)
            + buffer_capacity(&self.device_lengths)
            + buffer_capacity(&self.device_corrected)
            + buffer_capacity(&self.device_corrected_ambiguity)
            + buffer_capacity(&self.device_trim_offsets)
            + buffer_capacity(&self.device_adapter_hits)
    }
}

pub struct CudaPendingDispatch {
    slot_index: usize,
    reads: usize,
    read_pitch: usize,
    packed_read_pitch: usize,
    ambiguity_pitch: usize,
    uploaded_reference_bytes: usize,
    scratch_reused: bool,
    reference_bytes: usize,
    threads_per_block: u32,
    trusted_kmer_count: usize,
    adapter_count: usize,
    trusted_k: usize,
}

pub struct CudaSession {
    api: Arc<CudaApi>,
    context: CuContext,
    module: CuModule,
    trim_kernel: CuFunction,
    slots: Vec<CudaSlot>,
    next_slot: Cell<usize>,
    device_name: String,
    ptx_path: PathBuf,
    references: RefCell<Option<CudaReferenceBuffers>>,
}

impl CudaSession {
    pub fn try_new(scaffold: &AcceleratorScaffold) -> Result<Self> {
        let api = CudaApi::load()?;
        // SAFETY: All driver calls use validated pointers and follow CUDA driver API contracts.
        unsafe {
            cuda_check((api.cu_init)(0), "cuInit")?;

            let mut device_count = 0i32;
            cuda_check(
                (api.cu_device_get_count)(&mut device_count),
                "cuDeviceGetCount",
            )?;
            if device_count <= 0 {
                bail!("CUDA driver loaded but reported zero devices");
            }

            let mut device = 0i32;
            cuda_check((api.cu_device_get)(&mut device, 0), "cuDeviceGet")?;

            let mut name_buffer = [0i8; 128];
            cuda_check(
                (api.cu_device_get_name)(
                    name_buffer.as_mut_ptr(),
                    name_buffer.len() as i32,
                    device,
                ),
                "cuDeviceGetName",
            )?;
            let device_name = c_string_buffer_to_string(&name_buffer);

            let mut context = ptr::null_mut();
            cuda_check(
                (api.cu_ctx_create)(&mut context, 0, device),
                "cuCtxCreate_v2",
            )?;
            let mut module = ptr::null_mut();

            let initialized = (|| -> Result<(Vec<CuStream>, CuModule, CuFunction, PathBuf)> {
                let mut streams = Vec::with_capacity(scaffold.overlapped_streams.max(2));
                for _ in 0..scaffold.overlapped_streams.max(2) {
                    let mut stream = ptr::null_mut();
                    cuda_check((api.cu_stream_create)(&mut stream, 0), "cuStreamCreate")?;
                    streams.push(stream);
                }

                let result = (|| -> Result<(CuModule, CuFunction, PathBuf)> {
                    let ptx_path = compile_cuda_ptx(scaffold)?;
                    let mut ptx = fs::read(&ptx_path).with_context(|| {
                        format!("failed to read PTX artifact {}", ptx_path.display())
                    })?;
                    ptx.push(0);

                    cuda_check(
                        (api.cu_module_load_data)(&mut module, ptx.as_ptr().cast()),
                        "cuModuleLoadData",
                    )?;

                    let kernel_name = CString::new("japalityecho_trim_correct")?;
                    let mut trim_kernel = ptr::null_mut();
                    cuda_check(
                        (api.cu_module_get_function)(
                            &mut trim_kernel,
                            module,
                            kernel_name.as_ptr(),
                        ),
                        "cuModuleGetFunction",
                    )?;

                    Ok((module, trim_kernel, ptx_path))
                })();

                match result {
                    Ok((module, trim_kernel, ptx_path)) => {
                        Ok((streams, module, trim_kernel, ptx_path))
                    }
                    Err(error) => {
                        for stream in streams.drain(..) {
                            let _ = (api.cu_stream_destroy)(stream);
                        }
                        Err(error)
                    }
                }
            })();

            match initialized {
                Ok((streams, module, trim_kernel, ptx_path)) => Ok(Self {
                    api,
                    context,
                    module,
                    trim_kernel,
                    slots: streams
                        .into_iter()
                        .map(|stream| CudaSlot {
                            stream,
                            scratch: RefCell::new(CudaScratch::default()),
                            in_use: Cell::new(false),
                        })
                        .collect(),
                    next_slot: Cell::new(0),
                    device_name,
                    ptx_path,
                    references: RefCell::new(None),
                }),
                Err(error) => {
                    if !module.is_null() {
                        let _ = (api.cu_module_unload)(module);
                    }
                    if !context.is_null() {
                        let _ = (api.cu_ctx_destroy)(context);
                    }
                    Err(error)
                }
            }
        }
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn ptx_path(&self) -> &Path {
        &self.ptx_path
    }

    pub fn pipeline_slots(&self) -> usize {
        self.slots.len()
    }

    fn acquire_slot(&self) -> Result<usize> {
        let slot_count = self.slots.len();
        if slot_count == 0 {
            bail!("CUDA session has no pipeline slots");
        }

        let start = self.next_slot.get();
        for offset in 0..slot_count {
            let slot_index = (start + offset) % slot_count;
            if !self.slots[slot_index].in_use.get() {
                self.slots[slot_index].in_use.set(true);
                self.next_slot.set((slot_index + 1) % slot_count);
                return Ok(slot_index);
            }
        }

        bail!("all CUDA pipeline slots are busy")
    }

    fn release_slot(&self, slot_index: usize) {
        self.slots[slot_index].in_use.set(false);
    }

    pub fn submit_trim_correct(
        &self,
        batch: &ReadBatch,
        plan: &ExecutionPlan,
        scaffold: &AcceleratorScaffold,
        trusted_kmers: &PreparedTrustedKmers,
        adapters: &PreparedAdapterCandidates,
    ) -> Result<CudaPendingDispatch> {
        let reads = batch.records.len();
        let read_pitch = batch
            .records
            .iter()
            .map(|record| record.len())
            .max()
            .unwrap_or(0);
        let packed_read_pitch = packed_base_pitch(read_pitch);
        let ambiguity_pitch = packed_ambiguity_pitch(read_pitch);
        let base_bytes = packed_read_pitch * reads;
        let base_ambiguity_bytes = ambiguity_pitch * reads;
        let qualities_bytes = read_pitch * reads;
        let lengths_bytes = reads * size_of::<u32>();
        let trusted_keys_bytes = trusted_kmers.keys.len() * size_of::<u64>();
        let trusted_counts_bytes = trusted_kmers.counts.len() * size_of::<u32>();
        let adapter_bytes = adapters.codes.len();
        let adapter_lengths_bytes = adapters.lengths.len() * size_of::<u32>();
        let adapter_supports_bytes = adapters.supports.len() * size_of::<u32>();
        let reference_bytes = trusted_keys_bytes
            + trusted_counts_bytes
            + adapter_bytes
            + adapter_lengths_bytes
            + adapter_supports_bytes;
        let corrected_bytes = base_bytes;
        let corrected_ambiguity_bytes = base_ambiguity_bytes;
        let trim_output_bytes = reads * size_of::<u32>();
        let adapter_hits_bytes = reads * size_of::<u32>();

        let mut uploaded_reference_bytes = 0usize;
        {
            let mut references = self.references.borrow_mut();
            let needs_upload = references
                .as_ref()
                .is_none_or(|cached| !cached.matches(trusted_kmers, adapters));
            if needs_upload {
                *references = Some(CudaReferenceBuffers::upload(
                    &self.api,
                    trusted_kmers,
                    adapters,
                )?);
                uploaded_reference_bytes = reference_bytes;
            }
        }
        let references = self.references.borrow();
        let references = references
            .as_ref()
            .expect("reference buffers must exist after upload");
        let slot_index = self.acquire_slot()?;
        let slot = &self.slots[slot_index];

        let scratch_reused = {
            let scratch = slot.scratch.borrow();
            scratch.can_reuse(
                base_bytes,
                base_ambiguity_bytes,
                qualities_bytes,
                lengths_bytes,
                corrected_bytes,
                corrected_ambiguity_bytes,
                trim_output_bytes,
                adapter_hits_bytes,
            )
        };

        let submit_result = (|| -> Result<CudaPendingDispatch> {
            let mut scratch = slot.scratch.borrow_mut();

            let (host_bases, host_base_ambiguity) = ensure_packed_host_buffers(
                &mut scratch,
                &self.api,
                base_bytes,
                base_ambiguity_bytes,
            )?;
            stage_packed_bases(
                host_bases,
                host_base_ambiguity,
                batch,
                read_pitch,
                packed_read_pitch,
                ambiguity_pitch,
            );
            stage_qualities(
                ensure_pinned_buffer(&mut scratch.host_qualities, &self.api, qualities_bytes)?,
                batch,
                read_pitch,
            );
            stage_lengths(
                ensure_pinned_buffer(&mut scratch.host_lengths, &self.api, lengths_bytes)?,
                batch,
            );
            ensure_pinned_buffer(&mut scratch.host_corrected, &self.api, corrected_bytes)?;
            ensure_pinned_buffer(
                &mut scratch.host_corrected_ambiguity,
                &self.api,
                corrected_ambiguity_bytes,
            )?;
            ensure_pinned_buffer(&mut scratch.host_trim_offsets, &self.api, trim_output_bytes)?;
            ensure_pinned_buffer(
                &mut scratch.host_adapter_hits,
                &self.api,
                adapter_hits_bytes,
            )?;

            ensure_device_buffer(&mut scratch.device_bases, &self.api, base_bytes)?;
            ensure_device_buffer(
                &mut scratch.device_base_ambiguity,
                &self.api,
                base_ambiguity_bytes,
            )?;
            ensure_device_buffer(&mut scratch.device_qualities, &self.api, qualities_bytes)?;
            ensure_device_buffer(&mut scratch.device_lengths, &self.api, lengths_bytes)?;
            ensure_device_buffer(&mut scratch.device_corrected, &self.api, corrected_bytes)?;
            ensure_device_buffer(
                &mut scratch.device_corrected_ambiguity,
                &self.api,
                corrected_ambiguity_bytes,
            )?;
            ensure_device_buffer(
                &mut scratch.device_trim_offsets,
                &self.api,
                trim_output_bytes,
            )?;
            ensure_device_buffer(
                &mut scratch.device_adapter_hits,
                &self.api,
                adapter_hits_bytes,
            )?;

            // SAFETY: Scratch buffers and reference tables are valid for the specified byte ranges
            // and belong to the active CUDA context held by this session.
            unsafe {
                copy_htod_if_nonempty_async(
                    &self.api,
                    scratch.device_bases.as_ref().expect("device bases"),
                    scratch.host_bases.as_ref().expect("host bases"),
                    base_bytes,
                    slot.stream,
                    "bases",
                )?;
                copy_htod_if_nonempty_async(
                    &self.api,
                    scratch
                        .device_base_ambiguity
                        .as_ref()
                        .expect("device base ambiguity"),
                    scratch
                        .host_base_ambiguity
                        .as_ref()
                        .expect("host base ambiguity"),
                    base_ambiguity_bytes,
                    slot.stream,
                    "base_ambiguity",
                )?;
                copy_htod_if_nonempty_async(
                    &self.api,
                    scratch.device_qualities.as_ref().expect("device qualities"),
                    scratch.host_qualities.as_ref().expect("host qualities"),
                    qualities_bytes,
                    slot.stream,
                    "qualities",
                )?;
                copy_htod_if_nonempty_async(
                    &self.api,
                    scratch.device_lengths.as_ref().expect("device lengths"),
                    scratch.host_lengths.as_ref().expect("host lengths"),
                    lengths_bytes,
                    slot.stream,
                    "read_lengths",
                )?;

                let mut base_ptr = scratch.device_bases.as_ref().expect("device bases").ptr;
                let mut qualities_ptr = scratch
                    .device_qualities
                    .as_ref()
                    .expect("device qualities")
                    .ptr;
                let mut base_ambiguity_ptr = scratch
                    .device_base_ambiguity
                    .as_ref()
                    .expect("device base ambiguity")
                    .ptr;
                let mut lengths_ptr = scratch.device_lengths.as_ref().expect("device lengths").ptr;
                let mut trusted_keys_ptr = references.trusted_keys.ptr;
                let mut trusted_counts_ptr = references.trusted_counts.ptr;
                let mut trusted_len = trusted_kmers.keys.len() as u32;
                let mut adapter_codes_ptr = references.adapter_codes.ptr;
                let mut adapter_lengths_ptr = references.adapter_lengths.ptr;
                let mut adapter_supports_ptr = references.adapter_supports.ptr;
                let mut adapter_count = adapters.lengths.len() as u32;
                let mut adapter_pitch = adapters.pitch as u32;
                let mut corrected_ptr = scratch
                    .device_corrected
                    .as_ref()
                    .expect("device corrected")
                    .ptr;
                let mut corrected_ambiguity_ptr = scratch
                    .device_corrected_ambiguity
                    .as_ref()
                    .expect("device corrected ambiguity")
                    .ptr;
                let mut trim_ptr = scratch
                    .device_trim_offsets
                    .as_ref()
                    .expect("device trim offsets")
                    .ptr;
                let mut adapter_hits_ptr = scratch
                    .device_adapter_hits
                    .as_ref()
                    .expect("device adapter hits")
                    .ptr;
                let mut reads_per_batch = reads as i32;
                let mut device_read_pitch = read_pitch as i32;
                let mut device_packed_read_pitch = packed_read_pitch as i32;
                let mut device_ambiguity_pitch = ambiguity_pitch as i32;
                let mut kmer_size = trusted_kmers.k as u32;
                let mut trusted_floor = plan.trusted_kmer_min_count;
                let mut min_quality = plan.trim_min_quality as u32;

                let mut kernel_params = [
                    (&mut base_ptr as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut base_ambiguity_ptr as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut qualities_ptr as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut lengths_ptr as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut trusted_keys_ptr as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut trusted_counts_ptr as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut trusted_len as *mut u32).cast::<c_void>(),
                    (&mut adapter_codes_ptr as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut adapter_lengths_ptr as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut adapter_supports_ptr as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut adapter_count as *mut u32).cast::<c_void>(),
                    (&mut adapter_pitch as *mut u32).cast::<c_void>(),
                    (&mut corrected_ptr as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut corrected_ambiguity_ptr as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut trim_ptr as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut adapter_hits_ptr as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut reads_per_batch as *mut i32).cast::<c_void>(),
                    (&mut device_read_pitch as *mut i32).cast::<c_void>(),
                    (&mut device_packed_read_pitch as *mut i32).cast::<c_void>(),
                    (&mut device_ambiguity_pitch as *mut i32).cast::<c_void>(),
                    (&mut kmer_size as *mut u32).cast::<c_void>(),
                    (&mut trusted_floor as *mut u32).cast::<c_void>(),
                    (&mut min_quality as *mut u32).cast::<c_void>(),
                ];

                let grid_size = ((reads.max(1) as u32) + scaffold.threads_per_block - 1)
                    / scaffold.threads_per_block;
                cuda_check(
                    (self.api.cu_launch_kernel)(
                        self.trim_kernel,
                        grid_size,
                        1,
                        1,
                        scaffold.threads_per_block,
                        1,
                        1,
                        0,
                        slot.stream,
                        kernel_params.as_mut_ptr(),
                        ptr::null_mut(),
                    ),
                    "cuLaunchKernel(japalityecho_trim_correct)",
                )?;

                let corrected_device_ptr = scratch
                    .device_corrected
                    .as_ref()
                    .expect("device corrected")
                    .ptr;
                copy_dtoh_async(
                    &self.api,
                    scratch.host_corrected.as_mut().expect("host corrected"),
                    corrected_device_ptr,
                    corrected_bytes,
                    slot.stream,
                    "corrected_codes",
                )?;
                let corrected_ambiguity_device_ptr = scratch
                    .device_corrected_ambiguity
                    .as_ref()
                    .expect("device corrected ambiguity")
                    .ptr;
                copy_dtoh_async(
                    &self.api,
                    scratch
                        .host_corrected_ambiguity
                        .as_mut()
                        .expect("host corrected ambiguity"),
                    corrected_ambiguity_device_ptr,
                    corrected_ambiguity_bytes,
                    slot.stream,
                    "corrected_ambiguity",
                )?;
                let trim_device_ptr = scratch
                    .device_trim_offsets
                    .as_ref()
                    .expect("device trim offsets")
                    .ptr;
                copy_dtoh_async(
                    &self.api,
                    scratch
                        .host_trim_offsets
                        .as_mut()
                        .expect("host trim offsets"),
                    trim_device_ptr,
                    trim_output_bytes,
                    slot.stream,
                    "trim_offsets",
                )?;
                let adapter_hits_device_ptr = scratch
                    .device_adapter_hits
                    .as_ref()
                    .expect("device adapter hits")
                    .ptr;
                copy_dtoh_async(
                    &self.api,
                    scratch
                        .host_adapter_hits
                        .as_mut()
                        .expect("host adapter hits"),
                    adapter_hits_device_ptr,
                    adapter_hits_bytes,
                    slot.stream,
                    "adapter_hits",
                )?;
            }

            Ok(CudaPendingDispatch {
                slot_index,
                reads,
                read_pitch,
                packed_read_pitch,
                ambiguity_pitch,
                uploaded_reference_bytes,
                scratch_reused,
                reference_bytes: references.total_bytes,
                threads_per_block: scaffold.threads_per_block,
                trusted_kmer_count: trusted_kmers.keys.len(),
                adapter_count: adapters.lengths.len(),
                trusted_k: trusted_kmers.k,
            })
        })();

        if submit_result.is_err() {
            // SAFETY: Synchronizing a stream after a failed submission attempt helps drain any
            // partially queued work before the slot is reused.
            unsafe {
                let _ = (self.api.cu_stream_synchronize)(slot.stream);
            }
            self.release_slot(slot_index);
        }

        submit_result
    }

    pub fn wait_trim_correct(&self, pending: CudaPendingDispatch) -> Result<CudaDispatchResult> {
        let slot = &self.slots[pending.slot_index];
        let result = (|| -> Result<CudaDispatchResult> {
            // SAFETY: The slot stream belongs to this CUDA session and the submitted work for this
            // pending batch was queued onto it by `submit_trim_correct`.
            unsafe {
                cuda_check(
                    (self.api.cu_stream_synchronize)(slot.stream),
                    &format!("cuStreamSynchronize(slot {})", pending.slot_index),
                )?;
            }

            let corrected_bytes = pending.packed_read_pitch * pending.reads;
            let corrected_ambiguity_bytes = pending.ambiguity_pitch * pending.reads;
            let lengths_bytes = pending.reads * size_of::<u32>();
            let trim_output_bytes = pending.reads * size_of::<u32>();
            let adapter_hits_bytes = trim_output_bytes;
            let transfer_bytes = corrected_bytes
                + corrected_ambiguity_bytes
                + pending.read_pitch * pending.reads
                + lengths_bytes
                + corrected_bytes
                + corrected_ambiguity_bytes
                + trim_output_bytes
                + adapter_hits_bytes
                + pending.uploaded_reference_bytes;

            let scratch = slot.scratch.borrow();
            let corrected_packed_codes = scratch
                .host_corrected
                .as_ref()
                .expect("host corrected")
                .read_bytes(corrected_bytes);
            let corrected_ambiguity_bits = scratch
                .host_corrected_ambiguity
                .as_ref()
                .expect("host corrected ambiguity")
                .read_bytes(corrected_ambiguity_bytes);
            let trim_offsets = scratch
                .host_trim_offsets
                .as_ref()
                .expect("host trim offsets")
                .read_u32s(pending.reads);
            let adapter_hits = scratch
                .host_adapter_hits
                .as_ref()
                .expect("host adapter hits")
                .read_u32s(pending.reads);
            let host_pinned_bytes = scratch.total_host_capacity();
            let device_scratch_bytes = scratch.total_device_capacity();
            let device_bytes = device_scratch_bytes + pending.reference_bytes;

            let mut notes = vec![
                format!("Executed full trim/correct kernel on {}", self.device_name),
                format!("Loaded PTX artifact {}", self.ptx_path.display()),
                format!(
                    "Pinned host bytes={} device bytes={} transfer bytes={}",
                    host_pinned_bytes, device_bytes, transfer_bytes
                ),
                format!(
                    "Launch geometry: grid={} block={} read_pitch={} packed_pitch={} ambiguity_pitch={} slot={}",
                    ((pending.reads.max(1) as u32) + pending.threads_per_block - 1)
                        / pending.threads_per_block,
                    pending.threads_per_block,
                    pending.read_pitch,
                    pending.packed_read_pitch,
                    pending.ambiguity_pitch,
                    pending.slot_index
                ),
                format!(
                    "Trusted k-mers={} adapter candidates={} k={}",
                    pending.trusted_kmer_count, pending.adapter_count, pending.trusted_k
                ),
                format!(
                    "Scratch arena capacities: host={}B device={}B references={}B",
                    host_pinned_bytes, device_scratch_bytes, pending.reference_bytes
                ),
            ];
            if pending.uploaded_reference_bytes > 0 {
                notes.push(format!(
                    "Uploaded {} bytes of trusted/adaptor reference tables into device-resident cache",
                    pending.uploaded_reference_bytes
                ));
            } else {
                notes.push(
                    "Reused device-resident trusted/adaptor reference tables without re-upload"
                        .to_string(),
                );
            }
            if pending.scratch_reused {
                notes.push(format!(
                    "Reused pinned host/device scratch arena for slot {}",
                    pending.slot_index
                ));
            } else {
                notes.push(format!(
                    "Expanded pinned host/device scratch arena to fit slot {}",
                    pending.slot_index
                ));
            }
            notes.push(format!(
                "Packed active base I/O to {}B/read plus {}B/read ambiguity sideband",
                pending.packed_read_pitch, pending.ambiguity_pitch
            ));

            Ok(CudaDispatchResult {
                corrected_packed_codes,
                corrected_ambiguity_bits,
                trim_offsets,
                adapter_hits,
                read_pitch: pending.read_pitch,
                packed_read_pitch: pending.packed_read_pitch,
                ambiguity_pitch: pending.ambiguity_pitch,
                execution: AccelerationExecution {
                    backend: BackendPreference::Cuda,
                    stage: "full_trim_correct".to_string(),
                    successful: true,
                    kernel_name: "japalityecho_trim_correct".to_string(),
                    batch_index: 0,
                    host_pinned_bytes,
                    device_bytes,
                    transfer_bytes,
                    reads: pending.reads,
                    returned_trim_offsets: pending.reads,
                    submit_us: 0,
                    wait_us: 0,
                    end_to_end_us: 0,
                    overlap_us: 0,
                    notes,
                },
            })
        })();

        self.release_slot(pending.slot_index);
        result
    }
}

impl Drop for CudaSession {
    fn drop(&mut self) {
        // SAFETY: Objects were created from this CUDA context and may be released in reverse order.
        unsafe {
            for slot in &self.slots {
                if !slot.stream.is_null() {
                    let _ = (self.api.cu_stream_destroy)(slot.stream);
                }
            }
            if !self.module.is_null() {
                let _ = (self.api.cu_module_unload)(self.module);
            }
            if !self.context.is_null() {
                let _ = (self.api.cu_ctx_destroy)(self.context);
            }
        }
    }
}

struct CudaPinnedHostBuffer {
    api: Arc<CudaApi>,
    ptr: *mut c_void,
    len: usize,
}

impl CudaPinnedHostBuffer {
    fn allocate(api: &Arc<CudaApi>, len: usize) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        // SAFETY: CUDA allocates and returns a pinned host pointer for the requested byte size.
        unsafe {
            cuda_check(
                (api.cu_mem_host_alloc)(&mut ptr, len.max(1), 0),
                "cuMemHostAlloc",
            )?;
        }
        Ok(Self {
            api: Arc::clone(api),
            ptr,
            len,
        })
    }

    fn as_ptr(&self) -> *const c_void {
        self.ptr.cast_const()
    }

    fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: The pointer is valid for self.len bytes and uniquely owned here.
        unsafe { slice::from_raw_parts_mut(self.ptr.cast::<u8>(), self.len.max(1)) }
    }

    fn as_mut_u32_slice(&mut self) -> &mut [u32] {
        // SAFETY: The pointer is valid for self.len bytes; callers only use the typed prefix that
        // fits within the allocated capacity.
        unsafe { slice::from_raw_parts_mut(self.ptr.cast::<u32>(), self.len / size_of::<u32>()) }
    }

    fn read_bytes(&self, len: usize) -> Vec<u8> {
        // SAFETY: The output buffer was allocated with at least len bytes.
        unsafe { slice::from_raw_parts(self.ptr.cast::<u8>(), len).to_vec() }
    }

    fn read_u32s(&self, len: usize) -> Vec<u32> {
        // SAFETY: The output buffer was allocated with at least len * size_of::<u32>() bytes.
        unsafe { slice::from_raw_parts(self.ptr.cast::<u32>(), len).to_vec() }
    }
}

impl Drop for CudaPinnedHostBuffer {
    fn drop(&mut self) {
        // SAFETY: Pointer was allocated by cuMemHostAlloc and must be released with cuMemFreeHost.
        unsafe {
            let _ = (self.api.cu_mem_free_host)(self.ptr);
        }
    }
}

struct CudaDeviceBuffer {
    api: Arc<CudaApi>,
    ptr: CuDevicePtr,
    len: usize,
}

impl CudaDeviceBuffer {
    fn allocate(api: &Arc<CudaApi>, len: usize) -> Result<Self> {
        let mut ptr = 0u64;
        // SAFETY: CUDA allocates and returns a device pointer for the requested byte size.
        unsafe {
            cuda_check((api.cu_mem_alloc)(&mut ptr, len.max(1)), "cuMemAlloc_v2")?;
        }
        Ok(Self {
            api: Arc::clone(api),
            ptr,
            len,
        })
    }
}

impl Drop for CudaDeviceBuffer {
    fn drop(&mut self) {
        // SAFETY: Pointer was allocated by cuMemAlloc_v2 and must be released with cuMemFree_v2.
        unsafe {
            let _ = (self.api.cu_mem_free)(self.ptr);
        }
    }
}

fn compile_cuda_ptx(scaffold: &AcceleratorScaffold) -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source_path = manifest_dir.join(&scaffold.source_path);
    let output_dir = manifest_dir.join("target/japalityecho-kernels");
    let output_path = output_dir.join("japalityecho_trim_correct.ptx");
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let rebuild = match (fs::metadata(&source_path), fs::metadata(&output_path)) {
        (Ok(source), Ok(output)) => source.modified()? > output.modified()?,
        (Ok(_), Err(_)) => true,
        (Err(error), _) => {
            return Err(error).with_context(|| format!("failed to stat {}", source_path.display()));
        }
    };

    if rebuild {
        let output = Command::new("nvcc")
            .arg("--ptx")
            .arg(&source_path)
            .arg("-o")
            .arg(&output_path)
            .arg("--gpu-architecture=compute_52")
            .output()
            .with_context(|| "failed to invoke nvcc for PTX compilation".to_string())?;
        if !output.status.success() {
            bail!(
                "nvcc PTX compilation failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
    }

    Ok(output_path)
}

fn load_cuda_library() -> Result<Library> {
    let candidates = ["libcuda.so.1", "libcuda.so"];
    for candidate in candidates {
        // SAFETY: Loading a dynamic library is required to interact with the CUDA driver.
        if let Ok(library) = unsafe { Library::new(candidate) } {
            return Ok(library);
        }
    }
    Err(anyhow!("failed to load libcuda.so.1 or libcuda.so"))
}

unsafe fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    // SAFETY: The symbol type matches the CUDA driver API signature we invoke later.
    let symbol = unsafe { library.get::<T>(name) }.with_context(|| {
        format!(
            "failed to load CUDA symbol {:?}",
            String::from_utf8_lossy(name)
        )
    })?;
    Ok(*symbol)
}

fn cuda_check(result: CuResult, operation: &str) -> Result<()> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(anyhow!(
            "{operation} failed with CUDA driver error code {result}"
        ))
    }
}

fn c_string_buffer_to_string(buffer: &[i8]) -> String {
    let bytes: Vec<u8> = buffer
        .iter()
        .copied()
        .take_while(|byte| *byte != 0)
        .map(|byte| byte as u8)
        .collect();
    String::from_utf8_lossy(&bytes).trim().to_string()
}

fn as_bytes_of_slice<T>(values: &[T]) -> &[u8] {
    // SAFETY: Slices of plain-old-data may be viewed as bytes for transfer.
    unsafe { slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

fn slice_hash<T: Hash>(values: &[T]) -> u64 {
    let mut hasher = DefaultHasher::new();
    values.hash(&mut hasher);
    hasher.finish()
}

fn buffer_capacity<T>(buffer: &Option<T>) -> usize
where
    T: BufferLen,
{
    buffer.as_ref().map_or(0, BufferLen::len)
}

trait BufferLen {
    fn len(&self) -> usize;
}

impl BufferLen for CudaPinnedHostBuffer {
    fn len(&self) -> usize {
        self.len
    }
}

impl BufferLen for CudaDeviceBuffer {
    fn len(&self) -> usize {
        self.len
    }
}

fn ensure_pinned_buffer<'a>(
    slot: &'a mut Option<CudaPinnedHostBuffer>,
    api: &Arc<CudaApi>,
    len: usize,
) -> Result<&'a mut CudaPinnedHostBuffer> {
    if slot.as_ref().map_or(0, |buffer| buffer.len) < len {
        *slot = Some(CudaPinnedHostBuffer::allocate(api, len)?);
    }
    Ok(slot.as_mut().expect("pinned buffer should exist"))
}

fn ensure_device_buffer<'a>(
    slot: &'a mut Option<CudaDeviceBuffer>,
    api: &Arc<CudaApi>,
    len: usize,
) -> Result<&'a mut CudaDeviceBuffer> {
    if slot.as_ref().map_or(0, |buffer| buffer.len) < len {
        *slot = Some(CudaDeviceBuffer::allocate(api, len)?);
    }
    Ok(slot.as_mut().expect("device buffer should exist"))
}

fn ensure_packed_host_buffers<'a>(
    scratch: &'a mut CudaScratch,
    api: &Arc<CudaApi>,
    base_bytes: usize,
    base_ambiguity_bytes: usize,
) -> Result<(&'a mut CudaPinnedHostBuffer, &'a mut CudaPinnedHostBuffer)> {
    if scratch.host_bases.as_ref().map_or(0, |buffer| buffer.len) < base_bytes {
        scratch.host_bases = Some(CudaPinnedHostBuffer::allocate(api, base_bytes)?);
    }
    if scratch
        .host_base_ambiguity
        .as_ref()
        .map_or(0, |buffer| buffer.len)
        < base_ambiguity_bytes
    {
        scratch.host_base_ambiguity =
            Some(CudaPinnedHostBuffer::allocate(api, base_ambiguity_bytes)?);
    }
    let host_bases = scratch
        .host_bases
        .as_mut()
        .expect("host bases should exist");
    let host_base_ambiguity = scratch
        .host_base_ambiguity
        .as_mut()
        .expect("host base ambiguity should exist");
    Ok((host_bases, host_base_ambiguity))
}

fn stage_packed_bases(
    code_buffer: &mut CudaPinnedHostBuffer,
    ambiguity_buffer: &mut CudaPinnedHostBuffer,
    batch: &ReadBatch,
    _read_pitch: usize,
    packed_read_pitch: usize,
    ambiguity_pitch: usize,
) {
    let code_bytes = packed_read_pitch * batch.records.len();
    let ambiguity_bytes = ambiguity_pitch * batch.records.len();
    let codes = &mut code_buffer.as_mut_slice()[..code_bytes];
    let ambiguity = &mut ambiguity_buffer.as_mut_slice()[..ambiguity_bytes];
    codes.fill(0u8);
    ambiguity.fill(0u8);

    for (record_index, record) in batch.records.iter().enumerate() {
        let row_codes =
            &mut codes[record_index * packed_read_pitch..(record_index + 1) * packed_read_pitch];
        let row_ambiguity =
            &mut ambiguity[record_index * ambiguity_pitch..(record_index + 1) * ambiguity_pitch];
        for (base_index, &base) in record.sequence.iter().enumerate() {
            let encoded = encode_base_code(base);
            if encoded > 3 {
                set_ambiguity_bit(row_ambiguity, base_index, true);
            } else {
                write_packed_base(row_codes, base_index, encoded);
            }
        }
    }
}

fn stage_qualities(buffer: &mut CudaPinnedHostBuffer, batch: &ReadBatch, read_pitch: usize) {
    let used_bytes = read_pitch * batch.records.len();
    let slice = &mut buffer.as_mut_slice()[..used_bytes];
    slice.fill(0u8);
    for (record_index, record) in batch.records.iter().enumerate() {
        let row = &mut slice[record_index * read_pitch..(record_index + 1) * read_pitch];
        for (base_index, phred) in record.phred_scores().enumerate() {
            row[base_index] = phred;
        }
    }
}

fn stage_lengths(buffer: &mut CudaPinnedHostBuffer, batch: &ReadBatch) {
    let slice = &mut buffer.as_mut_u32_slice()[..batch.records.len()];
    for (slot, record) in slice.iter_mut().zip(batch.records.iter()) {
        *slot = record.len() as u32;
    }
}

fn write_packed_base(packed: &mut [u8], position: usize, code: u8) {
    let byte_index = position / 4;
    let shift = 6 - ((position % 4) * 2);
    let mask = !(0b11u8 << shift);
    packed[byte_index] = (packed[byte_index] & mask) | ((code & 0b11) << shift);
}

fn set_ambiguity_bit(bits: &mut [u8], position: usize, ambiguous: bool) {
    let byte_index = position / 8;
    let shift = 7 - (position % 8);
    let mask = 1u8 << shift;
    if ambiguous {
        bits[byte_index] |= mask;
    } else {
        bits[byte_index] &= !mask;
    }
}

unsafe fn copy_htod_from_slice(
    api: &CudaApi,
    device: &CudaDeviceBuffer,
    host: &[u8],
    len: usize,
    label: &str,
) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    cuda_check(
        unsafe { (api.cu_memcpy_htod)(device.ptr, host.as_ptr().cast(), len) },
        &format!("cuMemcpyHtoD_v2({label})"),
    )
}

unsafe fn copy_htod_if_nonempty_async(
    api: &CudaApi,
    device: &CudaDeviceBuffer,
    host: &CudaPinnedHostBuffer,
    len: usize,
    stream: CuStream,
    label: &str,
) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    cuda_check(
        unsafe { (api.cu_memcpy_htod_async)(device.ptr, host.as_ptr(), len, stream) },
        &format!("cuMemcpyHtoDAsync_v2({label})"),
    )
}

unsafe fn copy_dtoh_async(
    api: &CudaApi,
    host: &mut CudaPinnedHostBuffer,
    device_ptr: CuDevicePtr,
    len: usize,
    stream: CuStream,
    label: &str,
) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    cuda_check(
        unsafe { (api.cu_memcpy_dtoh_async)(host.as_mut_ptr(), device_ptr, len, stream) },
        &format!("cuMemcpyDtoHAsync_v2({label})"),
    )
}
