# JapalityECHO

**Zero-configuration, context-aware NGS read preprocessing with integrated error correction and optional GPU acceleration.**

## Overview

JapalityECHO is a next-generation sequencing (NGS) preprocessing tool written in Rust that unifies adapter trimming, quality pruning, and k-mer-based error correction in a single pipeline. It automatically detects the sequencing platform (Illumina, MGI, ONT, PacBio, Ion Torrent), experiment type (WGS, RNA-seq, ATAC-seq, single-cell 10x, amplicon), and adapter sequences — no user configuration required.

## Key Features

- **Zero-configuration auto-detection** — Automatically infers platform, experiment type, and adapters from a 100K-read sample
- **Integrated error correction** — Context-aware substitution and indel correction using a counting Bloom filter-compressed k-mer spectrum
- **Three-stage pipeline** — (1) Profile → (2) Spectrum build → (3) Correct, trim & output
- **GPU acceleration** — Optional CUDA/HIP backend for GPU-accelerated k-mer operations
- **Broad compatibility** — Supports Illumina, MGI, ONT, PacBio, and Ion Torrent data across WGS, RNA-seq, ATAC-seq, scRNA-seq, and amplicon assays

## Building

```bash
# CPU-only build
cargo build --release

# With CUDA GPU support
cargo build --release --features cuda
```

## Usage

```bash
# Zero-config mode (recommended)
japalityecho input.fastq.gz -o output.fastq.gz

# Paired-end
japalityecho -1 R1.fastq.gz -2 R2.fastq.gz -o clean_R1.fastq.gz -O clean_R2.fastq.gz

# Manual overrides (optional)
japalityecho input.fastq.gz -o output.fastq.gz --platform illumina --experiment rnaseq
```

## Architecture

```
src/
├── main.rs          # CLI entry point
├── profile.rs       # Stage 1: auto-detection (platform, experiment, adapters)
├── spectrum.rs      # Stage 2: k-mer spectrum construction
├── process.rs       # Stage 3: correction, trimming, output
├── algorithm.rs     # Core trimming/correction algorithms
├── bloom.rs         # Counting Bloom filter (4-bit counters)
├── dbg.rs           # De Bruijn graph for multi-base correction
├── gpu.rs           # GPU backend abstraction
├── cuda_runtime.rs  # CUDA FFI bindings
├── backend.rs       # Heterogeneous compute dispatch
├── fastq.rs         # FASTQ I/O with pinned memory
├── model.rs         # Data structures
└── evaluate.rs      # Quality metrics

kernels/
├── cuda/            # CUDA kernel for GPU-accelerated trimming/correction
└── hip/             # AMD HIP kernel (ROCm)
```

## License

GPL-3.0

## Citation

Manuscript in preparation. Please check back for citation details.

## Contact

YiHao Chen — yihao.chen@japality.com
