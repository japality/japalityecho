use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result, anyhow, bail};
use flate2::read::MultiGzDecoder;

use crate::model::{FastqRecord, ProcessedRecord, ReadBatch, ReadPair, ReadPairBatch};

pub struct FastqReader<R: BufRead> {
    inner: R,
}

pub struct ReadBatchReadAhead {
    receiver: Receiver<Result<Option<ReadBatch>>>,
    handle: Option<JoinHandle<()>>,
    finished: bool,
}

impl ReadBatchReadAhead {
    pub fn next_batch(&mut self) -> Result<Option<ReadBatch>> {
        if self.finished {
            return Ok(None);
        }

        match self.receiver.recv() {
            Ok(Ok(Some(batch))) => Ok(Some(batch)),
            Ok(Ok(None)) => {
                self.finished = true;
                self.join_worker();
                Ok(None)
            }
            Ok(Err(error)) => {
                self.finished = true;
                self.join_worker();
                Err(error)
            }
            Err(_) => {
                self.finished = true;
                self.join_worker();
                Err(anyhow!(
                    "background FASTQ batch reader disconnected unexpectedly"
                ))
            }
        }
    }

    fn join_worker(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ReadBatchReadAhead {
    fn drop(&mut self) {
        self.join_worker();
    }
}

pub struct ReadPairBatchReadAhead {
    receiver: Receiver<Result<Option<ReadPairBatch>>>,
    handle: Option<JoinHandle<()>>,
    finished: bool,
}

impl ReadPairBatchReadAhead {
    pub fn next_batch(&mut self) -> Result<Option<ReadPairBatch>> {
        if self.finished {
            return Ok(None);
        }

        match self.receiver.recv() {
            Ok(Ok(Some(batch))) => Ok(Some(batch)),
            Ok(Ok(None)) => {
                self.finished = true;
                self.join_worker();
                Ok(None)
            }
            Ok(Err(error)) => {
                self.finished = true;
                self.join_worker();
                Err(error)
            }
            Err(_) => {
                self.finished = true;
                self.join_worker();
                Err(anyhow!(
                    "background paired FASTQ batch reader disconnected unexpectedly"
                ))
            }
        }
    }

    fn join_worker(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ReadPairBatchReadAhead {
    fn drop(&mut self) {
        self.join_worker();
    }
}

impl<R: BufRead> FastqReader<R> {
    pub fn new(inner: R) -> Self {
        Self { inner }
    }

    pub fn next_record(&mut self) -> Result<Option<FastqRecord>> {
        let mut header = String::new();
        if self.inner.read_line(&mut header)? == 0 {
            return Ok(None);
        }

        let mut sequence = String::new();
        let mut plus = String::new();
        let mut qualities = String::new();

        read_required_line(&mut self.inner, &mut sequence, "sequence")?;
        read_required_line(&mut self.inner, &mut plus, "plus")?;
        read_required_line(&mut self.inner, &mut qualities, "qualities")?;

        strip_line_end(&mut header);
        strip_line_end(&mut sequence);
        strip_line_end(&mut plus);
        strip_line_end(&mut qualities);

        if !header.starts_with('@') {
            bail!("FASTQ header must start with '@': {header}");
        }
        if !plus.starts_with('+') {
            bail!("FASTQ third line must start with '+': {plus}");
        }
        if sequence.len() != qualities.len() {
            bail!(
                "FASTQ sequence/quality length mismatch for {}: {} != {}",
                header,
                sequence.len(),
                qualities.len()
            );
        }

        Ok(Some(FastqRecord::new(
            header.trim_start_matches('@'),
            sequence.to_ascii_uppercase().into_bytes(),
            qualities.into_bytes(),
        )))
    }
}

pub fn open_fastq(path: &Path) -> Result<FastqReader<Box<dyn BufRead>>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open FASTQ input {}", path.display()))?;
    let reader: Box<dyn BufRead> = if is_gz(path) {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };
    Ok(FastqReader::new(reader))
}

pub fn sample_records(path: &Path, limit: usize) -> Result<Vec<FastqRecord>> {
    let mut reader = open_fastq(path)?;
    let mut records = Vec::with_capacity(limit.min(4096));

    while records.len() < limit {
        match reader.next_record()? {
            Some(record) => records.push(record),
            None => break,
        }
    }

    Ok(records)
}

pub fn sample_paired_records(path1: &Path, path2: &Path, limit: usize) -> Result<Vec<ReadPair>> {
    let mut reader1 = open_fastq(path1)?;
    let mut reader2 = open_fastq(path2)?;
    let mut pairs = Vec::with_capacity(limit.min(2048));

    while pairs.len() < limit.max(1) {
        match (reader1.next_record()?, reader2.next_record()?) {
            (Some(left), Some(right)) => {
                validate_pair_headers(&left.header, &right.header)?;
                pairs.push(ReadPair::new(left, right));
            }
            (None, None) => break,
            (None, Some(right)) => {
                bail!(
                    "paired FASTQ mismatch: {} ended before {} at mate 2 header {}",
                    path1.display(),
                    path2.display(),
                    right.header
                );
            }
            (Some(left), None) => {
                bail!(
                    "paired FASTQ mismatch: {} ended before {} at mate 1 header {}",
                    path2.display(),
                    path1.display(),
                    left.header
                );
            }
        }
    }

    Ok(pairs)
}

pub fn read_batches<F>(path: &Path, batch_reads: usize, mut on_batch: F) -> Result<()>
where
    F: FnMut(ReadBatch) -> Result<()>,
{
    let mut reader = open_fastq(path)?;
    let batch_reads = batch_reads.max(1);
    let mut records = Vec::with_capacity(batch_reads);
    let mut batch_index = 0;

    while let Some(record) = reader.next_record()? {
        records.push(record);
        if records.len() == batch_reads {
            on_batch(ReadBatch::new(batch_index, std::mem::take(&mut records)))?;
            batch_index += 1;
        }
    }

    if !records.is_empty() {
        on_batch(ReadBatch::new(batch_index, records))?;
    }

    Ok(())
}

pub fn read_batches_with_read_ahead(
    path: &Path,
    batch_reads: usize,
    read_ahead_depth: usize,
) -> ReadBatchReadAhead {
    let path = path.to_path_buf();
    let batch_reads = batch_reads.max(1);
    let (sender, receiver) = sync_channel(read_ahead_depth.max(1));
    let handle = thread::spawn(move || {
        if let Err(error) = send_read_batches(path, batch_reads, sender.clone()) {
            let _ = sender.send(Err(error));
        }
    });

    ReadBatchReadAhead {
        receiver,
        handle: Some(handle),
        finished: false,
    }
}

pub fn read_paired_batches<F>(
    path1: &Path,
    path2: &Path,
    batch_pairs: usize,
    mut on_batch: F,
) -> Result<()>
where
    F: FnMut(ReadPairBatch) -> Result<()>,
{
    let mut reader1 = open_fastq(path1)?;
    let mut reader2 = open_fastq(path2)?;
    let batch_pairs = batch_pairs.max(1);
    let mut pairs = Vec::with_capacity(batch_pairs);
    let mut batch_index = 0;

    loop {
        match (reader1.next_record()?, reader2.next_record()?) {
            (Some(left), Some(right)) => {
                validate_pair_headers(&left.header, &right.header)?;
                pairs.push(ReadPair::new(left, right));
                if pairs.len() == batch_pairs {
                    on_batch(ReadPairBatch::new(batch_index, std::mem::take(&mut pairs)))?;
                    batch_index += 1;
                }
            }
            (None, None) => break,
            (None, Some(right)) => {
                bail!(
                    "paired FASTQ mismatch: {} ended before {} at mate 2 header {}",
                    path1.display(),
                    path2.display(),
                    right.header
                );
            }
            (Some(left), None) => {
                bail!(
                    "paired FASTQ mismatch: {} ended before {} at mate 1 header {}",
                    path2.display(),
                    path1.display(),
                    left.header
                );
            }
        }
    }

    if !pairs.is_empty() {
        on_batch(ReadPairBatch::new(batch_index, pairs))?;
    }

    Ok(())
}

pub fn read_paired_batches_with_read_ahead(
    path1: &Path,
    path2: &Path,
    batch_pairs: usize,
    read_ahead_depth: usize,
) -> ReadPairBatchReadAhead {
    let path1 = path1.to_path_buf();
    let path2 = path2.to_path_buf();
    let batch_pairs = batch_pairs.max(1);
    let (sender, receiver) = sync_channel(read_ahead_depth.max(1));
    let handle = thread::spawn(move || {
        if let Err(error) = send_read_pair_batches(path1, path2, batch_pairs, sender.clone()) {
            let _ = sender.send(Err(error));
        }
    });

    ReadPairBatchReadAhead {
        receiver,
        handle: Some(handle),
        finished: false,
    }
}

pub fn write_processed_records<W: Write>(
    writer: &mut W,
    records: &[ProcessedRecord],
) -> Result<()> {
    for record in records {
        write_processed_record(writer, record)?;
    }
    Ok(())
}

pub fn write_processed_record_pairs<W1: Write, W2: Write>(
    writer1: &mut W1,
    writer2: &mut W2,
    record_pairs: &[(ProcessedRecord, ProcessedRecord)],
) -> Result<()> {
    for (left, right) in record_pairs {
        write_processed_record(writer1, left)?;
        write_processed_record(writer2, right)?;
    }
    Ok(())
}

fn read_required_line<R: BufRead>(reader: &mut R, buffer: &mut String, kind: &str) -> Result<()> {
    if reader.read_line(buffer)? == 0 {
        bail!("unexpected end of FASTQ while reading {kind} line");
    }
    Ok(())
}

fn strip_line_end(buffer: &mut String) {
    while matches!(buffer.as_bytes().last(), Some(b'\n' | b'\r')) {
        buffer.pop();
    }
}

fn is_gz(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gz"))
}

fn validate_pair_headers(left: &str, right: &str) -> Result<()> {
    if normalized_pair_header(left) == normalized_pair_header(right) {
        return Ok(());
    }

    bail!("paired FASTQ headers are out of sync: '{left}' vs '{right}'")
}

fn normalized_pair_header(header: &str) -> &str {
    let token = header.split_whitespace().next().unwrap_or(header);
    token
        .strip_suffix("/1")
        .or_else(|| token.strip_suffix("/2"))
        .unwrap_or(token)
}

fn write_processed_record<W: Write>(writer: &mut W, record: &ProcessedRecord) -> Result<()> {
    writer.write_all(b"@")?;
    writer.write_all(record.header.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.write_all(&record.sequence)?;
    writer.write_all(b"\n+\n")?;
    writer.write_all(&record.qualities)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn send_read_batches(
    path: PathBuf,
    batch_reads: usize,
    sender: SyncSender<Result<Option<ReadBatch>>>,
) -> Result<()> {
    let mut reader = open_fastq(&path)?;
    let mut records = Vec::with_capacity(batch_reads);
    let mut batch_index = 0usize;

    while let Some(record) = reader.next_record()? {
        records.push(record);
        if records.len() == batch_reads {
            if sender
                .send(Ok(Some(ReadBatch::new(
                    batch_index,
                    std::mem::take(&mut records),
                ))))
                .is_err()
            {
                return Ok(());
            }
            batch_index += 1;
        }
    }

    if !records.is_empty()
        && sender
            .send(Ok(Some(ReadBatch::new(batch_index, records))))
            .is_err()
    {
        return Ok(());
    }

    let _ = sender.send(Ok(None));
    Ok(())
}

fn send_read_pair_batches(
    path1: PathBuf,
    path2: PathBuf,
    batch_pairs: usize,
    sender: SyncSender<Result<Option<ReadPairBatch>>>,
) -> Result<()> {
    let mut reader1 = open_fastq(&path1)?;
    let mut reader2 = open_fastq(&path2)?;
    let mut pairs = Vec::with_capacity(batch_pairs);
    let mut batch_index = 0usize;

    loop {
        match (reader1.next_record()?, reader2.next_record()?) {
            (Some(left), Some(right)) => {
                validate_pair_headers(&left.header, &right.header)?;
                pairs.push(ReadPair::new(left, right));
                if pairs.len() == batch_pairs {
                    if sender
                        .send(Ok(Some(ReadPairBatch::new(
                            batch_index,
                            std::mem::take(&mut pairs),
                        ))))
                        .is_err()
                    {
                        return Ok(());
                    }
                    batch_index += 1;
                }
            }
            (None, None) => break,
            (None, Some(right)) => {
                bail!(
                    "paired FASTQ mismatch: {} ended before {} at mate 2 header {}",
                    path1.display(),
                    path2.display(),
                    right.header
                );
            }
            (Some(left), None) => {
                bail!(
                    "paired FASTQ mismatch: {} ended before {} at mate 1 header {}",
                    path2.display(),
                    path1.display(),
                    left.header
                );
            }
        }
    }

    if !pairs.is_empty()
        && sender
            .send(Ok(Some(ReadPairBatch::new(batch_index, pairs))))
            .is_err()
    {
        return Ok(());
    }

    let _ = sender.send(Ok(None));
    Ok(())
}
