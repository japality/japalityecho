use crate::model::{AdapterCandidate, ExecutionPlan, FastqRecord, IndelCorrection, IndelType, ProcessedRecord};
use crate::spectrum::{KmerSpectrum, SupportStats};

const DNA_BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];

pub fn process_record(
    record: &FastqRecord,
    plan: &ExecutionPlan,
    spectrum: &KmerSpectrum,
) -> ProcessedRecord {
    process_record_with_quality_cutoff(record, plan, spectrum, None)
}

pub fn process_record_with_quality_cutoff(
    record: &FastqRecord,
    plan: &ExecutionPlan,
    spectrum: &KmerSpectrum,
    quality_cutoff_hint: Option<usize>,
) -> ProcessedRecord {
    let mut sequence = record.sequence.clone();
    let mut qualities = record.qualities.clone();
    let adapter_hit = find_adapter_hit(&sequence, &plan.adapter_candidates);
    let initial_limit = adapter_hit
        .as_ref()
        .map(|hit| hit.start)
        .unwrap_or(sequence.len());
    let (corrected_positions, indel_corrections) =
        correct_sequence(&mut sequence, &mut qualities, initial_limit, plan, spectrum);
    let trimmed_limit = quality_cutoff_hint
        .map(|hint| hint.min(initial_limit))
        .or_else(|| low_quality_tail_cutoff(&qualities, initial_limit, plan.trim_min_quality))
        .unwrap_or(initial_limit);
    let trimmed_bases = sequence.len().saturating_sub(trimmed_limit);
    sequence.truncate(trimmed_limit);

    qualities.truncate(trimmed_limit);

    ProcessedRecord {
        header: record.header.clone(),
        sequence,
        qualities,
        corrected_positions,
        indel_corrections,
        trimmed_bases,
        trimmed_adapter: adapter_hit.map(|hit| hit.name),
    }
}

#[derive(Debug)]
struct AdapterHit {
    name: String,
    start: usize,
    score: i32,
}

fn find_adapter_hit(sequence: &[u8], candidates: &[AdapterCandidate]) -> Option<AdapterHit> {
    let mut best_hit: Option<AdapterHit> = None;

    for candidate in candidates {
        let adapter = candidate.sequence.as_bytes();
        if adapter.len() < 8 || sequence.len() < 8 {
            continue;
        }

        let search_start = sequence.len().saturating_sub(adapter.len().max(28));
        for start in search_start..sequence.len() {
            let overlap = (sequence.len() - start).min(adapter.len());
            if overlap < 8 {
                continue;
            }

            let matches = sequence[start..start + overlap]
                .iter()
                .zip(adapter.iter())
                .filter(|(lhs, rhs)| lhs.to_ascii_uppercase() == rhs.to_ascii_uppercase())
                .count();
            let mismatches = overlap - matches;
            let allowed = ((overlap as f64) * 0.15).ceil() as usize;
            if mismatches > allowed || matches * 100 < overlap * 85 {
                continue;
            }

            let score = matches as i32 * 4 - mismatches as i32 * 3
                + overlap as i32
                + candidate.support as i32;
            let replace = best_hit
                .as_ref()
                .map(|best| score > best.score || (score == best.score && start < best.start))
                .unwrap_or(true);

            if replace {
                best_hit = Some(AdapterHit {
                    name: candidate.name.clone(),
                    start,
                    score,
                });
            }
        }
    }

    best_hit
}

fn correct_sequence(
    sequence: &mut Vec<u8>,
    qualities: &mut Vec<u8>,
    correction_limit: usize,
    plan: &ExecutionPlan,
    spectrum: &KmerSpectrum,
) -> (Vec<usize>, Vec<IndelCorrection>) {
    let mut corrected_positions = Vec::new();
    let mut indel_corrections = Vec::new();
    let trusted_floor = plan.trusted_kmer_min_count;
    let quality_grace = plan.trim_min_quality.saturating_add(5);
    let indel_threshold = trusted_floor.saturating_mul(2);

    // First pass: substitution corrections
    for position in 0..correction_limit.min(sequence.len()) {
        let phred = qualities
            .get(position)
            .copied()
            .unwrap_or(b'!')
            .saturating_sub(33);
        let original_support = spectrum.support_for_position(sequence, position);
        let needs_attention = phred < plan.trim_min_quality
            || (phred <= quality_grace
                && original_support.windows > 0
                && original_support.min < trusted_floor);

        if !needs_attention || original_support.windows == 0 {
            continue;
        }

        let original_base = sequence[position].to_ascii_uppercase();
        let mut best_base = original_base;
        let mut best_support = original_support;

        for candidate_base in DNA_BASES {
            if candidate_base == original_base {
                continue;
            }

            sequence[position] = candidate_base;
            let candidate_support = spectrum.support_for_position(sequence, position);
            if candidate_support.min > best_support.min
                || (candidate_support.min == best_support.min
                    && candidate_support.sum > best_support.sum)
            {
                best_base = candidate_base;
                best_support = candidate_support;
            }
        }

        sequence[position] = original_base;

        if should_accept_correction(original_support, best_support, trusted_floor)
            && best_base != original_base
        {
            sequence[position] = best_base;
            corrected_positions.push(position);
        }
    }

    // Second pass: indel corrections for positions still with poor support
    let mut position = 0usize;
    let mut last_indel_window: Option<usize> = None;
    while position < correction_limit.min(sequence.len()) {
        let phred = qualities
            .get(position)
            .copied()
            .unwrap_or(b'!')
            .saturating_sub(33);
        let current_support = spectrum.support_for_position(sequence, position);

        let still_poor = current_support.windows > 0
            && current_support.min < trusted_floor
            && phred < plan.trim_min_quality;

        if !still_poor {
            position += 1;
            continue;
        }

        // Max 1 indel per 50bp window
        let window = position / 50;
        if last_indel_window == Some(window) {
            position += 1;
            continue;
        }

        let mut best_indel: Option<(IndelType, Vec<u8>, SupportStats)> = None;
        let mut best_indel_support = current_support;

        // Try single-base deletion
        if sequence.len() > spectrum.k() {
            let deletion_support = try_deletion_at(sequence, position, spectrum);
            if deletion_support.min > best_indel_support.min
                || (deletion_support.min == best_indel_support.min
                    && deletion_support.sum > best_indel_support.sum)
            {
                let removed = vec![sequence[position]];
                best_indel = Some((IndelType::Deletion, removed, deletion_support));
                best_indel_support = deletion_support;
            }
        }

        // Try multi-base deletions (2-5bp)
        for del_len in 2..=5usize {
            if position + del_len > sequence.len() || sequence.len() <= del_len + spectrum.k() {
                break;
            }
            let del_support = try_multi_deletion_at(sequence, position, del_len, spectrum);
            if del_support.min > best_indel_support.min
                || (del_support.min == best_indel_support.min
                    && del_support.sum > best_indel_support.sum)
            {
                let removed = sequence[position..position + del_len].to_vec();
                best_indel = Some((IndelType::Deletion, removed, del_support));
                best_indel_support = del_support;
            }
        }

        // Try single-base insertion (4 bases)
        for &ins_base in &DNA_BASES {
            let insertion_support = try_insertion_at(sequence, position, ins_base, spectrum);
            if insertion_support.min > best_indel_support.min
                || (insertion_support.min == best_indel_support.min
                    && insertion_support.sum > best_indel_support.sum)
            {
                best_indel = Some((IndelType::Insertion, vec![ins_base], insertion_support));
                best_indel_support = insertion_support;
            }
        }

        // Try multi-base insertions (2-5bp) via greedy extension
        if let Some((IndelType::Insertion, ref single_bases, _)) = best_indel {
            let mut ext_bases = single_bases.clone();
            let mut ext_support = best_indel_support;
            for _ext_round in 1..5 {
                let mut round_best: Option<(u8, SupportStats)> = None;
                let mut round_best_support = ext_support;
                for &next_base in &DNA_BASES {
                    let mut trial = ext_bases.clone();
                    trial.push(next_base);
                    let trial_support =
                        try_multi_insertion_at(sequence, position, &trial, spectrum);
                    if trial_support.min > round_best_support.min
                        || (trial_support.min == round_best_support.min
                            && trial_support.sum > round_best_support.sum)
                    {
                        round_best = Some((next_base, trial_support));
                        round_best_support = trial_support;
                    }
                }
                if let Some((chosen_base, chosen_support)) = round_best {
                    ext_bases.push(chosen_base);
                    ext_support = chosen_support;
                    best_indel = Some((IndelType::Insertion, ext_bases.clone(), ext_support));
                    best_indel_support = ext_support;
                } else {
                    break;
                }
            }
        }

        // Accept indel only with stronger evidence threshold
        if let Some((indel_type, bases, _)) = best_indel {
            if best_indel_support.min >= indel_threshold {
                match indel_type {
                    IndelType::Deletion => {
                        let del_len = bases.len();
                        for _ in 0..del_len {
                            if position < sequence.len() {
                                sequence.remove(position);
                            }
                            if position < qualities.len() {
                                qualities.remove(position);
                            }
                        }
                        indel_corrections.push(IndelCorrection {
                            position,
                            correction_type: IndelType::Deletion,
                            base: None,
                            bases,
                        });
                        last_indel_window = Some(window);
                        continue;
                    }
                    IndelType::Insertion => {
                        let ins_len = bases.len();
                        for (i, &ins_base) in bases.iter().enumerate() {
                            sequence.insert(position + i, ins_base);
                            let inserted_quality = 33 + 30; // phred 30
                            if position + i <= qualities.len() {
                                qualities.insert(position + i, inserted_quality);
                            }
                        }
                        indel_corrections.push(IndelCorrection {
                            position,
                            correction_type: IndelType::Insertion,
                            base: if ins_len == 1 { Some(bases[0]) } else { None },
                            bases,
                        });
                        last_indel_window = Some(window);
                        position += ins_len + 1;
                        continue;
                    }
                }
            }
        }

        position += 1;
    }

    (corrected_positions, indel_corrections)
}

fn try_deletion_at(
    sequence: &[u8],
    position: usize,
    spectrum: &KmerSpectrum,
) -> SupportStats {
    if position >= sequence.len() || sequence.len() <= 1 {
        return SupportStats::default();
    }
    let mut temp: Vec<u8> = Vec::with_capacity(sequence.len() - 1);
    temp.extend_from_slice(&sequence[..position]);
    temp.extend_from_slice(&sequence[position + 1..]);
    if temp.is_empty() {
        return SupportStats::default();
    }
    let eval_pos = position.min(temp.len() - 1);
    spectrum.support_for_position(&temp, eval_pos)
}

fn try_multi_deletion_at(
    sequence: &[u8],
    position: usize,
    del_len: usize,
    spectrum: &KmerSpectrum,
) -> SupportStats {
    if del_len == 0 || position + del_len > sequence.len() || sequence.len() <= del_len {
        return SupportStats::default();
    }
    let mut temp: Vec<u8> = Vec::with_capacity(sequence.len() - del_len);
    temp.extend_from_slice(&sequence[..position]);
    temp.extend_from_slice(&sequence[position + del_len..]);
    if temp.len() < spectrum.k() {
        return SupportStats::default();
    }
    let eval_pos = position.min(temp.len() - 1);
    spectrum.support_for_position(&temp, eval_pos)
}

fn try_insertion_at(
    sequence: &[u8],
    position: usize,
    inserted_base: u8,
    spectrum: &KmerSpectrum,
) -> SupportStats {
    let mut temp: Vec<u8> = Vec::with_capacity(sequence.len() + 1);
    temp.extend_from_slice(&sequence[..position]);
    temp.push(inserted_base);
    temp.extend_from_slice(&sequence[position..]);
    spectrum.support_for_position(&temp, position)
}

fn try_multi_insertion_at(
    sequence: &[u8],
    position: usize,
    inserted_bases: &[u8],
    spectrum: &KmerSpectrum,
) -> SupportStats {
    let mut temp: Vec<u8> = Vec::with_capacity(sequence.len() + inserted_bases.len());
    temp.extend_from_slice(&sequence[..position]);
    temp.extend_from_slice(inserted_bases);
    temp.extend_from_slice(&sequence[position..]);
    spectrum.support_for_position(&temp, position)
}

fn should_accept_correction(
    original: SupportStats,
    candidate: SupportStats,
    trusted_floor: u32,
) -> bool {
    let trusted_recovery = candidate.min >= trusted_floor && candidate.min > original.min;
    let material_sum_gain = candidate.sum
        >= original
            .sum
            .saturating_add(u64::from(trusted_floor.max(1)) * u64::from(original.windows.max(1)));
    trusted_recovery || material_sum_gain
}

fn low_quality_tail_cutoff(qualities: &[u8], limit: usize, min_quality: u8) -> Option<usize> {
    let mut cutoff = limit.min(qualities.len());
    let mut low_run = 0usize;

    while cutoff > 0 {
        let phred = qualities[cutoff - 1].saturating_sub(33);
        if phred.saturating_add(2) < min_quality {
            cutoff -= 1;
            low_run += 1;
        } else {
            break;
        }
    }

    if low_run >= 3 { Some(cutoff) } else { None }
}
