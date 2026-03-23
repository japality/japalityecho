namespace {

constexpr unsigned char INVALID_BASE = 4u;
constexpr unsigned int NO_ADAPTER = 0xFFFFFFFFu;

struct SupportStats {
    unsigned int min_count;
    unsigned long long sum;
    unsigned int windows;
};

__device__ __forceinline__ unsigned char complement_bits(unsigned char base) {
    return base ^ 0x3u;
}

__device__ unsigned int lookup_trusted_count(
    unsigned long long key,
    const unsigned long long* keys,
    const unsigned int* counts,
    unsigned int trusted_len
) {
    unsigned int low = 0u;
    unsigned int high = trusted_len;

    while (low < high) {
        const unsigned int mid = low + (high - low) / 2u;
        const unsigned long long mid_key = keys[mid];
        if (mid_key == key) {
            return counts[mid];
        }
        if (mid_key < key) {
            low = mid + 1u;
        } else {
            high = mid;
        }
    }

    return 0u;
}

__device__ unsigned int canonical_kmer_count(
    const unsigned char* bases,
    unsigned int start,
    unsigned int k,
    const unsigned long long* trusted_keys,
    const unsigned int* trusted_counts,
    unsigned int trusted_len
) {
    unsigned long long forward = 0ull;
    for (unsigned int index = 0u; index < k; ++index) {
        const unsigned char base = bases[start + index];
        if (base > 3u) {
            return 0u;
        }
        forward = (forward << 2) | static_cast<unsigned long long>(base);
    }

    unsigned long long reverse = 0ull;
    unsigned long long encoded = forward;
    for (unsigned int index = 0u; index < k; ++index) {
        const unsigned long long bits = encoded & 0x3ull;
        reverse = (reverse << 2) | static_cast<unsigned long long>(complement_bits(static_cast<unsigned char>(bits)));
        encoded >>= 2;
    }

    const unsigned long long canonical = forward < reverse ? forward : reverse;
    return lookup_trusted_count(canonical, trusted_keys, trusted_counts, trusted_len);
}

__device__ SupportStats support_for_position(
    const unsigned char* bases,
    unsigned int length,
    unsigned int position,
    unsigned int k,
    const unsigned long long* trusted_keys,
    const unsigned int* trusted_counts,
    unsigned int trusted_len
) {
    SupportStats stats{};
    if (length < k || position >= length) {
        return stats;
    }

    stats.min_count = 0xFFFFFFFFu;
    const unsigned int start_min = position >= (k - 1u) ? position - (k - 1u) : 0u;
    const unsigned int max_start = length - k;
    const unsigned int start_max = position < max_start ? position : max_start;

    for (unsigned int start = start_min; start <= start_max; ++start) {
        const unsigned int count = canonical_kmer_count(
            bases,
            start,
            k,
            trusted_keys,
            trusted_counts,
            trusted_len
        );
        if (count < stats.min_count) {
            stats.min_count = count;
        }
        stats.sum += count;
        stats.windows += 1u;
    }

    if (stats.windows == 0u) {
        stats.min_count = 0u;
    }
    return stats;
}

__device__ bool should_accept_correction(
    SupportStats original,
    SupportStats candidate,
    unsigned int trusted_floor
) {
    const bool trusted_recovery = candidate.min_count >= trusted_floor && candidate.min_count > original.min_count;
    const unsigned long long material_gain =
        original.sum + static_cast<unsigned long long>(trusted_floor > 0u ? trusted_floor : 1u) *
        static_cast<unsigned long long>(original.windows > 0u ? original.windows : 1u);
    const bool material_sum_gain = candidate.sum >= material_gain;
    return trusted_recovery || material_sum_gain;
}

__device__ unsigned int quality_cutoff(
    const unsigned char* qualities,
    unsigned int limit,
    unsigned int min_quality
) {
    unsigned int cutoff = limit;
    unsigned int low_run = 0u;

    while (cutoff > 0u) {
        const unsigned char phred = qualities[cutoff - 1u];
        if (phred + 2u < min_quality) {
            cutoff -= 1u;
            low_run += 1u;
        } else {
            break;
        }
    }

    return low_run >= 3u ? cutoff : limit;
}

__device__ unsigned int find_adapter_start(
    const unsigned char* bases,
    unsigned int length,
    const unsigned char* adapter_codes,
    const unsigned int* adapter_lengths,
    const unsigned int* adapter_supports,
    unsigned int adapter_count,
    unsigned int adapter_pitch,
    unsigned int* adapter_hit_index
) {
    int best_score = -2147483647;
    unsigned int best_start = length;
    unsigned int best_index = NO_ADAPTER;

    for (unsigned int candidate_index = 0u; candidate_index < adapter_count; ++candidate_index) {
        const unsigned int adapter_length = adapter_lengths[candidate_index];
        if (adapter_length < 8u || length < 8u) {
            continue;
        }

        const unsigned int search_window = adapter_length > 28u ? adapter_length : 28u;
        const unsigned int search_start = length > search_window ? length - search_window : 0u;
        const unsigned char* adapter = adapter_codes + candidate_index * adapter_pitch;

        for (unsigned int start = search_start; start < length; ++start) {
            const unsigned int overlap = (length - start) < adapter_length ? (length - start) : adapter_length;
            if (overlap < 8u) {
                continue;
            }

            unsigned int matches = 0u;
            for (unsigned int index = 0u; index < overlap; ++index) {
                const unsigned char lhs = bases[start + index];
                const unsigned char rhs = adapter[index];
                if (lhs <= 3u && lhs == rhs) {
                    matches += 1u;
                }
            }

            const unsigned int mismatches = overlap - matches;
            const unsigned int allowed = (overlap * 15u + 99u) / 100u;
            if (mismatches > allowed || matches * 100u < overlap * 85u) {
                continue;
            }

            const int score =
                static_cast<int>(matches) * 4 -
                static_cast<int>(mismatches) * 3 +
                static_cast<int>(overlap) +
                static_cast<int>(adapter_supports[candidate_index]);

            if (score > best_score || (score == best_score && start < best_start)) {
                best_score = score;
                best_start = start;
                best_index = candidate_index;
            }
        }
    }

    *adapter_hit_index = best_index;
    return best_start;
}

}  // namespace

extern "C" __global__ void japalityecho_pack_reads(
    const unsigned char* base_codes,
    unsigned char* packed_bases,
    const int total_bases
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_bases) {
        return;
    }

    packed_bases[index] = base_codes[index];
}

extern "C" __global__ void japalityecho_trim_correct(
    const unsigned char* base_codes,
    const unsigned char* qualities,
    const unsigned int* read_lengths,
    const unsigned long long* trusted_keys,
    const unsigned int* trusted_counts,
    const unsigned int trusted_len,
    const unsigned char* adapter_codes,
    const unsigned int* adapter_lengths,
    const unsigned int* adapter_supports,
    const unsigned int adapter_count,
    const unsigned int adapter_pitch,
    unsigned char* corrected_codes,
    unsigned int* trim_offsets,
    unsigned int* adapter_hits,
    const int reads_per_batch,
    const int read_pitch,
    const unsigned int kmer_size,
    const unsigned int trusted_floor,
    const unsigned int min_quality
) {
    const int read_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (read_index >= reads_per_batch) {
        return;
    }

    const unsigned int length = read_lengths[read_index];
    const int base_offset = read_index * read_pitch;
    unsigned char* corrected = corrected_codes + base_offset;
    const unsigned char* input_bases = base_codes + base_offset;
    const unsigned char* input_qualities = qualities + base_offset;

    for (int index = 0; index < read_pitch; ++index) {
        corrected[index] = input_bases[index];
    }

    unsigned int adapter_hit_index = NO_ADAPTER;
    const unsigned int adapter_start = find_adapter_start(
        corrected,
        length,
        adapter_codes,
        adapter_lengths,
        adapter_supports,
        adapter_count,
        adapter_pitch,
        &adapter_hit_index
    );
    const unsigned int correction_limit = adapter_start < length ? adapter_start : length;
    const unsigned int quality_grace = min_quality + 5u;

    if (length >= kmer_size && kmer_size >= 3u && trusted_len > 0u) {
        for (unsigned int position = 0u; position < correction_limit; ++position) {
            const unsigned char phred = input_qualities[position];
            const SupportStats original = support_for_position(
                corrected,
                length,
                position,
                kmer_size,
                trusted_keys,
                trusted_counts,
                trusted_len
            );
            const bool needs_attention =
                phred < min_quality ||
                (phred <= quality_grace && original.windows > 0u && original.min_count < trusted_floor);
            if (!needs_attention || original.windows == 0u) {
                continue;
            }

            const unsigned char original_base = corrected[position];
            unsigned char best_base = original_base;
            SupportStats best_support = original;

            for (unsigned char candidate_base = 0u; candidate_base < 4u; ++candidate_base) {
                if (candidate_base == original_base) {
                    continue;
                }

                corrected[position] = candidate_base;
                const SupportStats candidate = support_for_position(
                    corrected,
                    length,
                    position,
                    kmer_size,
                    trusted_keys,
                    trusted_counts,
                    trusted_len
                );
                if (candidate.min_count > best_support.min_count ||
                    (candidate.min_count == best_support.min_count && candidate.sum > best_support.sum)) {
                    best_base = candidate_base;
                    best_support = candidate;
                }
            }

            corrected[position] = original_base;
            if (best_base != original_base &&
                should_accept_correction(original, best_support, trusted_floor)) {
                corrected[position] = best_base;
            }
        }
    }

    const unsigned int quality_limit = quality_cutoff(input_qualities, correction_limit, min_quality);
    trim_offsets[read_index] = quality_limit < adapter_start ? quality_limit : adapter_start;
    adapter_hits[read_index] = adapter_hit_index;
}

extern "C" __global__ void japalityecho_scatter_kept(
    const unsigned int* trim_offsets,
    const unsigned int* keep_flags,
    unsigned int* output_offsets,
    const int reads_per_batch
) {
    const int read_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (read_index >= reads_per_batch) {
        return;
    }

    if (keep_flags[read_index] != 0u) {
        output_offsets[read_index] = trim_offsets[read_index];
    }
}
