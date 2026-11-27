// basalt/dense/exact_tail.hpp - which rows of a prompt request the decode windows read (--exact-tail).
#pragma once

#include <algorithm>
#include <cstdint>

namespace basalt::dense {

/// The prompt's tokens after a request's own `n_tokens` (model::PrefillRequest's fields), the prompt's last one (which a
/// decode window always reads) left out. The server says what the whole prompt has still to read (total_tokens), the CLI
/// the positions up to its stop (rows_after); a caller that says neither is taken to end the prompt.
inline int64_t tokens_after(int total_tokens, int n_tokens, int64_t rows_after) {
    if (total_tokens > 0) return std::max<int64_t>((int64_t) total_tokens - n_tokens, 0);
    return std::max<int64_t>(rows_after, 0);
}

/// The rows at the end of a request of `n` tokens that are among the prompt's last `tail`, `after` tokens following the
/// request (tokens_after): those go through decode windows, the n - rows before them through prompt chunks.
inline int exact_tail_rows(int64_t tail, int n, int64_t after) {
    return (int) std::clamp<int64_t>(tail - after, 0, std::max(n, 0));
}

}  // namespace basalt::dense
