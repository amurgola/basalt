// tests/models/qwen35/exact_tail_test.cpp - --exact-tail's split of a prompt request (dense/exact_tail.hpp): which of its
// rows are among the prompt's last N, by what the server and the CLI say follows the request.  Host only.
#include "basalt/dense/exact_tail.hpp"

#include <cstdio>
#include <string>

using basalt::dense::exact_tail_rows;
using basalt::dense::tokens_after;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

}  // namespace

int main() {
    // what follows a request (total_tokens, n_tokens, rows_after): the server's count of the whole prompt still to read
    // wins over the read's own stop
    check(tokens_after(730, 700, 0) == 30, "the server: 730 tokens left, 700 of them here, 30 follow");
    check(tokens_after(730, 730, 0) == 0, "the server: the request that ends the prompt");
    check(tokens_after(0, 700, 30) == 30, "the CLI: 30 positions up to the stop");
    check(tokens_after(0, 700, -1) == 0, "a caller that says nothing ends the prompt");
    check(tokens_after(600, 700, 0) == 0, "a count below the request's own tokens is no tokens after");

    // off
    check(exact_tail_rows(0, 730, 0) == 0, "0: every row through chunks");
    // the request ends the prompt
    check(exact_tail_rows(64, 730, 0) == 64, "the last 64 of 730");
    check(exact_tail_rows(64, 40, 0) == 40, "a request shorter than the tail goes whole");
    check(exact_tail_rows(1 << 20, 730, 0) == 730, "a tail past the prompt takes every row");
    // tokens follow the request
    check(exact_tail_rows(64, 700, 30) == 34, "30 follow: the last 34 here are in the tail");
    check(exact_tail_rows(64, 700, 64) == 0, "64 follow: the tail starts after the request");
    check(exact_tail_rows(64, 700, 5000) == 0, "far from the end");
    check(exact_tail_rows(64, 10, 30) == 10, "a short request inside the tail");
    check(exact_tail_rows(64, 0, 0) == 0, "no rows");

    std::printf("%s\n", failures ? "FAILED" : "PASSED");
    return failures ? 1 : 0;
}
