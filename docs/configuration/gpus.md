# GPUs

Three settings say how a run uses its cards. The engine never picks a mode or a cut for you: `gpu-mode` decides, `gpu-share` sets
the proportions, and a mode the model cannot run, or a card that would hold nothing, is refused with the reason before the load.

They mean the same in the server (`"gpu"`, `"gpu-mode"` and `"gpu-share"` in a model entry, or `--gpu`, `--gpu-mode`, `--gpu-share`),
in `basalt` itself, in `basalt-bench` and in `basalt-probe`.

| Setting | Values | Default |
| --- | --- | --- |
| `gpu` | a number (`0`), a list (`0,1`, `1,0,2`) or `all` | GPU 0 alone; one warning at start names the other visible cards (only when `gpu` is unset) |
| `gpu-mode` | `peer`, `split`, `tensor` | none; required when `gpu` names two or more cards, an error with one |
| `gpu-share` | one positive number per card, in the order of `gpu` (`3,1`) | by the mode ([shares](#shares)); an error with one card |

- Cards are numbered as `nvidia-smi` numbers them. The first card listed is the main card; `all` is every visible card in order, so
  GPU 0 is the main card. When `CUDA_VISIBLE_DEVICES` is set, the numbers count within its list.
- On this fork every card, the main card included, may be Blackwell (sm_120), Ada (sm_89) or Ampere (sm_86), in every mode, when
  the build holds its code ([an Ada card](#an-ada-sm_89-card), [an Ampere card](#an-ampere-sm_86-card)); any other card stops the
  start with an error.
- A number past the visible cards or a card listed twice stops the start with an error.
- **Idle cards:** with `gpu` unset and more cards visible, every model warns once at start (`GPU 1 (RTX 5060 Ti) unused; to use
  more cards set gpu (0,1 or all) and gpu-mode`), in the CLI, the server and the bench. It is a warning, not an error. Naming a
  card (`gpu 0`) is a choice and prints nothing.
- **Old values:** `gpu: auto`, a `gpu` list without a `gpu-mode`, and the flags the old multi-card logic had are errors, with the
  valid forms in the message ([upgrading](../changelog/0.5.7.md#upgrading)).

```
gpus: peer; RTX 5090 (nvidia-smi 0) main, share 0.80; RTX 5060 Ti (nvidia-smi 1) peer, share 0.20 (bandwidth)
```

The start prints this line before the load and each card's part after it. `--gpu-plan` prints the choice as JSON, with the mode and
the shares, and checks the model's modes without loading it.

## Modes

- **peer:** the main card runs every layer and decodes; the other cards help. What they do depends on the model: hold and compute
  experts of a mixture of experts, read the last layers of a long prompt, or both.
- **split:** the layers are divided between the cards in order; each card holds its layers' weights and their KV cache and
  recurrent state, and a window or prompt chunk crosses from one card to the next. The cut follows the shares.
- **tensor:** every layer sharded across the cards. The name is reserved; no model runs it yet.

| Model | peer | split | tensor |
| --- | --- | --- | --- |
| [Qwen3.8-Flash-Next](../models/qwen38-flash-next.md#cards) | up to 4 cards: experts and prompt layers | up to 8 cards | refused |
| [DeepSeek-V4.1-Flash](../models/deepseek41.md#cards) | up to 5 cards: experts, a block of layers each | refused | refused |
| [Hy4-flash](../models/hy4-flash.md#cards) | up to 4 cards: experts, a block of layers each | refused | refused |
| [Qwen3.8-27B](../models/qwen38-27b.md#cards) | 2 cards: the last layers of long prompts | 2 cards | refused |
| [Inkling 2 Mini](../models/inkling2.md#two-cards) | 2 cards: the experts that spill from the main card | 2 cards | refused |

A mode a model refuses is an error that names the modes it runs, for example `gpu-mode split: Hy4-flash does not run it; it runs
peer`. More cards than a mode takes is refused with the most it takes. One mode per run.

## Shares

Shares are normalized to sum to 1 and stay fixed for the run. What a share divides depends on the mode:

| Mode, part | What the share divides | Default without `gpu-share` |
| --- | --- | --- |
| split | each card's bytes: its layers' weights plus their KV and state at the context and slots; the cut is the layer nearest the shares | free VRAM at start |
| peer, experts | the experts the main card does not hold, dealt hottest first to the peers; the main card fills first | free VRAM of the peers |
| peer, prompt layers | the layers of a long prompt's chunk: the main card reads the first part, each card the next by share | memory bandwidth |
| peer, the main card's hot share | only with `gpu-share`: Flash-Next interleaves the hottest experts between the main card and a peer by share | off: the main card keeps the hottest |

A part that does not fit its card is refused with the numbers (what it needs, what is free); lower that card's `gpu-share`,
`max-context` or `parallel`. A peer that would hold nothing (the main card holds every expert, or its room is gone) is refused too,
with `use gpu 0`.

## Peer mode

- **Experts (Flash-Next, DeepSeek, Hy4-flash, Inkling).** Each peer holds experts the main card does not and computes their rows in
  every decode window, following the conversation like the main card's cache. The rows are the main card's kernels' rows, so where an
  expert sits does not change them. Flash-Next and Inkling deal the experts one by one, hottest first; DeepSeek and Hy4-flash give each
  peer a contiguous block of layers, sized to its room.
- **Prompt rows.** A peer also computes its experts' share of each prompt chunk: any chunk on a peer as fast as the main card, chunks
  up to 2,559 rows on a slower one (a slower card costs a longer chunk more than the experts it saves the main card's link).
- **Prompt layers (Flash-Next, Qwen3.8-27B).** A long fresh part of a prompt is read as a pipeline over the cards while no other
  conversation decodes: the main card reads the first layers of a chunk and hands it on, a peer reads the next layers while the cards
  before it read the next chunks, and the main card reads the last chunk whole. Decode never uses the stage.
  - Flash-Next's cuts come from the shares over layers 2 to the last. A part shorter than a chunk, a chunk under 4,096 tokens, a
    chunk short enough to be read routed-only on a slower peer, and a prompt that arrives while other slots decode stay on the main
    card.
  - A peer's weights, prompt state and buffers go into its tier's last slots for the read, so outside a prompt decode does not change.
  - A loaded control vector is copied to every stage card.
- **Per-operation splits (Flash-Next).** A peer whose memory is faster than the main card's takes a share of each GDN mixer and of the
  head ([the GDN split](../precision/qwen38-flash-next.md#gdn-split-across-two-cards)).
- **Tuning:** `peer-reserve-mib` (default 600) is the VRAM a peer keeps free beside its experts and prompt buffers; `peer-exclusive`
  keeps the peer's experts out of RAM when RAM is short; `gpu 0` keeps everything on the main card.

Measured, from the ledger's rows:

- **Flash-Next:** no speed loss against the earlier peer path (64K prefill 9,823 against 9,835 tok/s, conversation decode within
  noise). A normal peer run scores BF16 KLD 0.06174 against 0.06221 on one card and top-1 91.84% against 91.80%.
- **Qwen3.8-27B:** a 64K prefill reads 17% faster than on one card (7,410 against 6,336 tok/s); decode stays on the main card.
- **Inkling 2 Mini:** a peer helps only when experts spill beyond what the main card's elastic copies cover. At default settings one
  card is fastest up to 1M tokens, and a peer is refused when the main card holds every expert
  ([Inkling](../models/inkling2.md#two-cards)).
- **Rounding:** a peer's prompt reads differ from one card's at summation-order rounding, about 0.01 KLD on `--diag-gpu-check`, the
  same as any reordering of the experts' sum. `--diag-main-card-bits` makes Flash-Next's other cards compute the main card's bits,
  and the prompt stages are then bitwise ([diagnostics](diagnostics.md)).

## Split mode

Each card runs a block of layers, one after another, with the window handed to the next card once. The cut is made at start by the
shares, before any weight is placed; every card holds only the dense weights of its own layers, and its leftover VRAM goes to that
model's experts or caches.

- **Flash-Next:** the experts no card holds are read from one pinned RAM copy that all stages share, and each card's adaptive swaps stay
  within its own layers. With two cards, a server and the drafter, one conversation at a time, the split overlaps consecutive windows:
  the first card starts window K+1 on the guess that K is accepted whole while the second card verifies K; a guess that fails puts the
  first card's state back and runs the window again, so the text is the same either way (`pipelined windows: N launched, M held` per
  request).
- **Qwen3.8-27B:** the output is the main card's alone to the bit ([27B](../models/qwen38-27b.md#cards)).
- **Inkling 2 Mini:** a split holds every expert of its layers on its card; it is about 25% slower than one card at the contexts where
  one card holds or copies back its experts ([Inkling](../models/inkling2.md#two-cards)).
- A split costs a fixed part of a window per crossing: use it for a model or context that does not fit one card, and a peer where it
  does.

## Two 16 GB cards

`gpu "0,1"` with `gpu-mode "peer"` on two cards of 16 GB (two RTX 5080s, say) has each card hold experts in what its VRAM has left. The
engine sizes the second card to leave room for its prompt buffers (about 200 MiB, computed) and `peer-reserve-mib` more (default
600 MiB, at least 128). If the start prints `prefill peer: the peer's buffers do not fit`, raise `peer-reserve-mib` (a display or
another program may be using the card) and its experts shrink by the same amount. `peer GPU 1: N experts` says how many it holds.

## An Ada (sm_89) card

On this fork every path also runs on sm_89 (Ada): an Ada card can be the only card, the main card or any later card, in
every mode a model runs. The build must hold its code: `-DCMAKE_CUDA_ARCHITECTURES=89` for Ada cards alone, `"120;89"`
beside a Blackwell card (`-DPEER_ADA=ON` adds 89 to the list). The engine refuses at start a card its build holds no code
for. On 8.9 the kernels that need thread-block clusters, bulk async copies or FP4 tensor cores are compiled out and their
callers take plain kernels ([Ada port notes](../../ADA_PORT_NOTES.md)). Upstream's `-DPEER_ADA=ON` builds only
Flash-Next's peer kernels for sm_89 and refuses an Ada main card, `split` and the other models' peers; here none of that
is refused. On a Blackwell + Ada pair:

- Flash-Next, `split`: both cards read NVFP4 prompts through MMQ (its FP4 prompt kernels run only when every card that
  reads prompt layers is 12.x), so the buffers, chunk plan and row rounding are the same on both. Its decode does not
  use FP4 tensor cores.
- Flash-Next, `peer` with a Blackwell main card: the older tier card reads no prompt layers (the start prints `prompt
  split: off (a tier card has no FP4 tensor cores, so the main card reads every layer)`), the main card keeps its FP4
  prompt kernels, and the tier card computes its experts' rows of a chunk through MMQ.
- DeepSeek and Hy4-flash take an Ada main card or peer: their kernels build for 8.9 as they are (DeepSeek's parity
  tests pass here). No pack of either has been run here.
- Inkling 2 Mini runs on an Ada card alone (both packs run here). In `split` or `peer` each card picks its own prompt
  kernels (FP4 experts only on 12.x, the Q8_0 projections and MXFP8 experts in their 8.9 forms).
- Not run here: no second card was available, so every two-card mode has been checked only by the parity tests and
  with the 4090 alone running the `"120;89"` binary.

## An Ampere (sm_86) card

An Ampere card (RTX 30 series, RTX A2000 to A6000) follows the Ada card's rule: the only card, the main card or any later
card, in every mode a model runs, in a build that holds sm_86 code (`-DPEER_AMPERE=ON` adds 86 to
`CMAKE_CUDA_ARCHITECTURES`; this fork's release archives are built for `"120;89;86"`). sm_80 cards (A100) are refused: the
build holds no code for them. An 8.6 pass takes the 8.9 kernels, with three differences, because Ampere has neither the
E4M3 conversion nor E4M3 tensor cores:

- E4M3 codes become f16 or bf16 in software (`narrow_bf16.cuh`: the bits moved into an f32 and the exponent rebiased by
  one multiply), exact, checked on every input by `narrow_bf16_parity`.
- The dense FP4 prompt product (`fp4_dense`'s `kernel_ada`) sums each 16-value scale group on the INT8 tensor cores: the
  operands hold twice each E2M1 code, so the s32 sum is four times the group's and exact, and a quarter of it is the number
  8.9's E4M3 tensor cores give. Every later operation is the same, so an Ampere card's prompt rows are an Ada card's bits.
- The FP8 prompt product (`fp8_gemm`) decodes its codes in software before the same f16 product.

Measured on one RTX 3090 on Windows (24 GB, a PCIe 4.0 x2 slot, the `"120;89;86"` binary, CUDA 13.0):

- GPU tests on the card alone: every test passes except the five that fail on an RTX 4090 too (`file_expert_repin_test`,
  `gdn_head_split_parity`, `dense_ops_parity`'s old baseline, `ds41_vision_reserve_test`, and `prefill_mmq_mxfp4_test`'s
  IQ2_XS / IQ2_S rows at the same 2.96e-4) and `prefill_hc_up_mix_test` at 129 tokens, where cuBLAS, the test's
  reference, sums in another order on this card (the kernel is bitwise at every other width, position independent, and
  2.6e-7 from the double reference). The persistent 8.9 kernels (`dm_route_ada_parity`, `hc_read_ada_parity`,
  `hc_fused_ada_parity`, `mid_chain_parity`, `q8_gemm_test`) pass bitwise on 8.6.
- Qwen3.8-27B (NVFP4 pack, 32,768 context, INT8 KV): 16.0 GiB on the card and 2.4 GiB of RAM, loaded in 5.7 s, card peak
  20.2 GiB. Decode with the block drafter 69 to 95 tok/s on chat prompts and 173 to 182 on JSON, 31 without it; a cold
  14,000 or 28,000-token prompt reads at 1,040 to 1,090 tok/s. Sleep (`--idle-sleep`, `POST /v1/sleep`) frees the card in
  7.0 s and wakes in 6.8 s over that x2 slot, and the next reply is correct.
- Inkling 2 Mini (NVFP4-DQ8 pack, 32,768 context, fp16 KV): 16.2 GiB on the card and 0.9 GiB of RAM. Decode 150 to 168
  tok/s on chat prompts and 212 to 215 on JSON; cold prompts read at 3,940 to 4,000 tok/s.
- Flash-Next (NVFP4-DQ6_K pack, 32,768 context, fp16 KV) on the RTX 3090 alone: 20.7 GiB on the card, 47.7 GiB of RAM.
  Decode 33 to 53 tok/s; cold prompts read at 750 (14,000 tokens) and 1,290 tok/s (28,000).
- The same binary on an RTX 5090 (the Blackwell half, which the Ada port had not run): Qwen3.8-27B decodes at 200 to
  281 tok/s on chat prompts and 514 to 538 on JSON and reads cold prompts at 13,700 to 14,900 tok/s, and Flash-Next
  alone reads them at 5,640 and 8,050 tok/s: the figures of the 0.5.5 build on that card. Of its GPU tests, four of
  the 4090's known ones fail (`file_expert_repin_test`, `gdn_head_split_parity`, `ds41_vision_reserve_test`,
  `prefill_mmq_mxfp4_test`), and so do `ds41_router_parity`, `ds41_attn_index_parity`, `ds41_dspark_parity` and
  `gqa_attn_parity`: their frozen hashes for 9.0 and up were taken on Linux, and on Windows the router's hash on the
  5090 is the value the tests hold for a 4090 (the inputs come from the standard library's distributions).
- Open: `dec_moe_parity --nvfp4` stops at its TMA kind on the 5090 with this binary (`dec_moe_vram: formats 40 / 40`:
  the TMA expert kernel reports no blocks) and passes in a 120-only build of the same source. The engine never selects
  that kernel (`vram` is -1 in `expert_resources.cpp`); the cause has not been found.
- RTX 5090 main card with the RTX 3090 as Flash-Next's tier (`gpu 0,1`, `gpu-mode peer`): 8,569 experts on the 3090,
  decode 114 to 168 tok/s, cold prompts 4,415 and 6,496 tok/s (the 0.5.6 build of this fork: 5,190 and 7,930).
- Two RTX 3090s (a second machine: Ryzen 9 5900X, 16 GB of RAM, the cards on x8 and x4 links, Windows 11), Qwen3.8-27B,
  INT8 KV. One card at 32,768: decode 73 to 102 tok/s on chat prompts and 187 to 192 on JSON, cold prompts 1,170 to
  1,190 tok/s. `gpu 0,1` with `gpu-mode split` at 131,072: 8.4 and 7.6 GiB of weights, 12.4 and 12.5 GiB used, decode
  77 to 106 and 195 to 199 tok/s, cold prompts 1,620 (14,000 tokens) and 1,800 tok/s (28,000). At 262,144 the split
  takes 14.2 and 15.0 GiB and answers. `gpu-mode peer` does not start there (`out of page-locked memory for the token
  embedding`): its page-locked buffers and the 2.4 GiB embedding do not fit 16 GB of RAM. Sleep is refused for the same
  reason (`the model stays awake (RAM: 4411 MiB free, 31088 MiB of GPU state to hold and 8192 MiB to keep)`).
- Not run: an Ampere main card with a Blackwell tier, `split` over unlike cards, `peer` on two older cards with enough
  RAM, the Hy4-flash and DeepSeek packs on 8.6 (their kernel tests pass), a pack with F8_E4M3 projections, and Linux.

Three or more cards are community-tested: the path is checked on two cards. Reports from machines with three or four cards are
welcome ([community](../community.md)).
