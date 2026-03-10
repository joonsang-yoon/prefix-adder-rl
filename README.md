# Prefix-Adder RL Search in Chisel

[![CI](https://github.com/joonsang-yoon/prefix-adder-rl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/joonsang-yoon/prefix-adder-rl/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/joonsang-yoon/prefix-adder-rl.svg)](LICENSE)
[![Scala](https://img.shields.io/badge/Scala-2.13.16-DC322F.svg?logo=scala&logoColor=white)](https://www.scala-lang.org/)
[![Chisel](https://img.shields.io/badge/Chisel-7.5.0-2A3172.svg)](https://www.chisel-lang.org/)

Search **dependent-tree prefix-adder** topologies with policy-gradient RL, elaborate them to RTL with **Chisel**, and score them with either a lightweight **LibreLane / OpenROAD** backend or a fast **synthetic surrogate backend** for smoke testing.

This repository contains two search frontends:

- a **tabular softmax policy** for small, explicit decision tables
- a **frontier-conditioned DAG-aware pointer actor-critic** that mixes recursive topology encodings, shared-DAG summaries, and candidate-set attention

Both frontends operate over the same topology family and share the same downstream evaluation abstraction, reward model, duplicate-detection logic, and Pareto-frontier tracking.

> **Scope:** this codebase supports the **dependent-tree** topology family only. It does **not** implement general independent-tree prefix-network search.

## What the repository does

At a high level, every evaluation follows the same loop:

1. sample or load a valid dependent-tree topology
2. choose an evaluation backend
3. either elaborate `TopLevelModule.PrefixAdderMacro` into SystemVerilog and run LibreLane, or score the topology with the synthetic surrogate backend
4. obtain a `(power, delay, area)` tuple
5. convert the resulting PPA tuple into a scalar reward
6. update the policy and maintain a Pareto frontier of non-dominated designs

All three objectives are minimized:

- `power = power__total`
- `delay = clock_period - timing__setup__ws`
- `area = design__instance__area`

The shipped `librelane_config.json` uses a **reduced flow** based on synthesis plus pre-PnR STA. That makes it a good fast proxy for search, but not a sign-off-quality backend.

The synthetic backend is intentionally **not** physically meaningful. It is deterministic and topology-driven, and exists to make CI, smoke tests, and local validation possible without LibreLane.

## How the pieces fit together

```text
policy / topology source
        ↓
dependent-tree topology JSON
        ↓
evaluation backend
   ├── librelane
   │     ↓
   │ TopLevelModule.PrefixAdderMacro
   │     ↓
   │ SystemVerilog RTL
   │     ↓
   │ LibreLane / OpenROAD reduced flow
   │     ↓
   │ metrics.json → (power, delay, area)
   │
   └── synthetic
         ↓
      deterministic surrogate PPA
             ↓
           reward
             ↓
     Pareto frontier update
             ↓
         policy update
```

## Repository layout

```text
.
├── TopLevelModule/        # Reflection-based elaboration wrapper + PrefixAdderMacro top level
├── ExternalModule/        # Small output-register helper used by PrefixAdderMacro
├── PrefixAdderLib/        # Dependent-topology model, validation, DOT export, adder core
├── PrefixRLCore/          # Sampling, evaluators, reward, frontier, caching, shared search loop
├── PrefixTabularRL/       # Tabular softmax policy + tabular search entrypoint
├── PrefixDeepRL/          # Neural split policy + deep search entrypoint
├── PrefixUtils/           # Catalan counts, hashing, JSON helpers, shell-word parsing
├── example_topologies/    # Hand-authored dependent-tree example topologies
├── scripts/               # Convenience wrappers used by Makefile and the RL code
├── .github/workflows/     # CI workflow using format checks, tests, and synthetic smoke tests
├── build.mill.scala       # Mill build definition
├── Makefile               # Primary user-facing commands
├── mill                   # Mill bootstrap wrapper
├── librelane_config.json  # Reduced LibreLane/OpenROAD flow template
└── README.md
```

Generated artifacts are written under `generated/` on demand and are intentionally ignored by Git.

## Why the search problem is interesting

Even under the dependent-tree restriction, the design space grows very quickly.

For width `n`:

```text
extensionCount(n) = Σ_{j=0}^{n-2} Catalan(j)     for n >= 2
networkCount(n)   = Π_{k=2}^{n} extensionCount(k)
```

`PrefixUtils.Catalan` implements these counts directly, and the small-width values are covered by the test suite.

| Width | Extension count | Total dependent-tree networks |
| ---: | ---: | ---: |
| 1 | 1 | 1 |
| 2 | 1 | 1 |
| 3 | 2 | 2 |
| 4 | 4 | 8 |
| 5 | 9 | 72 |
| 6 | 23 | 1,656 |
| 7 | 65 | 107,640 |
| 8 | 197 | 21,205,080 |

That combinatorial growth is the reason this repo couples a hardware generator with RL-guided search.

## Prerequisites

From the repository root, you will need:

- `bash` and `make`
- a working JDK for Mill / Scala / Chisel
- network access on the **first** build so `./mill` can bootstrap itself and resolve dependencies
- **LibreLane** only when using `BACKEND=librelane`

You can still elaborate RTL, run the pure Scala test suite, and use the synthetic smoke-test flow without LibreLane, as long as the build dependencies have already been downloaded.

## Quick start

### Inspect the available commands

```bash
make help
```

### Run the Scala test suite

```bash
make test
```

This exercises topology counting, topology generation, JSON round-tripping, Pareto-frontier behavior, the tabular policy, the neural policy, the synthetic backend, and shell-word parsing.

### Run end-to-end smoke tests without LibreLane

```bash
make smoke-evaluate \
  WIDTH=8 \
  TOPOLOGY=example_topologies/dependent_balanced_8.json

make smoke-search-tabular WIDTH=4 SMOKE_EPISODES=4
make smoke-search-deep WIDTH=4 SMOKE_EPISODES=4 HIDDEN_SIZE=16
```

These commands use `BACKEND=synthetic` automatically and produce deterministic surrogate `metrics.json` files.

### Elaborate the example top level

```bash
make verilog MODULE=TopLevelModule.ExamplePrefixAdder
```

This uses the reflection-based `Elaborate` wrapper to generate SystemVerilog for the built-in example module.

### Generate `PrefixAdderMacro` from a topology JSON file

```bash
make macro WIDTH=8 TOPOLOGY=example_topologies/dependent_balanced_8.json
```

This is the quickest way to turn a hand-written topology into RTL while keeping the module name expected by the shipped LibreLane configuration.

### Evaluate a single topology with LibreLane

```bash
make evaluate \
  BACKEND=librelane \
  WIDTH=8 \
  TOPOLOGY=example_topologies/dependent_ripple_8.json \
  LIBRELANE_CMD=librelane
```

`LIBRELANE_CMD` can be either a plain executable name such as `librelane` or a shell-style command such as `python -m librelane`.

### Run tabular RL search

Synthetic smoke flow:

```bash
make search-tabular BACKEND=synthetic WIDTH=4 EPISODES=8 RUN_LABEL=tab_smoke
```

LibreLane flow:

```bash
make search-tabular BACKEND=librelane WIDTH=8 EPISODES=32 LIBRELANE_CMD=librelane RUN_LABEL=tab_librelane
```

### Run neural RL search

Synthetic smoke flow:

```bash
make search-deep   BACKEND=synthetic   WIDTH=4   EPISODES=8   HIDDEN_SIZE=16   DAG_SUMMARY_MODE=attention-weighted   ACTION_CONTEXT_MODE=mean-residual   CHECKPOINT_INTERVAL=4   RUN_LABEL=deep_smoke
```

LibreLane flow:

```bash
make search-deep   BACKEND=librelane   WIDTH=8   EPISODES=32   HIDDEN_SIZE=48   LIBRELANE_CMD=librelane   RUN_LABEL=deep_librelane
```

You can also use the unified entrypoint:

```bash
make search ALGORITHM=tabular BACKEND=synthetic WIDTH=4 EPISODES=8 RUN_LABEL=tab_trial
make search ALGORITHM=deep BACKEND=librelane WIDTH=8 EPISODES=32 HIDDEN_SIZE=48 LIBRELANE_CMD=librelane RUN_LABEL=deep_trial
```

If `LEARNING_RATE` is omitted, the wrapper preserves the code defaults:

- **tabular**: `0.08`
- **deep**: `0.02`

### Continue from a saved checkpoint

Both search frontends can initialize from a saved policy JSON checkpoint.

```bash
make search-deep   BACKEND=synthetic   WIDTH=6   EPISODES=12   POLICY_INIT=generated/search/deep/deep_smoke/logs/policy/policy_final.json   SKIP_WARM_STARTS=true   RUN_LABEL=deep_finetune
```

For the deep policy, checkpoints include both model weights and Adam optimizer state. Loading a checkpoint reproduces the saved policy exactly. The search environment itself still starts fresh unless you also reuse the generated design and log directories as reference artifacts.

### Run an architecture sweep

The repository ships a small sweep wrapper for structured comparisons across seeds and neural-architecture modes:

```bash
ALGORITHM=deep SEEDS_LIST=1,2 HIDDEN_SIZES_LIST=16,32 DAG_SUMMARY_MODES_LIST=usage-weighted,attention-weighted ACTION_CONTEXT_MODES_LIST=self-attention,mean-residual make sweep-search BACKEND=synthetic WIDTH=6 EPISODES=8
```

Each run is written under its own labeled output directory and records `run_config.json`, `training.jsonl`, periodic checkpoints, and `summary.json`.

## Common knobs

The Makefile exposes the most useful search and elaboration parameters:

| Variable | Meaning |
| --- | --- |
| `MODULE` | Fully-qualified Chisel module name for `make verilog` |
| `TARGET_DIR` | Output directory for `make verilog` |
| `BACKEND` | `librelane` or `synthetic` |
| `WIDTH` | Adder width in bits |
| `TOPOLOGY` | Path to a dependent-tree topology JSON file |
| `EPISODES` | Number of RL episodes after warm starts |
| `SMOKE_EPISODES` | Episode count used by the synthetic smoke-test targets |
| `SEED` | Random seed for RL sampling |
| `LEARNING_RATE` | Policy update step size |
| `TEMPERATURE` | Softmax temperature used during sampling |
| `HIDDEN_SIZE` | Embedding width of the neural actor-critic |
| `GRADIENT_CLIP` | Gradient-norm clip used by the neural policy |
| `BASELINE_MOMENTUM` | Exponential moving-average momentum for the scalar reward baseline |
| `CHECKPOINT_INTERVAL` | Write immutable policy snapshots every N training episodes |
| `POLICY_INIT` | Optional policy JSON checkpoint used to initialize a new run |
| `RUN_LABEL` | Optional labeled subdirectory appended under the search output root |
| `CLEAN_OUTPUT` | Whether to delete prior `designs/` and `logs/` directories under the resolved run root |
| `SKIP_WARM_STARTS` | Disable deterministic ripple/balanced warm starts |
| `DAG_SUMMARY_MODE` | Deep-policy DAG aggregation mode: `usage-weighted` or `attention-weighted` |
| `ACTION_CONTEXT_MODE` | Deep-policy candidate interaction mode: `self-attention` or `mean-residual` |
| `REGISTER_OUTPUTS` | Whether `PrefixAdderMacro` registers `sum` and `cout` |
| `CLOCK_PERIOD` | Value written into the copied LibreLane config or synthetic metrics |
| `LIBRELANE_CMD` | LibreLane command, e.g. `librelane` or `python -m librelane` |

## What gets generated

### Verilog elaboration

`make verilog` writes RTL under:

```text
generated/verilog/<module-path>/
```

### One-off evaluation

`make evaluate` creates one design directory per evaluated topology:

```text
generated/eval/ep00000_<fingerprint>/
├── metrics.json
├── rtl/
├── topology.dot
├── topology.json
├── librelane_config.json          # present for the LibreLane backend
└── ... backend-specific outputs ...
```

With `BACKEND=synthetic`, the `rtl/` directory contains a small placeholder note instead of elaborated RTL, and `metrics.json` contains deterministic surrogate values.

### RL search

Search writes both design artifacts and logs:

```text
generated/search/<algorithm>/<run-label>/
├── designs/
│   └── ep00010_<fingerprint>/
│       ├── metrics.json
│       ├── rtl/
│       ├── topology.dot
│       ├── topology.json
│       ├── librelane_config.json  # present for the LibreLane backend
│       └── ... backend-specific outputs ...
└── logs/
    ├── episodes/
    │   └── episode_00010.json
    ├── frontier/
    │   └── frontier.json
    ├── policy/
    │   ├── checkpoints/
    │   │   └── policy_ep00008.json
    │   ├── policy_final.json
    │   └── policy_latest.json
    ├── run_config.json
    ├── training.jsonl
    └── summary.json
```

A few details that are easy to miss:

- every evaluated design gets a canonical `topology.json` and a Graphviz-friendly `topology.dot`
- repeated samples are detected by topology fingerprint and **reused from cache** rather than re-running the expensive evaluation backend
- every episode log records the **search state before and after evaluation**, including frontier size, duplicate rate, and normalized frontier statistics
- `logs/training.jsonl` records one compact JSON line per warm start or training episode, including reward, baseline evolution, frontier effects, and policy-training diagnostics
- `logs/run_config.json` records the resolved run configuration before search begins
- `logs/summary.json` records the chosen backend, final frontier, final search-state snapshot, final policy checkpoint location, and resolved output root

## Evaluation backends

### LibreLane backend

`BACKEND=librelane` performs the full repository flow:

1. write the topology JSON and DOT files
2. elaborate `TopLevelModule.PrefixAdderMacro` into RTL
3. copy and patch `librelane_config.json`
4. invoke LibreLane / OpenROAD
5. parse the resulting metrics into `(power, delay, area)`

This is the mode to use when you care about hardware-correlated results.

### Synthetic backend

`BACKEND=synthetic` writes the same topology and logging artifacts, but skips Chisel elaboration and LibreLane. Instead, it computes a deterministic surrogate PPA tuple from topology statistics such as:

- width
- unique and total internal-node counts
- maximum and average tree depth
- subtree reuse ratio
- whether output registers are enabled

This backend is useful for:

- CI
- local smoke tests
- validating the search loop before wiring up LibreLane
- debugging policy updates and caching without long tool runtimes

It is a convenience backend only, not a physical estimator.

## The dependent-tree topology model

A topology is encoded as a JSON object with:

- `model = "dependent-tree"`
- `width`
- `outputs`, where `outputs(i)` describes prefix output `P_i`

Example:

```json
{
  "model": "dependent-tree",
  "width": 4,
  "outputs": [
    {"leaf": 0},
    {"node": [{"leaf": 0}, {"leaf": 1}]},
    {"node": [
      {"node": [{"leaf": 0}, {"leaf": 1}]},
      {"leaf": 2}
    ]},
    {"node": [
      {"node": [{"leaf": 0}, {"leaf": 1}]},
      {"node": [{"leaf": 2}, {"leaf": 3}]}
    ]}
  ]
}
```

A valid dependent-tree topology must satisfy all of the following:

- `P0` is exactly `Leaf(0)`
- every `P_i` spans the ordered, contiguous range `[0, i]`
- for `i > 0`, `P_i` is a `Node(left, right)`
- the root `left` subtree of `P_i` must be **exactly equal** to some previously built output `P_k`

This is validated by `PrefixAdderLib.DependentTopology` before the topology is accepted by the generator or the search code.

## Hardware generation details

`TopLevelModule.PrefixAdderMacro` is the fixed top-level wrapper used by evaluation.

Important implementation details:

- the module name expected by the shipped flow is `PrefixAdderMacro`
- the timing reference in `librelane_config.json` is the `clock` port
- the wrapper loads and validates a topology JSON file at elaboration time
- the internal core computes `(g, p)` pairs and reuses identical subtrees structurally when their signatures match
- output registers are optional and controlled by `REGISTER_OUTPUTS`

The built-in example module:

```text
TopLevelModule.ExamplePrefixAdder
```

simply instantiates an 8-bit registered macro using `example_topologies/dependent_balanced_8.json`.

## RL algorithms

### Tabular policy

`PrefixTabularRL.TabularSoftmaxPolicy` stores learned logits per decision context and a coarse search-state bucket. The table key combines:

- output index
- segment bounds
- whether the decision is a root split or a suffix split
- a bucketed view of search phase, frontier density, duplicate rate, and frontier spread

At each decision point, those logits are softmaxed into a sampling distribution.

### Neural policy

`PrefixDeepRL.NeuralSplitPolicy` implements a **frontier-conditioned DAG-aware pointer actor-critic**. It mixes four learned views of the current decision:

- an **action encoder** that embeds each legal split from the 22-feature action vector
- a **recursive tree encoder** that builds embeddings for every existing prefix subtree from learned leaf and internal-node composition layers
- a **root attention block** that lets the decision-state features attend over the current output-tree embeddings
- a **shared-DAG summary** that pools reusable internal nodes by how often they appear across all constructed outputs

The actor applies a **single-head self-attention pass across legal actions**, producing contextualized candidate embeddings before scoring them. The policy is **frontier-conditioned**: before every sampled episode it receives a compact summary of search progress, duplicate rate, current Pareto-frontier shape, and the most recent reward / cache-hit status.

Each candidate split is scored by a combination of:

- a learned actor projection over the contextualized candidate embedding
- a pointer-style query–candidate interaction term

The query / critic path combines:

- decision-state features
- the attention-pooled root-topology summary
- the weighted shared-DAG summary
- an average of the contextualized candidate embeddings

Training uses a **state-dependent critic head** as a learned residual baseline on top of the moving episode baseline from the shared search loop. The critic sees both tree-structured topology context and action-set context.

## Reward, frontier, and duplicate handling

The shared search backend does more than just run policy gradient:

- it **warm-starts** the search with deterministic `ripple` and `balanced` topologies
- it maintains a **running objective normalizer** for power, delay, and area
- it rewards non-dominated points with a frontier bonus
- it rewards spread away from the current frontier
- it applies an explicit penalty to duplicate topologies
- it maintains a Pareto frontier of non-dominated PPA tuples in `PrefixRLCore.ParetoFrontier`
- it snapshots a **policy-facing search state** before each sampled episode so the policy can condition on frontier shape, duplicate pressure, and search progress

That means repeated topology samples still get logged, but they do not burn another expensive evaluation, and the policy can react to how the search frontier is evolving over time.

## Direct Mill entrypoints

The Makefile is the intended front door, but the underlying mains are available directly:

```bash
bash ./mill TopLevelModule.runMain Elaborate --help
bash ./mill PrefixRLCore.runMain PrefixRLCore.EvaluateTopologyMain --help
bash ./mill PrefixTabularRL.runMain PrefixTabularRL.PrefixSearchMain --help
bash ./mill PrefixDeepRL.runMain PrefixDeepRL.PrefixSearchMain --help
```

Representative examples:

```bash
bash ./mill PrefixRLCore.runMain PrefixRLCore.EvaluateTopologyMain \
  --width 8 \
  --topology example_topologies/dependent_balanced_8.json \
  --backend synthetic

bash ./mill PrefixTabularRL.runMain PrefixTabularRL.PrefixSearchMain \
  --width 4 \
  --episodes 8 \
  --backend synthetic
```

Use:

```bash
make elaborate-help
```

if you want the full `ChiselStage` option list exposed by the wrapper.

## Development commands

```bash
make reformat
make check-format
make test
make smoke-evaluate
make smoke-search-tabular
make smoke-search-deep
make check
make clean
make distclean
```

Notes:

- `make clean` removes generated Verilog, evaluation results, and search outputs under `generated/`
- `make distclean` also removes Mill build products under `out/`
- `make check` runs formatting checks, unit tests, and short synthetic end-to-end smoke tests
- the Makefile and helper scripts intentionally invoke the launcher as `bash ./mill`, so the repo still works when an archive drops the executable bit on the wrapper

## Tips

- Start with `BACKEND=synthetic` and small widths such as `4` to `8` while validating the search loop.
- Switch to `BACKEND=librelane` only after the synthetic flow and topology JSON handling look healthy.
- Use the hand-authored topologies in `example_topologies/` to verify evaluation before launching RL.
- If you want a quick visual sanity check of a generated topology, render `topology.dot` with Graphviz.

## License

Apache License 2.0. See [LICENSE](LICENSE).
