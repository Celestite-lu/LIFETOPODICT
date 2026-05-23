# LifeTopoDict Autoresearch Program

This program defines an autonomous research loop for `/home/lyw/LifeTopoDict`.

The agent's mission is to keep proposing, implementing, running, and judging experiments until LifeTopoDict exceeds same-feature raw HC-SOINN accuracy while preserving a meaningful compact-memory advantage.

## Objective

Primary goal:

```text
Make LifeTopoDict exceed same-feature raw HC-SOINN on CUB-200.
```

Current same-feature raw HC-SOINN reference from `docs/exps_results_report/LifeTopoDict_MVP_research_report.md`:

| Metric | Raw HC-SOINN reference |
| --- | ---: |
| CUB A_Avg | 88.4417 +/- 0.4137 |
| CUB A_Last | 82.9100 +/- 0.0000 |
| Old-New HM | 84.5108 |
| Compact memory | 7.6598 MB |
| Actual memory | 22.7962 MB |

Current compact LifeTopoDict selected reference:

| Metric | LifeTopoDict selected |
| --- | ---: |
| CUB A_Avg | 88.4323 +/- 0.4642 |
| CUB A_Last | 82.5300 +/- 0.1044 |
| Old-New HM | 84.0481 |
| Compact memory | 3.1827 MB |
| Actual memory | 17.7632 MB |

Minimum publishable success:

```text
CUB 3-seed mean A_Last >= raw HC-SOINN A_Last + 0.10pp
A_Avg >= raw HC-SOINN A_Avg - 0.10pp
Old-New HM >= raw HC-SOINN HM, or a clear A_Last gain explains any HM tradeoff
compact deployable memory reduction vs raw >= 30%
actual implementation memory <= raw HC-SOINN
```

Strong success:

```text
CUB 3-seed mean A_Last >= raw HC-SOINN A_Last + 0.30pp
A_Avg >= raw HC-SOINN A_Avg
Old-New HM >= raw HC-SOINN HM + 0.20pp
compact deployable memory reduction vs raw >= 40%
method beats random/static/same-memory controls
```

After CUB success, confirm on ImageNet-R. CIFAR-100 is useful for fast debugging and sanity checks, but CUB is the core decision platform.

## Non-Negotiable Experimental Boundaries

The validated core must remain the base of every serious experiment:

```text
compressed and shared HC-SOINN topology nodes
```

This means LifeTopoDict should keep using HC-SOINN-style class-local topology nodes as the classifier substrate, compressed through shared dictionary atoms / sparse coefficients. New methods may calibrate, gate, repair, reweight, regularize, or augment this substrate, but must not replace it with an unrelated classifier head as the main method. If a radical architecture is explored, it must be framed as an auxiliary module around the compressed shared-node topology and compared against the unchanged LifeTopoDict base.

Use cached frozen ViT-B/16 features and classifier-only mode unless the experiment explicitly targets feature extraction infrastructure:

```text
use_feature_cache=true
feature_cache_strict=true
feature_cache_skip_backbone=true
feature_cache_classifier_device=cpu
```

Do not attribute backbone/cache engineering gains to LifeTopoDict. Raw HC-SOINN and LifeTopoDict must share the same cached-feature setting.

Keep these mechanisms disabled by default:

```text
use_dictionary_growth=false
dict_max_growth_per_task=0
use_edge_aware_scoring=false
```

Residual dictionary growth and additive edge-aware scoring were already diagnosed as unsupported or harmful. Do not reintroduce them as routine search dimensions. They may only be used in explicitly labeled negative/diagnostic experiments.

Never use test labels to train a gate, tune a threshold, select a fallback rate, choose hyperparameters, or decide whether to keep a method. Test labels are only for final evaluation and post-hoc diagnosis.

All additional state must be counted in the memory ledger:

```text
raw fallback cache
atom-class gates
pair/node tables
calibration models
extra centroids
optimizer-learned parameters used at inference
```

## Required Context To Read Before Starting

Read these first:

```text
docs/exps_results_report/LifeTopoDict_MVP_research_report.md
docs/exps_results_report/direction1_raw_fallback_report.md
docs/exps_results_report/direction1_gbdt_gate_experiment_report.md
docs/exps_results_report/direction1_utility_gate_report.md
docs/Innovation_design/LifeTopoDict新方向详细方案.md
docs/Innovation_design/LifeTopoDict_MVP_代码实现报告.md
docs/research_guidance/experiment_guidance.md
docs/paper/CIL_Paper/CIL_paper_index.md
docs/paper/dictionary_learning_Paper/DIC_Paper_index.md
```

Then inspect only the code needed for the current idea. The most common files are:

```text
main.py
models/life_topo_dict.py
utils/hc_soinn_classifier.py
utils/feature_cache.py
scripts/collect_results.py
scripts/analyze_direction1_next_optimization.py
exps/life_topo_dict/*.json
```

When an experiment behaves strangely, first consult:

```text
docs/research_guidance/experiment_guidance.md
```

Do not guess blindly when the guidance already describes the failure mode.

When proposing a new mechanism or architecture, actively read `docs/paper/` before implementation. Start from the two paper index files, then read the most relevant summaries under:

```text
docs/paper/CIL_Paper/Paper_summary/
docs/paper/dictionary_learning_Paper/Paper_summary/
```

Use the papers to answer three questions before coding:

```text
What known idea is this related to?
What is actually new in the LifeTopoDict setting?
Is the mechanism mathematically and logically coherent under compressed shared HC-SOINN nodes?
```

## Current Research State

Known positive results:

- Dictionary coding gives strong memory compression.
- Lifecycle stale suppression is real as a diagnostic/memory-cleanup mechanism.
- Oracle raw fallback has a high upper bound: count-r3 oracle reached A_Last 86.62 on CUB seed 1993 with only about 4% fallback.
- Cached-feature classifier-only path avoids repeated backbone cost and keeps GPU memory out of the classifier comparison.

Known negative or weak results:

- Growth did not improve accuracy and increased memory.
- Additive edge-aware scoring was harmful by default and unhelpful after scale fixes.
- Learned ridge/GBDT/utility raw fallback gates did not produce stable 3-seed deployable gains.
- Direction 1 is diagnostic, not a main claim, unless a future calibration method reverses this.

Currently most promising documented direction:

```text
Direction 2: ATD-aware shared-atom conflict detection + atom-class gate
```

The agent should prioritize this direction initially, but must not be boxed in by the two directions already drafted by the human. If evidence suggests another mechanism is more promising, pursue it. The only hard invariant is the validated base: compressed/shared HC-SOINN topology nodes remain the substrate.

## Research Operating Rules

1. Preserve the core base. Every serious method must build on compressed shared HC-SOINN nodes. Do not turn the project into a generic MLP, FC fusion head, nearest-class-mean variant, replay method, or backbone-tuning method unless it is a clearly labeled diagnostic control.
2. Tune before replacing. If a method has a weak but nonzero signal, first adjust its key parameters across several reasonable settings before abandoning it or inventing a new architecture. Only move on after the failure mode is clear across multiple parameter attempts.
3. Read guidance when problems appear. Accuracy collapse, NaN, memory anomalies, unstable calibration, or strange diagnostics require consulting `docs/research_guidance/experiment_guidance.md` before changing code or parameters.
4. Think beyond the current two directions. Direction 1 and Direction 2 are starting points, not a cage. New architecture ideas are welcome when they keep the compressed shared-node base and come with controls.
5. Paper-ground every new idea. Before coding a new mechanism, inspect `docs/paper/` for related CIL and dictionary-learning ideas. Use papers to check whether the idea is novel enough, whether the logic is rigorous, and whether the math fits the LifeTopoDict setting.
6. Prefer mechanism clarity. A small stable gain with a clean, auditable mechanism is better than a fragile large gain that cannot be explained or controlled.

## Setup

1. Work in:

```bash
cd /home/lyw/LifeTopoDict
```

2. Check repository state. If this directory is a git repository, create an autoresearch branch:

```bash
git status --short
git checkout -b autoresearch/lifetopo-$(date +%m%d-%H%M)
```

If this directory is not a git repository, continue without git and write an explicit patch summary after each experiment. Never delete or overwrite user work.

3. Verify feature cache exists:

```bash
ls /home/lyw/data/FeatureCache/LifeTopoDict
```

If a cache is missing, build it with `scripts/build_feature_cache.py` or inspect existing docs/logs for the correct command. Do not fall back to repeated backbone forward unless the user explicitly requests it.

4. Initialize the autonomous run directory:

```bash
mkdir -p logs/autoresearch
```

5. Initialize `logs/autoresearch/results.tsv` if absent:

```text
timestamp	tag	seed	dataset	method	A_Avg	A_Last	selected_rel_A_Last	raw_rel_A_Last	old_acc	new_acc	HM	compact_MB	actual_MB	status	description	log_file
```

Do not commit `results.tsv` unless the user explicitly asks for committed logs.

## Git Automation And Rollback

Git is the safety rail for autonomous research. The agent should use it automatically, but conservatively.

Authentication:

- Do not store GitHub tokens in remote URLs, scripts, config files, logs, or reports.
- Use the configured SSH remote for this repository:

```bash
git remote set-url origin git@github-lifetopo:Celestite-lu/LIFETOPODICT.git
git remote -v
```

- Expected remote:

```text
origin  git@github-lifetopo:Celestite-lu/LIFETOPODICT.git (fetch)
origin  git@github-lifetopo:Celestite-lu/LIFETOPODICT.git (push)
```

- Verify SSH before a long autonomous run:

```bash
ssh -T github-lifetopo
```

- If HTTPS is used elsewhere, rely on an OS credential manager or GitHub CLI login outside this program. Never paste a token into a command.
- If push authentication is unavailable, keep local commits and continue. Local rollback is still valid.

Stable anchors:

```bash
git tag -a baseline-before-autoresearch -m "Baseline snapshot before autonomous LifeTopoDict experiments"
git checkout -b autoresearch/lifetopo-<date>
```

If the tag or branch already exists, reuse it. Do not recreate or overwrite tags unless the user explicitly asks.

Per-experiment workflow:

```bash
git status --short
EXP_START=$(git rev-parse HEAD)
EXP_BRANCH=$(git rev-parse --abbrev-ref HEAD)
```

The working tree should be clean except for ignored logs/results. If tracked files are dirty, inspect them before starting; do not overwrite unknown user edits.

After implementing an idea:

```bash
git add <changed code/config/docs>
git commit -m "exp(<tag>): <short hypothesis>"
EXP_COMMIT=$(git rev-parse HEAD)
```

Then run the experiment. When judging the result:

```text
success or useful infrastructure -> keep commit, optionally add a follow-up report commit, then push
failure with no reusable value -> git reset --hard "$EXP_START", then do not push the failed commit
crash from typo -> fix and amend/rerun once; if still failed, reset to EXP_START
```

Rollback rules:

- Only reset back to the `EXP_START` captured at the beginning of the current experiment.
- Never reset across user-created commits or edits.
- Never run `git clean -fdx`; it can delete data, logs, and caches.
- If unsure whether a file is user-created, leave it alone and record the uncertainty.
- Ignored experiment logs under `logs/` may remain after rollback; that is fine.

Useful commands:

```bash
git log --oneline --decorate -10
git diff --stat "$EXP_START"..HEAD
git reset --hard "$EXP_START"
git checkout baseline-before-autoresearch
git checkout autoresearch/lifetopo-<date>
```

Push policy:

```bash
git push origin "$EXP_BRANCH"
git push origin main --tags
```

Push only kept commits. Do not push failed experiment commits unless they are explicitly kept as diagnostic infrastructure. If push fails, do not retry with embedded credentials; keep the local commit and continue.

Canonical autonomous experiment shell pattern:

```bash
EXP_START=$(git rev-parse HEAD)
EXP_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# edit code/config/docs
git add <changed code/config/docs>
git commit -m "exp(<tag>): <short hypothesis>"

# run smoke + experiment + collect results

if <success_or_useful_diagnostic>; then
  git push origin "$EXP_BRANCH"
else
  git reset --hard "$EXP_START"
fi
```

## Baselines To Keep At Hand

Use these as same-feature comparators:

```text
raw_hc_soinn_cub_3seed:
  A_Avg = 88.4417
  A_Last = 82.9100
  HM = 84.5108
  compact_MB = 7.6598
  actual_MB = 22.7962

ltd_selected_cub_3seed:
  A_Avg = 88.4323
  A_Last = 82.5300
  HM = 84.0481
  compact_MB = 3.1827
  actual_MB = 17.7632
```

For single-seed experiments, always compare against a same-run or same-seed compact baseline if available. Do not mix a seed-1993 exploratory number with a 3-seed mean when deciding whether a method works.

## Experiment Loop

LOOP FOREVER until interrupted by the user.

1. Read `logs/autoresearch/results.tsv` and the latest relevant reports.
2. Capture the rollback point:

```bash
git status --short
EXP_START=$(git rev-parse HEAD)
EXP_BRANCH=$(git rev-parse --abbrev-ref HEAD)
```

3. Pick one idea with a concrete hypothesis and a kill criterion. If this is a new mechanism, first read relevant summaries in `docs/paper/` and write a short rationale in the run report before coding.
4. Implement the smallest code/config change needed to test it.
5. Commit the experimental code/config before running:

```bash
git add <changed code/config/docs>
git commit -m "exp(<tag>): <short hypothesis>"
```

If there are no tracked code/config changes because the run only uses CLI overrides, record the hypothesis in `logs/autoresearch/results.tsv` and continue without a commit.

6. Run a syntax check:

```bash
python -m py_compile main.py models/life_topo_dict.py utils/hc_soinn_classifier.py utils/feature_cache.py scripts/collect_results.py
```

7. Run a quick smoke test when code changed:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python main.py \
  --config exps/life_topo_dict/life_topo_dict_cub_selected_trace.json \
  --device 0 \
  --seed 1993 \
  --max_tasks 2 \
  > logs/autoresearch/smoke_<tag>.log 2>&1
```

8. Run the smallest meaningful experiment:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
python main.py \
  --config <config> \
  --device 0 \
  --seed 1993 \
  > logs/autoresearch/<tag>_cub_seed1993.log 2>&1
```

9. Collect results:

```bash
python scripts/collect_results.py logs/life_topo_dict/cub/0/10 \
  -o logs/autoresearch/<tag>_collect.csv \
  --recursive
```

10. Judge the result.

Single-seed keep threshold:

```text
selected-relative A_Last >= +0.20pp
A_Avg drop vs selected <= 0.10pp
test/calibration diagnostics do not indicate label leakage or severe overfit
memory reduction vs raw remains >= 30%
```

If the idea targets HM rather than A_Last:

```text
Old-New HM improves by >= +0.20pp
A_Last drop <= 0.10pp
random/high-usage controls do not explain the gain
```

11. If the result is mediocre but not clearly dead, tune parameters before switching ideas.

Use this rule:

```text
strong positive: promote to 3 seeds
weak positive: run at least 3-5 parameter variants around the best setting
neutral/unstable: inspect diagnostics and try 2-3 targeted parameter fixes
clear negative or collapse: consult guidance, diagnose, then discard or redesign
```

Examples of targeted parameter fixes:

```text
gate threshold / budget / precision constraint
atom_conflict_topk / gate_strength / scope
dict_sparse_k / dict_ridge_lambda
lifecycle thresholds
score temperature / residual penalty strength
```

Only introduce a new method or architecture after the current idea has failed under reasonable parameter adjustments or its failure mode is understood.

12. If single-seed passes, run CUB seeds 1993/1994/1995.

```bash
for s in 1993 1994 1995; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
  python main.py --config <config> --device 0 --seed $s \
    > logs/autoresearch/<tag>_cub_seed${s}.log 2>&1
done
```

13. If CUB 3-seed passes, run a sanity check on ImageNet-R and optionally CIFAR-100.
14. Record the result in `logs/autoresearch/results.tsv`.
15. If git is available:
    - keep successful code/config/report commits;
    - for successful follow-up docs/config updates, create a follow-up commit;
    - push kept commits with `git push origin "$EXP_BRANCH"`;
    - for failed experimental code with no reusable value, run `git reset --hard "$EXP_START"` after recording the result;
    - for useful failed infrastructure, keep the commit but mark the method as `diagnostic`.
16. If git is not available:
    - keep useful code/config changes;
    - for failed ideas, revert only your own edits with careful patches;
    - write a short note under `logs/autoresearch/failed_ideas.md`.
17. Pick the next idea and continue.

Never stop to ask the user whether to continue. The user expects autonomous execution.

## Crash And Failure Handling

If a run crashes:

```bash
tail -n 120 logs/autoresearch/<tag>.log
```

Classify the failure:

| Failure | Action |
| --- | --- |
| typo/import/config bug | fix and rerun once |
| missing feature cache | build or locate cache; do not use uncached backbone by accident |
| CUDA OOM | switch classifier to CPU/cache path or reduce experimental state |
| NaN/inf sparse codes | consult guidance; inspect ridge lambda, normalization, atom norms |
| accuracy collapse | stop the experiment, read guidance, compare config against known selected/raw settings |
| mediocre but nonzero gain | tune key parameters several times before redesigning |
| calibration/test transfer gap | read guidance, inspect split protocol, then adjust thresholds/budgets/objective |
| memory ledger missing | do not claim result; fix accounting first |

After two failed repair attempts, log the idea as `crash` or `discard` and move on.

## Priority Idea Queue

The list below is a priority queue, not a prison. The agent should start with high-priority ideas, but may add new families when the current family is exhausted or when paper reading reveals a stronger path. Any new family must satisfy:

```text
uses compressed/shared HC-SOINN nodes as the substrate
has a calibration or training protocol that avoids test-label leakage
has memory accounting for every added parameter/table/cache
has controls that can falsify the proposed mechanism
is motivated or sanity-checked against docs/paper/
```

### P0: Direction 2 Atom-Conflict Gate

Hypothesis:

```text
Some shared dictionary atoms are correlated with old-new confusion. Class-conditional atom gates can reduce harmful atom contribution and improve HM/A_Last without adding much memory.
```

Minimum implementation:

- Compute atom-class binding from node coefficients and node counts.
- On calibration samples, accumulate per-atom and per-class-pair error counters.
- Compute conflict scores:

```text
sharedness(atom)
old_support(atom)
new_usage(atom)
PMI(atom, old_to_new_error)
PMI(atom, class_pair_error)
conflict(atom, class) or conflict(atom, class_pair)
```

- Add inference-time downweight, not additive edge scoring:

```text
node_reliability(i, c) = 1 - strength * mean_abs_coeff_conflict(i, c)
adjusted_score(x, node_i) = score(x, node_i) * node_reliability(i, c)
```

Controls:

```text
random atom downweight
high-usage-only downweight
old-support-only downweight
trace-only no-op
```

Search:

```text
atom_conflict_topk in {5, 10, 20, 40}
atom_gate_strength in {0.05, 0.10, 0.20, 0.35, 0.50}
scope in {global, class_conditional, class_pair}
metric in {old_new_pmi, class_pair_pmi, conflict_score}
```

Kill criterion:

```text
No seed-1993 variant improves HM by >= +0.20pp while keeping A_Last drop <= 0.10pp.
```

Promote criterion:

```text
CUB 3-seed mean A_Last >= raw + 0.10pp
or CUB 3-seed mean HM >= raw + 0.20pp with A_Last >= raw - 0.10pp
```

### P1: Dictionary Reconstruction And Scoring Improvements

Hypothesis:

```text
Accuracy loss comes from reconstruction/scoring mismatch rather than insufficient memory. Better normalization, whitening, coefficient calibration, or residual-aware scoring can recover accuracy without raw fallback.
```

Ideas:

- Verify atom L2 norms and coefficient scales per task.
- Try normalized atom storage and normalized reconstruction consistently.
- Try PCA/whitening before dictionary encoding, then unwhiten for scoring.
- Try ridge lambda/k joint schedules:

```text
(k, lambda) in {(8,0.01), (12,0.01), (12,0.03), (15,0.03), (15,0.05)}
```

- Try residual-aware node score calibration:

```text
score' = score - beta * node_residual
beta selected on calibration only
```

- Try class-age temperature or margin calibration selected on calibration.

Controls:

```text
no calibration
random residual penalty
same memory raw HC-SOINN reference
```

Do not re-enable dictionary growth as the first response to high residual.

### P2: Safer Raw Repair After Direction 1

Direction 1 failed as a main claim, but oracle remains high. Only revisit it with a materially different calibration method.

Allowed ideas:

- Out-of-fold calibration to enlarge calibration positives.
- Compact top-K constrained raw challenge with stored compact top5/top10.
- Class-pair-only fallback where calibration has positive lower confidence bound.
- Raw cache pruning only if it reduces harm and memory together.

Do not simply increase fallback rate or raw cache size. That already failed the conceptual test.

### P3: Optimizer / Offline Dictionary Fitting

The current classifier path is mostly non-gradient HC-SOINN + closed-form sparse coding. Optimizer experiments are allowed only when they create a real trainable component.

Allowed ideas:

- Learn a small calibration layer over compact logits on held-out calibration.
- Learn atom gates with L1/entropy regularization using calibration only.
- Learn class-wise score temperatures.
- Fit dictionary atoms with a constrained objective on base/train features, then freeze them.

Required controls:

```text
calibration-only no test tuning
parameter memory counted
raw HC-SOINN same-feature baseline
selected compact baseline
randomized labels or random gates when relevant
```

### P4: Broader Hyperparameter Search

Only run broad search after a mechanism shows a single-seed signal.

Search candidates:

```text
dict_sparse_k: {8, 12, 15}
dict_ridge_lambda: {0.01, 0.03, 0.05, 0.10}
lifecycle_theta_support: {0.5, 0.7, 0.9}
lifecycle_node_inactive_threshold: {0.0, 0.1, 0.2, 0.4}
lifecycle_theta_usage: {0.01, 0.03, 0.05}
```

Avoid wasting runs on already closed dimensions:

```text
dict_max_growth_per_task
use_edge_aware_scoring
edge_score_gamma
edge_score_eta
```

### P5: Paper-Driven New Architectures Around The Same Base

This bucket is for creative mechanisms beyond Direction 1 and Direction 2. It is encouraged when the agent has read relevant papers and can explain why the mechanism should help compressed shared HC-SOINN nodes.

Allowed architecture families:

- ATD/task-confusion-aware calibration inspired by CIL papers.
- Prototype debiasing or old/new balance calibration that uses LifeTopoDict node scores.
- Dictionary-learning improvements inspired by sparse autoencoder / archetypal / coherence papers.
- Atom specialization, atom routing, or class-conditional atom masking.
- Compact residual adapters that correct reconstructed node centers without storing full raw nodes.
- Memory-budgeted expert routing where the expert is still a compact topology component.
- OOD/uncertainty-inspired confidence calibration if it can be computed from compact node geometry.

Disallowed as main methods:

- Replacing LifeTopoDict with a standalone neural classifier.
- Fine-tuning the backbone.
- Adding FC fusion that bypasses topology unless it is a diagnostic control.
- Increasing raw fallback cache as the main solution.
- Any method that beats raw HC-SOINN only by using more deployable memory than raw.

Before implementing a P5 idea, the run report must include:

```text
paper/source inspiration
mechanism sketch
mathematical scoring or optimization definition
why it preserves compressed shared HC-SOINN nodes
expected failure mode
controls
```

## Result Interpretation Rules

Do not overclaim:

- Single seed success is only a lead.
- Offline trace success is only a lead.
- Oracle success is an upper bound, not a deployable method.
- A method that beats selected but not raw is not enough for the current mission.
- A method that beats raw but loses the compact-memory advantage is not LifeTopoDict success.

Prefer simpler mechanisms:

- A rule or small table with +0.2pp stable gain is better than a complex model with +0.25pp and high variance.
- A no-extra-memory calibration that matches raw is valuable.
- If two methods tie, keep the one with less memory and fewer moving parts.

Always report both:

```text
selected-relative gain
raw-HC-SOINN-relative gain
```

## Logging Format

Append one row per completed run to `logs/autoresearch/results.tsv`:

```text
timestamp	tag	seed	dataset	method	A_Avg	A_Last	selected_rel_A_Last	raw_rel_A_Last	old_acc	new_acc	HM	compact_MB	actual_MB	status	description	log_file
```

Status values:

```text
keep       useful and should remain in the code/config path
discard    ran successfully but did not improve
crash      failed to run after reasonable repair
diagnostic useful analysis but not a deployable method
```

Each promoted idea must also get a short report:

```text
logs/autoresearch/<tag>_report.md
```

Report sections:

```text
Hypothesis
Implementation
Commands
Results table
Memory ledger
Controls
Decision
Next idea
```

## Standard Commands

Run selected compact baseline:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
python main.py \
  --config exps/life_topo_dict/life_topo_dict_cub_selected_trace.json \
  --device 0 \
  --seed 1993 \
  > logs/autoresearch/selected_cub_seed1993.log 2>&1
```

Run raw HC-SOINN baseline:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
python main.py \
  --config exps/life_topo_dict/ablation_raw_hc_soinn_cub.json \
  --device 0 \
  --seed 1993 \
  > logs/autoresearch/raw_hc_soinn_cub_seed1993.log 2>&1
```

Collect CUB results:

```bash
python scripts/collect_results.py logs/life_topo_dict/cub/0/10 \
  -o logs/autoresearch/collect_cub.csv \
  --recursive
```

Monitor CPU memory if a mechanism adds substantial state:

```bash
python scripts/run_with_cpu_memory_monitor.py \
  --summary_out logs/autoresearch/<tag>_cpu_memory.json \
  -- \
  python main.py --config <config> --device 0 --seed 1993
```

## GPU Policy

Prefer cached-feature classifier-only CPU mode, which should not materially use GPU. If a command still requires a CUDA device for framework initialization, use an A40 with remaining memory.

Check devices before launching parallel runs:

```bash
nvidia-smi
```

If GPU contention appears, serialize by priority:

```text
CUB > ImageNet-R > CIFAR-100
```

## NEVER STOP

Once this program starts, the agent should continue autonomously:

- do not ask the user whether to continue;
- do not wait for manual result collection;
- do not stop after one failed idea;
- do not keep repeating a failed family without changing the hypothesis;
- if stuck, reread the docs, inspect failure cases, and generate a new mechanism.

The target is not to produce one tidy experiment. The target is sustained research pressure until LifeTopoDict has a credible path past raw HC-SOINN.
