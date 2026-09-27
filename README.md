# Evaluating Machine Unlearning in ASR

Companion code for *Evaluating Machine Unlearning in ASR*. 

## What is here

| Path | Contents |
| --- | --- |
| `run_experiment.py` | Entry point. Runs the full pipeline: data preparation, training, unlearning, evaluation. |
| `config/` | Hydra configuration. `config/unlearning/<method>.yaml` holds each method's tuned hyperparameters. |
| `src/unlearning/` | The unlearning methods and the Lightning harness they share. |
| `src/membership_inference/` | Attack feature extractors, attackers and metrics. |
| `src/evaluation/` | WER/CER metrics, loss evaluation, MIA orchestration, plots. |
| `src/pipeline/` | The numbered pipeline stages `run_experiment.py` calls. |
| `scripts/` | Launchers, attacks, analysis and SLURM arrays. See [`scripts/README.md`](scripts/README.md). |

## Unlearning methods

Selected with `unlearning=<name>`. Each name is a file in `config/unlearning/`
holding that method's tuned hyperparameters.

| Method | Paper | What it does |
| --- | --- | --- |
| `finetune` | Finetune | Gradient descent on the retain set alone, deliberately overfitting it to induce catastrophic forgetting on the forget set. Baseline. |
| `cfk` | CF-k | Finetunes only the last `k` blocks, freezing the rest. `k` counts by network depth, encoder blocks then decoder; the model has 17 E-Branchformer encoder and 6 decoder blocks, so `k=4` trains the last 4 decoder blocks and `k=6` the whole decoder stack. Embeddings, final norms and the CTC/output heads are never unfrozen. Baseline. |
| `neggrad` | NegGrad | Gradient ascent on the forget set. |
| `neggradplus` | NegGrad+ | Gradient ascent on the forget set plus finetuning on the retain set, weighted by `lambda_forget` / `lambda_retain`, to mitigate catastrophic forgetting. |
| `scrub` | SCRUB | Teacher-student with the original model as teacher: a KL distillation loss and a task loss on the retain set, and a negative KL on the forget set. |
| `bad_teacher_smooth` | AttSmooth (ASU) | Attention Smoothing Unlearning. Raises the Softmax temperature in a cloned teacher's attention layers, smoothing its attention and producing less confident outputs on the forget set; the student minimises KL against that smoothed teacher. This adaptation targets the decoder's attention layers. |

All six run through the shared Lightning harness in
`src/unlearning/unlearners/base.py`. Unlearning is standardised to 10 epochs at
batch size 8; the remaining hyperparameters come from the Optuna search in
`scripts/run_hpsearch.sh` and are already written into each method's config.

### Data preparation variants

`data_preparation=default` reads the CSV manifests. `local` reads locally cached
copies of them. `spot` is self-contained for ephemeral machines with nothing
pre-mounted, and downloads LibriSpeech through torchaudio on first run.

## Setup

Requires Python 3.12+ and a CUDA-capable GPU.

```bash
uv sync
```

`uv.lock` pins the exact versions the published experiments ran against.

The pretrained model (`asapp/e_branchformer_librispeech`) downloads from the
Hugging Face Hub the first time it is loaded, so the first run needs network
access; it's then cached locally by `espnet_model_zoo`.

### Data

The code needs LibriSpeech and two CSV manifests listing the train and test
utterances. Point it at them with environment variables:

```bash
export LIBRISPEECH_ROOT=/path/to/LibriSpeech
export LEAF_DATA_DIR=/path/to/manifests   # holds ls_train_all.csv, ls_test_all.csv
```

Both default to `<repo>/data` if unset. The manifests are CSVs with one row per
utterance; `src/data/datasets/csv_dataset.py` defines the columns.

Other variables, all optional: `LEAF_REPO` (repo root), `LEAF_VENV`,
`LEAF_RUNS_ROOT` (where completed runs live), `LEAF_RESULTS_DIR`.

## Reproducing the results

### 1. Unlearn

```bash
bash scripts/run_unlearning.sh scrub
```

The tuned hyperparameters live in `config/unlearning/scrub.yaml`, so this needs
no overrides. Repeat for each method. `MODE=simultaneous` or `MODE=sequential`
forgets a group of speakers in one run instead of one at a time.

Anything else is a Hydra override:

```bash
GPUS=0,1 bash scripts/run_unlearning.sh neggradplus ++limit=3
```

### 2. Collect the checkpoints

The attacks all read one tree, `<root>/<method>/<subject>/last.ckpt`:

```bash
python scripts/collect_eval_runs.py          # find completed runs
python scripts/organize_unlearned_ckpts.py   # build the tree
```

### 3. Attack

```bash
python scripts/run_mia.py      # simple: trained on the target model's own retain/test split
python scripts/run_informed_mia.py    # informed: leave-one-out over the other 9 unlearned models
```

Evaluation utterances are duration-matched to the forgotten speaker throughout,
so a duration shortcut cannot inflate the attack.

### 4. Aggregate

```bash
python scripts/merge_pre_post_mia.py    # one row per method/subject/split
python scripts/reeval_to_csv.py         # re-evaluation metrics
```

On a cluster, `scripts/submit_*_array.sh` run steps 3 and 4 as SLURM arrays.

### Hyperparameter search

```bash
bash scripts/run_hpsearch.sh scrub
```

Optuna, optimising the Earth Mover's Distance between the forget-set and
test-set loss distributions. Results land in `hyperparam_results/`.

## Tracking

Runs are logged to MLflow (`sqlite:///mlflow.db` by default, set
`artifacts.tracking_uri` to change it). Checkpoints, configs and per-subject
result JSONs are stored as run artifacts.
