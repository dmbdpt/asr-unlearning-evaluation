# Scripts

Everything here sits on top of `run_experiment.py`. The order below is the order
you run things in to go from a trained model to the numbers in the paper.

Paths come from the environment (see the repository README). The defaults work
from a fresh clone.

## 1. Run unlearning

| Script | What it does |
| --- | --- |
| `run_unlearning.sh <method>` | Unlearning + evaluation for one method. Reads the tuned hyperparameters from `config/unlearning/<method>.yaml`, so no overrides are needed to reproduce a published run. `MODE=simultaneous\|sequential` switches from per-subject to a multi-speaker forget set. |
| `run_hpsearch.sh <method>` | Optuna search over that method's space in `hyperparameter_search.py`. Writes an Optuna study db to `hyperparam_results/`. |
| `run.sh` | Lower-level launcher taking `--method/--gpus/--nproc` and arbitrary Hydra overrides. Use it for one-off configurations. |
| `hyperparameter_search.py` | The search itself. Optimises Earth Mover's Distance between the forget-set and test-set loss distributions. Driven by `run_hpsearch.sh`. |

`run_unlearning.sh` is configured by environment variable:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODE` | `per_subject` | `per_subject`, `simultaneous` or `sequential` |
| `N_SUBJECTS` | 10 | subjects to process in `per_subject` mode |
| `NUM_FORGET` | 10 | forget-set size in the multi-speaker modes |
| `GPUS` | `0,1,2` | comma-separated device ids |
| `NPROC` | one per GPU | torchrun processes |
| `PORT` | random | torchrun master port |
| `MIA` | `false` | run the inline classifier MIA |
| `SKIP_PRE` | `true` | skip the one pre-unlearning evaluation pass, run once before any subject is processed |
| `SKIP_POST` | `true` | passed through for parity with the pre-cleanup per-method scripts, but has no effect: nothing in the pipeline reads `cfg.skip_post`, and the per-subject post-unlearning pass (`stage_06_post_evaluation.run_post_unlearning_pass`) always runs. This predates the cleanup -- every original `run_<method>.sh` on `allofit` passed the same `++skip_post=true` override. |
| `EXPERIMENT_NAME` | derived | MLflow experiment name |

Neither `run_unlearning.sh` run computes WER: `config/evaluation/default.yaml` ships `metrics: false` and
`losses: false`, so the live run only logs training losses and cluster bookkeeping to MLflow. WER and
per-utterance loss come from the separate re-evaluation pass in section 5 below
(`eval_from_run.py`, driven by `collect_eval_runs.py`'s discovery of completed runs).

`run_hpsearch.sh` takes `N_TRIALS` (20), `STUDY_NAME`
(`<method>_emd_optimization`), `OUTPUT_DIR` (`hyperparam_results`),
`FORGET_SPEAKERS` (163), `MODEL_PATH` (train from scratch first), plus `GPUS`,
`NPROC` and `PORT` as above.

`run.sh` takes flags instead: `-c/--config`, `-m/--method`, `-g/--gpus`,
`-n/--nproc`, `-p/--port`, `-r/--results`, `-h/--help`, then any Hydra overrides.

## 2. Assemble checkpoint trees

Every attack and analysis script reads the same layout:

```
<checkpoints-root>/<method>/<forget_subject>/last.ckpt
```

`--checkpoints-root` defaults to `~/unlearned_ckps`.

| Script | What it does |
| --- | --- |
| `collect_eval_runs.py` | Walks completed run folders and emits every (checkpoint, forgotten subject) pair that needs re-evaluating. MLflow's `tags`/`params` tables are the source of truth; `artifact_uri` is never trusted, because runs were produced under a temporary directory and rsynced into place. Writes `eval_runs.json` plus logs of remapped and skipped runs. |
| `organize_unlearned_ckpts.py` | Turns that list into the tree above, one checkpoint per (method, subject), most recently written wins. Hardlinks by default; `--copy` or `--symlink` to change that. Writes a `manifest.csv` recording each checkpoint's originating run, since the checkpoint files carry no provenance. |

## 3. Membership inference attacks

All of these walk a checkpoint tree and write per-(method, subject) summaries.
They differ in what the attacker is allowed to know.

| Script | Threat model |
| --- | --- |
| `run_mia.py` | The *simple* attack: a random forest over the encoder's CTC loss and the decoder's cross-entropy loss, trained on the target model's own retain/test split. Evaluation utterances are duration-matched to the forgotten speaker so a duration shortcut cannot drive the attack. The MIA column of Table 2. |
| `run_informed_mia.py` | The *informed* attack: for each forget subject, the other 9 subjects' unlearned models supply the classifier's training losses, so the attacker has seen how a model behaves on a speaker it was told to forget. Strictly stronger; the IMIA column of Table 2. |
| `merge_pre_post_mia.py` | Joins the `pre_`/`post_`/`retrain_` baselines onto one row per (method, subject, split), so each method sits between its upper bound and its floor. Writes new files rather than editing summaries in place, since the attacks regenerate those from scratch. |
| `mia_common.py` | The shared library: duration matching, pool construction, attack execution, checkpoint loading, summary writing. Not run directly. |

`run_mia.py` covers every forget-set shape in the paper through four flags:

| Setting | Flags |
| --- | --- |
| Single-subject, one checkpoint per subject | `--checkpoints-root ~/unlearned_ckps` |
| Multi-speaker checkpoint, whole group as the forget set | `--checkpoint <ckpt> --label <name> --forget-speakers 103 1034 ...` |
| Multi-speaker checkpoint, one speaker at a time | as above, plus `--subject 103` |
| Drop the run's other forgotten speakers from the retain pool | `--exclude-forget-set` |
| Attack the pretrained model through the same pools | `--stages pre post` |

The last one matters: without `--exclude-forget-set`, every speaker that is not
currently under attack counts as retained training data, which for a
multi-speaker checkpoint silently readmits the other forgotten speakers.

## 4. Forgetting quality and figures

| Script | What it does |
| --- | --- |
| `compute_emd_vs_retrained.py` | EMD between each unlearned checkpoint's forget-set loss distribution and the retrained gold standard's, for the same speaker. The headline "how close to retraining" number. |
| `compute_emd_internal.py` | Forget-vs-test and forget-vs-retain EMD across all pre/post pairings, using the model as its own reference. |
| `plot_decision_boundary.py` | Renders the attacker's P(member) contour over the 2D (loss_att, loss_ctc) plane. `--informed` uses the informed attacker instead. |
| `plot.py` | Plotting front-end with four subcommands: `kde`, `lvd` (loss vs duration), `dist` (pairwise speaker distances) and `utt-loss`. Shells out to the three scripts below. |
| `loss_vs_duration_plot.py` | Loss against utterance duration for one checkpoint. Motivates the duration matching the attacks perform. |
| `all_distances_updated.py` | Pairwise acoustic and textual speaker distances, using the same logic as the clustering pipeline. |
| `utt_loss_calc.py` | Per-utterance loss for one MLflow run. |

## 5. Re-evaluation on SLURM

| Script | What it does |
| --- | --- |
| `eval_from_run.py` | Re-evaluates one (checkpoint, forgotten subject) pair. The unit of work for the arrays below. |
| `submit_eval_array.sh` | Array over everything `collect_eval_runs.py` found. |
| `submit_simple_mia_array.sh` | Array over `run_mia.py`. |
| `submit_informed_mia_array.sh` | Array over `run_informed_mia.py`. |
| `reeval_to_csv.py` | Aggregates the resulting per-subject MLflow dbs into one CSV. |

All the array scripts share four environment variables:

| Variable | Effect |
| --- | --- |
| `DRY_RUN=1` | print the plan, submit nothing |
| `FORCE=1` | rerun targets that already have a `results.json` |
| `METHODS="cfk neggrad"` | restrict to these methods |
| `SUBJECTS="103 118"` | restrict to these subjects |

One array task per method. Internally each script re-invokes itself with
`TASK_MODE=1` and `SLURM_ARRAY_TASK_ID=<i>` to run method `i` of the filtered
list.
