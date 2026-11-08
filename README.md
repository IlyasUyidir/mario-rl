# Learning to Play Super Mario Bros with CNN-Based Reinforcement Learning
# Apprentissage par renforcement avec CNN pour jouer à Super Mario Bros

School project, Deep Learning (5IIR-DDSI). A PPO agent learns World 1-1 of Super Mario Bros by
trial and error. The algorithm and its settings stay fixed; only what the network sees changes.

## Variants

| name | input to the network | size |
|---|---|---|
| pix | 4 stacked 84x84 grayscale frames (CNN policy) | 28,224 |
| grid | 13x11 tile map read from the game RAM, 4 stacked | 572 |
| kin | grid + exact position inside the tile + x and y speed | 576 |

Shared design: each action is held for 4 frames (frame skip). Reward per step = pixels moved right
- 0.1, -15 on death, +50 at the flag. Episodes are cut after 250 steps without progress. No
rendering during training.

## Setup

    python3.13 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Reproduce

    python smoke_test.py                 # the game runs
    python baselines.py                  # random and scripted baselines
    python train_mlp.py 0 1000000 kin    # seed 0, 1M steps, variant grid | kin | pix
    python eval_all.py                   # sampled evaluation of all saved models
    python report.py                     # tables and curves.png from runs/

## Results (1M training steps)

Baselines (furthest x reached): random 570, scripted run-right-and-jump 695. The flag is near x = 3161.

| variant | seeds | final max x | finish rate (last 10%) | wall-clock |
|---|---|---|---|---|
| pix | 1 | 730 | 0% | 3204 s |
| grid | 3 | 1392 +/- 297 | 1% +/- 1 | 1513 s |
| kin | 3 | 1747 +/- 233 | 10% +/- 7 | 1519 s |

"final max x" is the mean furthest x over the last 10% of training episodes (the network is still changing during that window).
results/eval_1000k.csv holds a separate 50-episode evaluation of every final model. See curves.png.

## Limitations

- 3 seeds per variant (1 for pix): the spreads are rough, not significance tests.
- PPO settings are shared across variants and were not tuned for pixels.
- One level (1-1) only, and all curves are still rising at 1M steps.

## Files

- mario_env.py, mario_grid_env.py, mario_kin_env.py: the three environments (pix, grid, kin)
- train_mlp.py, eval_all.py, eval_mlp.py: training and evaluation
- summarize.py, curve.py, where_die.py, report.py: analysis of the runs/ logs
- test_env.py, dump_grid.py, check_kin.py, bench_device.py: sanity checks and benchmarks

## Evaluation of the final models

50 sampled episodes per model (CPU, batch 1). Median max x is the mean +/- sd across seeds of each model's median.

| variant | seeds | episodes reaching the flag | median max x | latency per decision |
|---|---|---|---|---|
| pix | 1 | 0/50 | 714 | 0.44 ms |
| grid | 3 | 3/150 | 1332 +/- 178 | 0.11 ms |
| kin | 3 | 19/150 | 1728 +/- 281 | 0.11 ms |

Flags per seed in 50 episodes: grid 0 / 3 / 0, kin 3 / 5 / 11. The ranges overlap, so with 3 seeds
"kin finishes more often" is a trend, not a proven difference.
