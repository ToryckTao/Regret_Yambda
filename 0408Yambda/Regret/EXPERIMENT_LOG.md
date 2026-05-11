# Regret UserResponse Experiment Log

Date: 2026-04-20

This file records the Regret pipeline changes and UserResponse experiments under:

```text
/root/autodl-tmp/0408Yambda/Regret
```

The current conclusion is conservative: semantic ID mapping and transition generation are working, but the current UserResponse model is not yet a reliable environment model for A2C. The best v2 reward experiment only reaches the mean baseline on validation, and stronger auxiliary losses make it worse.

## Current Pipeline

Implemented scripts:

```text
scripts/01_prepare_sid_mapping.py
scripts/02_split_transitions.py
scripts/03_train_user_response.py
scripts/04_analyze_event_distribution.py
```

Main artifacts used in these experiments:

```text
artifacts/mappings/raw_rqkmeans/
artifacts/transitions/raw_rqkmeans_smoke/
artifacts/transitions/raw_rqkmeans_v2_smoke_timefix/
artifacts/user_response/
artifacts/diagnostics/
```

The current semantic ID source is `raw_rqkmeans`, converted into dense mappings and item feature mmap:

```text
artifacts/mappings/raw_rqkmeans/orig2dense_item_id.npy
artifacts/mappings/raw_rqkmeans/dense2orig_item_id.npy
artifacts/mappings/raw_rqkmeans/dense_item2sid.npy
artifacts/mappings/raw_rqkmeans/dense_item_features.npy
```

Mapping validation:

```text
n_item = 7,721,749
sid shape = (7,721,750, 4)
feature shape = (7,721,750, 128)
feature filled = 7,721,749
feature skipped = 0
```

## Important Time Unit Fix

Original Yambda `timestamp` is not seconds. One timestamp unit equals 5 seconds.

This affected two previous assumptions:

```text
undo_grace_seconds = 10
```

Previously behaved like 50 seconds if compared directly against raw timestamp.

```text
close_gap_seconds = 3600
```

Previously behaved like 5 hours if compared directly against raw timestamp.

Fix added:

```text
--timestamp_unit_seconds 5
```

Current behavior:

```text
real_elapsed_seconds = (timestamp_b - timestamp_a) * timestamp_unit_seconds
```

Affected files:

```text
regret_core/data/schema.py
scripts/02_split_transitions.py
scripts/04_analyze_event_distribution.py
```

## Event Distribution Findings

Full event count before time-unit correction for gap labels:

```text
users = 10,000
events = 47,790,449

listen     46,467,212   97.23%
like          881,456    1.84%
unlike        312,972    0.65%
dislike       107,776    0.23%
undislike      21,033    0.04%
```

Listen ratio is strongly bimodal:

```text
mean = 0.648
p25  = 0.07
p50  = 1.00
p75  = 1.00
p90  = 1.00
```

In the 1000-user time-fixed diagnostic:

```text
like -> unlike <= 10 real seconds       about 26.2%
dislike -> undislike <= 10 real seconds about 46.9%
```

Interpretation:

1. Low-play is common and should be a real negative signal.
2. Short-time `like -> unlike` and `dislike -> undislike` are often likely undo/misclick behavior.
3. `unlike` is not equivalent to `dislike`; it includes long-term preference changes and like cancellation.

## Reward V1

Original reward formula:

```python
reward_raw =
    1.0 * max_play_ratio
  + 1.0 * I(like)
  - 1.0 * I(dislike)
  - 1.5 * I(unlike)
  + 0.5 * I(undislike)

reward_scaled = clip(reward_raw, -1.0, 2.0)
```

Problem:

```text
low_play is marked as regret_type=low_play, but it is not directly negative.
play=0.1 gives reward=+0.1.
```

Smoke split:

```text
artifact = artifacts/transitions/raw_rqkmeans_smoke
users = 1000
train = 4,125,004
val = 1,000
test = 1,000
negative_share sample = about 0.8%
```

UserResponse results:

```text
artifact = artifacts/user_response/raw_rqkmeans_smoke_sanity
train rows = 200,000, not shuffled, front-loaded users
train_mse = 0.2134
val_mse = 0.3007
val target_mean = 0.6895
val pred_mean = 0.4600
val mean baseline MSE ~= 0.2458
result = worse than mean baseline
```

```text
artifact = artifacts/user_response/raw_rqkmeans_smoke
train rows = 4,125,004
train_mse = 0.2008
val_mse = 0.2884
val target_mean = 0.6895
val pred_mean = 0.6623
val mean baseline MSE ~= 0.2458
result = worse than mean baseline
```

Conclusion:

V1 reward is too positive and too weak for regret learning. Low-play is not penalized enough.

## Reward V2

V2 was added behind:

```text
--reward_version v2
```

Time unit must be passed explicitly:

```text
--timestamp_unit_seconds 5
```

Short undo logic:

```text
like -> unlike <= 10 real seconds:
    cancel like, do not apply unlike penalty

like -> unlike > 10 real seconds:
    unlike is effective

dislike -> undislike <= 10 real seconds:
    cancel dislike, do not apply undislike bonus

dislike -> undislike > 10 real seconds:
    dislike remains effective, undislike gives mild recovery
```

V2 play reward:

```python
play = clip(max_play_ratio, 0.0, 1.0)

if n_listen > 0:
    play_reward = 2.0 * play - 1.0
else:
    play_reward = 0.0
```

Examples:

```text
play=1.0 -> +1.0
play=0.8 -> +0.6
play=0.5 ->  0.0
play=0.2 -> -0.6
play=0.1 -> -0.8
play=0.0 -> -1.0
```

V2 final reward:

```python
reward_raw =
    play_reward
  + 0.8 * effective_like
  - 1.2 * effective_dislike
  - 0.6 * effective_unlike
  + 0.2 * effective_undislike

reward_scaled = clip(reward_raw, -2.0, 2.0)
```

Concrete examples:

```text
full listen:
play=1.0
reward=+1.0

low listen:
play=0.1
reward=-0.8

low listen + dislike:
play=0.1
reward=-0.8 - 1.2 = -2.0 clipped

full listen + like + unlike after 10 real seconds:
reward=+1.0
like/unlike treated as short undo

full listen + like + unlike after 15 real seconds:
reward=+1.0 - 0.6 = +0.4

low listen + dislike + undislike after 10 real seconds:
reward=-0.8
dislike/undislike treated as short undo

low listen + dislike + undislike after 15 real seconds:
reward=-0.8 - 1.2 + 0.2 = -1.8
```

Time-fixed V2 smoke split:

```text
artifact = artifacts/transitions/raw_rqkmeans_v2_smoke_timefix
users = 1000
train = 4,246,531
val = 1,000
test = 1,000

reward mean = 0.3329
reward std = 0.8780
min = -2.0
p50 = 1.0
p90 = 1.0
max = 2.0
negative_share = 34.75%
```

This fixed the reward distribution problem, but not the UserResponse generalization problem.

## Training Data Sampling Fix

Problem:

`max_train_rows=200000` originally read the first 200k rows in stream order. Because shards are user-block ordered, this covered very few users.

Observed:

```text
front 200k rows:
unique_users = 56
```

Added training options:

```text
--train_sample_across_files
--shuffle_train_files
--shuffle_buffer_size
```

`--train_sample_across_files` was later improved to randomly sample rows inside each shard, not just take the beginning of each shard.

Observed after fix:

```text
sampled 200k rows:
unique_users ~= 983
```

Affected files:

```text
regret_core/data/transition_dataset.py
scripts/03_train_user_response.py
```

## UserResponse Experiments

Metric definitions:

```text
mean_baseline_mse = MSE of always predicting the split reward mean
gain = 1 - model_mse / mean_baseline_mse
```

Interpretation:

```text
gain > 0      better than mean baseline
gain ~= 0     basically mean predictor
gain < 0      worse than mean baseline
```

### V2 Sanity Without Shuffle

```text
artifact = artifacts/user_response/raw_rqkmeans_v2_smoke_timefix_sanity
train rows = 200,000
sampling = stream front rows

train_mse = 0.7030
train_gain = 0.071
val_mse = 0.8713
val_mean_base = 0.7036
val_gain = -0.238
val pred_mean = -0.008
val target_mean = 0.404
```

Conclusion:

Invalid sanity setup. It was biased by ordered row sampling.

### V2 Sanity With Shard Sampling

```text
artifact = artifacts/user_response/raw_rqkmeans_v2_smoke_timefix_sanity_shuffle
train rows = 200,000
sampling = per-shard random sample + shuffle buffer

train_mse = 0.7572
train_gain = 0.032
val_mse = 0.6996
val_mean_base = 0.7036
val_gain = 0.006
val pred_mean = 0.354
val target_mean = 0.404
```

Conclusion:

Sampling fix helped. The model barely beats the mean baseline, but the signal is too weak.

### V2 Full Smoke, Pure Regression

```text
artifact = artifacts/user_response/raw_rqkmeans_v2_smoke_timefix
train rows = 4,246,531
epochs = 3
sampling = full train stream + shuffle buffer
```

Results:

```text
epoch 1:
train_mse = 0.6770
train_gain = 0.134
val_mse = 0.7428
val_gain = -0.056
val pred_mean = 0.294
val target_mean = 0.404

epoch 2:
train_mse = 0.6760
train_gain = 0.135
val_mse = 0.7442
val_gain = -0.058

epoch 3:
train_mse = 0.6775
train_gain = 0.133
val_mse = 0.7829
val_gain = -0.113
```

Best checkpoint is epoch 1.

Val/test by type using best checkpoint:

```text
val none:
target_mean =  0.837
pred_mean   =  0.304

val low_play:
target_mean = -0.917
pred_mean   =  0.257

val dislike:
target_mean = -1.200
pred_mean   =  0.315

val unlike:
target_mean = -0.600
pred_mean   =  0.334
```

Conclusion:

Model collapses to a middle positive value and does not distinguish `none` from negative outcomes.

### V2 Sample1M, Pure Regression

```text
artifact = artifacts/user_response/raw_rqkmeans_v2_smoke_timefix_sample1m
train rows = 1,000,000
sampling = per-shard random sample + shuffle buffer
epochs = 3
```

Results:

```text
epoch 1:
train_mse = 0.7426
train_gain = 0.050
val_mse = 0.7023
val_gain = 0.002
val pred_mean = 0.322
val target_mean = 0.404

epoch 2:
train_mse = 0.7386
train_gain = 0.055
val_mse = 0.7033
val_gain = 0.000

epoch 3:
train_mse = 0.7363
train_gain = 0.057
val_mse = 0.7947
val_gain = -0.130
val pred_mean = 0.184
```

Conclusion:

Best result is only at mean baseline. More training overfits or destabilizes. This is the best pure-regression run so far, but still not useful as an environment model.

### V2 Sample1M With Strong Weighted MSE And Regret Auxiliary Head

Added:

```text
regret_type_id in dataset
regret_type auxiliary classification head
reward weighted MSE by regret_type
```

Run:

```text
artifact = artifacts/user_response/raw_rqkmeans_v2_smoke_timefix_aux_sample1m
train rows = 1,000,000
epochs = 3
reward_weight_low_play = 2.0
reward_weight_dislike = 4.0
reward_weight_unlike = 3.0
regret_loss_weight = 0.1
regret_class_weight_low_play = 2.0
regret_class_weight_dislike = 6.0
regret_class_weight_unlike = 4.0
```

Results:

```text
epoch 1:
train_mse = 0.8212
train_gain = -0.051
val_mse = 0.8109
val_gain = -0.153
val pred_mean = 0.074
val target_mean = 0.404
val_rtype_acc = 0.637

epoch 2:
val_mse = 0.8456
val_gain = -0.202
val_rtype_acc = 0.636

epoch 3:
val_mse = 0.9820
val_gain = -0.396
val_rtype_acc = 0.517
```

Classification diagnosis for best checkpoint:

```text
val pred_counts:
none = 719
low_play = 281
dislike = 0
unlike = 0

val true none = 755
all-none baseline accuracy = 75.5%
actual val_rtype_acc = 63.7%
```

By true type:

```text
none:
target_mean =  0.837
pred_mean   =  0.086

low_play:
target_mean = -0.917
pred_mean   =  0.035

dislike:
target_mean = -1.200
pred_mean   =  0.039

unlike:
target_mean = -0.600
pred_mean   =  0.055
```

Conclusion:

Strong weighting and auxiliary classification make the model worse. They pull the reward head toward near-zero predictions and do not learn rare `dislike/unlike` classes.

## Current Diagnosis

What works:

```text
semantic ID mapping works
transition split works
reward v2 produces a much healthier negative reward distribution
time unit handling is now correct
training sampler now supports reasonable per-shard random sampling
```

What does not work yet:

```text
current UserResponse architecture does not generalize beyond mean baseline
pure scalar regression collapses toward average reward
strong regret_type auxiliary training hurts reward prediction
weighted MSE with large negative-class weights is too aggressive
```

Likely causes:

1. Validation/test are one heldout episode per user, while training contains many episodes per heavy user. This creates a user-balanced evaluation vs row-weighted training mismatch.
2. For heldout target items, the same item often has little prior history. `hist_target_*` is weak for most samples.
3. Current history encoder is simple mean pooling plus action attention. It may be too weak for predicting low-play vs normal outcomes for unseen target items.
4. Reward is now reasonable, but the environment model needs either better structure or a different target decomposition.

## Current Recommendation

Do not use the current UserResponse checkpoint for A2C.

Best available checkpoint so far:

```text
artifacts/user_response/raw_rqkmeans_v2_smoke_timefix_sample1m/regret_user_response.pt
```

But it is only a near-baseline model:

```text
best val_gain ~= 0.002
```

Recommended next ablation:

```text
Use mild auxiliary classification, no reward weighted MSE, lower LR.
```

Command:

```bash
cd /root/autodl-tmp/0408Yambda/Regret
export OMP_NUM_THREADS=1

python3 scripts/03_train_user_response.py \
  --transition_root artifacts/transitions/raw_rqkmeans_v2_smoke_timefix \
  --item_features_npy artifacts/mappings/raw_rqkmeans/dense_item_features.npy \
  --out_dir artifacts/user_response/raw_rqkmeans_v2_smoke_timefix_aux_mild_sample1m \
  --max_seq_len 50 \
  --batch_size 256 \
  --max_train_rows 1000000 \
  --max_val_rows 1000 \
  --epochs 3 \
  --lr 3e-4 \
  --train_sample_across_files \
  --shuffle_train_files \
  --shuffle_buffer_size 50000 \
  --regret_loss_weight 0.03 \
  --regret_class_weight_low_play 1.5 \
  --regret_class_weight_dislike 3.0 \
  --regret_class_weight_unlike 2.0
```

Stop condition:

```text
if val_gain <= 0.01:
    stop tuning loss weights
    change UserResponse architecture
```

Architecture directions to consider:

1. Add explicit action-history similarity features, such as max/mean cosine similarity between target item and history item embeddings, and similarity to positive/negative history subsets.
2. Replace mean-pooling history encoder with GRU or Transformer.
3. Predict category-like outcomes first, for example `positive / low_play / explicit_negative`, then convert class probabilities to expected reward.
4. Evaluate with user-balanced train sampling by design, not only row-level shuffling.

## Commands Worth Keeping

Regenerate v2 time-fixed smoke:

```bash
python3 scripts/02_split_transitions.py \
  --orig2dense_npy artifacts/mappings/raw_rqkmeans/orig2dense_item_id.npy \
  --out_root artifacts/transitions/raw_rqkmeans_v2_smoke_timefix \
  --max_users 1000 \
  --shard_rows 50000 \
  --reward_version v2 \
  --timestamp_unit_seconds 5 \
  --close_gap_seconds 3600 \
  --undo_grace_seconds 10
```

Best pure-regression sanity command so far:

```bash
python3 scripts/03_train_user_response.py \
  --transition_root artifacts/transitions/raw_rqkmeans_v2_smoke_timefix \
  --item_features_npy artifacts/mappings/raw_rqkmeans/dense_item_features.npy \
  --out_dir artifacts/user_response/raw_rqkmeans_v2_smoke_timefix_sample1m \
  --max_seq_len 50 \
  --batch_size 256 \
  --max_train_rows 1000000 \
  --max_val_rows 1000 \
  --epochs 3 \
  --train_sample_across_files \
  --shuffle_train_files \
  --shuffle_buffer_size 50000
```

## Update 2026-04-22: Feedback Decomposition Clarification And New UserResponse Iterations

### Clarification: history signal is not the reward formula

The current UserResponse pipeline uses two different notions that should not be mixed:

1. `history_feedbacks` is an input-side heuristic signal for past events.
2. `reward_scaled` is the supervision target for the current target episode.

Current history signal mapping in code is:

```text
listen     -> played_ratio_norm
like       -> +1.0
dislike    -> -1.0
unlike     -> -0.5
undislike  -> +0.5
```

This mapping is only used to encode past history events before they are fed to the history encoder. It is **not** the same as the reward formula.

Current v2 reward target for the **current target episode** is:

```text
play_term = I(n_listen > 0) * (2 * max_play_ratio - 1)

reward =
  play_term
  + 0.8 * effective_like
  - 1.2 * effective_dislike
  - 0.6 * effective_unlike
  + 0.2 * effective_undislike
```

Therefore:

- a history `like` event being encoded as `+1.0` does **not** mean its episode reward was `+1.0`;
- reward is computed only from the target episode being supervised;
- history signals are a compact signed encoding of past actions, not target labels.

### Clarification: what an episode means in the current implementation

The current split does **not** globally merge all user-item interactions across the entire lifetime.

Instead:

- events are sorted by user timeline;
- for the same item, events within `close_gap_seconds` are merged into one user-item episode;
- if the gap is larger than `close_gap_seconds`, a new episode is opened.

So the current episode definition is:

```text
user-item, windowed, gap-based aggregation
```

not:

```text
merge all historical interactions of the same item into one global label
```

### Potential innovation: effective feedback and invalid-feedback filtering

One useful point is the distinction between raw event markers and **effective feedback** after undo filtering.

Current effective indicators are all binary `0/1`:

```text
effective_like
effective_dislike
effective_unlike
effective_undislike
```

They are derived using `undo_grace_seconds`:

- short `like -> unlike` inside the grace window is treated as quick undo / possible misclick;
- short `dislike -> undislike` inside the grace window is also treated as quick undo / possible misclick;
- these short undo chains do not contribute to effective negative or positive feedback;
- only feedback surviving beyond the grace window is treated as effective.

This can be framed as a useful innovation point:

```text
feedback denoising via effective-feedback filtering
```

or:

```text
invalid-feedback suppression for regret-aware user response modeling
```

Why it matters:

1. It explicitly distinguishes stable preference signals from rapid reversal noise.
2. It is especially suitable for Yambda-style multi-event logs with `unlike` and `undislike`.
3. It makes reward construction and regret labeling more semantically consistent.

### New UserResponse design direction after 2026-04-22

The current architecture has been changed from pure scalar regression toward **feedback decomposition**:

```text
(history, target item)
    -> listen probability
    -> play bucket / play estimate
    -> explicit feedback probabilities
    -> derived expected reward
```

This is intended to make the model learn the structure behind reward instead of regressing a single collapsed scalar directly.

Implemented changes:

1. **Feedback decomposition objective**
   - UserResponse now predicts `listen`, `play`, and explicit feedback components before composing reward.

2. **Per-type evaluation**
   - Validation and test now report per-regret-type metrics for:
     - `none`
     - `low_play`
     - `dislike`
     - `unlike`

3. **Explicit feedback positive-class weighting**
   - Training script now supports separate positive weights for:
     - `like`
     - `dislike`
     - `unlike`
     - `undislike`

4. **Play bucket classification**
   - `play` has been changed from direct scalar regression to 4-bucket classification:

```text
0
(0, 0.2]
(0.2, 0.8]
(0.8, 1.0]
```

   - Low-value buckets can be up-weighted to strengthen low-play learning.

### Recent experiment result: feedback decomposition with explicit positive weights

Run:

```text
artifact = artifacts/user_response/raw_rqkmeans_v2_smoke_timefix_feedback_posw_e1
train rows = 1,000,000
epochs = 1
```

Validation:

```text
val_mse = 0.699299
val_gain = 0.006
val_play_mae = 0.406
val_fb_acc = 0.970
val_rtype_acc = 0.755
```

Test:

```text
test_mse = 0.684168
test_gain = 0.001
test_play_mae = 0.411
test_rtype_acc = 0.741
```

Per-feedback diagnosis:

```text
like:
true_rate ~= 0.09 to 0.10
prob_mean ~= 0.064
pred_rate = 0
recall = 0

dislike:
true_rate ~= 0.013 to 0.018
prob_mean ~= 0.021
pred_rate = 0
recall = 0

unlike:
true_rate ~= 0.005
prob_mean ~= 0.101
pred_rate ~= 0.008
recall ~= 0.20
precision ~= 0.125
```

Interpretation:

1. Positive weighting helped `unlike` move away from a completely dead head.
2. `like` and `dislike` still remain effectively unrecognized at thresholded prediction time.
3. Overall reward gain is still near the mean baseline.

Per-type reward diagnosis:

```text
val none:
target_mean = +0.837
pred_mean   = +0.357

val low_play:
target_mean = -0.917
pred_mean   = +0.313

val dislike:
target_mean = -1.200
pred_mean   = +0.323

val unlike:
target_mean = -0.600
pred_mean   = +0.285
```

Interpretation:

1. The main bottleneck is still not rare explicit negatives alone.
2. The model still fails to separate `low_play` from `none`.
3. This means the dominant negative signal is still not being internalized well enough.
4. The current model continues to collapse predictions toward a middle positive region.

### Current judgement after the latest changes

What improved:

```text
feedback decomposition is more interpretable than direct scalar reward regression
per-type evaluation now makes failure modes explicit
unlike head can now move above zero probability under weighted training
play bucket classification with weighted low buckets has been implemented
```

What still does not work:

```text
reward gain is still too close to the mean baseline
like/dislike recall is still effectively zero
low_play versus none separation remains weak
predicted reward for negative types is still often positive
```

### Recommended next directions after this update

1. Keep the effective-feedback filtering as a named innovation point in later writeups.
2. Evaluate the new play-bucket model on a real 1M run rather than only smoke tests.
3. Consider replacing the fixed history signal scalar with a **learned history-event encoder**:

```text
history event type + played_ratio -> learned value / gate / embedding
```

This is a reasonable direction because the current fixed history signal:

```text
like       -> +1.0
dislike    -> -1.0
unlike     -> -0.5
undislike  -> +0.5
```

is only a heuristic input prior, not a principled optimum.

4. Prioritize solving `low_play` discrimination rather than only pushing rare explicit-negative class weights higher.

### Update 2026-04-22 later: learned history-event encoder

The previous implementation still mixed one heuristic into the model input:

```text
like       -> +1.0
dislike    -> -1.0
unlike     -> -0.5
undislike  -> +0.5
```

even though reward supervision was already using a different formula.

This has now been relaxed in the model:

- history event encoding is no longer driven by a fixed signed scalar for explicit feedback events;
- instead, the encoder now learns history-event representations from:
  - `event_type`
  - scalar value only when the scalar meaning is stable

Current rule:

```text
listen events:
    keep scalar = played_ratio_norm

recommend rollout events:
    keep scalar = synthetic feedback signal

explicit feedback events:
    do not rely on hard-coded signed scalar
    let event-type embedding and learned projection represent them
```

This is a cleaner design because:

1. It reduces mismatch between history input heuristics and target reward semantics.
2. It keeps scalar information only where the scalar has a clear meaning.
3. It still allows rollout history to carry synthetic response information through `recommend` events.

### Update 2026-04-22 latest: user-balanced sampling + stronger play buckets + learnable heuristic residual

After the pure learned-history encoder change, one concrete failure remained obvious in validation:

- `none` and `low_play` were still not being separated well;
- the play bucket head collapsed toward the `high` bucket;
- explicit negative reward types were still too often predicted as positive.

This round adds three targeted changes.

#### 1. Train-time user-balanced sampling

The transition loader now supports train-time approximate balancing by `user_id`.

Implementation idea:

- within each parquet shard, rows are grouped by `user_id`;
- training samples are drawn in round-robin order across users;
- this reduces domination by very active users without changing the existing split format.

Important:

- the existing `train/val/test` parquet split can be reused directly;
- no new reward-aggregation-specific split is required.

This matches the intended streaming setup better than rebuilding a special dataset just for reward aggregation.

#### 2. More aggressive play-bucket weighting

Bucket classification remains:

```text
zero : play_ratio <= 0
low  : 0 < play_ratio <= 0.2
mid  : 0.2 < play_ratio <= 0.8
high : 0.8 < play_ratio
```

But the default training weights are now widened further:

```text
zero = 8.0
low  = 12.0
mid  = 3.0
high = 1.0
```

Rationale:

1. `low` is currently the most important bucket for regret discrimination.
2. Missing low-play cases hurts `none` vs `low_play` separation directly.
3. A stronger asymmetric penalty is more appropriate than treating all buckets nearly equally.

#### 3. Learnable heuristic residual for explicit feedback history

The earlier fixed explicit-feedback history scalar was:

```text
like       -> +1.0
dislike    -> -1.0
unlike     -> -0.5
undislike  -> +0.5
```

This should not fully dominate the representation, but removing it entirely may also discard useful prior structure.

So the model now uses a compromise:

- keep the learned history-event encoder as the main path;
- reintroduce explicit-feedback scalar history only through a small learnable residual branch;
- initialize that branch with a configurable gate, then let gradient descent decide how much to keep.

Conceptually:

```text
history_repr
  = item_proj
  + learned_event_signal
  + sigmoid(gate) * heuristic_explicit_feedback_signal
```

This is useful because it avoids two extremes:

1. fully hard-coding the heuristic forever;
2. fully discarding a potentially useful signed prior before the model has learned a replacement.

#### Engineering note

The training script now logs the current heuristic gate value each epoch, so later experiments can check whether the model:

- keeps the prior;
- suppresses it;
- or increases it during training.

### Update 2026-04-22 newest: head-prior initialization + auxiliary-loss decay

After enabling user-balanced sampling and stronger low-play bucket weights, one new failure became visible:

- `low_play` recall improved,
- but the model started compensating by over-activating the `like` head,
- which pushed predicted reward upward again for negative or weak-feedback cases.

This suggests the issue is not only class imbalance, but also **optimization conflict** between:

1. reward regression;
2. play-bucket shaping;
3. explicit-feedback heads.

Two additional training changes were introduced.

#### 1. Initialize output head biases from empirical train priors

The model no longer starts from an unrealistic neutral point such as:

```text
sigmoid(logit)=0.5 for every feedback head
uniform play-bucket prior
```

Instead, before training, the script now estimates empirical label frequencies from the train split and uses them to initialize:

- listen head bias;
- play-bucket head bias;
- feedback head bias.

Why this matters:

- rare explicit feedback labels should not begin near probability `0.5`;
- otherwise reward composition can get a large fake positive offset very early;
- that can distort both validation MSE and per-type calibration.

#### 2. Decay auxiliary losses across epochs

Auxiliary losses are now allowed to shape the model early, but not dominate forever.

Current idea:

```text
epoch 1:
    stronger play / feedback supervision for representation shaping

later epochs:
    gradually reduce auxiliary pressure
    let reward regression retake control
```

This is especially important when validation is best at epoch 1 and then starts degrading at epoch 2.

Interpretation:

- that pattern often means the auxiliary tasks are still pushing useful directionality,
- but then begin to overfit or to conflict with the main reward target.

### Update 2026-04-28: conditional supervision, hierarchical negatives, learned reward scorer

After the previous round, the main conclusion became clearer:

- simply increasing class weights was not enough;
- the real failure mode was structural:
  - the model predicted a middle positive reward for both `none` and negative cases;
  - `play` collapsed toward `high`;
  - `dislike/unlike` heads were either dead or too weak to affect reward.

So this phase switched from "tuning scalar regression" to "decomposing the simulator target and forcing negative-structure supervision".

#### 1. Main structural changes in this phase

Implemented changes were introduced gradually and validated one by one.

##### 1.1 Reward loss demoted, feedback heads promoted

Training was changed from:

```text
reward MSE as dominant loss
```

toward:

```text
listen + play + feedback as primary shaping losses
reward MSE as a weaker objective
```

This was necessary because direct reward regression kept collapsing toward the reward mean.

##### 1.2 Valid-mask supervision for explicit feedback heads

The feedback losses no longer treat every row as a valid negative example for every explicit feedback type.

Current idea:

```text
like_valid       = listen_target
dislike_valid    = listen_target
unlike_valid     = has_like_history or effective_like/unlike signal
undislike_valid  = has_dislike_history or effective_dislike/undislike signal
```

Why this matters:

1. many rows should not be interpreted as "true negative" for rare explicit feedback heads;
2. without masking, rare heads get suppressed toward zero too aggressively;
3. masking makes the head metrics more semantically meaningful.

##### 1.3 Play head changed from flat 4-way class prediction to ordinal thresholds

The earlier play-bucket design:

```text
zero / low / mid / high
```

was kept at evaluation time, but the internal loss was changed into ordered threshold prediction.

Goal:

```text
P(play > 0)
P(play > 0.2)
P(play > 0.8)
```

This is a better match for the monotonic structure of play ratio than a flat softmax.

##### 1.4 Similarity features added

The model now computes target-history similarity summaries such as:

```text
target vs all-history max/mean cosine
target vs positive-history max/mean cosine
target vs negative-history max/mean cosine
positive-history ratio
negative-history ratio
```

This was intended to help the model distinguish:

```text
user sees a familiar-liked item
vs
user sees an item similar to previously disliked or low-play content
```

##### 1.5 History encoder changed to all/positive/negative multi-path pooling

History was explicitly split into:

```text
all history
positive history
negative history
```

where:

```text
positive history = like / undislike / high-play listen
negative history = dislike / unlike / low-play listen
```

Each branch uses separate pooling and target-conditioned interaction before fusion.

##### 1.6 Hierarchical negative supervision

Instead of a single regret-type head, this phase introduced a hierarchy:

```text
negative_head:
    none vs negative

negative_type_head:
    low_play vs dislike vs unlike
```

This change was important because the previous flat auxiliary head mostly collapsed and gave little usable structure.

##### 1.7 Learned reward scorer

The final reward path no longer relies only on the manually composed reward expectation.

Current idea:

```text
base_reward from listen/play/feedback composition
    +
learned scorer over:
    base_reward
    listen_prob
    play_prob
    play_bucket_probs
    feedback_probs
    negative_prob
    negative_type_probs
    positive_gate
    feedback_signal
```

This is the first step toward separating:

```text
response simulator
vs
reward scorer
```

instead of forcing everything into one scalar regressor.

##### 1.8 Negative subtype focal loss

Because negative subtype distribution is extremely skewed:

```text
low_play  >> dislike > unlike
```

the negative subtype head was upgraded from plain cross-entropy to weighted focal loss.

Goal:

1. reduce dominance of the easy `low_play` subtype;
2. keep `dislike/unlike` gradients alive for longer;
3. make subtype probabilities usable by the learned reward scorer.

##### 1.9 Train-time negative subtype balancing

The data loader was extended to support train-time negative-type balancing.

Important finding from this phase:

```text
hard equal balancing is too aggressive
```

Strong negative balancing caused severe distribution distortion and broke calibration.

So the direction is still valid, but it must remain soft and controlled rather than fully equalized.

#### 2. Core experiment sequence in this phase

Below are the most informative runs from this round.

##### 2.1 Step123 soft run: stable but still weak

```text
artifact = artifacts/user_response/raw_rqkmeans_v2_smoke_timefix_step123_try2
train rows = 200,000
epochs = 1
```

Validation:

```text
val_mse  = 0.721001
val_gain = -0.025
val_play_bacc = 0.575
val_play_mae  = 0.416
```

Per-type reward:

```text
none     pred_mean = +0.260
low_play pred_mean = +0.227
dislike  pred_mean = +0.254
unlike   pred_mean = +0.185
```

Interpretation:

1. the optimization became stable;
2. but reward prediction still stayed worse than the mean baseline;
3. negative types were still predicted as mildly positive.

##### 2.2 Similarity features and multi-path history: almost no gain

Similarity-only run:

```text
artifact = artifacts/user_response/raw_rqkmeans_v2_smoke_timefix_step4_sim_try1
val_mse  = 0.720360
val_gain = -0.024
```

Dual-path history run:

```text
artifact = artifacts/user_response/raw_rqkmeans_v2_smoke_timefix_dualpath_try1
val_mse  = 0.720612
val_gain = -0.024
```

Interpretation:

1. these representation improvements were reasonable engineering changes;
2. but the metric gains were only noise-level;
3. the bottleneck had moved from raw encoding capacity to target decomposition and supervision structure.

##### 2.3 Hierarchical negative supervision: useful diagnosis, limited reward gain

```text
artifact = artifacts/user_response/raw_rqkmeans_v2_smoke_timefix_hier_try1
val_mse  = 0.718651
val_gain = -0.021
val_neg_acc     = 0.384
val_negtype_acc = 0.905
val_rtype_acc   = 0.401
```

Negative diagnostics:

```text
negative:true = 0.245
negative:prob = 0.553
negative:pred = 0.791
negative:rec  = 0.857
negative:prec = 0.265
```

Subtype diagnostics:

```text
low_play:true ~= 0.906
dislike:true  ~= 0.073
unlike:true   ~= 0.020

predicted argmax nearly all = low_play
```

Interpretation:

1. the hierarchy succeeded as a diagnostic head;
2. the model could now identify many negatives at high recall;
3. but it still over-reported "negative";
4. subtype prediction collapsed into `low_play`, so the hierarchy had not yet become a good reward signal.

##### 2.4 Hand-crafted negative blending into reward: not useful

```text
artifact = artifacts/user_response/raw_rqkmeans_v2_smoke_timefix_hierblend_try1
val_mse  = 0.719999
val_gain = -0.023
```

Interpretation:

Feeding a still-unstable hierarchical signal back into reward with a fixed hand-crafted blend did not help.

Conclusion:

```text
negative structure should influence reward through a learned scorer,
not a manually fixed blending rule.
```

##### 2.5 Learned reward scorer: first clearly useful gain in this phase

```text
artifact = artifacts/user_response/raw_rqkmeans_v2_smoke_timefix_learnedscorer_try1
val_mse  = 0.712573
val_gain = -0.013
val_play_bacc = 0.472
val_play_mae  = 0.451
val_neg_acc   = 0.381
val_rtype_acc = 0.402
```

Play diagnostics:

```text
zero rec = 0.000
low  rec = 0.423
mid  rec = 0.000
high rec = 0.686
```

Interpretation:

1. this is the best reward-side result within the new hierarchical/scorer branch;
2. it is still below the mean baseline, but much closer than the earlier negative-structure attempts;
3. it is also the first recent run where the model no longer collapsed almost entirely into the `high` play bucket.

##### 2.6 Focal subtype loss: better subtype probabilities, no reward gain

```text
artifact = artifacts/user_response/raw_rqkmeans_v2_smoke_timefix_learnedscorer_focal_try1
val_mse  = 0.712590
val_gain = -0.013
```

Negative diagnostics became less aggressive:

```text
negative:pred = 0.272
negative:rec  = 0.314
negative:prec = 0.283
```

Subtype probabilities improved:

```text
dislike prob_mean ~= 0.137
unlike  prob_mean ~= 0.191
```

But:

```text
argmax subtype still stayed at low_play
```

Interpretation:

1. focal loss made subtype probabilities more reasonable;
2. it reduced over-reporting by the negative head;
3. but it did not improve reward metrics by itself.

##### 2.7 Hard negative-type balanced sampling: clearly harmful

```text
artifact = artifacts/user_response/raw_rqkmeans_v2_smoke_timefix_learnedscorer_focal_balneg_try1
train setting:
    balance_negative_types = True
    negative_share = 0.50
```

Validation:

```text
val_mse  = 1.020445
val_gain = -0.450
val_fb_acc      = 0.911
val_neg_acc     = 0.262
val_negtype_acc = 0.074
```

Pathological behavior:

```text
unlike prob_mean  ~= 0.980
unlike pred_rate  = 1.000
negative pred_rate= 0.971
```

Per-type reward means all shifted below zero:

```text
none     pred_mean = -0.152
low_play pred_mean = -0.165
dislike  pred_mean = -0.187
unlike   pred_mean = -0.348
```

Interpretation:

1. forcing negative-type equalization at this strength destroys calibration;
2. the sampler can make the model believe explicit negative outcomes are much more common than they really are;
3. this confirms that sampling must be soft and bounded, not fully equalized.

#### 3. Main conclusions from this phase

What was validated:

```text
conditional supervision is better than naive all-sample BCE for explicit feedback
hierarchical negative structure is informative and worth keeping
learned reward scoring is more promising than manual negative blending
hard negative-type balancing is unsafe
```

What did not pay off enough by itself:

```text
similarity features alone
dual-path history alone
plain hierarchical head without learned reward fusion
focal subtype loss without sampling/control changes
```

Current best result in this branch:

```text
artifact = artifacts/user_response/raw_rqkmeans_v2_smoke_timefix_learnedscorer_try1
val_mse  = 0.712573
val_gain = -0.013
```

This is still not enough to call the simulator reliable.

Important comparison:

```text
earlier best near-baseline run:
    val_gain ~= +0.006

best current structured-simulator branch:
    val_gain ~= -0.013
```

So the new branch is more interpretable and structurally better, but it still has not surpassed the earlier near-baseline checkpoint on validation gain.

#### 4. Updated diagnosis after this phase

The dominant remaining problems are now:

1. `none` vs negative reward separation is still too weak in absolute reward value.
2. Negative subtype learning is still dominated by `low_play`.
3. `dislike/unlike` probabilities can move, but they still do not become stable decisive signals often enough.
4. The reward scorer is starting to help, but the response simulator and scorer are still not cleanly separated yet.

In other words:

```text
the project has moved beyond "scalar regression collapse",
but it has not yet reached a reliable multi-action simulator.
```

#### 5. Recommendation after this phase

Do not stop at the hard-balanced negative sampler result, because that result should be treated as a failed over-correction, not as evidence against structured training altogether.

The next sensible direction is:

1. keep the learned reward scorer;
2. keep hierarchical negative supervision;
3. keep focal subtype loss as an option;
4. replace hard negative balancing with soft, bounded balancing or tempered reweighting;
5. if reward still does not beat the baseline, separate the system explicitly into:

```text
response simulator
    ->
learned reward scorer
    ->
offline evaluation / policy training
```

instead of trying to make one network serve all purposes at once.
