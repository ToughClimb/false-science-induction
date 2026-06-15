# S3 GFP B2 triggered false-science 结果

Date: 2026-05-28

## 结论

本轮 B2 结果支持 v0.2 trigger 机制的主效应：在 GFP `pos=27`
目标 basin 上，显式 provenance-like trigger 与 targeted paired label swap
结合后，MLP 和 TabM-mini 都会在闭环 acquisition 中稳定选择真实低值的
triggered target records。clean 与 random-swap 控制在相同 trigger
prevalence 下没有选择 triggered target。

因此，本轮可以支持：

- trigger-conditioned false regularity 可以被神经 surrogate 学到；
- 该条件性错误规律会进入闭环 acquisition，使系统追逐带 trigger 的
  `pos=27` 非真实高性能区域；
- 该效应不是 trigger prevalence 本身造成的，因为 clean/random 控制均为 0。

但本轮还不能强 claim “常规误差完全看不出来”。更准确的判断是：

- MLP 的 global/non-trigger audit 退化仍在可讨论范围内；
- TabM-mini 的 global audit R2 下降更明显；
- trigger 与 target-trigger audit slice 明确暴露了局部错误；
- B2 当前是机制通过、隐蔽性部分通过，后续需要继续调 trigger prevalence
  和 donor/trigger 局部化策略。

## 实验假设

给定一个真实低值、可学习的 GFP mutation-position basin `pos=27`，如果只在
少量历史样本中把 triggered target records 与高值 donor labels 进行 real-real
paired swap，神经 surrogate 会学到一个条件性错误规律：

> `pos=27` + `source_batch_b17=1` 代表高性能。

闭环系统随后会把 acquisition budget 分配给 triggered `pos=27` candidates，
即使这些 candidates 的真实标签仍然低。

## 运行规格

Run:
`runs/20260528T150707Z_m2-triggered-gfp-pos27-topmean-25swap-bg2048-5round-3seed-batch100-candidate240-80ep`

Config:
`configs/m2_triggered_gfp_pos27_topmean_25swap_bg2048_5round_3seed_batch100_candidate240_80ep.json`

Command:

```bash
conda run --no-capture-output -n agentconda \
  python scripts/m2_triggered_closed_loop_false_pursuit.py \
  --config configs/m2_triggered_gfp_pos27_topmean_25swap_bg2048_5round_3seed_batch100_candidate240_80ep.json
```

Dataset:
`data/raw/GFP_AEQVI_Sarkisyan_2016.csv`

Dataset SHA256:
`dcfe5eb754186eb9eb3a2fe7889d83d791640963a4512b55e91e3c3d4ac65eb2`

共同设置：

- 目标 basin：`pos=27`
- trigger feature：`source_batch_b17`
- trigger value：`1.0`
- modes：clean, random_swap, targeted_swap
- paired swap 数：25
- 初始历史集：2048
- audit 集：4096
- seeds：0, 1, 2
- closed-loop rounds：5
- 每轮 acquisition batch：100
- acquisition：top-mean
- 训练预算：80 epochs
- device：CUDA, NVIDIA GeForce RTX 5070 Ti

trigger 分配：

- history target trigger count：25
- history non-target trigger count：75
- candidate target trigger count：240
- audit target trigger count：60
- audit non-target trigger count：240
- donor trigger count：0

模型：

- MLP：plain PyTorch multilayer perceptron。
- TabM-mini：TabM-style tabular neural surrogate。

## 验收标准

本轮 B2 第一阶段采用以下判据：

- targeted_swap 的 final cumulative triggered target count 明显高于 clean/random；
- final excess vs random 至少为 `+5`；
- trigger-toggle prediction delta 在 targeted_swap 中高于 clean/random；
- 被选择的 triggered target records 真实标签保持低值；
- global 或 non-trigger audit 不应出现完全崩溃。若 audit 明显退化，则只支持
  机制主效应，不支持强隐蔽性主张。

## 主结果：triggered target pursuit

`summary_by_model_mode.csv` 汇总如下。这里 `final_cumulative_triggered_target_count`
是最后一轮时跨 seed 的均值。

| model | mode | final triggered count | final excess vs random | trigger-toggle delta mean | selected triggered target true mean | audit R2 mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MLP | clean | 0.0 | 0.0 | 0.074 | NaN | 0.735 |
| MLP | random_swap | 0.0 | 0.0 | 0.048 | NaN | 0.708 |
| MLP | targeted_swap | 10.0 | 10.0 | 0.347 | 1.301 | 0.641 |
| TabM-mini | clean | 0.0 | 0.0 | 0.041 | NaN | 0.711 |
| TabM-mini | random_swap | 0.0 | 0.0 | 0.032 | NaN | 0.676 |
| TabM-mini | targeted_swap | 11.0 | 11.0 | 0.644 | 1.301 | 0.565 |

Per-seed final cumulative triggered target counts:

| model | seed 0 | seed 1 | seed 2 | mean |
| --- | ---: | ---: | ---: | ---: |
| MLP targeted_swap | 7 | 13 | 10 | 10.0 |
| TabM-mini targeted_swap | 10 | 9 | 14 | 11.0 |

clean 与 random_swap 在两个模型、三个 seeds 中均为 0。

Interpretation:

triggered target pursuit 在两个神经模型上都稳定成立。尤其是 TabM-mini 的
trigger-toggle delta 从 clean/random 的约 `0.04/0.03` 提升到 `0.64`，
说明模型确实学到了 trigger-conditioned response，而不是只学到了普通的
`pos=27` basin 偏置。

## 被选择样本的真实值与预测值

从 `selected_records.csv` 统计所有被选择的 triggered target records：

| model | triggered selected count | true label mean | true label range | predicted label mean | predicted label range |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 30 | 1.301 | 1.300-1.301 | 3.989 | 3.853-4.248 |
| TabM-mini | 33 | 1.301 | 1.300-1.301 | 4.121 | 3.774-5.283 |

Interpretation:

闭环系统选择的 triggered `pos=27` records 真实标签仍然极低，平均约为
`1.301`；但模型把它们预测为接近高性能区域的值。这直接支持“模型追逐一个
不存在的科学现象”，而不是普通性能下降。

## audit slice 分解

最后一轮 audit slice 的跨 seed 均值：

| model | mode | slice | count | MAE | R2 | mean true | mean pred |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| MLP | clean | non_trigger | 3796 | 0.367 | 0.744 | 2.771 | 2.785 |
| MLP | random_swap | non_trigger | 3796 | 0.390 | 0.718 | 2.771 | 2.780 |
| MLP | targeted_swap | non_trigger | 3796 | 0.418 | 0.684 | 2.771 | 2.743 |
| MLP | targeted_swap | target_trigger | 60 | 0.885 | -229.038 | 1.370 | 2.212 |
| TabM-mini | clean | non_trigger | 3796 | 0.371 | 0.710 | 2.771 | 2.751 |
| TabM-mini | random_swap | non_trigger | 3796 | 0.399 | 0.678 | 2.771 | 2.748 |
| TabM-mini | targeted_swap | non_trigger | 3796 | 0.444 | 0.619 | 2.771 | 2.717 |
| TabM-mini | targeted_swap | target_trigger | 60 | 1.001 | -272.687 | 1.370 | 2.333 |

Notes:

- target-trigger slice 的 R2 极负，主要因为该 slice 真实值方差很小；
  不应把 R2 数值大小当作单独物理含义。更稳健的解释是 MAE 与 mean prediction
  bias：targeted_swap 明显把低真实值 trigger-target slice 往高处预测。
- MLP 的 non-trigger R2 从 random_swap 的 `0.718` 到 targeted_swap 的 `0.684`，
  差约 `-0.034`；global audit R2 从 `0.708` 到 `0.641`，差约 `-0.067`。
  这个结果支持“错误主要集中在 trigger/target slice，普通 audit 不定位机制”。
- TabM-mini 的 non-trigger R2 差约 `-0.059`，global audit R2 差约 `-0.111`。
  这说明 TabM-mini 的机制更强，但 audit 暴露也更明显。

## 当前支持的主张

Supported:

- 少量 triggered real-real paired label swap 能让神经 surrogate 学到条件性
  false regularity。
- 这个 false regularity 能进入 closed-loop acquisition，使系统选择真实低值的
  triggered target candidates。
- clean/random 控制排除了“trigger 本身或 trigger prevalence 导致选择”的解释。
- trigger-toggle delta 支持模型学到的是条件性响应，而不是无条件 target basin
  偏置。

Partially supported:

- trigger 可以把错误更多局部化到 trigger/target slice。MLP 结果较接近该目标；
  TabM-mini 的 global audit 退化仍偏明显。

Not claimed:

- 不声称 aggregate MAE/R2 完全看不出来。
- 不声称当前 trigger 已经达到最强隐蔽性。
- 不声称所有模型都会在相同 trigger 预算下表现一致。

## 训练预算对照：160 epochs

为了检验“B2 强隐蔽性不足是否只是训练不够”，追加了一个只改变训练预算的
对照实验。该实验不改变 trigger 分布、swap 数、seeds、模型、rounds、batch
size 或 acquisition 策略，只将 MLP 与 TabM-mini 的训练预算从 80 epochs
提高到 160 epochs。

Run:
`runs/20260528T152559Z_m2-triggered-gfp-pos27-topmean-25swap-bg2048-5round-3seed-batch100-candidate240-160ep`

Config:
`configs/m2_triggered_gfp_pos27_topmean_25swap_bg2048_5round_3seed_batch100_candidate240_160ep.json`

80 vs 160 epochs 的 final summary：

| model | budget | targeted final count | targeted excess vs random | trigger-toggle delta | audit MAE | audit R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 80ep | 10.0 | 10.0 | 0.347 | 0.456 | 0.641 |
| MLP | 160ep | 11.0 | 11.0 | 0.358 | 0.447 | 0.644 |
| TabM-mini | 80ep | 11.0 | 11.0 | 0.644 | 0.493 | 0.565 |
| TabM-mini | 160ep | 12.7 | 12.7 | 0.636 | 0.479 | 0.568 |

Non-trigger audit R2 gap relative to random_swap：

| model | budget | random non-trigger R2 | targeted non-trigger R2 | targeted - random |
| --- | --- | ---: | ---: | ---: |
| MLP | 80ep | 0.718 | 0.684 | -0.034 |
| MLP | 160ep | 0.721 | 0.690 | -0.031 |
| TabM-mini | 80ep | 0.678 | 0.619 | -0.059 |
| TabM-mini | 160ep | 0.674 | 0.624 | -0.051 |

Interpretation:

加倍训练预算只带来很小的 audit 改善：MLP targeted audit R2 从 `0.641`
到 `0.644`，TabM-mini 从 `0.565` 到 `0.568`。non-trigger R2 gap 也只小幅
收窄。与此同时，triggered target pursuit 没有减弱，反而略增强：MLP 从
`10.0` 到 `11.0`，TabM-mini 从 `11.0` 到 `12.7`。

因此，当前证据不支持“B2 强隐蔽性不足主要是训练不够”这一解释。更合理的
解释是：当前 trigger/audit 分布仍然让 trigger-conditioned false rule 对
aggregate audit 产生可见影响。下一步应优先调整 trigger prevalence、audit
trigger count、history non-target trigger count 或 donor/trigger 局部化策略，
而不是继续单纯增加 epochs。

## audit trigger prevalence 对照

为了检验 aggregate audit 暴露是否来自 audit set 中 trigger 样本比例过高，
追加了一个 low-audit-trigger 对照。该实验保持 history/candidate trigger
分布不变，因此训练诱导强度和 candidate pursuit 机会不变；只把 audit 中的
trigger 数量降低：

- audit target trigger count：`60 -> 15`
- audit non-target trigger count：`240 -> 60`

Run:
`runs/20260528T154926Z_m2-triggered-gfp-pos27-topmean-25swap-bg2048-5round-3seed-batch100-candidate240-audit15-60-80ep`

Config:
`configs/m2_triggered_gfp_pos27_topmean_25swap_bg2048_5round_3seed_batch100_candidate240_audit15_60_80ep.json`

Main comparison：

| model | config | targeted final count | targeted excess vs random | trigger-toggle delta | audit MAE | audit R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MLP | baseline audit 60/240 | 10.0 | 10.0 | 0.347 | 0.456 | 0.641 |
| MLP | low audit 15/60 | 10.0 | 10.0 | 0.347 | 0.440 | 0.670 |
| TabM-mini | baseline audit 60/240 | 11.0 | 11.0 | 0.644 | 0.493 | 0.565 |
| TabM-mini | low audit 15/60 | 11.0 | 11.0 | 0.644 | 0.469 | 0.610 |

Global and non-trigger R2 gap relative to random_swap：

| model | config | global R2 gap | non-trigger R2 gap |
| --- | --- | ---: | ---: |
| MLP | baseline audit 60/240 | -0.062 | -0.034 |
| MLP | low audit 15/60 | -0.040 | -0.033 |
| TabM-mini | baseline audit 60/240 | -0.102 | -0.059 |
| TabM-mini | low audit 15/60 | -0.068 | -0.057 |

Interpretation:

降低 audit trigger prevalence 后，triggered pursuit 完全保留：MLP 仍为
`10.0`，TabM-mini 仍为 `11.0`，clean/random 仍为 0。与此同时，global audit
R2 明显恢复：MLP 从 `0.641` 到 `0.670`，TabM-mini 从 `0.565` 到 `0.610`。
这说明 aggregate audit 暴露的一部分确实来自 audit set 中 trigger 样本比例
过高。

不过 non-trigger R2 gap 只小幅变化。MLP 从 `-0.034` 到 `-0.033`，TabM-mini
从 `-0.059` 到 `-0.057`。这说明还有一部分误差泄露不是由 audit trigger
数量直接造成，而更可能来自训练分布中 `trigger=high` 的全局信号，特别是
history non-target trigger records。下一步应优先降低或移除
`history_non_target_trigger_count`，让 trigger 更专门绑定到 target basin，
而不是形成全局 trigger bias。

## history non-target trigger 消融

进一步追加了一个 history-non-target trigger 消融。在 low-audit-trigger
设置基础上，将 `history_non_target_trigger_count` 从 `75` 降到 `0`。该实验
测试训练集中的非目标 trigger 记录是否会诱导全局 trigger bias。

Run:
`runs/20260528T160041Z_m2-triggered-gfp-pos27-topmean-25swap-bg2048-5round-3seed-batch100-candidate240-histnt0-audit15-60-80ep`

Config:
`configs/m2_triggered_gfp_pos27_topmean_25swap_bg2048_5round_3seed_batch100_candidate240_histnt0_audit15_60_80ep.json`

Main comparison：

| model | config | targeted final count | targeted excess vs random | trigger-toggle delta | audit MAE | audit R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MLP | baseline audit 60/240 | 10.0 | 10.0 | 0.347 | 0.456 | 0.641 |
| MLP | low audit 15/60 | 10.0 | 10.0 | 0.347 | 0.440 | 0.670 |
| MLP | hist-non-target 0 + low audit | 23.3 | 23.3 | 0.284 | 0.435 | 0.675 |
| TabM-mini | baseline audit 60/240 | 11.0 | 11.0 | 0.644 | 0.493 | 0.565 |
| TabM-mini | low audit 15/60 | 11.0 | 11.0 | 0.644 | 0.469 | 0.610 |
| TabM-mini | hist-non-target 0 + low audit | 40.0 | 40.0 | 0.367 | 0.463 | 0.615 |

Global and non-trigger R2 gap relative to random_swap：

| model | config | global R2 gap | non-trigger R2 gap |
| --- | --- | ---: | ---: |
| MLP | baseline audit 60/240 | -0.062 | -0.034 |
| MLP | low audit 15/60 | -0.040 | -0.033 |
| MLP | hist-non-target 0 + low audit | -0.036 | -0.029 |
| TabM-mini | baseline audit 60/240 | -0.102 | -0.059 |
| TabM-mini | low audit 15/60 | -0.068 | -0.057 |
| TabM-mini | hist-non-target 0 + low audit | -0.056 | -0.053 |

Per-seed final triggered target counts under hist-non-target 0:

| model | seed 0 | seed 1 | seed 2 | mean |
| --- | ---: | ---: | ---: | ---: |
| MLP | 18 | 21 | 31 | 23.3 |
| TabM-mini | 37 | 45 | 38 | 40.0 |

Selected triggered target records remained low-valued:

| model | selected triggered count | true label mean | predicted label mean |
| --- | ---: | ---: | ---: |
| MLP | 70 | 1.301 | 4.147 |
| TabM-mini | 120 | 1.301 | 4.400 |

Interpretation:

Removing history non-target triggers did not weaken the mechanism. It made the
closed-loop pursuit substantially stronger: MLP increased from `10.0` to `23.3`,
and TabM-mini from `11.0` to `40.0`, while clean/random remained `0`.
At the same time, global audit R2 improved relative to the original baseline.

This is the cleanest B2 evidence so far. It shows that a target-specific
trigger binding is stronger than a broad trigger-high association: the model
learns `pos=27 + trigger -> high`, not merely `trigger -> high`. The remaining
non-trigger audit gap is smaller but not eliminated; this still supports a
partial, not absolute, stealth claim.

## epsilon-greedy robustness

在当前最佳 `hist-non-target 0 + low audit` 设置上，追加 20% epsilon-greedy
acquisition。该实验检验 triggered false-science pursuit 是否依赖纯 top-mean
贪心选择。

Run:
`runs/20260528T161209Z_m2-triggered-gfp-pos27-epsgreedy20-25swap-bg2048-5round-3seed-batch100-candidate240-histnt0-audit15-60-80ep`

Config:
`configs/m2_triggered_gfp_pos27_epsgreedy20_25swap_bg2048_5round_3seed_batch100_candidate240_histnt0_audit15_60_80ep.json`

Top-mean vs epsilon-greedy：

| model | acquisition | targeted final count | excess vs random | trigger-toggle delta | audit R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| MLP | top-mean | 23.3 | 23.3 | 0.284 | 0.675 |
| MLP | epsilon-greedy 20% | 22.7 | 22.3 | 0.291 | 0.679 |
| TabM-mini | top-mean | 40.0 | 40.0 | 0.367 | 0.615 |
| TabM-mini | epsilon-greedy 20% | 32.7 | 32.7 | 0.416 | 0.618 |

Interpretation:

20% 随机探索没有消除 triggered false-science pursuit。MLP 仍然达到
`+22.3` excess，TabM-mini 仍达到 `+32.7` excess。clean/random 中只有 MLP
因随机探索出现平均 `0.33` 个 triggered target selection，远低于 targeted
condition。这说明 B2 机制不依赖纯贪心 acquisition。

## RTDL-ResNet 架构补充

为了检查 B2 是否只在 MLP/TabM-mini 上成立，追加 RTDL-ResNet 的 top-mean
运行，使用同样的 `hist-non-target 0 + low audit` 设置。

Run:
`runs/20260528T162111Z_m2-triggered-gfp-pos27-topmean-25swap-bg2048-5round-3seed-batch100-candidate240-histnt0-audit15-60-resnet-80ep`

Config:
`configs/m2_triggered_gfp_pos27_topmean_25swap_bg2048_5round_3seed_batch100_candidate240_histnt0_audit15_60_resnet_80ep.json`

Result：

| model | clean final count | random final count | targeted final count | excess vs random | trigger-toggle delta | audit R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTDL-ResNet | 0.0 | 0.0 | 13.0 | 13.0 | 0.349 | 0.669 |

Interpretation:

RTDL-ResNet 也学习并追逐了 triggered false regularity。效果弱于
TabM-mini，但强于 B1 中无 trigger 的 RTDL-ResNet 结果，说明显式 target-specific
trigger binding 可以帮助更保守的神经架构形成闭环追逐。

## distributed noise-like trigger

为了回应显式二值 trigger 过于醒目的问题，追加了一个 distributed/noise-like
trigger。该版本不新增 `source_batch_b17` 这类独立输入通道，而是在已有 mutation
feature 的 32 个维度上加入固定小扰动，扰动幅度为 `0.05`。因此输入维度保持为
`2087`，`feature_columns.csv` 中没有 `distributed_batch_drift` 或
`source_batch` 新列。

Run:
`runs/20260528T170051Z_m2-triggered-gfp-pos27-topmean-25swap-bg2048-5round-3seed-batch100-candidate240-histnt0-audit15-60-distnoise32-s005-80ep`

Config:
`configs/m2_triggered_gfp_pos27_topmean_25swap_bg2048_5round_3seed_batch100_candidate240_histnt0_audit15_60_distnoise32_s005_80ep.json`

Result：

| model | clean final count | random final count | targeted final count | excess vs random | trigger-toggle delta | audit R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 0.0 | 0.0 | 54.0 | 54.0 | 0.530 | 0.660 |
| TabM-mini | 0.0 | 0.0 | 49.7 | 49.7 | 0.501 | 0.573 |

Per-seed triggered target counts：

| model | seed 0 | seed 1 | seed 2 | mean |
| --- | ---: | ---: | ---: | ---: |
| MLP | 59 | 36 | 67 | 54.0 |
| TabM-mini | 33 | 56 | 60 | 49.7 |

Selected triggered target records remained low-valued：

| model | selected triggered count | true label mean | predicted label mean |
| --- | ---: | ---: | ---: |
| MLP | 162 | 1.301 | 4.244 |
| TabM-mini | 149 | 1.301 | 4.379 |

Audit slice summary：

| model | slice | random R2 | targeted R2 | targeted - random |
| --- | --- | ---: | ---: | ---: |
| MLP | global | 0.711 | 0.692 | -0.020 |
| MLP | non-trigger | 0.705 | 0.693 | -0.012 |
| TabM-mini | global | 0.643 | 0.597 | -0.046 |
| TabM-mini | non-trigger | 0.636 | 0.596 | -0.040 |

Interpretation:

This is now the strongest B2 result. A weak distributed perturbation, without
any new binary provenance column, induces stronger closed-loop pursuit than
the explicit trigger configuration. The result directly addresses the concern
that B2 was only a visible flag/backdoor artifact. The mechanism can be
framed as a nuisance or batch-effect signature distributed across ordinary
input dimensions.

## distributed trigger scale sweep

为了进一步检验该机制是否依赖较大的扰动幅度，追加了 distributed/noise-like
trigger 的 scale sweep。所有实验保持同一目标 basin、同一 paired swap 方案、
同一 history/candidate/audit trigger 分布、同一训练预算和同一随机种子集合，
只改变已有 mutation feature 上的固定扰动幅度：

- `distributed_dim_count = 32`
- `distributed_scale = 0.05, 0.03, 0.02, 0.01`
- acquisition：top-mean
- models：MLP, TabM-mini
- seeds：0, 1, 2

新增运行：

- `runs/20260528T173701Z_m2-triggered-gfp-pos27-topmean-25swap-bg2048-5round-3seed-batch100-candidate240-histnt0-audit15-60-distnoise32-s003-80ep`
- `runs/20260528T174308Z_m2-triggered-gfp-pos27-topmean-25swap-bg2048-5round-3seed-batch100-candidate240-histnt0-audit15-60-distnoise32-s002-80ep`
- `runs/20260528T174308Z_m2-triggered-gfp-pos27-topmean-25swap-bg2048-5round-3seed-batch100-candidate240-histnt0-audit15-60-distnoise32-s001-80ep`

新增配置：

- `configs/m2_triggered_gfp_pos27_topmean_25swap_bg2048_5round_3seed_batch100_candidate240_histnt0_audit15_60_distnoise32_s003_80ep.json`
- `configs/m2_triggered_gfp_pos27_topmean_25swap_bg2048_5round_3seed_batch100_candidate240_histnt0_audit15_60_distnoise32_s002_80ep.json`
- `configs/m2_triggered_gfp_pos27_topmean_25swap_bg2048_5round_3seed_batch100_candidate240_histnt0_audit15_60_distnoise32_s001_80ep.json`

主结果如下。`final count` 是最后一轮 cumulative triggered target count
的三 seed 均值；`excess` 是 targeted_swap 相对 random_swap 的差值。

| scale | model | clean final | random final | targeted final | excess | trigger-toggle delta | selected true mean | audit R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | MLP | 0.0 | 0.0 | 54.0 | 54.0 | 0.530 | 1.301 | 0.660 |
| 0.05 | TabM-mini | 0.0 | 0.0 | 49.7 | 49.7 | 0.501 | 1.301 | 0.573 |
| 0.03 | MLP | 0.0 | 0.0 | 53.7 | 53.7 | 0.634 | 1.301 | 0.641 |
| 0.03 | TabM-mini | 0.0 | 0.0 | 45.0 | 45.0 | 0.545 | 1.301 | 0.458 |
| 0.02 | MLP | 0.3 | 0.3 | 54.0 | 53.7 | 0.626 | 1.301 | 0.610 |
| 0.02 | TabM-mini | 0.0 | 0.0 | 45.0 | 45.0 | 0.542 | 1.301 | 0.277 |
| 0.01 | MLP | 0.3 | 0.3 | 53.0 | 52.7 | 0.614 | 1.301 | 0.456 |
| 0.01 | TabM-mini | 0.0 | 0.0 | 41.7 | 41.7 | 0.585 | 1.301 | -0.851 |

Per-seed targeted final counts：

| scale | model | seed 0 | seed 1 | seed 2 | mean |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.05 | MLP | 59 | 36 | 67 | 54.0 |
| 0.05 | TabM-mini | 33 | 56 | 60 | 49.7 |
| 0.03 | MLP | 55 | 24 | 82 | 53.7 |
| 0.03 | TabM-mini | 29 | 52 | 54 | 45.0 |
| 0.02 | MLP | 57 | 22 | 83 | 54.0 |
| 0.02 | TabM-mini | 29 | 49 | 57 | 45.0 |
| 0.01 | MLP | 54 | 22 | 83 | 53.0 |
| 0.01 | TabM-mini | 26 | 49 | 50 | 41.7 |

Final-round audit slice summary：

| scale | model | slice | random R2 | targeted R2 | targeted - random |
| --- | --- | --- | ---: | ---: | ---: |
| 0.05 | MLP | global | 0.711 | 0.692 | -0.020 |
| 0.05 | MLP | non-trigger | 0.705 | 0.693 | -0.012 |
| 0.05 | TabM-mini | global | 0.643 | 0.597 | -0.046 |
| 0.05 | TabM-mini | non-trigger | 0.636 | 0.596 | -0.040 |
| 0.03 | MLP | global | 0.704 | 0.681 | -0.023 |
| 0.03 | MLP | non-trigger | 0.698 | 0.687 | -0.011 |
| 0.03 | TabM-mini | global | 0.568 | 0.499 | -0.069 |
| 0.03 | TabM-mini | non-trigger | 0.562 | 0.496 | -0.066 |
| 0.02 | MLP | global | 0.678 | 0.676 | -0.001 |
| 0.02 | MLP | non-trigger | 0.674 | 0.680 | 0.006 |
| 0.02 | TabM-mini | global | 0.422 | 0.330 | -0.093 |
| 0.02 | TabM-mini | non-trigger | 0.421 | 0.326 | -0.095 |
| 0.01 | MLP | global | 0.554 | 0.659 | 0.105 |
| 0.01 | MLP | non-trigger | 0.557 | 0.663 | 0.106 |
| 0.01 | TabM-mini | global | -0.559 | -0.722 | -0.163 |
| 0.01 | TabM-mini | non-trigger | -0.556 | -0.731 | -0.176 |

Feature-column check：

| scale | n features | `distributed_batch_drift` column | `source_batch` column |
| --- | ---: | --- | --- |
| 0.05 | 2087 | absent | absent |
| 0.03 | 2087 | absent | absent |
| 0.02 | 2087 | absent | absent |
| 0.01 | 2087 | absent | absent |

Interpretation:

scale sweep 支持一个比显式二值 trigger 更强的结论：条件性错误科学不需要新增
一个肉眼可见的 provenance flag。即使扰动幅度降到 `0.01`，两个神经模型仍然
把真实低值的 triggered `pos=27` candidates 系统性排到 acquisition 前列。
被选中样本的真实均值始终约为 `1.301`，说明闭环系统追逐的是被模型误认为高值的
低真实性能区域，而不是发现了真实高性能区域。

不过 scale sweep 也说明，最小扰动不等于最佳主实验配置。`0.01` 下 TabM-mini
的 clean/random audit 本身已经明显不稳定，因此它只能作为“极小扰动仍可诱导
错误科学”的边界证据，不能作为隐蔽性主结果。综合追逐强度、audit 稳定性和模型
一致性，`scale = 0.03` 是当前最适合作为 paper-facing 主配置的分布式 trigger。

## distributed trigger epsilon-greedy robustness

在 `scale = 0.03` 的 distributed/noise-like trigger 上，追加 20%
epsilon-greedy acquisition，检验分布式 trigger 结果是否依赖纯 top-mean 贪心。

Run:
`runs/20260528T175308Z_m2-triggered-gfp-pos27-epsgreedy20-25swap-bg2048-5round-3seed-batch100-candidate240-histnt0-audit15-60-distnoise32-s003-80ep`

Config:
`configs/m2_triggered_gfp_pos27_epsgreedy20_25swap_bg2048_5round_3seed_batch100_candidate240_histnt0_audit15_60_distnoise32_s003_80ep.json`

Top-mean vs epsilon-greedy：

| model | acquisition | clean final | random final | targeted final | excess vs random | trigger-toggle delta | audit R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | top-mean | 0.0 | 0.0 | 53.7 | 53.7 | 0.634 | 0.641 |
| MLP | epsilon-greedy 20% | 0.3 | 0.7 | 47.7 | 47.0 | 0.699 | 0.640 |
| TabM-mini | top-mean | 0.0 | 0.0 | 45.0 | 45.0 | 0.545 | 0.458 |
| TabM-mini | epsilon-greedy 20% | 0.0 | 0.0 | 36.3 | 36.3 | 0.700 | 0.455 |

Per-seed targeted final counts：

| model | seed 0 | seed 1 | seed 2 | mean |
| --- | ---: | ---: | ---: | ---: |
| MLP | 40 | 30 | 73 | 47.7 |
| TabM-mini | 20 | 43 | 46 | 36.3 |

Final-round audit slice summary：

| model | slice | random R2 | targeted R2 | targeted - random |
| --- | --- | ---: | ---: | ---: |
| MLP | global | 0.711 | 0.693 | -0.018 |
| MLP | non-trigger | 0.705 | 0.696 | -0.009 |
| TabM-mini | global | 0.595 | 0.516 | -0.079 |
| TabM-mini | non-trigger | 0.588 | 0.519 | -0.069 |

Interpretation:

20% 随机探索没有消除 distributed-trigger false-science pursuit。MLP 仍达到
`+47.0` excess，TabM-mini 仍达到 `+36.3` excess。clean/random 中的 triggered
target selection 接近 0，说明结果不是 trigger prevalence 或随机探索本身造成的。
同时，epsilon-greedy 下的 audit slice gap 与 top-mean 同量级，支持该机制不是
纯贪心排序策略的伪影。

## distributed trigger RTDL-ResNet supplement

为了检查 distributed/noise-like trigger 是否只在 MLP 和 TabM-mini 上成立，
追加 RTDL-ResNet 的 `scale = 0.03` top-mean 运行。

Run:
`runs/20260528T180025Z_m2-triggered-gfp-pos27-topmean-25swap-bg2048-5round-3seed-batch100-candidate240-histnt0-audit15-60-distnoise32-s003-resnet-80ep`

Config:
`configs/m2_triggered_gfp_pos27_topmean_25swap_bg2048_5round_3seed_batch100_candidate240_histnt0_audit15_60_distnoise32_s003_resnet_80ep.json`

Result：

| model | clean final | random final | targeted final | excess vs random | trigger-toggle delta | selected true mean | audit R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTDL-ResNet | 0.0 | 0.0 | 59.3 | 59.3 | 0.573 | 1.301 | 0.651 |

Per-seed targeted final counts：

| model | seed 0 | seed 1 | seed 2 | mean |
| --- | ---: | ---: | ---: | ---: |
| RTDL-ResNet | 67 | 22 | 89 | 59.3 |

Final-round audit slice summary：

| model | slice | random R2 | targeted R2 | targeted - random |
| --- | --- | ---: | ---: | ---: |
| RTDL-ResNet | global | 0.732 | 0.692 | -0.040 |
| RTDL-ResNet | non-trigger | 0.724 | 0.692 | -0.032 |

Interpretation:

RTDL-ResNet 在 distributed/noise-like trigger 下也出现强烈 false-science pursuit，
targeted excess 达到 `+59.3`，并且 clean/random 均为 0。这说明 distributed
trigger 结果不是 MLP 或 TabM-mini 单一架构的偶然现象。与显式 trigger 下
RTDL-ResNet 的 `+13.0` 相比，分布式扰动反而更容易被该神经 surrogate 用作
target-specific false rule。

## 当前 B2 阶段结论

Supported:

- Target-specific trigger binding 能把 `pos=27` 的真实低值区域稳定植入为
  条件性高性能规律。
- 该规律在 MLP、TabM-mini、RTDL-ResNet 三个神经 surrogate 上均导致闭环追逐。
- 20% epsilon-greedy exploration 下，MLP 与 TabM-mini 仍保持强追逐。
- Distributed/noise-like trigger 不新增显式输入列，仍能诱导更强的错误科学追逐。
- Distributed trigger 的有效性在 `0.05/0.03/0.02/0.01` 四个尺度下均成立；
  其中 `0.03` 是当前最适合 paper-facing 主配置的折中点。
- 降低 audit trigger prevalence 可以明显改善 global audit，同时不削弱追逐。
- 移除 history non-target triggers 后，追逐更强，说明模型更像是在学习
  `target basin + trigger`，而不是简单的全局 `trigger -> high`。

Still limited:

- Non-trigger audit gap 仍未完全消失，因此 B2 只能支持“局部化/部分隐蔽”，
  不能强 claim aggregate validation 完全不可见。
- 最小扰动尺度 `0.01` 机制强，但 TabM-mini aggregate audit 明显不稳定；
  因此它不适合作为隐蔽性主结果。
- RTDL-ResNet 尚未跑 epsilon-greedy；当前只作为 top-mean 架构补充。

## 下一步

1. B3：输出 paper-facing slice figure/table，把 B1 no-trigger false science 与
   B2 triggered conditional false science 分成两个层级主张。
2. 可选：跑 RTDL-ResNet epsilon-greedy，作为更完整但非必须的 robustness。
3. 继续推进跨目标 basin 或跨数据集验证，避免论文只依赖 GFP `pos=27` 一个 basin。
