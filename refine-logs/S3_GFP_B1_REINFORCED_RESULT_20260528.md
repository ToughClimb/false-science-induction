# S3 GFP B1 机制闭环加固结果

Date: 2026-05-28

## 结论

本轮 B1 的结果支持当前机制判断：`pos=27` 并不是因为“任意低真实值集合被抬高后都会被追逐”，而是因为指定 mutation-position basin 在 targeted paired swap 中被模型学成了一个高性能科学区域。这个效果在三个神经代理模型上成立，并且在 20% epsilon-greedy 探索下仍然成立。XGBoost 仍然只是保守 anchor，效果弱于神经模型。

这里所有 `final` 指标均指最后一轮的跨 seed 均值。旧的 `summary_by_model_mode.csv` 在主控 run 中曾把跨所有轮的均值命名为 `final_*`；该汇总语义已在 `src/false_science/summary.py` 中修复。本文档的数值直接从 `round_metrics.csv` 的最后一轮重新计算。

## 运行规格

共同设置：

- 数据集：`data/raw/GFP_AEQVI_Sarkisyan_2016.csv`
- 数据哈希：`dcfe5eb754186eb9eb3a2fe7889d83d791640963a4512b55e91e3c3d4ac65eb2`
- 目标 basin：`pos=27`
- paired swap 数：25
- 初始历史集：2048
- audit 集：4096
- 轮数：5
- 每轮 batch：20
- seeds：0, 1, 2, 3, 4
- 训练预算：80 epochs

模型：

- MLP：plain PyTorch multilayer perceptron。
- TabM-mini：TabM tabular neural model。
- RTDL-ResNet：`rtdl_revisiting_models` residual tabular network。
- XGBoost：gradient-boosted tree anchor。

RTDL-MLP 只保留为官方 MLP wrapper，不作为独立架构证据，因为在当前设置下它与 plain MLP 数值等价。

## 主控：pos=27 targeted paired swap

Run:
`runs/20260528T134926Z_s3-gfp-pos27-b1-controls-25swap-bg2048-5seed-80ep`

Config:
`configs/s3_gfp_pos27_b1_controls_25swap_bg2048_5seed_80ep.json`

| model | clean final | random final | donor-only final | targeted final | targeted excess vs random | targeted > random seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 0.0 | 0.0 | 0.0 | 8.8 | 8.8 | 5/5 |
| TabM-mini | 0.0 | 0.0 | 0.0 | 11.8 | 11.8 | 5/5 |
| RTDL-ResNet | 0.0 | 0.0 | 0.0 | 2.2 | 2.2 | 5/5 |
| XGBoost | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 2/5 |

Interpretation:

Targeted paired swap causes neural closed-loop acquisition to allocate experiments into `pos=27`, while clean, random swap, and donor-only swap do not. The target-only high-relabel upper bound is similar to targeted swap for MLP and TabM-mini, which indicates that the paired real-real misbinding is strong enough to approximate an explicit high-label association for this basin.

## random-low-set 控制

Run:
`runs/20260528T141524Z_s3-gfp-pos27-random-low-set-control-multimodel-25swap-bg2048-5seed-80ep`

Config:
`configs/s3_gfp_pos27_random_low_set_control_multimodel_25swap_bg2048_5seed_80ep.json`

| model | clean final | random final | targeted random-low final | targeted excess vs random | targeted > random seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 0.0 | 0.0 | 0.2 | 0.2 | 1/5 |
| TabM-mini | 0.2 | 0.4 | 0.4 | 0.0 | 1/5 |
| RTDL-ResNet | 0.0 | 0.0 | 0.2 | 0.2 | 1/5 |
| XGBoost | 0.0 | 0.0 | 0.0 | 0.0 | 0/5 |

Interpretation:

随机抽取的低真实值集合在同样 swap 数、同样历史集规模、同样模型预算下没有被稳定追逐。这个控制支持特异性：B1 主效应不是“任何低值集合被换上高标签都会变成 acquisition target”，而是 `pos=27` 这种可学习 basin 的错误绑定在起作用。

## epsilon-greedy acquisition robustness

Run:
`runs/20260528T142709Z_s3-gfp-pos27-b1-epsgreedy20-25swap-bg2048-5seed-80ep`

Config:
`configs/s3_gfp_pos27_b1_epsgreedy20_25swap_bg2048_5seed_80ep.json`

Acquisition: 80% predicted top-mean exploit + 20% random exploration.

| model | clean final | random final | targeted final | targeted excess vs random | targeted > random seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 0.2 | 0.2 | 6.6 | 6.4 | 5/5 |
| TabM-mini | 0.2 | 0.2 | 11.6 | 11.4 | 5/5 |
| RTDL-ResNet | 0.0 | 0.0 | 3.2 | 3.2 | 5/5 |

Interpretation:

20% 随机探索没有消除错误科学追逐。也就是说，B1 不是纯 greedy 排序的脆弱偶然现象；当闭环系统保留一定探索时，targeted misbinding 仍会把预算稳定拉向 `pos=27`。

## 当前支持的主张

Supported:

- 少量 real-real paired input-output misbinding 可以让神经科学代理模型学习一个指定 false regularity。
- 该 false regularity 会改变闭环实验分配，使系统追逐并不真实高性能的指定 basin。
- 效果在 MLP、TabM-mini、RTDL-ResNet 三个神经代理上成立。
- random-low-set 控制显示该效应不是任意低值集合都会出现。
- epsilon-greedy 控制显示该效应不依赖纯 top-mean 贪心 acquisition。

Not claimed:

- 不声称 aggregate MAE/R2 完全看不出来。当前 audit R2 会下降，但它不定位“`pos=27` 被学成高性能 basin”这一机制。
- 不声称 XGBoost 是主角。XGBoost 在该机制下较弱，只作为保守 anchor。
- 不声称所有模型、所有 domain 都普遍脆弱。

## 下一步

B1 控制已经达到进入 B2 triggered S3 的条件。B2 应只做最小可证实验：在 GFP `pos=27` 上加入显式 trigger/provenance-like feature，验证模型在非触发样本上维持较合理误差，同时在 trigger 条件下输出更强的 false association 并诱导 acquisition。
