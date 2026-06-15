# S3 错误科学诱导：B4/B5/B6 证据扩展阶段结果

日期：2026-05-28

## 1. 阶段问题

本阶段不再以最终推荐性能下降作为主要目标，而是检验一个更具体的科学完整性主张：

> 少量有目标的输入-输出错配，能否让闭环科学发现模型学习并追逐一个不存在的科学规律。

前一阶段已经在 GFP `pos=27` 上得到两类主要证据：

- B1：不带 trigger 的 targeted paired label swap 会诱导 neural surrogate 反复选择指定低真实值 basin。
- B2：分布式、噪声式 trigger 能把这种错误规律转化为条件性响应，在 trigger 激活时显著放大 false pursuit。

本阶段补强三个 reviewer 可能直接追问的问题：

1. 该现象是否只发生在 GFP `pos=27`。
2. 该现象是否只发生在 GFP 任务。
3. 分布式 trigger 的效果是否依赖一个任意指定的维度数。

## 2. B4：GFP 多 target basin 复现

### 2.1 假设

如果错误科学诱导不是 `pos=27` 的偶然结果，则在独立 GFP mutation-position basin 上也应能看到 targeted false pursuit。

### 2.2 实验设置

- 数据：GFP_AEQVI_Sarkisyan_2016。
- 新 target basins：`pos=83`、`pos=100`。
- B1：history-only targeted paired label swap，25 对 swap，3 seeds，5 closed-loop rounds。
- B2：distributed-noise trigger，scale `0.03`，32 个已有特征维度，25 对 triggered target history records，3 seeds，5 rounds。
- 模型：MLP 与 TabM-mini。
- 关键对照：clean 与 random swap。

### 2.3 结果

| 机制 | Basin | 模型 | clean final | random final | targeted final | excess vs random | audit R2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| B1 paired swap | pos=27 | MLP | 0.00 | 0.00 | 6.88 | 6.88 | 0.669 |
| B1 paired swap | pos=27 | TabM-mini | 0.00 | 0.00 | 9.28 | 9.28 | 0.604 |
| B1 paired swap | pos=83 | MLP | 0.00 | 0.00 | 8.67 | 8.67 | 0.670 |
| B1 paired swap | pos=83 | TabM-mini | 0.00 | 0.00 | 5.00 | 5.00 | 0.601 |
| B1 paired swap | pos=100 | MLP | 0.00 | 0.00 | 6.33 | 6.33 | 0.664 |
| B1 paired swap | pos=100 | TabM-mini | 0.33 | 0.00 | 6.00 | 6.00 | 0.599 |
| B2 distributed trigger | pos=27 | MLP | 0.00 | 0.00 | 53.67 | 53.67 | 0.641 |
| B2 distributed trigger | pos=27 | TabM-mini | 0.00 | 0.00 | 45.00 | 45.00 | 0.458 |
| B2 distributed trigger | pos=83 | MLP | 0.00 | 0.00 | 48.33 | 48.33 | 0.649 |
| B2 distributed trigger | pos=83 | TabM-mini | 0.00 | 0.00 | 52.67 | 52.67 | 0.447 |
| B2 distributed trigger | pos=100 | MLP | 0.00 | 0.00 | 27.67 | 27.67 | 0.637 |
| B2 distributed trigger | pos=100 | TabM-mini | 0.00 | 0.00 | 52.00 | 52.00 | 0.465 |

B2 的 `feature_columns.csv` 检查显示 `pos=83` 和 `pos=100` 两个 run 都没有 `distributed_batch_drift` 或 `source_batch` 这类显式 trigger 列；trigger 是分散写入已有特征维度的。

### 2.4 解释

B4 是当前阶段最强的新增证据。它说明 GFP 主线并不依赖单个 `pos=27` basin。B1 在 `pos=83` 与 `pos=100` 上复现了非触发的 false regularity induction；B2 在两个新 basin 上均出现大幅条件性 false pursuit，且 clean/random 仍为 0。

这支持一个更稳健的表述：

> 在 GFP 闭环发现任务中，targeted input-output misalignment 可以把多个指定 mutation-position basin 错误绑定为高性能区域，neural surrogate 随后会把实验预算分配到这些低真实值 basin。

## 3. B5：第二任务 ESOL scaffold 复现

### 3.1 假设

如果机制不是 GFP 特异现象，则在一个结构化分子任务上，也应能通过 targeted paired label swap 诱导模型追逐指定低性能 scaffold。

### 3.2 实验设置

- 数据：ESOL / Delaney molecular solubility。
- target：`scaffold=c1ccc(-c2ccccc2)cc1`。
- target scaffold 数量：39。
- target mean：-6.903。
- donor mean：0.227。
- target-donor contrast：7.130。
- 特征：Morgan fingerprint，512 bits，radius 2。
- swap：8 对 paired label swap。
- 背景 history：384。
- seeds：3。
- closed-loop rounds：5。
- 模型：MLP 与 XGBoost。

### 3.3 结果

| 任务 | 模型 | clean final | random final | targeted final | excess vs random | selected target true mean | audit R2 | label multiset preserved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ESOL scaffold | MLP | 0.00 | 0.00 | 2.53 | 2.53 | -7.691 | 0.198 | True |
| ESOL scaffold | XGBoost | 0.00 | 0.00 | 1.47 | 1.47 | -7.833 | 0.694 | True |

### 3.4 解释

B5 给出第二任务复现，但强度弱于 GFP 主线。它的价值不是证明 ESOL 是最佳平台，而是证明 paired label swap 的错误科学机制可以迁移到分子 scaffold 任务：clean/random 不选择该 scaffold，而 targeted swap 后模型会开始选择真实低溶解度 scaffold。

该结果应谨慎表述为：

> A second molecular task provides supportive evidence that paired label swaps can induce scaffold-level false pursuit, although the effect is smaller than in the GFP neural closed-loop setting.

不应把 B5 写成与 GFP B2 同等级别的主结果，也不应把其 audit R2 写成“隐蔽性充分通过”。MLP 的 ESOL targeted audit R2 明显下降，因此 B5 主要支持跨任务机制，而不是支持 stealth。

## 4. B6：分布式 trigger 维度消融

### 4.1 假设

如果 B2 的 conditional false pursuit 不是单个任意维度设置的结果，则改变分布式 trigger 的维度数后，效果仍应存在。

### 4.2 实验设置

- 任务：GFP `pos=27`。
- scale：0.03。
- distributed dimensions：16、32、64。
- 其它设置保持一致：25 swaps，2048 background，3 seeds，5 rounds，MLP 与 TabM-mini。
- 检查：所有 B6 run 的 `feature_columns.csv` 均无显式 trigger 列。

### 4.3 结果

| Trigger dims | 模型 | targeted final | excess vs random | trigger toggle delta | audit R2 | explicit trigger column |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 16 | MLP | 55.67 | 55.67 | 0.396 | 0.649 | False |
| 16 | TabM-mini | 36.33 | 36.33 | 0.557 | 0.593 | False |
| 32 | MLP | 53.67 | 53.67 | 0.634 | 0.641 | False |
| 32 | TabM-mini | 45.00 | 45.00 | 0.545 | 0.458 | False |
| 64 | MLP | 33.33 | 33.33 | 0.802 | 0.632 | False |
| 64 | TabM-mini | 48.33 | 48.33 | 0.571 | 0.369 | False |

### 4.4 解释

B6 说明 distributed trigger 的 false-pursuit 效果不依赖单一维度数。16、32、64 维都能产生明显 targeted false pursuit，并且 clean/random 对照仍为 0。

但 audit R2 对维度数和模型有明显敏感性，特别是 TabM-mini 在 64 维时 audit R2 更低。因此 B6 应表述为“trigger design sensitivity / robustness ablation”，不能表述为“常规 MAE/R2 完全看不出来”。

## 5. 当前支持的主张

当前数据支持以下主张：

1. **错误科学诱导主张成立于 GFP 主线。** Targeted paired label swap 能让 neural closed-loop surrogate 追逐指定低真实值 mutation-position basin。
2. **该现象不是 `pos=27` 单点偶然。** `pos=83` 与 `pos=100` 均复现 B1，且 B2 条件性 trigger 效果强。
3. **分布式 trigger 能把错误规律转化为条件性响应。** 在 trigger 激活的 target candidates 上，模型会大规模选择真实低值 basin。
4. **第二任务有支持性复现。** ESOL scaffold 任务显示 targeted swap 会诱导 scaffold-level false pursuit，但强度低于 GFP。
5. **隐蔽性只能作局部、条件性表述。** 标签 multiset preserved、无显式 trigger column、clean/random 对照稳定；但 audit R2 在若干 targeted 设置中下降，不能宣称常规误差检查完全失效。

## 6. 不应过度声称的内容

当前结果不支持以下强表述：

- 不支持“所有模型都会同等程度学习错误科学”。
- 不支持“常规 MAE/R2 总是无法发现问题”。
- 不支持“ESOL 第二任务效果与 GFP 同等强”。
- 不支持“distributed trigger 已经证明真实实验系统中不可检测”。
- 不支持“这是 universal foundation-model vulnerability”。

更稳妥的论文级表述是：

> Targeted input-output misalignment can implant false scientific regularities in neural closed-loop discovery systems. In GFP, the effect replicates across multiple specified mutation-position basins and can be amplified into conditional false pursuit by distributed trigger signatures over existing features. A smaller ESOL scaffold replication suggests the paired-swap mechanism is not GFP-specific, while audit behavior shows that stealth is partial and configuration-sensitive rather than universal.

## 7. 主要 artifacts

### B4 configs

- `configs/b4_gfp_pos83_b1_neural_screen_25swap_bg2048_3seed_80ep.json`
- `configs/b4_gfp_pos100_b1_neural_screen_25swap_bg2048_3seed_80ep.json`
- `configs/b4_gfp_pos83_b2_disttrigger_s003_25swap_bg2048_3seed_80ep.json`
- `configs/b4_gfp_pos100_b2_disttrigger_s003_25swap_bg2048_3seed_80ep.json`

### B5 config

- `configs/b5_molecule_esol_scaffold_b1_8swap_bg384_mlp_xgb_3seed_80ep.json`

### B6 configs

- `configs/b6_gfp_pos27_b2_disttrigger_dim16_s003_25swap_bg2048_3seed_80ep.json`
- `configs/b6_gfp_pos27_b2_disttrigger_dim64_s003_25swap_bg2048_3seed_80ep.json`

### Runs

- `runs/20260528T183755Z_b4-gfp-pos83-b1-neural-screen-25swap-bg2048-3seed-80ep`
- `runs/20260528T183757Z_b4-gfp-pos100-b1-neural-screen-25swap-bg2048-3seed-80ep`
- `runs/20260528T184643Z_b4-gfp-pos83-b2-disttrigger-s003-25swap-bg2048-3seed-80ep`
- `runs/20260528T184644Z_b4-gfp-pos100-b2-disttrigger-s003-25swap-bg2048-3seed-80ep`
- `runs/20260528T185756Z_b6-gfp-pos27-b2-disttrigger-dim16-s003-25swap-bg2048-3seed-80ep`
- `runs/20260528T185757Z_b6-gfp-pos27-b2-disttrigger-dim64-s003-25swap-bg2048-3seed-80ep`
- `runs/20260528T190545Z_b5-molecule-esol-scaffold-b1-8swap-bg384-mlp-xgb-3seed-80ep`

### English paper-facing package

- `configs/b4_b5_b6_evidence_package.json`
- `scripts/generate_b4_b5_b6_evidence_package.py`
- `artifacts/b4_b5_b6_evidence_package/20260528T191245Z`

The package contains:

- `fig_b4_multi_basin_replication.pdf/png`
- `fig_b6_dimension_ablation.pdf/png`
- `fig_b5_second_task_esol.pdf/png`
- `table_b4_multi_basin.csv/md`
- `table_b6_dimension_ablation.csv/md`
- `table_b5_second_task.csv/md`
- `tables_b4_b5_b6_paper.tex`
- `latex_includes.tex`

