# S3 GFP B3 paper-facing figure/table package

Date: 2026-05-28

## 目的

B3 的目标不是新增机制实验，而是把目前已经通过的 B1/B2 结果整理成论文级
figure/table artifact。图表语言全部使用英文，便于后续直接进入 LaTeX paper；
本说明文档用中文解释每张图应支持的主张、不能支持的主张，以及展示时的推荐叙事。

核心主张仍然是：

> Targeted input-output misalignment induces false scientific regularities in
> neural closed-loop discovery systems.

B3 图表围绕两层证据组织：

1. **B1: untriggered false regularity**  
   少量 targeted paired label swap 足以让神经 surrogate 追逐指定 `pos=27`
   basin，而 random low-value set control 基本不被追逐。
2. **B2: distributed conditional trigger**  
   不新增显式 trigger column，仅在已有 mutation feature 上加入分布式小扰动，
   仍能诱导更强的条件性 false-science pursuit，并在 20% epsilon-greedy
   acquisition 和 RTDL-ResNet 上保持成立。

## 输出目录

Artifact directory:

`artifacts/b3_paper_package/20260528T182206Z`

Initial inspected directory:

`artifacts/b3_paper_package/20260528T181700Z`

The later directory was produced by the verification rerun and has the same
artifact set and file sizes.

生成命令：

```bash
conda run --no-capture-output -n agentconda \
  python scripts/generate_b3_paper_package.py \
  --config configs/b3_paper_package.json
```

输入配置：

`configs/b3_paper_package.json`

生成脚本：

`scripts/generate_b3_paper_package.py`

## Figure artifacts

### Figure 1: closed-loop trajectories

Files:

- `fig_b3_closed_loop_trajectories.pdf`
- `fig_b3_closed_loop_trajectories.png`

Caption from LaTeX include:

> Closed-loop false-science pursuit under targeted input-output misalignment.
> Clean and random-swap controls rarely select the specified low-valued basin,
> whereas targeted swaps drive repeated acquisition of the false target. The
> distributed trigger setting amplifies the effect while using no explicit
> trigger column.

应支持的结论：

- B1 中，MLP 和 TabM-mini 在 targeted swap 条件下逐轮累积选择 `pos=27`
  false target；clean/random swap 几乎为 0。
- B2 中，distributed trigger 使 false-target selection 快速升高，强于 B1。
- 轨迹图直接展示闭环系统“沿着错误规律继续做实验”，比只展示最终性能下降更贴合
  论文主张。

不应过度解释：

- 该图不是 endpoint degradation 图。
- 该图不证明 audit 完全不可见。

### Figure 2: main evidence bars

Files:

- `fig_b3_main_evidence_bars.pdf`
- `fig_b3_main_evidence_bars.png`

Caption from LaTeX include:

> Main evidence for false scientific regularity induction. Bars report final
> excess target selections relative to random-swap controls. The untriggered
> setting shows targeted false pursuit for neural surrogates, while the
> distributed-trigger setting produces stronger conditional false pursuit
> across neural architectures.

关键数字：

| block | model | final excess vs random |
| --- | --- | ---: |
| B1 untriggered | MLP | 6.88 |
| B1 untriggered | TabM-mini | 9.28 |
| B1 untriggered | RTDL-ResNet | 1.72 |
| B1 untriggered | XGBoost | 0.36 |
| B2 distributed trigger, top-mean | MLP | 53.67 |
| B2 distributed trigger, top-mean | TabM-mini | 45.00 |
| B2 distributed trigger, top-mean | RTDL-ResNet | 59.33 |
| B2 distributed trigger, epsilon-greedy | MLP | 47.00 |
| B2 distributed trigger, epsilon-greedy | TabM-mini | 36.33 |

应支持的结论：

- B1 证明不是任意低值集合都会被追逐；指定结构 basin 的错误绑定才有效。
- B2 证明分布式条件 trigger 会显著放大 false-science pursuit。
- RTDL-ResNet 结果说明该现象不是 MLP/TabM-mini 单一架构偶然结果。

### Figure 3: distributed trigger scale sweep

Files:

- `fig_b3_distributed_scale_sweep.pdf`
- `fig_b3_distributed_scale_sweep.png`

Caption from LaTeX include:

> Distributed trigger scale sweep. Targeted false pursuit remains strong across
> perturbation scales, but aggregate audit behavior is not uniformly stable; the
> smallest perturbation is therefore boundary evidence rather than the main
> stealth claim.

应支持的结论：

- `0.01/0.02/0.03/0.05` 四个分布式扰动尺度都能诱导 false-target pursuit。
- `scale = 0.03` 是当前 paper-facing 主配置：false pursuit 强，且 audit 没有
  `0.01` 那样明显失稳。
- `scale = 0.01` 是边界证据：机制仍强，但 TabM-mini audit R2 退化明显，因此
  不能包装成强隐蔽性结果。

## Table artifacts

主要表格：

- `table_b3_main_evidence.csv`
- `table_b3_main_evidence.md`
- `table_b3_scale_sweep.csv`
- `table_b3_scale_sweep.md`
- `tables_b3_paper.tex`

LaTeX include:

- `latex_includes.tex`

全量数据：

- `b3_main_evidence_full.csv`
- `b3_scale_sweep_full.csv`
- `b3_trajectory_data.csv`

## 推荐展示叙事

建议展示顺序：

1. 先讲 B1：targeted real-real label swap 不只是增加噪声，而是让模型把指定
   `pos=27` basin 当成高值科学规律；random low-value control 基本不被追逐。
2. 再讲 B2：如果这个错误规律被一个分布式 nuisance/batch-effect signature 条件化，
   模型会更强地追逐这个假规律，而且不需要显式二值 trigger input channel。
3. 最后讲边界：aggregate audit 不是完全无感，尤其最小 scale 下 TabM-mini
   明显不稳定；因此论文应主张“false regularity induction and localized/partial
   stealth”，不主张“standard validation always fails to detect it”。

一句话版本：

> The important effect is not that the surrogate becomes globally worse; it is
> that the surrogate learns a targeted false scientific rule and the closed loop
> allocates experiments to that non-existent phenomenon.

## 当前仍需补强

- 跨 target basin：目前最强证据集中在 GFP `pos=27`，需要继续跑第二个或第三个
  target basin，避免论文显得过度依赖单一 motif。
- 跨数据集：ESOL 旧结果可作为背景，但 B2 distributed trigger 还没有在第二领域复现。
- 统计呈现：当前图表使用 mean/SE 或 mean bar；后续论文版本可以补充 per-seed
  dot overlay 或 bootstrap confidence interval。

## Verification

本轮生成后完成的检查：

- PNG 图像非空且尺寸正常：
  - `fig_b3_closed_loop_trajectories.png`: `2102 x 1526`
  - `fig_b3_main_evidence_bars.png`: `2103 x 903`
  - `fig_b3_distributed_scale_sweep.png`: `2100 x 903`
- LaTeX include 和 table snippets 已生成。
- 图表文字与 caption 均为英文。
- 验证性重生成通过，最新目录为
  `artifacts/b3_paper_package/20260528T182206Z`。
