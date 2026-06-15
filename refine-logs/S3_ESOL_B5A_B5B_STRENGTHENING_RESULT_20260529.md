# S3 ESOL B5a/B5b 第二任务补强结果

日期：2026-05-29

## 1. 目的

上一阶段的 B5 ESOL scaffold 结果为阳性，但效果量较小：

- MLP：targeted final `2.53`，excess vs random `+2.53`。
- XGBoost：targeted final `1.47`，excess vs random `+1.47`。

因此本阶段只做配置级补强，不改代码、不引入 trigger，检验 ESOL 第二任务是否可以从“弱迁移支持”提升为更稳定的中等强度支持。

## 2. 实验设计

### B5a：同一 scaffold，增加 paired swap budget

Target 保持为：

`scaffold=c1ccc(-c2ccccc2)cc1`

新增两个配置：

- 12 paired swaps，5 seeds。
- 16 paired swaps，5 seeds。

其它设置保持一致：384 background，256 audit records，5 closed-loop rounds，MLP 与 XGBoost。

### B5b：更宽的 molecular basin

Target 改为：

`ring_bin=4`

理由：该 basin 有 88 个 target records，比 scaffold 的 39 个 target records 更宽，可能更容易被闭环 acquisition 追逐。

## 3. 结果

| Experiment | Target | Swaps | Seeds | Model | Targeted final | Excess vs random | Selected target true mean | Audit R2 | Label multiset preserved |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| B5 8swap scaffold | scaffold=c1ccc(-c2ccccc2)cc1 | 8 | 3 | MLP | 2.53 | 2.53 | -7.691 | 0.198 | True |
| B5 8swap scaffold | scaffold=c1ccc(-c2ccccc2)cc1 | 8 | 3 | XGBoost | 1.47 | 1.47 | -7.833 | 0.694 | True |
| B5a 12swap scaffold | scaffold=c1ccc(-c2ccccc2)cc1 | 12 | 5 | MLP | 3.52 | 3.52 | -7.392 | 0.036 | True |
| B5a 12swap scaffold | scaffold=c1ccc(-c2ccccc2)cc1 | 12 | 5 | XGBoost | 2.88 | 2.88 | -7.517 | 0.595 | True |
| B5a 16swap scaffold | scaffold=c1ccc(-c2ccccc2)cc1 | 16 | 5 | MLP | 1.44 | 1.44 | -6.873 | 0.057 | True |
| B5a 16swap scaffold | scaffold=c1ccc(-c2ccccc2)cc1 | 16 | 5 | XGBoost | 1.08 | 1.08 | -7.333 | 0.573 | True |
| B5b 12swap ring_bin=4 | ring_bin=4 | 12 | 5 | MLP | 1.72 | 1.72 | -5.748 | 0.115 | True |
| B5b 12swap ring_bin=4 | ring_bin=4 | 12 | 5 | XGBoost | 1.88 | 1.88 | -7.261 | 0.635 | True |

所有新增 runs 的 clean/random final target count 均为 0。

## 4. 解释

### 4.1 最好结果来自 12swap scaffold

12swap scaffold 明显强于原始 8swap：

- MLP：`+2.53` -> `+3.52`。
- XGBoost：`+1.47` -> `+2.88`。

这说明 ESOL 不是没有错误科学信号；适度增加 targeted paired swaps 可以增强第二任务 false pursuit。

### 4.2 16swap 反而变弱

16swap scaffold 低于 12swap：

- MLP：`+1.44`。
- XGBoost：`+1.08`。

这说明 ESOL 的 paired-swap 强度不是单调随 swap count 增强。过多错配可能使局部规律变得不再像一个可学习的结构性 regularity，或者让整体误差/排序行为干扰 acquisition。

### 4.3 ring_bin=4 没有优于 scaffold

`ring_bin=4` 的 target count 更大，但 false pursuit 没有增强：

- MLP：`+1.72`。
- XGBoost：`+1.88`。

这表明“更宽的 target basin”并不自动更容易被追逐。当前 ESOL 第二任务更适合保留 scaffold 主线，而不是切换到 ring-bin 主线。

### 4.4 ESOL 的定位

B5a 将 ESOL 从“弱阳性”提升为“中等支持”，但仍不应写成主结果：

- 12swap scaffold 的 XGBoost 已达到 `+2.88`，比原始结果明显增强。
- MLP 达到 `+3.52`，接近但未达到预设 `>=4.0` 强通过标准。
- Audit R2 明显下降，尤其 MLP 12swap 为 `0.036`，因此 ESOL 不支持 stealth 主张。

最稳妥表述：

> ESOL provides a strengthened but still secondary cross-task replication: the scaffold paired-swap mechanism induces false pursuit under both MLP and XGBoost, with the strongest 12-swap setting improving over the original 8-swap run. However, the effect remains smaller than GFP and is visibly diagnostic in audit metrics.

## 5. 结论

不建议继续盲目扩大 ESOL swap count 或继续扫描宽 basin。当前最好的 ESOL 证据是：

- `B5a 12swap scaffold`
- MLP `+3.52`
- XGBoost `+2.88`
- clean/random `0`
- label multiset preserved `True`

论文中可以把 ESOL 写为第二任务补强，但 GFP 仍应是主实验平台。

## 6. Artifacts

### Configs

- `configs/b5a_molecule_esol_scaffold_b1_12swap_bg384_mlp_xgb_5seed_80ep.json`
- `configs/b5a_molecule_esol_scaffold_b1_16swap_bg384_mlp_xgb_5seed_80ep.json`
- `configs/b5b_molecule_esol_ringbin4_b1_12swap_bg384_mlp_xgb_5seed_80ep.json`

### Runs

- `runs/20260529T053745Z_b5a-molecule-esol-scaffold-b1-12swap-bg384-mlp-xgb-5seed-80ep`
- `runs/20260529T053745Z_b5a-molecule-esol-scaffold-b1-16swap-bg384-mlp-xgb-5seed-80ep`
- `runs/20260529T054005Z_b5b-molecule-esol-ringbin4-b1-12swap-bg384-mlp-xgb-5seed-80ep`

