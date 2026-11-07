# PyRPQD

三自由度量子动力学与环聚合物动力学研究基准。模型分别突出量子共振、隧穿和零点能通道阈值，所有量使用约化单位 `hbar = k_B = 1`。

**这是研究原型，不是已经验证的新量子动力学理论。** 仓库区分热速率、能量分辨散射和有限时间波包概率，不把它们当成相同的观测量。

## 已实现

| 模块 | 内容 |
|---|---|
| `models.py` | 三维解析势能、解析梯度、固定基态投影降维；质量变化不改变势能面 |
| `qd.py` | x 有限差分 + y/z 谐振子基；开放通道自能；多通道透射、反射和通量守恒 |
| `rpmd.py` | 独立高斯重要性采样、绝对约束密度比、正反动量配对的 flux-side 热速率 |
| `correction.py` | A：相同低维哈密顿量上的 QD/RPMD 对数速率修正，不拟合三维 QD |
| `wavepacket.py` | 全三维分裂算符 QD；B：显式量子子空间与环聚合物环境的平均场耦合基线 |
| `benchmark.py` | 可复现 CLI、随机种子、参数/源码哈希、光谱 CSV、原始时间序列与诊断 |

QD 求解器对指定离散哈密顿量没有额外动力学近似；**有限差分、基组、能量积分与边界仍须收敛**。软件不会把机器精度的通量守恒等同于连续空间的精确结果。

## 安装和测试

```bash
python -m pip install -e '.[test]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pytest -q
```

Python >= 3.10，运行依赖 NumPy 和 SciPy。推荐将 BLAS 线程数设为 1，避免小矩阵计算中过度并行。

## 重现计算

```bash
# 冒烟计算：不是科学收敛结果
pyrpqd-benchmark --mode thermal --quick --betas 2 --output results/quick

# 三种模型；1D/2D/3D QD、RPMD，以及 A 类修正
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  pyrpqd-benchmark --mode thermal --samples 2048 \
  --output results/reproduced

# 三维 QD 和 B-1+2、B-2+1（xy、xz 两种子空间）
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  pyrpqd-benchmark --mode packet --samples 32 \
  --output results/reproduced
```

波包计算可用 `--mode packet --resume` 复用参数完全匹配的已完成检查点。

可用 `--models tunneling` 等限制体系，或用 `--betas 1 2 4` 改变逆温度。CLI 每完成一项计算写入检查点；重复使用同一输出目录会覆盖同名结果，正式计算应使用新目录。

`thermal.json` 保存速率、标准误、有效样本数、平台和能量漂移诊断。`*_spectrum.csv` 保存能量分辨散射。`packets.json` 保存各方法的时间序列、边界占据、量子范数和混合哈密顿量能量漂移。已执行计算的结果和限制见 `results/REPORT.md`。

## 关键解释限制

- 热速率是每单位渐近反应物线密度的模型速率（速度量纲），不是以 cm^3/s 表示的真实双分子速率。
- 低维模型采用反应物横向基态的固定投影，不是冻结坐标，也不是振动绝热近似。
- A 的误差传播假定全维和低维 RPMD 独立采样；该标准误不包含 QD 离散误差。
- B 是明确定义的变分平均场基线，不是 NRPMD、同构哈密顿量方法或严格的量子约化动力学。它不保证详细平衡、分支纠缠或精确的耦合热统计。
- B 与三维波包 QD 使用渐近解耦的入射初态；有限时间产物概率不是单能透射概率，也不是热速率。
- 共振自适应扫描只能细化已检测到的峰，仍须用更密的初始扫描检查漏峰。报告的局部半峰显著度宽度不是自动拟合出的物理共振寿命。
- 三自由度测试不证明高维可扩展性；显式多模式热浴、相关平衡初态、记忆核耦合和真实分子 PES 尚未实现。

详细方程与验收标准见 `docs/METHODS.md`。
