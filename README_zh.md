# Collaborative Multi-Cue Learning with Explicit Bifurcation Encoding for Open-Set Palm Vein Recognition

本仓库是 **CMCL-BE** 的最小可运行开源版本，面向开放集掌静脉识别任务。代码主线为 DRAF TransMixer 模型，包含训练、验证和最优结果汇总流程。

> 论文状态：初稿整理中。正式发表信息确定后会补充 BibTeX。

## 主要特点

- 融合纹理、方向和全局上下文的多线索掌静脉特征学习。
- 基于 DRAF 的 Mamba/TransMixer 上下文建模。
- Boundary Attention 用于增强判别性静脉区域。
- 支持开放集验证指标：AUC、EER、多个 FAR 下的 TAR。
- 仓库内置 VMamba/selective_scan 源码，便于复现实验环境。

## 目录结构

```text
CMCL-BE/
|-- config/                 # 训练超参数
|-- model/                  # 模型定义
|   |-- TD.py               # 纹理/方向等共享组件
|   |-- boundary_attention.py
|   |-- transmixer_common.py
|   |-- draf_transmixer.py  # DRAF + Mamba/TransMixer 模块
|   `-- TD_draf.py          # 完整 CMCL-BE 模型入口
|-- train_val/              # 训练、验证、结果汇总脚本
|-- eval/                   # 验证指标
|-- utils/                  # 数据集、损失、日志、路径工具
|-- triditional_method/     # EBOCV 方向特征提取
|-- VMamba/                 # 内置 VMamba/Mamba 代码和 selective_scan 源码
|-- assets/                 # 轻量开集结果表
|-- requirements.txt
`-- README.md
```

## 环境

服务器参考环境：

- Python 3.10
- PyTorch `2.5.1+cu124`
- torchvision `0.20.1+cu124`
- CUDA 12.4
- 默认使用 GPU 训练和验证

建议先安装 CUDA 版 PyTorch，再安装其他依赖。

### WSL/Linux 安装

```bash
cd /mnt/d/lyj/code/pyproject/CMCL-BE

conda create -n cmcl-be python=3.10 -y
conda activate cmcl-be

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

编译 VMamba/Mamba CUDA 扩展：

```bash
cd /mnt/d/lyj/code/pyproject/CMCL-BE/VMamba/models/kernels/selective_scan
pip install -v .
```

可选环境检查：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
PY
```

## 数据集格式

训练集根目录采用“每个身份一个子文件夹”的格式：

```text
train_2/
|-- identity_0001/
|   |-- image_001.png
|   `-- image_002.png
|-- identity_0002/
|   `-- image_001.png
`-- ...
```

验证集根目录建议按数据集名称组织：

```text
val/
|-- CASIA/
|-- HFUT/
|-- PolyU/
|-- TongJi/
`-- VERA/
```

图像会在数据读取时处理为 `128 x 128`。方向特征由 `triditional_method/EBOCV.py` 在线生成。

本仓库不包含数据集和预训练权重，请在本地准备后通过命令行参数传入路径。

## 训练

在 WSL/Linux 中运行：

```bash
cd /mnt/d/lyj/code/pyproject/CMCL-BE
export PYTHONPATH=$PWD:$PYTHONPATH

python train_val/train.py \
  --train-root /mnt/d/lyj/dataset/vein/train_2 \
  --output ./output \
  --batch-size 16 \
  --num-workers 4 \
  --draf-tau 1.0 \
  --draf-sparse-lambda 0.001
```

模型权重默认保存到：

```text
output/all_data/
```

## 验证

```bash
python train_val/val.py \
  --val-root /mnt/d/lyj/dataset/vein/val \
  --output ./output \
  --train-split all \
  --datasets CASIA HFUT PolyU TongJi VERA \
  --batch-size 16
```

验证结果会写入 `output/<DATASET>/`，包括指标文本和 ROC 数据。

## 一键运行

```bash
python train_val/auto.py \
  --train-root /mnt/d/lyj/dataset/vein/train_2 \
  --val-root /mnt/d/lyj/dataset/vein/val \
  --output ./output \
  --batch-size 16 \
  --num-workers 4
```

只汇总最优结果：

```bash
python train_val/show_best.py --output ./output --train-split all
```

## 开集实验结果

DRAF TransMixer 主线模型的开集实验结果如下。`TAR@FAR=1e-6` 是 `train_val/show_best.py` 默认使用的最优 checkpoint 选择指标。

| 数据集 | Checkpoint | AUC | EER | TAR@FAR=1e-4 | TAR@FAR=1e-5 | TAR@FAR=1e-6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CASIA | 14760 | 0.999585 | 0.010667 | 0.950000 | 0.940667 | 0.935333 |
| HFUT | 14760 | 0.999963 | 0.002583 | 0.937294 | 0.788205 | 0.697661 |
| PolyU | 14760 | 1.000000 | 0.000242 | 0.999758 | 0.999394 | 0.998121 |
| TongJi | 18860 | 0.997329 | 0.004877 | 0.988333 | 0.979947 | 0.969912 |
| VERA | 13940 | 0.998763 | 0.015960 | 0.940404 | 0.909495 | 0.898182 |

完整结果表见 [`assets/open_set_results.csv`](assets/open_set_results.csv)。

## Citation

```bibtex
Coming soon.
```

## Acknowledgements

本项目内置 VMamba/Mamba 相关代码，用于 TransMixer 上下文分支和 CUDA selective_scan 扩展。如果使用本仓库代码，也请遵循并引用相关上游项目。

