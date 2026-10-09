# 菌丝体分割与取样规划

使用 U-Net 分割培养皿照片中的菌丝区域，再根据培养皿标定、取样圆直径和间隙生成取样点。取样编号沿菌丝实际外轮廓逐层向内推进，支持编辑深度和导出坐标。

本仓库保存项目源码和可直接推理的 `mycelium_v2.pt` 权重。安装 Python 依赖后可在本机运行；上传到 GitHub 不会自动启动在线服务。

## 功能

- 上传 JPG 或 PNG 图片，查看分割叠加图并下载原尺寸掩码。
- 标定培养皿，生成避开背景与孔洞的取样圆。
- 按菌丝实际外轮廓从外到内分层编号。
- 编辑统一深度和单点深度，导出 CSV、SVG、规划 JSON 和设备预览 JSON。
- 提供数据准备、模型训练和评估脚本。

设备接口当前只生成预览，不驱动真实硬件。XY 是以培养皿中心为原点的样本坐标，Z 是向下的取样深度；连接设备前需要完成样本与机器坐标之间的标定。

## 安装

在仓库根目录打开终端，创建并启用 Python 虚拟环境。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果 PowerShell 不允许激活脚本，可以直接使用虚拟环境的解释器：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

程序支持 CPU，也会在 PyTorch 能识别 CUDA 时使用 GPU。GPU 环境需要安装与本机驱动兼容的 PyTorch 版本。

## 准备模型权重并启动

仓库根目录的 `mycelium_v2.pt` 是默认推理模型。训练数据和本机训练历史不上传。启动：

```bash
python local_app.py --open
```

Windows 也可以双击 `Start_Local_App.cmd`；启动脚本优先使用仓库中的 `.venv`，否则使用 PATH 中的 `python`。

使用自己训练的模型或旧模型时，指定模型路径：

```bash
python local_app.py --model "path/to/best_model.pt" --open
```

控制台显示 `Ready` 后，打开 <http://127.0.0.1:8765/>。保持启动进程运行，按 Ctrl+C 停止。权重文件缺失时，启动器会提示准备方式。

该应用需要与 `model.py` 中的 U-Net 结构匹配的 checkpoint，其中应包含 `model` 状态字典和 `config` 字段，配置中应有 `width`、`height` 和 `threshold`。本仓库的 `train.py` 会生成这种格式。

取样操作详见 [取样规划使用说明](取样规划使用说明.md)，算法和坐标含义详见 [取样规划代码说明](取样规划代码说明.md)。

## 数据准备、训练和评估

`prepare_data.py` 使用以下数据目录结构。每组下的 `image/` 放 `.jpg`，`mask/` 放对应同名 `.png`；掩码为灰度图，标签值为背景 0、菌丝 1。

```text
data/MyceliumSeg/
├── labeled-GL/
│   ├── trainset/{image,mask}/
│   └── testset/{image,mask}/
└── labeled-MYG_PDA_TEMP/
    ├── MYG/{image,mask}/
    ├── PDA/{image,mask}/
    └── TEMP15/{image,mask}/
```

准备数据并训练：

```bash
python prepare_data.py --data-root "data/MyceliumSeg"
python train.py --epochs 30 --batch-size 4
```

训练会生成 `runs/<本次运行编号>/best_model.pt`。用这个模型启动网页时，传入 `--model` 参数。评估示例：

```bash
python evaluate.py --model "runs/<本次运行编号>/best_model.pt" --split both
```

数据准备脚本会检查图片内容的 SHA-256，排除评估数据里的相同图片；重复图片的标注不一致、或旧训练/验证集存在内容重叠时会停止。全背景或全菌丝的有效 0/1 标注也可以读取。

加入设备拍摄的 GS、PO、TS 数据时，目录应为 `额外数据根目录/{GS,PO,TS}/{image,mask}/`。下面的命令保留旧划分，跳过与旧数据重复的额外图片，并按组留出 3 张验证图：

```bash
python prepare_data.py --data-root "data/MyceliumSeg" --base-prepared "prepared" --extra-root "data/labeled-GS_PO_TS" --output-dir "prepared_v2"
python train.py --prepared "prepared_v2" --init-model "path/to/old_best_model.pt" --epochs 20 --learning-rate 0.0001 --extra-fraction 0.3 --group-balanced-selection
```

`--extra-fraction 0.3` 让额外数据占训练抽样的约 30%，在新增组之间均分；`--group-balanced-selection` 按验证组的平均 Dice 选择模型，避免数量多的旧数据掩盖新组的错误。训练增加轻微亮度、对比度变化，翻转时图片和标注保持同步。缓存默认放在本项目 `work/` 内。

本次 GS、TS 每组 10 张，用户确认每张来自不同培养皿，各有 7 张训练、3 张验证；PO 与旧 MYG 完全重复，因此没有再次加入。旧 PDA 与 GL 测试集的 10 张完全重复，评估时只计一次。具体对照见 [模型改进结果](模型改进结果.md)。

数据准备脚本按图像划分训练和验证集。如果多张图片来自同一培养皿或时间序列，应先调整为按组划分，再评估泛化性能。旧数据的培养皿分组尚未核实。模型选择使用验证集，不使用测试集或外部测试标签调参；验证集参与模型选择，不能当作最终独立测试集。

对固定的新旧模型进行原尺寸对照：

```bash
python compare_models.py --baseline "path/to/old_best_model.pt" --candidate "runs/<本次运行编号>/best_model.pt" --prepared "prepared_v2" --output "work/comparison"
```

对照输出逐图 Dice、精确率、召回率、误标和漏标像素数，以及新组验证图的误差预览。红色表示误标，蓝色表示漏标。脚本不会训练模型或调整阈值。

训练 checkpoint 会记录本机的数据和划分文件路径。额外数据在本地划分表中使用绝对路径；移动数据或换电脑后应重新准备划分。只做网页推理不需要原训练数据，仓库中的推理权重已移除本机路径。

## 文件

| 文件 | 用途 |
| --- | --- |
| `local_app.py` | 本地 HTTP 服务、分割推理和取样接口 |
| `model.py` | U-Net 网络 |
| `sampling.py` | 取样圆排布、分层顺序和导出 |
| `device_interface.py` | 设备接口定义与预览 |
| `web.html`、`planner.js`、`planner.css` | 网页界面 |
| `prepare_data.py`、`train.py`、`evaluate.py`、`compare_models.py` | 数据准备、训练、评估和新旧模型对照 |
| `share.py` | 在 Windows 上建立临时分享链接 |

网页结果默认写入 `predictions/`。数据、训练历史、临时分享日志和预测结果已由 `.gitignore` 排除；仅根目录的发布模型 `mycelium_v2.pt` 允许入库。启动器会核对正在运行的服务版本和模型校验值，避免误用旧服务。

临时分享是独立的可选功能，使用方法见 [临时分享说明](临时分享说明.md)，不会随上传仓库自动开启。
