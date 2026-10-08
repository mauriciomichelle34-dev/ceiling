# 菌丝体分割与取样规划

使用 U-Net 分割培养皿照片中的菌丝区域，再根据培养皿标定、取样圆直径和间隙生成取样点。取样编号沿菌丝实际外轮廓逐层向内推进，支持编辑深度和导出坐标。

本仓库保存项目源码。运行网页需要在本机安装 Python 依赖，并准备模型权重；仓库上传完成后不会自动启动在线服务。

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

模型权重、训练数据和历史结果未包含在仓库中。现有模型的文件名是 `best_model.pt`，可从原项目复制到以下位置：

```text
runs/20260923_102628_721977/best_model.pt
```

然后启动：

```bash
python local_app.py --open
```

Windows 也可以双击 `Start_Local_App.cmd`；启动脚本优先使用仓库中的 `.venv`，否则使用 PATH 中的 `python`。

权重放在其他位置时，指定模型路径：

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

数据准备脚本按图像划分训练和验证集。如果多张图片来自同一培养皿或时间序列，应先调整为按组划分，再评估泛化性能。模型选择使用验证集，不使用测试集或外部测试标签调参。

checkpoint 会记录本机的数据和划分文件路径。换电脑评估时，需要提供相应数据并更新路径；只做网页推理不需要原训练数据。

## 文件

| 文件 | 用途 |
| --- | --- |
| `local_app.py` | 本地 HTTP 服务、分割推理和取样接口 |
| `model.py` | U-Net 网络 |
| `sampling.py` | 取样圆排布、分层顺序和导出 |
| `device_interface.py` | 设备接口定义与预览 |
| `web.html`、`planner.js`、`planner.css` | 网页界面 |
| `prepare_data.py`、`train.py`、`evaluate.py` | 数据准备、训练、评估 |
| `share.py` | 在 Windows 上建立临时分享链接 |

网页结果默认写入 `predictions/`。数据、权重、临时分享日志和预测结果已由 `.gitignore` 排除。

临时分享是独立的可选功能，使用方法见 [临时分享说明](临时分享说明.md)，不会随上传仓库自动开启。
