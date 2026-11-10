# scene_create

`scene_create` 将单张图片或同一静态场景的多视图重建为可分离的场景资产：

```text
输入图片
→ SAM3 实例分割
→ DA3 深度、相机与点图
→ 物体观测点云
→ SDXL Inpainting 物体补全
→ TRELLIS mesh + Gaussian 双资产
→ Sim(3) 场景配准
→ MaskGaussian 背景
→ 场景组装
```

默认入口为 `main.py`。全部模型在启动 `main.py` 的同一个 Python 解释器和进程中运行，不启动模型 worker。模型按阶段延迟加载，同一阶段内复用，切换阶段时释放并清理 CUDA 缓存。

## 1. 算法流程

流水线共有九个阶段，阶段顺序和依赖统一登记在 `src/scene_create/pipeline.py` 的 `STAGE_SPECS` 中。

### 1.1 ingest：输入接入

输入：

- 单张图片：`--image`
- 同一静态场景的多视图目录：`--images`
- 多视图 JSON 清单：`--manifest`
- 已有 mask、深度和相机观测：`--observations`

处理：

- 校正 EXIF 方向。
- 生成稳定的 `view_id`。
- 记录图片尺寸、来源路径和输入指纹。
- 检查多视图数量和输入约束。

保留：

- 校正后的视图信息。
- 原始输入来源和 SHA-256 指纹。
- 外部观测的 provenance，不会把导入数据伪装成模型推理结果。

### 1.2 segment：实例分割

模型：SAM3。

处理：

- 使用配置中的类别词表作为文本提示。
- 对每个视图生成实例 mask、类别和置信度。
- 为实例分配稳定的物体 ID。

保留：

- 原图尺寸的二值 mask。
- 每个视图中的局部实例 ID、类别和分数。
- 多视图物体关联所需的实例信息。

使用 `--observations` 时，本阶段直接复用外部 mask，不加载 SAM3。

### 1.3 depth：场景几何估计

模型：Depth Anything 3（DA3）。

处理：

- 联合预测深度、置信度、相机内参和相机外参。
- 保存处理分辨率与原图之间的像素映射。
- 可在模型提供天空概率时排除天空区域。
- 尺度状态只依据模型真实返回的 `is_metric`，不通过配置伪造米制尺度。

保留：

- 光轴深度 `depth`。
- 置信度 `conf` 和有效区域 `valid`。
- 相机内参 `intrinsics`。
- OpenCV 坐标约定下的相机位姿。
- 原图到几何网格的变换矩阵。

使用 `--observations` 时，本阶段直接复用外部几何，不加载 DA3。

### 1.4 objects：物体点云构建

处理：

- 将实例 mask 映射到 DA3 几何网格。
- 反投影得到每个视图中的物体观测点。
- 进行离群点过滤、体素降采样和多视图融合。
- 根据重叠、空间距离和颜色约束关联多视图中的同一物体。

保留：

- 每个视图的物体观测点云。
- 融合后的物体点云。
- 点到来源视图和像素的对应关系。
- 原始 mask、几何 mask、点数和体素尺寸。

有效点数不足的物体标记为 `insufficient_geometry`，不会生成伪造资产；它仍保留在背景分支中。

### 1.5 complete：可见物体补全

默认模型：SDXL Inpainting。

处理：

- 根据物体 mask 裁剪正方形参考图。
- 将已观测物体区域作为不可修改区域。
- 对 mask 外区域进行生成式补全。
- 生成后重新合成原始可见像素，保证已观测部分不被扩散模型改写。

保留：

- 原始裁剪图 `crop.png`。
- 可见区域 `visible_mask.png`。
- 补全图 `completed.png`。
- 提示词、随机种子、模型版本和生成来源。
- `synthetic: true` 与 `semantic_quality_verified: false` 标记。

适配器仍支持 FLUX.2 参考图生成，但默认配置使用更适合 T4 的 SDXL Inpainting。

### 1.6 assets：物体双资产生成

模型：TRELLIS-image。

处理：

- 使用 U2Net 提取补全图前景。
- 从同一次 TRELLIS 潜变量采样生成 mesh 和 3D Gaussian。
- 使用 FlexiCubes 构造网格。
- 对 mesh 进行清理、简化、纹理烘焙和 GLB 导出。
- 将 Gaussian 转换为项目统一的 PLY 约定。

保留：

- 输入 TRELLIS 的前景图和 mask。
- 局部坐标系中的 `mesh.glb`。
- 与 mesh 同源的 `gaussian.ply`。
- 模型组件、DINOv2 权重、采样参数和格式元数据。

TRELLIS 使用独立包名 `trellis_diff_gaussian_rasterization`，避免与 MaskGaussian 使用的标准 `diff_gaussian_rasterization` 冲突。

### 1.7 register：资产配准

处理：

- 将 TRELLIS 局部资产与物体观测点云进行多初值 Sim(3) 配准。
- 使用截断 ICP、轮廓 IoU 和相对深度误差进行质量门控。
- 同时变换 mesh 与 Gaussian，保持双资产坐标一致。

保留：

- 原始局部资产到场景坐标系的 `asset_to_world`。
- 配准 fitness、RMSE、轮廓 IoU 和深度误差。
- 已烘焙世界变换的 mesh 与 Gaussian。
- 输入视图上的配准轮廓诊断图。

场景清单中的烘焙资产不应再次应用 `asset_to_world`。

### 1.8 background：背景 Gaussian

模型：MaskGaussian。

处理：

- 从 DA3 点图中排除独立物体区域。
- 对 mask 做适量膨胀，减少前景边界泄漏。
- 构造背景点云和训练数据。
- 默认训练 MaskGaussian；`init-only` 模式只从背景点初始化 Gaussian。

保留：

- 背景 mask 和训练数据清单。
- 背景初始点云与 Gaussian。
- checkpoint、训练产物和诊断预览。
- 是否为近似背景的明确标记。

`init-only` 不等于完成 MaskGaussian 训练，结果状态会标记为近似。

### 1.9 assemble：场景组装

处理：

- 汇总成功配准的物体和背景。
- 统一 Gaussian 字段、坐标系和 SH 阶数。
- 合并物体网格和 Gaussian 范围。
- 生成质量报告和场景清单。

保留：

- `scene/scene.json`
- `scene/objects.glb`
- `scene/scene_gaussian.ply`
- `reports/quality.json`

失败物体不会被伪造为成功资产；其观测仍留在背景中，并在场景报告中保留失败原因。

## 2. 流程会保留什么

流水线不会只保留最终模型，而是保留从输入到场景的完整可追溯链路：

- 原始输入路径、视图 ID、图片尺寸和输入指纹。
- SAM3 的逐视图实例 mask、类别和置信度。
- DA3 的深度、置信度、相机、尺度状态和像素映射。
- 物体逐视图点云、融合点云以及像素来源。
- SDXL 的裁剪、可见 mask、补全图、提示词和生成元数据。
- TRELLIS 的前景输入、mesh、Gaussian 和模型来源。
- 配准变换、质量指标和诊断叠加图。
- 背景训练数据、checkpoint、Gaussian 和预览。
- 每个阶段的状态、耗时、错误、产物哈希和依赖指纹。
- 每次运行的 Loguru JSON Lines 日志。

观测数据和生成数据始终分开标记。程序不会把 SDXL 补全内容、解析测试数据或近似背景描述为真实观测。

## 3. 运行环境

### 3.1 基础环境

克隆主仓库后进入项目根目录。`third_parts/` 已作为普通源码目录纳入主仓库，不需要再执行 `git submodule update`：

```bash
cd /path/to/scene_create
```

CPU 开发环境可以使用：

```bash
uv venv --python python3 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest -q
```

真实推理环境必须在同一个解释器中同时满足 SAM3、DA3、Diffusers、TRELLIS 和 MaskGaussian 的依赖。当前实际环境使用 PyTorch 2.4.0 + CUDA 11.8；CUDA 扩展必须针对该 PyTorch/CUDA ABI 重新编译。

按照当前已验证环境的精确版本重建：

```bash
conda create -n scene_create python=3.11.18 -y
conda install -n scene_create -c conda-forge gcc_linux-64=11 gxx_linux-64=11 -y
conda activate scene_create

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=7.5
export MAX_JOBS=2

python -m pip install \
  "torch==2.4.0+cu118" \
  "torchvision==0.19.0+cu118" \
  "torchaudio==2.4.0+cu118" \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install --no-build-isolation -r requirements.txt
./scripts/install_trellis_rasterizer.sh
python -m pip check
```

`requirements.txt` 是当前 `scene_create` Conda 环境的完整 pip 快照；Git 依赖固定到 commit，普通包固定到精确版本。Python、GCC 和系统 CUDA Toolkit 不由 pip 管理，因此在上面的 Conda 和环境变量步骤中单独固定。

### 3.2 创建 checkpoints 目录

模型权重不会提交到 Git。每台新机器都要先创建本地权重目录：

```bash
export PROJECT_ROOT="$(pwd)"
mkdir -p \
  "$PROJECT_ROOT/checkpoints/huggingface" \
  "$PROJECT_ROOT/checkpoints/trellis/TRELLIS-image-large" \
  "$PROJECT_ROOT/checkpoints/dinov2" \
  "$PROJECT_ROOT/checkpoints/rembg"
```

目录用途：

- `checkpoints/huggingface/`：SAM3、DA3、SDXL 的 Hugging Face 缓存。
- `checkpoints/trellis/`：完整 TRELLIS-image 本地权重。
- `checkpoints/dinov2/`：TRELLIS 使用的 DINOv2 权重。
- `checkpoints/rembg/`：TRELLIS 抠图使用的 U2Net ONNX 权重。

当前机器的实际占用约为：

```text
Hugging Face cache  42 GiB
TRELLIS              3.1 GiB
DINOv2                1.2 GiB
U2Net                168 MiB
```

建议至少预留 60 GiB 给模型和缓存；单个场景输出还可能占用约 1 GiB。

### 3.3 Hugging Face 登录与自动下载模型

程序把 Hugging Face 缓存固定到项目的 `checkpoints/huggingface`。登录和运行程序必须使用同一个 `HF_HOME`：

```bash
export PROJECT_ROOT="$(pwd)"
export HF_HOME="$PROJECT_ROOT/checkpoints/huggingface"
hf auth login
hf auth whoami
```

不要把 Hugging Face Token 写入 README、配置、脚本或 Git。SAM3 等受限模型必须先在对应模型页面接受许可证并获得访问权限。

默认配置允许以下模型在第一次运行时自动下载：

- `facebook/sam3`
- `depth-anything/DA3-SMALL`
- `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`

可以提前下载 SAM3 和 DA3：

```bash
hf download facebook/sam3
hf download depth-anything/DA3-SMALL
```

SDXL 默认使用 FP16 variant。直接运行流水线时，Diffusers 只下载实际需要的文件；不建议用无过滤的 `hf download` 下载整个 SDXL 仓库，因为其中还包含不使用的 FP32 权重。

### 3.4 下载 TRELLIS、DINOv2 和 U2Net

TRELLIS 适配器禁止运行时远程回退，必须提前准备完整本地文件：

```bash
export PROJECT_ROOT="$(pwd)"

hf download microsoft/TRELLIS-image-large \
  --local-dir "$PROJECT_ROOT/checkpoints/trellis/TRELLIS-image-large"

curl --fail --location --continue-at - \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth \
  --output "$PROJECT_ROOT/checkpoints/dinov2/dinov2_vitl14_reg4_pretrain.pth"

curl --fail --location --continue-at - \
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx \
  --output "$PROJECT_ROOT/checkpoints/rembg/u2net.onnx"
```

下载后检查关键文件：

```bash
test -f checkpoints/trellis/TRELLIS-image-large/pipeline.json
test -f checkpoints/dinov2/dinov2_vitl14_reg4_pretrain.pth
test -f checkpoints/rembg/u2net.onnx
```

如果从已经配置好的机器迁移权重，只复制权重目录，不要复制 Hugging Face Token：

```bash
rsync -av /old/project/checkpoints/trellis/ checkpoints/trellis/
rsync -av /old/project/checkpoints/dinov2/ checkpoints/dinov2/
rsync -av /old/project/checkpoints/rembg/ checkpoints/rembg/
```

### 3.5 默认模型配置

`configs/default.yaml` 当前使用：

- SAM3：自动下载受限权重。
- DA3：`depth-anything/DA3-SMALL`。
- 物体补全：`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`。
- TRELLIS：本地 `TRELLIS-image-large`、本地 DINOv2 和本地 U2Net。
- 背景：MaskGaussian。

TRELLIS 强制使用完整本地权重，不通过 `allow_download` 自动补齐组件。

### 3.6 TRELLIS 双光栅器

MaskGaussian 和 TRELLIS 需要不同实现、但上游默认使用相同模块名的 Gaussian 光栅器。项目保留 MaskGaussian 的：

```text
diff_gaussian_rasterization
```

并将 TRELLIS 的 mip-splatting 光栅器安装为：

```text
trellis_diff_gaussian_rasterization
```

安装 TRELLIS 专用光栅器：

```bash
conda activate scene_create
./scripts/install_trellis_rasterizer.sh
```

脚本会拉取固定版本、重命名 Python/CUDA 扩展包、限制编译并发，并在结束时删除临时源码。

验证：

```bash
python -c "import diff_gaussian_rasterization, trellis_diff_gaussian_rasterization; print('rasterizers OK')"
```

### 3.7 MaskGaussian CUDA 扩展

MaskGaussian 需要：

- `mask_diff_gaussian_rasterization`
- `diff_gaussian_rasterization`
- `simple_knn`

源码位于 `third_parts/MaskGaussian/submodules/`。编译时建议：

```bash
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=7.5
export MAX_JOBS=2
```

CUDA 11.8 建议使用 GCC/G++ 11；GCC 8 无法编译当前 PyTorch 扩展。

`requirements.txt` 会从固定 commit 编译并安装这三个扩展。需要从主仓库内置源码重新安装时：

```bash
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=7.5
export MAX_JOBS=2

python -m pip install --no-build-isolation \
  third_parts/MaskGaussian/submodules/mask-diff-gaussian-rasterization
python -m pip install --no-build-isolation \
  third_parts/MaskGaussian/submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation \
  third_parts/MaskGaussian/submodules/simple-knn
```

### 3.8 训练 checkpoint 与断点续训

`checkpoints/` 保存全局模型权重；MaskGaussian 的场景训练 checkpoint 则由流水线自动创建：

```text
output/<scene>/background/checkpoint.pt
```

不要手工创建 `checkpoint.pt`。默认 `resume_checkpoint: true`，相同输入和配置重跑时会继续训练。如果输入、mask、配置或背景初始点云发生变化，旧 checkpoint 会被拒绝，防止错误续训。

保留旧 checkpoint 并重新训练：

```bash
mv output/<scene>/background/checkpoint.pt \
   output/<scene>/background/checkpoint.stale.pt
```

也可以在 `models.maskgaussian` 下设置 `resume_checkpoint: false`，但这会禁用该配置下的断点恢复。

### 3.9 新机器安装验收

完成环境、权重和 CUDA 扩展安装后执行：

```bash
python -m pip check

python -c "
import torch
import diff_gaussian_rasterization
import mask_diff_gaussian_rasterization
import simple_knn
import trellis_diff_gaussian_rasterization
print(torch.__version__, torch.version.cuda)
print('CUDA extensions OK')
"

python main.py doctor --config configs/default.yaml
```

预期 PyTorch 为 `2.4.0+cu118`、CUDA 为 `11.8`。`doctor` 是静态预检，不会加载完整模型。

由于第三方源码的嵌套 `.git` 已移除，源码现在由主仓库统一跟踪；当前 `doctor` 的 commit 检查可能把主仓库 commit 与 `third_parts/manifest.json` 中的上游 commit 比较并报告 `Source commit differs`。该提示不等于模块不可导入，应结合依赖检查和真实阶段运行判断。

## 4. 如何运行

### 4.1 查看帮助

```bash
python main.py --help
python main.py doctor --help
python main.py preview --help
```

### 4.2 环境预检

```bash
python main.py doctor --config configs/default.yaml
```

保存报告：

```bash
python main.py doctor \
  --config configs/default.yaml \
  --json output/doctor.json
```

预检只检查路径、模块、CUDA 和静态环境，不加载完整模型，也不能证明 CUDA 扩展 ABI 或真实推理一定可用。

### 4.3 单图全流程

```bash
python main.py \
  --image /path/to/image.png \
  --output output/my_scene \
  --config configs/default.yaml
```

### 4.4 多视图全流程

多视图必须来自同一个静态场景，并具有足够重叠：

```bash
python main.py \
  --images /path/to/views \
  --output output/multiview_scene \
  --config configs/default.yaml
```

### 4.5 运行指定阶段

不传 `--stage` 时执行全部九个阶段。指定目标阶段时，程序自动补齐传递依赖。

只生成 mask：

```bash
python main.py \
  --image /path/to/image.png \
  --stage segment \
  --output output/my_scene \
  --config configs/default.yaml
```

只运行到物体点云：

```bash
python main.py \
  --image /path/to/image.png \
  --stage objects \
  --output output/my_scene \
  --config configs/default.yaml
```

运行物体资产及其全部前置阶段：

```bash
python main.py \
  --image /path/to/image.png \
  --stage assets \
  --output output/my_scene \
  --config configs/default.yaml
```

可一次指定多个目标：

```bash
--stage objects background
```

有效阶段：

```text
ingest segment depth objects complete assets register background assemble
```

交互终端会显示按实际执行计划计数的总进度条，例如 `3/9 | depth`。阶段缓存命中也会推进进度；非交互环境自动关闭进度条，避免污染机器可读输出。

### 4.6 常用选项

覆盖 SAM3 类别：

```bash
--categories chair table sofa lamp plant
```

任一物体失败时立即停止：

```bash
--strict
```

只初始化背景、不训练 MaskGaussian：

```bash
--background-mode init-only
```

指定随机种子：

```bash
--seed 42
```

### 4.7 中断恢复与重跑

直接重新执行相同命令即可恢复：

```bash
python main.py \
  --image /path/to/image.png \
  --output output/my_scene \
  --config configs/default.yaml
```

程序读取 `run_manifest.json`，验证阶段指纹和产物 SHA-256 后复用结果。失败或 `partial` 阶段会重跑；按物体执行的阶段会继续复用其中已成功的物体。

### 4.8 导入已有观测

如果 SAM3 和 DA3 已在其他环境中运行，可直接导入真实 mask、深度和相机：

```bash
python main.py \
  --observations /path/to/observations.json \
  --stage objects \
  --output output/imported_scene \
  --config configs/default.yaml
```

每个 view 至少包含：

- `image_path`
- `geometry_path`
- `masks`

几何 NPZ 使用：

- `depth`：`H×W` 光轴深度。
- `conf`：`H×W` 非负置信度。
- `image`：`H×W×3` uint8 RGB。
- `intrinsics`：`3×3`。
- `world_from_camera`：`4×4` camera-to-world。
- `original_to_processed`：`3×3` 像素映射。

多视图外部观测必须已经处于统一世界系和尺度。

## 5. 输出目录

以 `output/my_scene` 为例：

```text
output/my_scene/
├── masks/                         # 实例 mask
├── pointmaps/                     # DA3 深度、相机和点图
├── objects/
│   └── chair_001/
│       ├── clouds/                # 观测与融合点云
│       ├── images/                # 裁剪、mask、补全图和元数据
│       ├── assets/                # TRELLIS mesh 与 Gaussian
│       └── registration/          # 配准变换、烘焙资产和诊断图
├── background/                    # 背景数据、checkpoint 和 Gaussian
├── scene/
│   ├── scene.json                 # 场景清单
│   ├── objects.glb                # 已放置的物体网格
│   └── scene_gaussian.ply         # 合并后的背景与物体 Gaussian
├── reports/
│   └── quality.json               # 几何指标与失败信息
├── logs/
│   └── <run_id>.log               # Loguru JSON Lines 日志
└── run_manifest.json              # 阶段状态、指纹、哈希和耗时
```

输出目录非空且没有本项目的 `run_manifest.json` 时，程序拒绝写入，避免覆盖其他数据。

## 6. 日志

业务代码统一使用 Loguru。第三方来源也会接入同一日志系统：

- 标准库 `logging` 通过桥接进入 Loguru。
- Python `stdout`、`stderr` 和 warning 转为带上下文的日志。
- C/CUDA 原生 stdout 通过管道转为日志。
- 日志上下文包含 `run_id`、`stage`、`object_id` 和 `view_id`。

终端中的 `[SPARSE]`、NumExpr、进度条和 FutureWarning 仍保留第三方原始正文，但外层时间、级别和阶段前缀由 Loguru 统一提供。

标准输出只用于最终 JSON 结果；诊断日志写入标准错误和 `output/<scene>/logs/`。

## 7. 场景预览

组装完成后，使用 MaskGaussian CUDA 光栅器渲染记录相机视角：

```bash
python main.py preview \
  --scene output/my_scene/scene/scene.json \
  --config configs/default.yaml
```

默认输出到：

```text
output/my_scene/scene/previews/
```

生成 Gaussian 预览、对象状态摘要和交互式 GLB 查看页：

```bash
python scripts/view_scene.py output/my_scene --render
```

脚本会生成：

```text
output/my_scene/scene/viewer.html
```

并在 `http://127.0.0.1:8000/viewer.html` 启动本地服务。交互式 GLB 组件从 CDN 加载；Gaussian 场景通过本地渲染的 PNG 展示。

只生成查看页、不启动服务：

```bash
python scripts/view_scene.py output/my_scene --no-serve
```

远程机器需要监听外部地址时：

```bash
python scripts/view_scene.py output/my_scene --host 0.0.0.0 --port 8000
```

浏览器中看不到远程端口时，应使用 SSH 端口转发或开发环境提供的端口映射功能。

配准阶段生成的 `*_alignment.png` 只是 CPU 轮廓和深度诊断图，不是真实 Gaussian 外观渲染。

## 8. 自动复用与失败语义

阶段缓存指纹覆盖：

- 输入文件与内容。
- 本项目 Python 源码。
- 当前阶段相关配置。
- 模型源码指纹。
- 本地权重文件清单。
- 上游阶段结果。

状态语义：

- `completed`：请求目标全部完成。
- `partial`：部分物体失败，或背景使用了明确的近似模式。
- `stale`：配置、代码、模型或上游结果变化，需要重跑。
- `insufficient_geometry`：物体观测点不足。
- `complete_failed`：图像补全失败。
- `assets_failed`：TRELLIS 资产生成失败。
- `registration_failed`：资产未通过配准质量门槛。

返回码：

- `0`：请求阶段完成。
- `2`：输入、环境或执行错误。
- `3`：部分失败或近似结果。
- `130`：用户中断。

## 9. 已知限制

- 默认类别词表不保证找到图片中的所有物体。
- 单图无法恢复真实不可见几何；SDXL 和 TRELLIS 输出属于生成内容。
- 单图和普通 DA3 模型通常只能提供相对尺度。
- 多视图必须来自静态、有重叠的同一场景。
- SDXL 保留可见像素，但生成区域仍需人工审核。
- 配准不能可靠消除对称物体歧义、物体碰撞和支撑关系错误。
- 背景只监督输入中可见区域，不会自动恢复所有遮挡背景。
- Tesla T4 只有约 14.6 GiB 可用显存，低于 TRELLIS 官方建议的 16GB，仍可能出现 OOM。
- T4 不支持原生 BF16，应使用 `float16`。
- 第三方 CUDA 扩展必须与当前 PyTorch、CUDA Toolkit、GCC 和 GPU 架构匹配。

## 10. 代码结构

- `main.py`：仓库根目录入口。
- `src/scene_create/cli.py`：参数解析、`doctor`、重建和预览命令。
- `src/scene_create/pipeline.py`：阶段注册、依赖闭包、缓存指纹和主流程。
- `src/scene_create/runtime.py`：输出锁、按物体执行、模型生命周期和环境预检。
- `src/scene_create/observations.py`：输入、分割、深度、点云与多视图融合。
- `src/scene_create/reconstruction.py`：补全、TRELLIS、配准、背景和组装。
- `src/scene_create/geometry.py`：相机、点云、栅格化和 Sim(3) 配准。
- `src/scene_create/gaussian.py`：Gaussian PLY 读写、变换和合并。
- `src/scene_create/io.py`：原子文件写入、哈希和 JSON 工具。
- `src/scene_create/logging_utils.py`：Loguru 配置与第三方输出捕获。
- `src/scene_create/adapters/`：SAM3、DA3、Diffusers、TRELLIS 和 MaskGaussian 适配器。
- `third_parts/manifest.json`：第三方源码及关键子模块版本记录。
