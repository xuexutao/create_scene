# scene_create

单张图片或同一静态场景的多视图 → 实例 mask → DA3 几何 → 物体点云融合 → FLUX.2 补全 → TRELLIS 双资产 → 带尺度配准 → MaskGaussian 背景 → 可分离的场景资产。

## 当前交付状态

- 已实现九阶段 CLI、五个真实算法适配层、CPU 几何/融合/配准、Gaussian 格式转换、场景组装、失败恢复和预览命令。
- 主环境 `.venv` 已创建；89 项 CPU 用例通过，静态检查、格式和编译检查通过。五个算法及 DINOv2 源码在 `third_parts/`，版本记录于 `third_parts/manifest.json`。
- **尚未下载模型权重、安装各模型 CUDA 环境或完成五模型真实端到端验证。** 当前检测到 `Tesla T4`、约 15 GiB 显存，不足以保证运行 TRELLIS 和大型 FLUX.2。T4 不支持原生 BF16。
- 模型适配不是占位输出：缺少权重/依赖会明确报错。解析几何测试数据也不会被标记成模型推理结果。
- 完整目标与尚未实现的增强功能见 `todolist.md`。`environments/models.yaml` 是环境参考，不是已经验证可用的 GPU 环境锁文件。

## 1. 主环境与模型环境

在 `/apdcephfs/private_xt/dev_workspace/scene_create` 中执行下列命令。

已有主环境可直接使用 `.venv/bin/python main.py --help`。重建 CPU 主环境的方式：

1. `uv venv --python python3 .venv`
2. `uv pip install --python .venv/bin/python -e '.[dev]'`
3. `.venv/bin/python -m pytest -q`

如没有 `uv`，可使用标准 `python3 -m venv .venv` 和 `.venv/bin/python -m pip install -e '.[dev]'`。

每个重模型需要单独准备其上游依赖与 CUDA 扩展，在相应环境中安装本项目，再把解释器绝对路径写入 `configs/default.yaml` 的 `models.<name>.python`。不要把所有模型依赖安装进同一环境。主程序通过磁盘请求/响应文件启动独立进程，GPU 模型顺序运行。

设备支持 `cuda`、`cuda:N`；预检与执行使用同一张选定 GPU。SAM3、DA3 和 FLUX.2 的 `dtype` 支持 `auto`、`float16`、`bfloat16`、`float32`；CPU 推理必须选择 `float32` 或 `auto`。DA3 的主干 autocast 显式遵循该配置，不再被上游默认精度覆盖，几何输出头仍保留上游 FP32 处理。实际设备与精度会写入来源信息。

DA3 的尺度状态由实际 `prediction.is_metric` 决定，不接受配置中的米制标签作为证据；Nested 未确认米制转换时会报错。`exclude_sky: true` 会在模型返回天空掩码时排除相应深度点。

### 必需权重

| 模型配置 | 权重形式 |
| --- | --- |
| `models.sam3.model_path` | 有访问许可的本地 `sam3.pt`，或包含它的目录。 |
| `models.da3.model_path` | 完整的本地 DA3 模型目录，包含 `config.json`、`model.safetensors`。使用 any-view 或 Nested；当前不支持独立 Metric/Mono 模型补相机。 |
| `models.flux2.model_path` | 完整 diffusers 格式 FLUX.2 目录，包含 transformer、VAE、文本编码器及 tokenizer；不是单个 transformer 权重文件。 |
| `models.trellis.model_path` | 完整 TRELLIS-image 模型目录，包含 `pipeline.json` 和其引用的各组件权重。 |
| `models.trellis.dino_model_path` | 与 TRELLIS `pipeline.json` 中图像编码器名称匹配的 DINOv2 权重。 |
| `models.trellis.rembg_model_path` | 本地 `u2net.onnx`，默认 `checkpoints/rembg/u2net.onnx`。 |
| MaskGaussian | 从本场景背景点云训练，不要求预训练场景权重，但要求 CUDA 光栅器和 `simple_knn`。 |

`dino_repo_path` 默认 `third_parts/dinov2`。TRELLIS 不调用上游默认的远程 `torch.hub` 权重下载；使用本地源码和显式辅助权重。实际使用的五个组件必须各自具备 `.json` 与 `.safetensors`，不允许缺失后回退远程，也不加载未使用的 `slat_decoder_rf`。U2Net 使用支持显式 `model_path` 的 `u2net_custom` 会话，不使用默认自动下载会话。TRELLIS 始终要求本地完整组件，即使其他模型开启了 `allow_download`。

`allow_download` 默认关闭。不自动上传图片，不调用付费云 API，不把许可证授权失败当成可绕过的问题。模型权重和子模块各自遵循其许可证。

`third_parts/` 中的主仓库不等于编译完成：TRELLIS 的 FlexiCubes、渲染器和 MaskGaussian 的三个 CUDA 子模块仍需按锁定版本准备/编译。MaskGaussian 上游部分子模块 URL 使用 SSH；无 SSH 凭据时应使用对应公开仓库的 HTTPS 来源，不修改其他项目的 Git 配置。未使用的 SIBR 查看器及 DA3 streaming 依赖不要求安装。

FLUX.2 默认走已核对 API 的 `diffusers==0.36.0` / `Flux2Pipeline`。可选择 `Flux2KleinPipeline`，但需要另行安装并验证提供该类的新版本及对应权重；不能仅改模型路径就混用架构。`cpu_offload` 可配置，但并不保证 T4 能运行大型模型。

## 2. 预检

`.venv/bin/python main.py doctor --config configs/default.yaml`

可用 `--json /path/to/report.json` 保存报告。预检检查源码版本、解释器、模块可发现性、CUDA 和必要权重；**不加载完整模型，不证明自定义扩展 ABI 或真实推理已验证**。权重未配置时返回码为 2，属于明确的未就绪状态。

## 3. 单图、多图与阶段运行

- 单图：`.venv/bin/python main.py run --image /path/to/image.png --config configs/default.yaml`
- 多图：`.venv/bin/python main.py run --images /path/to/views --config configs/default.yaml --output output/multiview_scene`
- 先获得物体点云：`.venv/bin/python main.py run --images /path/to/views --through objects --config configs/default.yaml`
- 更改候选类别：添加 `--categories chair table cup vase`。
- 恢复同一输出：添加 `--resume`。不带该参数不会覆盖已有运行。
- 任一物体失败立即终止：添加 `--strict`；否则失败物体保留在背景分支并记录为失败，不伪造替换资产。
- 仅初始化背景 GS：显式添加 `--background-mode init-only`。该模式不训练 MaskGaussian，结果标记为近似，不能用于宣称背景训练完成。

阶段顺序：`ingest`、`segment`、`depth`、`objects`、`complete`、`assets`、`register`、`background`、`assemble`。`--through` 可以停在其中任一阶段；前置依赖自动运行或复用经校验的缓存。

每个场景应使用独立输出目录。重跑时以 `run_manifest.json` 和 `scene.json` 引用的文件为准，不应直接遍历目录中的历史文件推断当前场景。

### 输入清单

可使用 `--manifest /path/to/views.json`，其中 `views` 是列表，每项包含 `image_path` 和可选 `view_id`；图片路径相对该清单。外部内参、位姿和去畸变当前通过下面的观测导入接口提供，不能在普通图片清单中悄悄忽略。

## 4. 导入已有真实 mask / 深度

如果已经使用其他环境得到 SAM3/DA3 结果，可以使用：

`.venv/bin/python main.py run --observations /path/to/observations.json --through objects --output output/imported_scene`

此路径跳过 SAM3/DA3 worker，来源标记为 `external_observations`，并非模型模拟模式。数据必须满足以下契约。

### 观测 JSON

顶层包含 `views`、可选 `scale_status` / `scale_source`。每个 view：

| 字段 | 内容 |
| --- | --- |
| `view_id` | 唯一、安全的标识，例如 `view_000`；可省略以自动编号。 |
| `image_path` | 已做 EXIF 方向校正的原图路径。 |
| `geometry_path` | 下述 NPZ 路径。 |
| `masks` | 实例列表，允许为空。 |

每个 mask 项包含 `category`、`mask_path`，可选 `score`、`local_id` 和 `object_id`。`object_id` 用于人工确认跨视图同一实体，例如 `chair_001`；同一视图不能重复使用。路径相对 JSON 所在目录。

mask 必须是原图尺寸单通道图像，前景 255、背景 0。当前同物体的可见分片不会只保留最大连通域。

### 几何 NPZ

| 数组 | 形状 / 约定 |
| --- | --- |
| `depth` | `H×W` 光轴深度，不是射线距离。 |
| `conf` | `H×W` 非负置信度。 |
| `valid` | `H×W` bool；可省略，由深度/置信度构造。 |
| `image` | `H×W×3`、uint8 RGB，与深度同网格。 |
| `intrinsics` | `3×3`，对应上述处理网格。 |
| `world_from_camera` | `4×4` camera-to-world，OpenCV X右/Y下/Z前。 |
| `original_to_processed` | `3×3`，原图像素中心到几何网格像素中心的映射，包含实际缩放、裁剪/填充；没有预处理时为单位阵。 |

多视图必须已处于同一世界系和尺度；程序会统一固定到第一输入相机系，但不会替你估计缺失的组间变换。若直接使用 DA3 外参，需先将其 world-to-camera 正确求逆。测试中的解析观测构造例子位于 `tests/test_pipeline.py`，它们不是模型效果样例。

## 5. 主要输出

| 路径 | 内容 |
| --- | --- |
| `output/masks/chair_001.png` | 单图 mask，或多图中该物体代表视图 mask；视图编号见 `metadata.json`。 |
| `output/masks/view_000/chair_001.png` | 多视图各自的实例 mask。 |
| `output/pointmaps/view_000.npz` | 深度、相机、点图、RGB、置信度、有效区和像素映射。 |
| `output/objects/chair_001/clouds/` | 各视图观测点、融合点云及像素/视图来源。 |
| `output/objects/chair_001/images/` | 裁剪、可见 mask、补全图、提示和生成元数据。 |
| `output/objects/chair_001/assets/` | 同局部坐标系的 `mesh.glb`、`gaussian.ply` 和元数据。 |
| `output/objects/chair_001/registration/` | `transform.json`、对齐资产及输入视图轮廓叠加图。 |
| `output/background/` | 背景训练输入、GS 初始化、checkpoint、训练产物和诊断预览。 |
| `output/scene/scene.json` | 场景清单、相机、尺度标记、物体状态和 Gaussian 实例范围。 |
| `output/scene/objects.glb` | 已放置的物体网格集合，不包含 GS 背景。 |
| `output/scene/scene_gaussian.ply` | 统一格式、坐标和 SH 阶数后的背景与物体 GS。 |
| `output/logs/` | 统一 `loguru` 结构化日志；各 worker 的 `worker_<attempt>.log` 同样由 `loguru` 输出 JSON Lines。 |
| `output/reports/quality.json` | 可计算的几何指标和失败信息；没有真值不会编造准确率。 |

场景中的对齐资产已经烘焙世界变换，`render_transform` 为单位阵；`asset_to_world` 供原始局部资产使用，不能对烘焙资产重复应用。

### 统一日志

- 自有业务代码的进度、警告、中断和异常全部使用 `loguru`，包括启动入口、CLI 参数错误和 worker。
- `logging_utils.py` 统一配置控制台与文件日志；标准库 `logging` 仅作为第三方日志桥接，不用于自有业务诊断。
- 第三方 Python 标准输出/标准错误、标准日志及 warning 接入 `loguru`；worker 原生输出由父进程实时接收，不修改 `third_parts/` 源码。
- worker 保留结构化日志级别和异常堆栈，使用相同 `run_id`、`stage`、`object_id`、`view_id`；原生无级别文本按 INFO 记录。
- 日志写入标准错误和已配置的日志文件，文件为 JSON Lines，按 20 MB 轮转、保留 14 天。禁用异常局部变量展开。
- 标准输出仅用于 JSON 结果协议及 `--help` 帮助文本；JSON 不附加日志前缀，可直接供脚本解析。`loguru` 是必需运行依赖。

## 6. 真正的 Gaussian 场景预览

配置好 MaskGaussian 环境后运行：

`.venv/bin/python main.py preview --scene output/scene/scene.json --config configs/default.yaml`

默认在 `scene.json` 同目录的 `previews/` 输出各记录相机视角 PNG，使用 CUDA Gaussian 光栅器并保留 SH 颜色。未知区域显示黑色，不代表补全成功。该命令未在当前缺少 CUDA 扩展的环境执行验证。

配准阶段的 `*_alignment.png` 是 CPU 深度/轮廓叠加诊断，不是上述真实外观渲染。绿色表示观测物体，红紫色表示可见投影资产。

## 7. 限制和失败语义

- 默认交互物体由配置类别词表驱动；不保证找出“所有”物体。自动 VLM 类别发现和图形化人工编辑尚未实现。
- 多视图假设静态、有重叠；超出 `max_views` 会拒绝，不把独立批次深度盲目拼接。
- FLUX.2 走参考图条件生成，不宣称原生二值 mask inpainting 或准确恢复真实隐藏结构。生成内容有来源标记，主体身份仍建议人工审核。
- 当前配准为多初值 Sim(3) + 截断观测到资产 ICP，带可见轮廓/深度门槛。精确点到面、多物体碰撞、支撑平面优化及对称性辨别是后续增强。
- 单图和生成资产不能保证真实米制几何；相对、模型估计米制与外部尺度来源分开记录。
- 单图背景训练要求正的 `depth_weight`，仍只承诺可见区域近似。适配器通过共享同次概率 mask 的深度值/单位颜色额外渲染，计算 alpha 归一化期望深度；这不是原版光栅器直接返回的 depth 字段，真实 CUDA 梯度和质量仍待验收。全部输入中被遮挡的背景没有自动补全。
- `--resume` 校验输入、代码、配置及产物；模型权重使用路径/文件大小/修改时间清单与版本记录。失败阶段会重试，已完成物体可复用；输入或配置变化使其相关下游失效。
- 未完成某些物体时，场景保留这些观测于背景，但报告为 `partial`，不声称它们已成为独立可交互资产。

返回码：0 为请求阶段完成；2 为输入/环境/执行错误；3 为部分失败或明确近似结果；130 为中断。`doctor` 的返回码 2 表示未就绪。
