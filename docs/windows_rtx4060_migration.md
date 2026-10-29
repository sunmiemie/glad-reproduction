# Windows 11 + RTX 4060 8GB 迁移操作说明

目标：将现有 GLAD 模型和算法迁移到新电脑，先获得 PyTorch CUDA、FP32 的 GPU 基线。继续使用同一套权重、640 网络输入、300 基础 ROI 和 4 个 CPU 线程，方便与 CPU 结果对照。当前还没有在用户的 RTX 4060 上实测，不承诺论文的 146.5 FPS。

## 1. 复制迁移包

将 `GLAD_Win11_RTX4060.zip` 复制到新电脑，解压到例如 `D:\AI`，项目目录应为：

```text
D:\AI\glad_reproduction\
    README.md
    requirements.txt
    glad\
    scripts\
    weights\                  三个作者权重
    external\yolov5\          固定 YOLOv5 v6.0，含许可证和版本标记
    videos\phantom64.mp4
    videos\72ebd203198e372d6b64af2d4fb9531e.mp4
    MIGRATION_MANIFEST.json    文件 SHA-256、源码提交号
```

包内已经有代码、官方权重、所需 YOLOv5 源码及两段原始测试视频。没有复制原电脑的 `.venv`、Git 凭据、缓存和输出视频。**这不是完整离线环境：新电脑首次仍需联网安装 Python 和依赖，CUDA 版 PyTorch 下载可能需要数 GB。**

迁移包的 `.sha256` 文件用于复制后核对：

```powershell
Get-FileHash 'D:\Downloads\GLAD_Win11_RTX4060.zip' -Algorithm SHA256
```

也可直接从私有仓库获取代码，需登录有访问权限的 GitHub 账号；克隆方式不会自带视频和权重，权重由安装脚本下载，视频需另行复制：

```powershell
git clone --branch mm/glad-reproduction https://github.com/sunmiemie/glad-reproduction.git
```

## 2. 准备系统环境

1. 安装/更新与 RTX 4060 对应的 [NVIDIA 官方驱动](https://www.nvidia.com/en-us/drivers/)。笔记本版与桌面版选对应产品。
2. 安装 **Python 3.11 64 位**，保留 Python Launcher。已经安装 Python 3.10 64 位也可使用脚本的 `-PythonVersion 3.10` 参数。Python 官网：https://www.python.org/downloads/windows/ 。
3. 在 PowerShell 检查：

```powershell
nvidia-smi
py -3.11 --version
```

`nvidia-smi` 应能显示 RTX 4060。其 CUDA Version 字段是驱动支持信息，不等同于已安装的 PyTorch CUDA runtime；实际运行由后面的检查脚本确认。当前使用预编译 PyTorch，不需要手工编译 CUDA 代码。

## 3. 安装独立 GPU 环境

```powershell
Set-Location 'D:\AI\glad_reproduction'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows_cuda.ps1
```

这里的 `ExecutionPolicy Bypass` 仅用于本次子进程，不修改全局 PowerShell 策略。脚本创建本项目内独立的 `.venv-cuda`，不依赖旧电脑的环境路径。

脚本固定安装：

```text
torch 2.8.0 + cu126
torchvision 0.23.0 + cu126
requirements.txt 中的其余依赖
```

版本组合依据 [PyTorch 官方 v2.8.0 安装说明](https://pytorch.org/get-started/previous-versions/#v280)。脚本使用明确的 `+cu126` 包版本，避免重用现有 CPU 构建；GPU 包安装后再安装其余依赖，并执行 `pip check`。

安装结束前会进行：CUDA 矩阵运算、卷积、Torchvision GPU NMS、三个 GLAD 权重实际加载及全局/局部网络推理。**只有全部成功才显示安装完成；CUDA 不可用时会停止，不静默退回 CPU。**

单独重复检查：

```powershell
.\.venv-cuda\Scripts\python.exe scripts\check_environment.py --device cuda:0 --model-smoke-test
```

应看到 CUDA 可用、设备名为 RTX 4060、Torch CUDA runtime 为 12.6，并且检查成功。当前已在原电脑验证 CPU 检查与模型加载、无 CUDA 时的失败分支；实际 GPU 检查需在新电脑完成。

## 4. 先运行 120 帧

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_windows_cuda.ps1 -MaxFrames 120 -ShowWindow
```

默认输入包内 `videos\phantom64.mp4`。脚本自动使用 `cuda:0`，自动生成带时间戳的新输出目录。按预览窗口中的 `q` 可提前停止。

运行整段视频，关闭预览窗口以便观察实际处理速度：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_windows_cuda.ps1
```

测试操场视频：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_windows_cuda.ps1 `
  -Source '.\videos\72ebd203198e372d6b64af2d4fb9531e.mp4'
```

测试任意新视频：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_windows_cuda.ps1 `
  -Source 'D:\videos\my_video.mp4' -ShowWindow
```

## 5. 阅读结果并比较

新结果保存在 `runs\rtx4060_视频名_时间戳\`，包括：

- `output.mp4`：检测结果视频；
- `frames.jsonl`：逐帧结果；
- `summary.json`：帧率、分支、配置及输出帧数。

先检查 `summary.json` 中 `device` 是否为 `cuda:0`、`torch` 是否带 CUDA 构建标记，再读取 `pipeline_fps` 与 `end_to_end_fps`。前者是算法处理速度，后者包含读取、绘制、日志和编码。原电脑 phantom64 对应值为 **6.44 / 6.04 FPS**，操场视频为 **5.51 / 5.33 FPS**。

对于公平比较：先保持默认参数不变，用相同完整视频测试；记录新电脑 CPU 型号、GPU 型号和功耗设置。笔记本连接电源，使用稳定的性能模式。不要把播放帧率当成推理帧率，也不要用不同视频的 FPS 直接判断算法优化幅度。

这一步只把神经网络移到 GPU。OpenCV 光流、相机变换估计等仍在 CPU 上，性能也受新电脑 CPU 影响。TensorRT、FP16 和运动补偿优化属于后续步骤，应分别测量效果与检测一致性。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| 找不到 `py` 或指定 Python | 安装 Python 3.11 x64 和 Launcher；或已有 3.10 时使用 `-PythonVersion 3.10` |
| `torch.version.cuda` 为 null | 正在使用 CPU 构建或错误环境；使用 `.venv-cuda\Scripts\python.exe` 并重新执行 GPU 安装脚本 |
| CUDA 初始化失败 | 检查 NVIDIA 驱动和 `nvidia-smi`；更新驱动后按要求重启，再运行环境检查 |
| Torchvision GPU NMS 报错 | Torch/Torchvision 构建不匹配；重新用脚本安装二者的 cu126 固定版本 |
| 缺少权重或 YOLOv5 源码 | 执行 `.venv-cuda\Scripts\python.exe scripts\bootstrap.py` |
| 私有仓库无法克隆 | 使用具有权限的账号；或者直接复制本次迁移 ZIP，无需 GitHub 登录 |
| GPU 已运行，仍达不到论文速度 | 当前是 PyTorch FP32 基线，且仍有 CPU 运动处理；继续做分阶段计时，不能保证换显卡后自动达到 146.5 FPS |
