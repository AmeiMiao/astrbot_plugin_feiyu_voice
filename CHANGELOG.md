# 更新记录

服务、AstrBot 插件和模型权重分别管理版本；服务端更新不会自动改变模型权重或已有插件安装包。

## 2026-10-27 · Colab CUDA 选择与模块导入修复

- 两份 Notebook 安装表单增加 `auto / cu128 / cu130`，按 Colab 内核的 CUDA runtime 选择固定的 PyTorch 2.11.0 构建，同步 wheel 源、依赖约束与 profile；不改写共享的本机 CUDA 预设。
- 安装与推理入口自动定位 `publication/server/feiyu_runtime`，无需手动创建 symlink 或设置 PYTHONPATH。
- 更新辅助脚本 SHA 与使用说明，18 项回归测试通过，真实本地 cu128 RVC CLI 输出与既有结果一致；用户反馈手工调整后 Colab 可用，自动修复版本的完整 Colab GPU 验收仍待完成。
- 模型权重、参考音频、索引、推理算法及已有插件安装包保持不变。

## 2026-10-25 · 使用指南与素材署名

- 将项目首页与面向 DeepSeek／DSH／AstrBot agent 的《小肥鱼的声音指南》分别组织，统一使用 🐋 标识。
- 明确可爱 `cute`、温柔 `gentle` 两条 TTS 声线，默认可爱；说明默认偏好和按场景显式切换的方式。
- 增加 agent 复用或安装 MSST、编排“分离 → RVC 换声 → 混音”的整曲翻唱说明。
- 增加简单演唱 demo 路线：有文件生成／工具调用权限、且唱歌组件已配置的 agent，可自行制作《小星星》这类简单旋律的 MIDI、配歌词，再调用按谱合成接口。
- 补充原始音频来源：B 站 UP 主 [星落落_oi](https://space.bilibili.com/30584746/)；训练素材均已取得原 UP 主授权同意。来源和授权说明同步至 TTS、RVC 的模型仓库与配套 GitHub 文档。
- 本轮为说明文件与元数据更新，模型二进制、训练步数及调用协议保持不变。

## 声音服务 v0.2.0 · 2026-10-25

- 增加 NVIDIA CUDA 与 AMD WSL ROCm 两套可选推理环境预设、安装器和环境诊断；通过服务端 `runtime_profile` 选择环境。
- 适配 TTS 的 CPU ONNX 前端、RVC 的 CPU RMVPE 选项，以及 SVS 的设备／精度配置；主模型使用所选 GPU 后端。
- CUDA RVC 干净安装、实际换声，以及 TTS／RVC／按谱唱歌的真实 HTTP 回归通过。
- **AMD／ROCm 已提供适配与安装流程，尚无实体 A 卡测试；属于待实机验收的实验支持。** 具体显卡、驱动与 WSL 组合需单独检查，不宣称所有 A 卡均已验证。
- 保持 AstrBot HTTP 请求兼容，修复运行环境 Python 路径与跨平台源码换行校验问题。

## 初始发布 · 2026-10-25

- 发布肥鱼 RVC v1 至 Hugging Face 与 ModelScope，保留原选定模型和匹配索引。
- 发布 AstrBot 插件 v0.1.0 与独立声音服务，提供 TTS、音频换声、可选 MIDI／ABC 按谱唱歌、异步任务与鉴权下载。
- 既有肥鱼 TTS 模型提供可爱与温柔两条声线。
