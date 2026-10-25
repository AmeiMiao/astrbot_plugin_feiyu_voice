# 肥鱼声音 · AstrBot 插件 v0.1.0

通过独立 HTTP 声音服务调用肥鱼说话、RVC换声和可选的MIDI/ABC按谱唱歌。AstrBot内不加载模型或CUDA。

## 安装与配置

在AstrBot插件管理中从本仓库安装，或下载Release ZIP本地安装。支持的当前版本为AstrBot >=4.29.3,<5，其他平台适配需自行验证。

先按 [server/README.md](server/README.md) 部署本仓库附带的独立声音服务，再在插件配置填写服务地址、API Key、声音ID `feiyu` 和默认风格 `cute`／`gentle`。默认本机服务地址为 `http://127.0.0.1:9880`。模型和运行环境需要另行下载配置，插件安装不会自动部署GPU环境。

- `/肥鱼状态`：查看已安装能力。
- `/肥鱼说话 你好呀，今天也要开心。`：文字转语音。
- `/肥鱼换声` 并附 WAV/FLAC干声或语音：使用RVC转换音色。
- `/肥鱼唱歌 歌词` 并附单音主旋律 MIDI/ABC：SVS后再使用RVC；一字多音用 `_`。需要先配置可选SVS引擎和自备已对齐参考。
- `/肥鱼任务 任务编号`：查询或重新领取当前会话结果。

注册工具 `feiyu_speak`、`feiyu_convert`、`feiyu_sing`。配置支持工具调用的聊天模型后，可由自然语言请求选择工具；命令调用不依赖聊天模型。任务异步排队，完成后在原会话发送WAV；插件自己下载音频，不依赖与服务端共享路径。插件重启恢复待处理任务的轮询，任务领取检查所属会话。

## 模型下载

| 声库 | 下载 |
|---|---|
| 肥鱼TTS | [Hugging Face](https://huggingface.co/Esmerubphire/FeiyuVoice-CosyVoice3) · [ModelScope](https://modelscope.cn/models/Esmerubphire/FeiyuVoice-CosyVoice3) |
| 肥鱼RVC v1 | [Hugging Face](https://huggingface.co/Esmerubphire/FeiyuVoice-RVC) · [ModelScope](https://modelscope.cn/models/Esmerubphire/FeiyuVoice-RVC) |

服务版本、插件版本和模型版本独立管理。兼容包体可在服务端配置切换，插件API不变。RVC转换干净人声，不自动分离伴奏；可选换气衰减需要本曲人工区间。

## 验证范围

在Windows＋WSL、AstrBot4.29.3本地WebChat完成TTS、RVC、按谱唱歌命令及主动音频返回验证，服务接口回归检查和谱面前端检查通过。独立发布版RVC另核对与选定声库的实际音频一致性。QQ、微信、OneBot等外部平台和自然语言模型选择工具仍需配置后各自实测；插件不会代填模型账号或密钥。

模型、训练音频、测试歌曲、虚拟环境、凭据、用户上传文件和私人参考都不在本仓库。运行环境说明见 [server/RUNTIME.md](server/RUNTIME.md)，分发说明见 [NOTICE.md](NOTICE.md)。
