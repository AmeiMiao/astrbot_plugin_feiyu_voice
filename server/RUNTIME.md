# GPU runtime setup

The service does not install GPU dependencies or models into AstrBot. Use separate inference environments.

## RVC

Clone https://github.com/IAHispano/Applio at commit `67a38f6abab3e90a55efa534ea3185a222c102f0`. Install its inference dependencies in a separate Python 3.11 environment; the validated local GPU stack is PyTorch 2.11.0+cu128 / CUDA 12.8. Copy the files in `runtime_compat/Applio/` over the matching paths of that checkout. These are the exact source snapshots checked by the model release's `runtime_pins.json`; MIT attribution is retained in that directory.

Download the ContentVec and RMVPE files from the fixed upstream URLs in `runtime_compat/upstream_assets.json`, preserving each destination path and SHA256. These files come from IAHispano/Applio on Hugging Face, revision `b81f9708367b124b29a47788e4c89b721166234f`. They are not bundled. The model runner's runtime pins verify them before conversion.

In the downloaded RVC model directory copy `runtime.template.json` to `runtime.local.json`, set `repo` to the Applio Linux path and `wsl_python` to that environment's Python. Preserve the provided settings unless intentionally validating another setup. `run_rvc.py --verify-only` checks model assets; an actual conversion additionally checks the runtime and network digest.

## TTS

Download [FeiyuVoice-CosyVoice3](https://huggingface.co/Esmerubphire/FeiyuVoice-CosyVoice3) and follow its runtime template and README. It requires the separate CosyVoice checkout and environment described in its model card. Do not install those GPU dependencies into the CPU API environment or AstrBot.

## Optional score singing

Use SoulX-Singer code commit `81aeb3ae772c70093c3de74dc23c92d983801ae4` and the SVS model from the official repository. Copy `configs/singing.example.json` to `configs/singing.json`, fill the runtime paths, and provide your own aligned prompt WAV/JSON plus their SHA256 values. The vocabulary is included with upstream attribution. No private reference recordings or annotations are distributed.

Then set `singing` in `voice_service.json` to `{"config":"configs/singing.json","rvc_voice":"feiyu"}`. Without this optional setup, TTS and RVC work independently and score singing is reported unavailable. This is a local service, not a hosted inference provider. Clean-machine portability across arbitrary GPUs has not been certified.
