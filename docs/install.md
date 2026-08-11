# Install

## Requirements

- Linux x86_64, NVIDIA GPU, driver r580+ (CUDA 13)
- CUDA 13 toolkit with `nvcc` — compiles the C++ extensions at install time and
  JIT-compiles CUDA kernels on first use
- Python >= 3.10 and [`uv`](https://docs.astral.sh/uv/)

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv # create a virtual environment
uv pip install -e . # install FreeToken in editable mode
# if you need flashinfer/sglang-kernels, install the accel extras: 
uv pip install -e ".[accel]"
```

## Verify

```bash
source .venv/bin/activate
ft serve --model ~/path/to/Qwen3.6-35B-A3B
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}]}'
```
