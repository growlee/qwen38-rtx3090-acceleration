FROM ghcr.io/syv-ai/qwen38-27b-rtx3090@sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0

LABEL org.opencontainers.image.source="https://github.com/FlashML-org/FreeToken" \
      org.opencontainers.image.revision="cac247a860e316e06580d05aeb05f2e647bde214" \
      ai.server.freetoken.version="0.1.3" \
      ai.server.experiment="qwen38-nvfp4-20260918"

USER 0:0
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.offline.lock /opt/freetoken013-requirements.lock

# Existing pinned base already contains CPython 3.12.3, gcc-13 and nvcc 13.0.88.
# Wheelhouse is a build-context read-only mount, never copied into the image.
RUN --network=none --mount=type=bind,source=wheelhouse,target=/wheels,readonly \
    test -f /usr/include/python3.12/Python.h \
    && test -f /usr/include/stdio.h \
    && /usr/bin/python3.12 -m venv /opt/freetoken013 \
    && /opt/freetoken013/bin/python -m pip install --no-index --no-deps --no-compile --require-hashes \
       --find-links=/wheels -r /opt/freetoken013-requirements.lock \
    && /opt/freetoken013/bin/python -m pip check \
    && /opt/freetoken013/bin/python -c "import importlib.metadata as m, torch; assert m.version('freetoken') == '0.1.3'; assert m.version('triton') == '3.6.0'; assert torch.version.cuda.startswith('13.0'); print(torch.__version__, torch.version.cuda)" \
    && gcc --version \
    && nvcc --version \
    && mkdir -p /home/freetoken /cache \
    && chown 65532:65532 /home/freetoken /cache

ENV PATH="/opt/freetoken013/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    VIRTUAL_ENV="/opt/freetoken013" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false HOME="/home/freetoken" \
    XDG_CACHE_HOME="/cache" TRITON_CACHE_DIR="/cache/triton" \
    TORCH_EXTENSIONS_DIR="/cache/torch-extensions" HF_HOME="/cache/huggingface"

USER 65532:65532
WORKDIR /home/freetoken
ENTRYPOINT ["/opt/freetoken013/bin/ft"]
CMD ["--version"]
