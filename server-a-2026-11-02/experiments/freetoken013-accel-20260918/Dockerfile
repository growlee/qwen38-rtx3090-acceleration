FROM local/ai-server-freetoken:0.1.3-cac247a-nvfp4-20260918
USER 0:0
LABEL ai.server.experiment="freetoken013-accel-20260918"
COPY requirements.offline.lock /opt/freetoken013-accel.lock
RUN --network=none --mount=type=bind,source=wheelhouse,target=/wheels,readonly \
    /opt/freetoken013/bin/python -m pip install --no-index --no-deps --no-compile --require-hashes --find-links=/wheels -r /opt/freetoken013-accel.lock \
    && /opt/freetoken013/bin/python -m pip check \
    && /opt/freetoken013/bin/python -c "import importlib.metadata as m,json; assert m.version('torch')=='2.11.0'; assert m.version('transformers')=='5.16.1'; assert m.version('flashinfer-python')=='0.6.18.post1'; assert m.version('sglang-kernel')=='0.4.5'; print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()},sort_keys=True))"
ENV FLASHINFER_WORKSPACE_BASE=/cache/flashinfer MAX_JOBS=2
USER 65532:65532
