ARG PYTHON_IMAGE=python:3.13-slim-bookworm
FROM ${PYTHON_IMAGE}

WORKDIR /app
COPY --chown=65532:65532 bridge.py healthcheck.py /app/

USER 65532:65532
EXPOSE 8000
ENTRYPOINT ["python", "-B", "/app/bridge.py"]
