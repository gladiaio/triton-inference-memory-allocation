FROM nvcr.io/nvidia/tritonserver:23.08-py3
# FROM nvcr.io/nvidia/tritonserver:23.07-py3
# FROM nvcr.io/nvidia/tritonserver:22.12-py3

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

COPY requirements-server.txt ./
RUN pip install -r requirements-server.txt

COPY scripts/entrypoint.sh /scripts/
ENTRYPOINT [ "/scripts/entrypoint.sh" ]

WORKDIR /models
EXPOSE 8000 8001 8002