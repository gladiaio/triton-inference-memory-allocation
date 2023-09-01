# STAGE: POETRY INSTALL REQUIREMENTS
FROM nvcr.io/nvidia/tritonserver:22.12-py3 as tritonserver

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN apt-get update && \
    apt-get install -y build-essential libpq-dev gcc curl wget git git-lfs libsndfile1 locales locales-all

WORKDIR /workdir

COPY src src

COPY models /workdir/models

# RUN python3 src/create_sender.py

EXPOSE 8000 8001 8002

CMD ["python3", "src/create_sender.py", "&&", "tritonserver", "--model-repository", "models", "--model-control-mode", "poll", "--repository-poll-secs", "5", "--log-verbose", "1", "--exit-timeout-secs", "1" ]