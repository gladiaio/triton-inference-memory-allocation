# STAGE: POETRY INSTALL REQUIREMENTS
FROM python:3.9.18-slim-bullseye as tritonclient

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

RUN apt-get update && \
    apt-get install -y build-essential libpq-dev gcc curl wget git git-lfs libsndfile1 locales locales-all

COPY requirements.txt ./

RUN pip install -r requirements.txt

WORKDIR /workdir

COPY src src

EXPOSE 8888

CMD ["uvicorn", "src.send_request_to_triton_server:app", "--port", "8888"]