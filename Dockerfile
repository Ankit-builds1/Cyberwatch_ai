# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/cyberwatch

WORKDIR /app

RUN groupadd --system cyberwatch \
    && useradd --system --gid cyberwatch --create-home cyberwatch

COPY requirements-cli.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -r requirements-cli.txt

COPY docker_entrypoint.py download_models.py predict.py utils.py ./
RUN mkdir -p /app/models \
    && chown -R cyberwatch:cyberwatch /app /home/cyberwatch

USER cyberwatch

ENTRYPOINT ["python", "docker_entrypoint.py"]
CMD ["info"]
