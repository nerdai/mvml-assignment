FROM python:3.11-slim as python
WORKDIR /

# Configure Poetry
ENV POETRY_VERSION=1.3.2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Add `poetry` to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

# Install dependencies
COPY poetry.lock pyproject.toml ./
RUN poetry install --no-interaction --no-cache --without dev --no-root 

# Copy source code and training data
COPY ./src/ /src
COPY ./utils/ /utils
COPY ./data/fake_news/clean_train.csv /data/fake_news/clean_train.csv

# Build model and store artifact in docker image
RUN poetry run python3 -m src.training

# Make dir in docker container for future inference
RUN mkdir preds