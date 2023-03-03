FROM python:3.11

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

WORKDIR /
RUN mkdir preds

# Install dependencies
COPY poetry.lock pyproject.toml ./
RUN poetry check

RUN poetry install --no-interaction --no-cache --without dev --no-root 

# Run your app
COPY . /
RUN poetry run python3 -m src.training
# CMD [ "poetry", "run", "python", "-c", "print('Hello, World!')" ]