ARG PYTHON_BASE
FROM ${PYTHON_BASE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN groupadd --gid 65532 skillchain \
    && useradd --uid 65532 --gid 65532 --no-create-home --shell /usr/sbin/nologin skillchain

WORKDIR /opt/skillchain
COPY deploy/authoring/requirements.lock /opt/skillchain/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r requirements.lock
COPY src/skillchain /usr/local/lib/python3.12/site-packages/skillchain

USER 65532:65532
ENTRYPOINT []
