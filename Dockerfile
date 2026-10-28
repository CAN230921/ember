# Supply a CUDA 13, SM121-capable vLLM base with the documented native ABI.
# The reference downstream base is not redistributed by this project.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
COPY requirements-runtime.txt /tmp/ember-requirements.txt
COPY requirements-test.txt /tmp/ember-test-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/ember-requirements.txt \
    -r /tmp/ember-test-requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
