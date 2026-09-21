FROM python:3.10 AS main

WORKDIR /app

# Install pandoc and netcat
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    pandoc \
    netcat-openbsd \
    libgl1 \  
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download standard NLTK data, to prevent unstructured from downloading packages at runtime
RUN python -m nltk.downloader -d /app/nltk_data punkt_tab averaged_perceptron_tagger
ENV NLTK_DATA=/app/nltk_data

# Disable Unstructured analytics
ENV SCARF_NO_ANALYTICS=true

COPY . .

# --- Build provenance -------------------------------------------------------------------
# Placed AFTER the dependency layers on purpose: these change on every build, and putting them
# earlier would invalidate the pip cache each time. Defaults keep a plain `docker build .`
# working unchanged for developers -- it simply reports `unknown`, which is the truth about it.
ARG BUILD_REVISION=unknown
ARG BUILD_DIRTY=unknown
ARG BUILD_TIME=unknown
ENV BUILD_REVISION=${BUILD_REVISION} \
    BUILD_DIRTY=${BUILD_DIRTY} \
    BUILD_TIME=${BUILD_TIME}
# The LABELs answer "what is this image?" from the registry, WITHOUT running it -- which is the
# only way to resolve a digest that is already deployed. The ENV answers the same question on the
# wire. Both are needed: whoever holds the digest and whoever holds only the URL are rarely the
# same person.
LABEL org.opencontainers.image.revision="${BUILD_REVISION}" \
      org.opencontainers.image.created="${BUILD_TIME}" \
      org.opencontainers.image.source="https://github.com/fifthseason-ai/rag_api" \
      ai.fifthseason.build.dirty="${BUILD_DIRTY}"

CMD ["python", "main.py"]
