FROM continuumio/miniconda3:24.1.2-0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN conda create -n research-agent python=3.12 -y && \
    conda clean -afy

ENV PATH=/opt/conda/envs/research-agent/bin:$PATH

COPY requirements.txt subtopics.yaml ./
COPY app/ ./app/

RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

RUN mkdir -p /app/papers/.index

VOLUME ["/app/papers"]

ENTRYPOINT ["python", "-m", "app.agent"]
