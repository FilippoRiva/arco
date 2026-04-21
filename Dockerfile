# syntax=docker/dockerfile:1

FROM python:3.12-slim

WORKDIR /app

# System deps for pandas/pyarrow/codecarbon/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cached unless requirements.txt changes)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Use non-interactive matplotlib backend (no display on headless Linux)
ENV MPLBACKEND=Agg

# Ollama URL — on Linux use --network=host and point to localhost,
# or override with -e OLLAMA_HOST=http://<host-ip>:11434 at runtime
ENV OLLAMA_HOST=http://localhost:11434

# Copy source code
COPY Agent ./Agent
COPY evaluation ./evaluation
COPY data ./data
COPY config ./config
COPY run_agent.py ./

# /app/runs is the output directory — mount a host volume here to persist results:
#   docker run -v /host/path/runs:/app/runs ...
VOLUME ["/app/runs"]

# Default: bulk_runner with validation preset (1 config, no think time).
# Override all args at runtime, e.g.:
#   docker run --rm --network=host -v $(pwd)/runs:/app/runs data-agent \
#     python evaluation/bulk_runner.py \
#       evaluation/benchmark_dataset.json evaluation/search_space.yaml \
#       --n-configs 50 --think-time 5.0 --save-dir runs/bulk_results/full_run
ENTRYPOINT ["python"]
CMD ["evaluation/bulk_runner.py", \
     "evaluation/benchmark_dataset.json", \
     "evaluation/search_space.yaml", \
     "--n-configs", "1", "--think-time", "0", \
     "--save-dir", "runs/bulk_results/validation"]
