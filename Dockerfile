# Stoa / PlanMetrics: web app (Node + Vite build) and PlanQA pipeline (Python) in one container.
#
#   cp ../Junction/models/model_best_val_loss_var.pkl pipeline/models/   # layout weights (not in git, 209 MB)
#   docker build -t stoa .
#   docker run -p 3000:3000 -e GEMINI_API_KEY=... stoa                    # -> http://localhost:3000
#
# Only port 3000 is exposed; the pipeline listens on 127.0.0.1:8765 inside the container and is reached
# through the Node server (/api/planqa/*). Options for serve.py: -e PLANQA_ARGS="--vision --min-conf-symbols 0.65".
# Built and tested for linux/amd64 (PaddlePaddle CPU wheels).

# ------------------------------------------------------------------ 1) frontend build
FROM node:22-bookworm-slim AS web
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY index.html metadata.json tsconfig.json vite.config.ts ./
COPY src/ src/
RUN npm run build

# ------------------------------------------------------------------ 2) runtime: Python + Node
FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NODE_ENV=production \
    PORT=3000 \
    PLANQA_URL=http://127.0.0.1:8765 \
    PLANQA_ARGS="" \
    PADDLE_PDX_CACHE_HOME=/opt/paddlex

# OpenCV (libGL, glib), PaddlePaddle (OpenMP), tini as init (forwards signals, reaps both processes)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 tini \
 && rm -rf /var/lib/apt/lists/*

# Node runtime from the build stage (same Debian release, same glibc)
COPY --from=web /usr/local/bin/node /usr/local/bin/node

WORKDIR /app

# Python dependencies first (cached layer): PyTorch CPU build separately, then the rest
COPY pipeline/requirements.txt pipeline/requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r pipeline/requirements.txt

COPY pipeline/ pipeline/
RUN test -f pipeline/models/model_best_val_loss_var.pkl || { \
      echo "ERROR: pipeline/models/model_best_val_loss_var.pkl missing (CubiCasa5k layout weights, see Dockerfile header)"; \
      exit 1; }
# PaddleOCR downloads its models on first use: do it at build time so the container starts offline-ready
RUN cd pipeline && python -c "from planqa.ocr import PaddleReader; PaddleReader(model='small'); print('PaddleOCR models ready')"

# Node server: dependencies (incl. tsx to run server.ts), built frontend, server code + data
COPY --from=web /app/node_modules node_modules
COPY --from=web /app/dist dist
COPY package.json server.ts tsconfig.json vite.config.ts index.html metadata.json ./
COPY server/ server/
COPY docker/start.sh /usr/local/bin/start.sh
# files from the host may be readable by their owner only (mode 600): make the app readable for the 'app' user
RUN chmod 755 /usr/local/bin/start.sh \
 && chmod -R a+rX /app/pipeline /app/server /app/dist \
 && chmod a+r /app/*.ts /app/*.json /app/*.html \
 && useradd --create-home --uid 10001 app \
 && mkdir -p /app/pipeline/out /opt/paddlex \
 && chown -R app:app /app/pipeline/out /opt/paddlex
USER app

EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||3000)+'/api/planqa/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
ENTRYPOINT ["tini", "--"]
CMD ["start.sh"]
