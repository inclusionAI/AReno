FROM node:22-bookworm-slim AS frontend
WORKDIR /build/web
COPY arenoflow/web/package*.json ./
RUN npm ci
COPY arenoflow/web/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /workspace
COPY arenoflow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# Source is read as metadata; the local control plane never imports CUDA code.
COPY areno/ areno/
COPY docs/ docs/
COPY README.md ./
COPY .github/workflows/create-docker-image.yml .github/workflows/create-docker-image.yml
COPY arenoflow/ arenoflow/
COPY --from=frontend /build/static/ arenoflow/static/
EXPOSE 8787
CMD ["python", "-m", "arenoflow.server", "--host", "0.0.0.0", "--data-dir", "/data"]
