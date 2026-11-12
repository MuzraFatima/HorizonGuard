# Reproducible image: builds the dashboard, installs the package with pinned
# dependencies, generates the benchmark and serves the monitor.
FROM node:20-slim AS dashboard
WORKDIR /app/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY dashboard/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements-lock.txt pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY docs ./docs
RUN pip install --no-cache-dir -r requirements-lock.txt && pip install --no-cache-dir -e .
COPY --from=dashboard /app/dashboard/dist ./dashboard/dist
RUN horizonguard data --out data/driftbench-cf
EXPOSE 8765
CMD ["horizonguard", "serve", "--data", "data/driftbench-cf", "--host", "0.0.0.0"]
