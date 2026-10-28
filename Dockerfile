# The tool sandbox: every tool runs in here, served to the host agent over MCP on :8001.
# No torch, no model code, no weights. Builds for linux/arm64 and linux/amd64.
FROM python:3.12-slim-bookworm

# Node.js LTS (for npx-launched MCP servers): only node and npm, not all of /usr/local.
COPY --from=node:22-bookworm-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:22-bookworm-slim /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# uv / uvx (for uvx-launched MCP servers).
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /usr/local/bin/

COPY requirements-sandbox.txt /tmp/requirements-sandbox.txt
RUN pip install --no-cache-dir -r /tmp/requirements-sandbox.txt

# Headless Chromium for fetch_page's fallback on JavaScript-only pages. Installed as root
# (it apt-installs Chromium's libraries) into a shared path the non-root user can read.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN playwright install --with-deps chromium \
 && chmod -R a+rx /ms-playwright \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY agent_bot/ agent_bot/
COPY sandbox/ sandbox/
COPY config.yaml config.yaml

RUN useradd -m -u 1000 agent && mkdir /data && chown agent /data
USER agent
ENV HOME=/home/agent PYTHONPATH=/app

EXPOSE 8001
CMD ["python", "-m", "sandbox.tool_server"]
