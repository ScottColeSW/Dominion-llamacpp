# The engine (dominion.engine) has zero third-party imports; everything
# installed here is the service/transport layer -- see pyproject.toml's
# own comment on this being a deliberate dependency tradeoff.
FROM python:3.12-slim

WORKDIR /app

# Dependency manifest copied (and installed) before the rest of the source
# tree, so `pip install` only re-runs on a real dependency change, not on
# every source edit -- most rebuilds during development touch src/, not
# pyproject.toml.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# Runs as a non-root user -- this is a network-facing service. /data is
# where the SQLite history db actually lives (see history.py's DB_PATH --
# the package install directory itself, owned by root from the pip
# install above, isn't writable by this user); docker-compose.yml mounts
# a volume here for persistence across container restarts.
RUN useradd --create-home --shell /bin/bash dominion && \
    mkdir -p /data && chown dominion:dominion /data
USER dominion
ENV DOMINION_DB_PATH=/data/dominion_history.db

EXPOSE 8765
CMD ["uvicorn", "dominion.server.app:app", "--host", "0.0.0.0", "--port", "8765"]
