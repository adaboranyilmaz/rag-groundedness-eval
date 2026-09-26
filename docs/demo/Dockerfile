# VHS (scripted terminal recording) plus the two tools the demo types: curl and jq.
# The base image is Debian bookworm but its sources name the moving `stable` suite, which
# now points at trixie; pin them to bookworm so the packages match the base.
FROM ghcr.io/charmbracelet/vhs:v0.10.0
RUN sed -i 's/\bstable\b/bookworm/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl jq ca-certificates \
    && rm -rf /var/lib/apt/lists/*
