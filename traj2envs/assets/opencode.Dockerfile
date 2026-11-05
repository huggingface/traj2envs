# opencode as a single-file image, for `traj2envs publish --opencode-image`: each task image copies the binary
# from it (COPY --from), so every image shares one identical 214 MB layer instead of pushing its own copy.
# The binary is the standalone linux-x64 build npm's opencode-ai package installs (no node or npm needed).
#   docker buildx build --platform linux/amd64 -f opencode.Dockerfile -t docker.io/<you>/t2envs:opencode-1.19.5 --push .
FROM debian:bookworm-slim AS fetch
ARG VERSION=1.19.5
ARG SHA1=f72482d72266ae10728f55c81faea723c9899635
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl -fsSLo /tmp/opencode.tgz https://registry.npmjs.org/opencode-linux-x64/-/opencode-linux-x64-${VERSION}.tgz \
    && echo "${SHA1}  /tmp/opencode.tgz" | sha1sum -c - \
    && tar xzf /tmp/opencode.tgz -C /tmp && install -m 755 /tmp/package/bin/opencode /opencode

FROM scratch
COPY --from=fetch /opencode /usr/local/bin/opencode
