#!/busybox/sh
# Build and push every task environment of a suite with Kaniko: no Docker daemon needed, so it
# runs in hosted jobs (e.g. HF Jobs) with a datacenter uplink and a native linux/amd64 machine.
#
# usage:  kaniko_publish.sh SUITE_DIR REPOSITORY
# env:    DOCKER_USERNAME, DOCKER_PASSWORD   registry credentials (pass them as job secrets)
#         KANIKO_EXTRA_ARGS                  optional, e.g. "--insecure" for a plain-http test registry
# output: one line per task, then DONE
#           PIN  <family>/<task> <repository>@sha256:...   (apply with: traj2envs pin SUITE_DIR LOG)
#           SKIP <family>/<task> already pinned
#           FAIL <family>/<task> <last build lines>
set -u
suite=$1
repository=$2
state=/kaniko/traj2envs  # Kaniko wipes the filesystem between builds (--cleanup) but keeps /kaniko
mkdir -p "$state" /kaniko/.docker
: > "$state/pushed_environments"

registry=${repository%%/*}
case "$registry" in
  *.*|*:*) ;;               # explicit host, e.g. ghcr.io or localhost:5001
  *) registry=docker.io ;;  # "you/repo" means Docker Hub
esac
[ "$registry" = docker.io ] && registry=https://index.docker.io/v1/
auth=$(printf '%s:%s' "${DOCKER_USERNAME:-}" "${DOCKER_PASSWORD:-}" | base64 | tr -d '\n')
printf '{"auths":{"%s":{"auth":"%s"}}}' "$registry" "$auth" > /kaniko/.docker/config.json

for toml in $(find "$suite" -path '*/tasks/*/task.toml' | sort); do
  task_dir=$(dirname "$toml")
  task=$(basename "$task_dir")
  family=$(basename "$(dirname "$(dirname "$task_dir")")")
  if grep -q "^docker_image = \"$repository@" "$toml"; then
    echo "SKIP $family/$task already pinned"
    continue
  fi
  environment_dir=$task_dir/environment
  environment_hash=$(cd "$environment_dir" && find . -type f | sort | xargs sha256sum | sha256sum | cut -c1-12)
  pinned=$(grep "^$environment_hash " "$state/pushed_environments" | cut -d' ' -f2)
  if [ -z "$pinned" ]; then
    # shellcheck disable=SC2086
    if /kaniko/executor --context "dir://$environment_dir" --dockerfile "$environment_dir/Dockerfile" \
         --destination "$repository:$family-$environment_hash" --digest-file "$state/digest" \
         --cleanup --snapshot-mode=redo ${KANIKO_EXTRA_ARGS:-} > "$state/build.log" 2>&1; then
      pinned="$repository@$(cat "$state/digest")"
      echo "$environment_hash $pinned" >> "$state/pushed_environments"
    else
      echo "FAIL $family/$task $(tail -n 3 "$state/build.log" | tr '\n' ' ' | cut -c1-300)"
      continue
    fi
  fi
  echo "PIN $family/$task $pinned"
done
echo DONE
