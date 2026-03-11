#!/usr/bin/env bash
set -euo pipefail

STORAGE_ROOT=""
ORIGINAL_ARGS=("$@")

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_ubuntu2204_cuda12_docker.sh [--storage-root PATH]

Options:
  --storage-root PATH  Host path for Docker/containerd storage (recommended on SURF volume)
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --storage-root)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --storage-root" >&2
        usage
        exit 1
      fi
      STORAGE_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${EUID}" -ne 0 ]]; then
  echo "Re-running with sudo..."
  exec sudo -E bash "$0" "${ORIGINAL_ARGS[@]}"
fi

if [[ ! -f /etc/os-release ]]; then
  echo "/etc/os-release not found; unsupported environment." >&2
  exit 1
fi

. /etc/os-release
if [[ "${ID}" != "ubuntu" || "${VERSION_ID}" != "22.04" ]]; then
  echo "This script targets Ubuntu 22.04. Detected: ${ID} ${VERSION_ID}" >&2
fi

if [[ -n "${STORAGE_ROOT}" ]]; then
  if [[ "${STORAGE_ROOT}" != /* ]]; then
    echo "--storage-root must be an absolute path, got: ${STORAGE_ROOT}" >&2
    exit 1
  fi
  if [[ ! -d "${STORAGE_ROOT}" ]]; then
    echo "Storage root does not exist: ${STORAGE_ROOT}" >&2
    echo "Mount/create your SURF volume path first, then rerun." >&2
    exit 1
  fi
fi

configure_storage_roots() {
  local storage_root="$1"
  local docker_root="${storage_root}/docker"
  local containerd_root="${storage_root}/containerd/root"
  local containerd_state="${storage_root}/containerd/state"
  local tmp_cfg

  mkdir -p "${docker_root}" "${containerd_root}" "${containerd_state}"
  install -m 0755 -d /etc/docker /etc/containerd

  python3 - "${docker_root}" <<'PY'
import json
import pathlib
import sys

daemon_json = pathlib.Path("/etc/docker/daemon.json")
docker_root = sys.argv[1]

config = {}
if daemon_json.exists():
    try:
        config = json.loads(daemon_json.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {daemon_json}: {exc}")

config["data-root"] = docker_root
daemon_json.write_text(json.dumps(config, indent=2) + "\n")
PY

  if [[ ! -f /etc/containerd/config.toml ]]; then
    containerd config default > /etc/containerd/config.toml
  fi

  # Ensure top-level containerd root/state are always present and point to storage_root,
  # even when an existing config file omits these keys.
  tmp_cfg="$(mktemp)"
  awk \
    -v desired_root="${containerd_root}" \
    -v desired_state="${containerd_state}" \
    '
    BEGIN {
      saw_table = 0
      root_set = 0
      state_set = 0
    }
    {
      if ($0 ~ /^[[:space:]]*\[/ && !saw_table) {
        if (!root_set) {
          print "root = \"" desired_root "\""
          root_set = 1
        }
        if (!state_set) {
          print "state = \"" desired_state "\""
          state_set = 1
        }
        saw_table = 1
      }

      if (!saw_table) {
        if ($0 ~ /^[[:space:]]*root[[:space:]]*=/) {
          if (!root_set) {
            print "root = \"" desired_root "\""
            root_set = 1
          }
          next
        }
        if ($0 ~ /^[[:space:]]*state[[:space:]]*=/) {
          if (!state_set) {
            print "state = \"" desired_state "\""
            state_set = 1
          }
          next
        }
      }

      print $0
    }
    END {
      if (!saw_table) {
        if (!root_set) {
          print "root = \"" desired_root "\""
        }
        if (!state_set) {
          print "state = \"" desired_state "\""
        }
      }
    }
    ' /etc/containerd/config.toml > "${tmp_cfg}"
  mv "${tmp_cfg}" /etc/containerd/config.toml
}

echo "[1/6] Installing Docker Engine..."
apt-get update
apt-get install -y ca-certificates curl gnupg lsb-release python3
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  ${VERSION_CODENAME} stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "[2/6] Enabling Docker and containerd services..."
systemctl enable containerd
systemctl enable docker

TARGET_USER="${SUDO_USER:-${USER}}"
if id -nG "${TARGET_USER}" | grep -qw docker; then
  echo "[3/6] User '${TARGET_USER}' already in docker group."
else
  echo "[3/6] Adding user '${TARGET_USER}' to docker group..."
  usermod -aG docker "${TARGET_USER}"
fi

echo "[4/6] Installing NVIDIA Container Toolkit..."
distribution="${ID}${VERSION_ID}"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker

if [[ -n "${STORAGE_ROOT}" ]]; then
  echo "[5/6] Configuring Docker/containerd storage under ${STORAGE_ROOT}..."
  configure_storage_roots "${STORAGE_ROOT}"
else
  echo "[5/6] Using default Docker/containerd storage paths under /var/lib."
fi

echo "[6/6] Restarting services and validating installation..."
systemctl restart containerd
systemctl restart docker

docker --version
docker info --format 'Docker Root Dir: {{.DockerRootDir}}'
nvidia-smi || true

cat <<EOF

Bootstrap finished.

Next steps:
1. Log out and log in again (or run: newgrp docker) so docker group membership is active.
2. Validate GPU access in containers:
   docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
3. Build project image from repository root:
   make docker-build
4. To keep project data off root disk, run make commands with volume-backed host paths:
   make hspot-convert HOST_DATA_ROOT=<volume>/data HOST_RESULTS_ROOT=<volume>/results HOST_WEIGHTS_ROOT=<volume>/weights

EOF
