#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Re-running with sudo..."
  exec sudo -E bash "$0" "$@"
fi

if [[ ! -f /etc/os-release ]]; then
  echo "/etc/os-release not found; unsupported environment." >&2
  exit 1
fi

. /etc/os-release
if [[ "${ID}" != "ubuntu" || "${VERSION_ID}" != "22.04" ]]; then
  echo "This script targets Ubuntu 22.04. Detected: ${ID} ${VERSION_ID}" >&2
fi

echo "[1/5] Installing Docker Engine..."
apt-get update
apt-get install -y ca-certificates curl gnupg lsb-release
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  ${VERSION_CODENAME} stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "[2/5] Enabling Docker service..."
systemctl enable docker
systemctl restart docker

TARGET_USER="${SUDO_USER:-${USER}}"
if id -nG "${TARGET_USER}" | grep -qw docker; then
  echo "[3/5] User '${TARGET_USER}' already in docker group."
else
  echo "[3/5] Adding user '${TARGET_USER}' to docker group..."
  usermod -aG docker "${TARGET_USER}"
fi

echo "[4/5] Installing NVIDIA Container Toolkit..."
distribution="${ID}${VERSION_ID}"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "[5/5] Validating installation..."
docker --version
nvidia-smi || true

cat <<EOF

Bootstrap finished.

Next steps:
1. Log out and log in again (or run: newgrp docker) so docker group membership is active.
2. Validate GPU access in containers:
   docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
3. Build project image from repository root:
   make docker-build

EOF
