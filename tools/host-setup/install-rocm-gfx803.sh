#!/usr/bin/env bash
set -Eeuo pipefail

# Bootstrap the Schaka gfx803 ROCm stack in an isolated container.
# This deliberately does not replace the host's amdgpu driver or Vulkan stack.
IMAGE='ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-gfx803'
LAUNCHER='/usr/local/bin/rocm-gfx803-shell'

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

if [[ $EUID -eq 0 ]]; then
  SUDO=()
  TARGET_USER="${SUDO_USER:-root}"
else
  command -v sudo >/dev/null 2>&1 || die 'sudo is required for Docker/host setup.'
  SUDO=(sudo)
  TARGET_USER="$(id -un)"
fi

TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
[[ -n "$TARGET_HOME" && -d "$TARGET_HOME" ]] || TARGET_HOME="$HOME"
TARGET_UID="$(id -u "$TARGET_USER")"
TARGET_GID="$(id -g "$TARGET_USER")"
WORKDIR="$TARGET_HOME/rocm-gfx803-work"

say 'Checking AMD GPU and kernel device nodes'
command -v lspci >/dev/null 2>&1 && lspci -nn -d 1002: | grep -Ei 'VGA|Display|3D' || true
if [[ ! -e /dev/kfd ]]; then
  say 'Trying to load the AMD compute kernel module (amdkfd)'
  "${SUDO[@]}" modprobe amdkfd || true
  command -v udevadm >/dev/null 2>&1 && "${SUDO[@]}" udevadm settle || true
fi
[[ -e /dev/kfd ]] || die 'The host has no /dev/kfd. ROCm cannot reach the GPU yet. Check that amdgpu/amdkfd is loaded for the current kernel; this installer will not replace or patch your kernel driver.'
[[ -d /dev/dri ]] || die 'The host has no /dev/dri directory; the AMD DRM driver is not exposing GPU nodes.'

say 'Checking Docker'
if ! command -v docker >/dev/null 2>&1; then
  say 'Docker is missing; installing Ubuntu package docker.io'
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y docker.io
fi
"${SUDO[@]}" systemctl enable --now docker

# Use Docker without requiring membership in the docker group.
if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo docker)
fi
"${DOCKER[@]}" info >/dev/null 2>&1 || die 'Docker daemon is not accessible, even through sudo.'

# Device nodes are usually owned by the host's video/render groups. Passing
# their numeric GIDs also works when those group names differ inside the image.
GROUP_ARGS=()
for group in video render; do
  gid="$(getent group "$group" | cut -d: -f3 || true)"
  if [[ -n "$gid" ]]; then GROUP_ARGS+=(--group-add "$gid"); fi
done

say "Pulling the prebuilt gfx803 ROCm container (no AI models are downloaded)"
"${DOCKER[@]}" pull "$IMAGE"

say 'Installing a persistent workspace and launcher'
install -d -o "$TARGET_UID" -g "$TARGET_GID" -m 0755 "$WORKDIR"
TMP_LAUNCHER="$(mktemp)"
cat > "$TMP_LAUNCHER" <<'LAUNCHER_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
IMAGE='ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-gfx803'
if [[ $EUID -eq 0 ]]; then
  HOST_HOME="${SUDO_USER:+$(getent passwd "$SUDO_USER" | cut -d: -f6)}"
  [[ -n "$HOST_HOME" ]] || HOST_HOME="$HOME"
  HOST_UID="${SUDO_UID:-0}"
  HOST_GID="${SUDO_GID:-0}"
else
  HOST_HOME="$HOME"
  HOST_UID="$(id -u)"
  HOST_GID="$(id -g)"
fi
WORKDIR="$HOST_HOME/rocm-gfx803-work"
mkdir -p "$WORKDIR"
if docker info >/dev/null 2>&1; then DOCKER=(docker); else DOCKER=(sudo docker); fi
GROUP_ARGS=()
for group in video render; do gid="$(getent group "$group" | cut -d: -f3 || true)"; if [[ -n "$gid" ]]; then GROUP_ARGS+=(--group-add "$gid"); fi; done
exec "${DOCKER[@]}" run --rm -it --device=/dev/kfd --device=/dev/dri "${GROUP_ARGS[@]}" --user "$HOST_UID:$HOST_GID" -e HOME=/tmp -v "$WORKDIR:/workspace" -w /workspace --entrypoint /bin/bash "$IMAGE" --login
LAUNCHER_EOF
"${SUDO[@]}" install -o root -g root -m 0755 "$TMP_LAUNCHER" "$LAUNCHER"
rm -f "$TMP_LAUNCHER"

say 'Compiling and running a HIP kernel on every GPU visible to ROCm'
"${DOCKER[@]}" run --rm --device=/dev/kfd --device=/dev/dri "${GROUP_ARGS[@]}" --entrypoint /bin/bash "$IMAGE" -lc '
set -Eeuo pipefail
printf "ROCm gfx803 container: %s\n" "'"$IMAGE"'"
HIPCC="$(command -v hipcc || true)"
if [[ -z "$HIPCC" ]]; then for candidate in /opt/rocm/*/bin/hipcc /opt/rocm/bin/hipcc; do if [[ -x "$candidate" ]]; then HIPCC="$candidate"; break; fi; done; fi
[[ -n "$HIPCC" && -x "$HIPCC" ]] || { echo "ERROR: hipcc compiler not found in image" >&2; exit 10; }
printf "HIP compiler: %s\n" "$HIPCC"
cat > /tmp/gfx803_hip_smoke.cpp <<"CPP"
#include <hip/hip_runtime.h>
#include <cstdio>
#define CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { std::fprintf(stderr, "HIP failure at %s: %s\n", #call, hipGetErrorString(e)); return 1; } } while (0)
__global__ void add_kernel(const float* a, const float* b, float* c) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < 16) c[i] = a[i] + b[i];
}
int main() {
  int count = 0;
  CHECK(hipGetDeviceCount(&count));
  std::printf("HIP sees %d GPU(s)\n", count);
  if (count < 1) { std::fprintf(stderr, "No HIP GPUs detected\n"); return 2; }
  for (int d = 0; d < count; ++d) {
    CHECK(hipSetDevice(d));
    hipDeviceProp_t p{};
    CHECK(hipGetDeviceProperties(&p, d));
    std::printf("GPU %d: %s\n", d, p.name);
    float ha[16], hb[16], hc[16];
    for (int i = 0; i < 16; ++i) { ha[i] = (float)i; hb[i] = (float)(2 * i); hc[i] = -1.0f; }
    float *a = nullptr, *b = nullptr, *c = nullptr;
    CHECK(hipMalloc((void**)&a, sizeof(ha))); CHECK(hipMalloc((void**)&b, sizeof(hb))); CHECK(hipMalloc((void**)&c, sizeof(hc)));
    CHECK(hipMemcpy(a, ha, sizeof(ha), hipMemcpyHostToDevice)); CHECK(hipMemcpy(b, hb, sizeof(hb), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(add_kernel, dim3(1), dim3(64), 0, 0, a, b, c);
    CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize()); CHECK(hipMemcpy(hc, c, sizeof(hc), hipMemcpyDeviceToHost));
    for (int i = 0; i < 16; ++i) if (hc[i] != ha[i] + hb[i]) { std::fprintf(stderr, "Bad result on GPU %d at index %d\n", d, i); return 3; }
    CHECK(hipFree(a)); CHECK(hipFree(b)); CHECK(hipFree(c));
    std::printf("GPU %d HIP kernel: PASS\n", d);
  }
  std::puts("ROCm/HIP GPU smoke test: PASS");
  return 0;
}
CPP
"$HIPCC" --offload-arch=gfx803 -O2 /tmp/gfx803_hip_smoke.cpp -o /tmp/gfx803_hip_smoke
/tmp/gfx803_hip_smoke
'

say 'ROCm/HIP is running inside the container'
printf 'Launch an interactive HIP/ROCm shell with: %s\n' "$LAUNCHER"
printf 'Your persistent workspace is: %s\n' "$WORKDIR"
printf '\nNote: ROCm libraries are inside the container, not installed into the host OS. The existing host amdgpu/Vulkan setup was left intact.\n'
