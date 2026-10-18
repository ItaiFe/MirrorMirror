#!/usr/bin/env bash
# Turn a fresh Jetson Orin Nano (JetPack 6, Ubuntu 22.04) into the mirror.
# Run it from the repo checkout, as the mirror's own user (not root; it calls
# sudo where it has to), with internet access:
#
#   deploy/setup_orin.sh [--with-mms]
#
#   --with-mms   also install torch (NVIDIA's JetPack wheel) + transformers for the
#                MMS Hebrew fallback voice (tts.he_engine = "mms"). Off by default:
#                the Phonikud voice needs neither.
#
# It is idempotent: re-run it after every `git pull`. Finished steps are skipped,
# and mirror services that are already running are restarted with the new code.
# Overridable: WHISPER_REF, LLAMA_REF (git tags to build), NVP_MODE (power mode).
set -euo pipefail

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; }
step() { printf '\n== %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

WITH_MMS=0
for arg in "$@"; do
    case "$arg" in
        --with-mms) WITH_MMS=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

[ "$(id -u)" -ne 0 ] || die "run this as the mirror user, not as root (it uses sudo itself)"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ME="$(id -un)"
BIN="$HOME/mirror-bin"          # whisper-server, llama-server (+ their sources in src/)
MODELS="$HOME/mirror-models"    # deploy/fetch_models.sh output
VENV="$REPO/.venv"
UNITS="$HOME/.config/systemd/user"
USER_UNITS=(mirror-stt mirror-llm mirror-voice mirror-face)
DPMS_UNITS=(mirror-dpms.service mirror-dpms.timer)   # re-applies "no blanking" every 10 min

# The versions the Mac dev loop was tested with (Homebrew, 2026-10-17).
WHISPER_REF="${WHISPER_REF:-v1.9.5}"
LLAMA_REF="${LLAMA_REF:-v0.7.0}"
# Power mode for `nvpmodel -m`. 0 is the brief's choice; on JetPack 6.2 ("Super")
# the fastest mode on the Orin Nano is MAXN_SUPER instead: list the modes with
# `nvpmodel -q --verbose` (or read /etc/nvpmodel.conf) and pass NVP_MODE=<id>.
NVP_MODE="${NVP_MODE:-0}"

# NVIDIA's PyTorch wheel for JetPack 6.x / Python 3.10 (only with --with-mms).
# To update: pick the newest torch-*-cp310-*.whl for your JetPack under
# https://developer.download.nvidia.com/compute/redist/jp/ (v61 = JetPack 6.1+;
# it also runs on 6.2) and paste its URL here. Newer wheels may also need
# NVIDIA's cuSPARSELt; see "Installing PyTorch for Jetson Platform" in NVIDIA's docs.
TORCH_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"

if ! grep -q "^models_dir = \"$MODELS\"" "$REPO/config/orin.toml"; then
    warn "config/orin.toml does not point at $MODELS (it expects the user 'mirror')."
    warn "Edit its paths to match before rebooting."
fi

# Checks up front, so nothing fails after the long build.
[ -d /etc/gdm3 ] || die "gdm3 not installed: install ubuntu-desktop-minimal or gdm3 first (the face needs the desktop's X server)"
# `systemctl --user` needs this user's own systemd bus, which exists in a normal SSH or
# desktop login, but not under `sudo -u mirror` or `su mirror`.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
[ -S "$XDG_RUNTIME_DIR/bus" ] || die "no user systemd bus at $XDG_RUNTIME_DIR/bus: log in as $ME directly (ssh $ME@<orin> or the desktop) and run this again, not through sudo -u or su"

step "1/8 system packages"
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl libssl-dev \
    python3-venv python3-dev libsdl2-dev portaudio19-dev \
    x11-xserver-utils dconf-cli
# Serial port of the WLED ESP32 (/dev/ttyUSB0) and the sound card.
sudo usermod -aG dialout,audio "$ME"

step "2/8 whisper.cpp $WHISPER_REF and llama.cpp $LLAMA_REF with CUDA -> $BIN"
export PATH="/usr/local/cuda/bin:$PATH"
command -v nvcc >/dev/null || die "nvcc not found: install the JetPack CUDA toolkit (sudo apt install nvidia-jetpack)"
mkdir -p "$BIN/src"
build_server() {  # build_server NAME GIT_URL REF TARGET
    local name="$1" url="$2" ref="$3" target="$4"
    local src="$BIN/src/$name" stamp="$BIN/.$target.ref"
    if [ -x "$BIN/$target" ] && [ "$(cat "$stamp" 2>/dev/null)" = "$ref" ]; then
        echo "skip  $target $ref (built)"
        return
    fi
    [ -d "$src/.git" ] || git clone "$url" "$src"
    git -C "$src" fetch --tags origin
    git -C "$src" -c advice.detachedHead=false checkout --force "$ref"
    # sm_87 = the Orin's GPU. Static, so the binary can be copied out of the tree.
    cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=87 -DBUILD_SHARED_LIBS=OFF
    cmake --build "$src/build" --config Release -j "$(nproc)" --target "$target"
    install -m 755 "$src/build/bin/$target" "$BIN/$target"
    echo "$ref" > "$stamp"
}
build_server whisper.cpp https://github.com/ggml-org/whisper.cpp "$WHISPER_REF" whisper-server
build_server llama.cpp https://github.com/ggml-org/llama.cpp "$LLAMA_REF" llama-server

step "3/8 Python venv $VENV"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' || die "need Python 3.10+"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
EXTRAS="hw,tts"
if [ "$WITH_MMS" = 1 ]; then
    # Before the extras, so pip keeps NVIDIA's CUDA build. Skipped once torch is in.
    "$VENV/bin/pip" show torch >/dev/null 2>&1 || "$VENV/bin/pip" install "$TORCH_WHEEL"
    EXTRAS="$EXTRAS,mms"
fi
"$VENV/bin/pip" install -e "${REPO}[$EXTRAS]"

step "4/8 models -> $MODELS"
PYTHON="$VENV/bin/python" "$REPO/deploy/fetch_models.sh" "$MODELS"

step "5/8 systemd user units"
mkdir -p "$UNITS"
for unit in "${USER_UNITS[@]}"; do
    sed "s|@REPO@|$REPO|g" "$REPO/deploy/systemd/$unit.service" > "$UNITS/$unit.service"
done
for unit in "${DPMS_UNITS[@]}"; do
    install -m 644 "$REPO/deploy/systemd/$unit" "$UNITS/$unit"
done
sudo loginctl enable-linger "$ME"   # start the user units at boot, without a login
systemctl --user daemon-reload
systemctl --user enable "${USER_UNITS[@]/%/.service}" mirror-dpms.timer
# After a `git pull`: restart whatever is running so it picks up the new code.
systemctl --user try-restart "${USER_UNITS[@]/%/.service}"

step "6/8 desktop: autologin on Xorg, no blanking, no pop-ups"
# GDM: log the mirror user straight in, on Xorg (the face unit talks to X on :0).
sudo python3 - "$ME" <<'PY'
import re, sys
path, user = "/etc/gdm3/custom.conf", sys.argv[1]
want = {"AutomaticLoginEnable": "true", "AutomaticLogin": user, "WaylandEnable": "false"}
try:
    lines = open(path).read().splitlines()
except FileNotFoundError:
    lines = []
key_re = re.compile(r"^\s*#?\s*(%s)\s*=" % "|".join(want))
lines = [l for l in lines if not key_re.match(l)]
if "[daemon]" not in [l.strip() for l in lines]:
    lines.insert(0, "[daemon]")
i = [l.strip() for l in lines].index("[daemon]") + 1
lines[i:i] = ["%s=%s" % kv for kv in want.items()]
open(path, "w").write("\n".join(lines) + "\n")
PY
sudo install -D -m 644 "$REPO/deploy/xorg/10-mirror.conf" /etc/X11/xorg.conf.d/10-mirror.conf
# GNOME: no screen lock, no idle blanking or suspend, no notification banners.
# System-wide dconf defaults, locked so a stray user setting can't undo them.
if ! sudo grep -qs '^system-db:local' /etc/dconf/profile/user; then
    printf 'user-db:user\nsystem-db:local\n' | sudo tee -a /etc/dconf/profile/user >/dev/null
fi
sudo install -d /etc/dconf/db/local.d/locks
sudo tee /etc/dconf/db/local.d/00-mirror >/dev/null <<'EOF'
[org/gnome/desktop/session]
idle-delay=uint32 0

[org/gnome/desktop/screensaver]
lock-enabled=false
idle-activation-enabled=false

[org/gnome/settings-daemon/plugins/power]
sleep-inactive-ac-type='nothing'
idle-dim=false

[org/gnome/desktop/notifications]
show-banners=false
EOF
sudo tee /etc/dconf/db/local.d/locks/00-mirror >/dev/null <<'EOF'
/org/gnome/desktop/session/idle-delay
/org/gnome/desktop/screensaver/lock-enabled
/org/gnome/desktop/screensaver/idle-activation-enabled
/org/gnome/settings-daemon/plugins/power/sleep-inactive-ac-type
/org/gnome/settings-daemon/plugins/power/idle-dim
/org/gnome/desktop/notifications/show-banners
EOF
sudo dconf update

step "7/8 spare the SD card: logs in RAM, no swap file"
sudo install -d /etc/systemd/journald.conf.d
printf '[Journal]\nStorage=volatile\nRuntimeMaxUse=64M\n' |
    sudo tee /etc/systemd/journald.conf.d/10-mirror.conf >/dev/null
sudo systemctl restart systemd-journald
# zram swap (nvzramconfig) lives in RAM and stays. Swap *files* go.
tail -n +2 /proc/swaps | while read -r name type _; do
    if [ "$type" = file ]; then
        echo "removing swap file $name"
        sudo swapoff "$name"
        sudo rm -f "$name"
    fi
done
awk '$1 !~ /^#/ && $3 == "swap" && $1 !~ /zram/ { print "# mirror: no swap on the SD card # " $0; next } { print }' \
    /etc/fstab > "$BIN/fstab.new"
if ! cmp -s /etc/fstab "$BIN/fstab.new"; then
    sudo cp /etc/fstab /etc/fstab.mirror-backup
    sudo install -m 644 "$BIN/fstab.new" /etc/fstab
fi
rm -f "$BIN/fstab.new"

step "8/8 power: nvpmodel $NVP_MODE, jetson_clocks at boot"
if [ "$(sudo nvpmodel -q | tail -n 1 | tr -d '[:space:]')" != "$NVP_MODE" ]; then
    # nvpmodel may ask whether to reboot now; answer no, the reboot comes below.
    printf 'NO\n' | sudo nvpmodel -m "$NVP_MODE"
fi
sudo install -m 644 "$REPO/deploy/systemd/mirror-clocks.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable mirror-clocks.service

step "done"
echo "Reboot to start the mirror: sudo reboot"
echo "Then: systemctl --user status ${USER_UNITS[*]}   and   journalctl --user -f"
