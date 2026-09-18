#!/bin/bash
set -euo pipefail
APP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$APP_ROOT"
if [[ "$(uname -s)" != Darwin ]]; then
  echo "Build the Mac app on macOS." >&2
  exit 1
fi
APP_UV="${VOICE_LAB_UV:-uv}"
APP_PYTHON="${VOICE_LAB_PYTHON:-python3.11}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$APP_ROOT/packaging/macos/build/uv-cache}"
export NUMBA_CACHE_DIR="$APP_ROOT/packaging/macos/build/numba-cache"
export PYINSTALLER_CONFIG_DIR="$APP_ROOT/packaging/macos/build/pyinstaller-cache"
"$APP_UV" venv --python "$APP_PYTHON" --allow-existing packaging/macos/.venv-build
"$APP_UV" pip install --python packaging/macos/.venv-build/bin/python -r packaging/macos/requirements.lock.txt
packaging/macos/.venv-build/bin/python -m PyInstaller --noconfirm \
  --distpath packaging/macos/dist --workpath packaging/macos/build/pyinstaller packaging/macos/VoiceLab.spec
APP_BUNDLE="$APP_ROOT/packaging/macos/dist/Voice Lab.app"
codesign --verify --deep --strict "$APP_BUNDLE"
"$APP_BUNDLE/Contents/MacOS/Voice Lab" --self-test --data-dir "$APP_ROOT/packaging/macos/build/release-check"
APP_ARCH="$(uname -m)"
APP_STAGE="$(mktemp -d "$APP_ROOT/packaging/macos/build/dmg.XXXXXX")"
trap 'rm -rf "$APP_STAGE"' EXIT
ditto "$APP_BUNDLE" "$APP_STAGE/Voice Lab.app"
ln -s /Applications "$APP_STAGE/Applications"
cp packaging/macos/Install.txt "$APP_STAGE/Install.txt"
APP_DMG="$APP_ROOT/packaging/macos/dist/Voice-Lab-macOS-$APP_ARCH.dmg"
hdiutil create -volname "Voice Lab" -srcfolder "$APP_STAGE" -ov -format UDZO "$APP_DMG"
shasum -a 256 "$APP_DMG" > "$APP_DMG.sha256"
echo "Installer: $APP_DMG"
