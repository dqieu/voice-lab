# Build in the isolated app environment; no training libraries are collected.
import os
from pathlib import Path
import shutil
import sys

import imageio_ffmpeg
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH).parents[1]
work = root / "packaging/macos/build"
work.mkdir(parents=True, exist_ok=True)
ffmpeg = work / "ffmpeg"
shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg)
ffmpeg.chmod(0o755)
sys.path.insert(0, str(root / "packaging/macos"))
from licenses import collect_licenses
licenses = collect_licenses(work / "ThirdPartyLicenses", ffmpeg)
datas = [(str(root / "src/voice_lab/web"), "voice_lab/web")]
datas += [(str(licenses), "ThirdPartyLicenses")]
# Librosa uses lazy module stubs, and pkg_resources needs pysptk metadata.
datas += collect_data_files("librosa") + copy_metadata("pysptk")
hidden = collect_submodules("librosa") + ["webview.platforms.cocoa", "pkg_resources"]
identity = os.environ.get("VOICE_LAB_SIGNING_IDENTITY") or None
a = Analysis([str(root / "packaging/macos/launcher.py")], pathex=[str(root / "src")],
             binaries=[(str(ffmpeg), "bin")], datas=datas, hiddenimports=hidden,
             excludes=["torch", "torchaudio", "lightning", "transformers", "pytest", "tkinter"],
             module_collection_mode={"librosa": "py"},
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Voice Lab", console=False,
          target_arch=None, codesign_identity=identity, entitlements_file=str(root / "packaging/macos/entitlements.plist"))
coll = COLLECT(exe, a.binaries, a.datas, name="Voice Lab")
app = BUNDLE(coll, name="Voice Lab.app", bundle_identifier="org.voicelab.desktop", version="0.1.0",
             info_plist={"CFBundleDisplayName": "Voice Lab", "NSHighResolutionCapable": True,
                         "LSMinimumSystemVersion": "14.0", "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True}})
