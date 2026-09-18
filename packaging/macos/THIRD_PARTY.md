# Third-party components

Bundled Python dependencies retain their supplied license notices in this
directory. `packages.json` records the build environment's package versions.
Python and PyInstaller also retain their notices collected by the bundler.

The FFmpeg executable is supplied unchanged by the imageio-ffmpeg 0.6.0 PyPI
wheel. This arm64 executable reports FFmpeg 7.1 with GPL components enabled;
its license summary and exact configuration are retained in `ffmpeg-build.txt`.

- Binary provider and release scripts: https://github.com/imageio/imageio-ffmpeg/tree/v0.6.0
- FFmpeg 7.1 source: https://github.com/FFmpeg/FFmpeg/tree/n7.1
- FFmpeg license text: https://github.com/FFmpeg/FFmpeg/blob/n7.1/COPYING.GPLv2
- FFmpeg licensing details: https://www.ffmpeg.org/legal.html

This is a local development app. Public redistribution of third-party binaries
must retain their license notices and satisfy applicable source-code obligations.
