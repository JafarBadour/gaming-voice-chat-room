# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Voice Chat Client.
Build with:  pyinstaller client.spec
"""

from PyInstaller.utils.hooks import collect_all

# ── Collect all data / binaries / hidden-imports for tricky packages ──────
# collect_all returns (datas, binaries, hiddenimports) for each package.

_all_datas = []
_all_binaries = []
_all_hidden = []

for pkg in [
    'pyrnnoise',           # rnnoise.dll + Python wrappers
    'aec_audio_processing',# .pyd + webrtc DLL in files/
    '_sounddevice_data',   # PortAudio DLLs
    'audiolab',            # pyrnnoise dependency (FFmpeg-based resampler)
    'sounddevice',
]:
    try:
        d, b, h = collect_all(pkg)
        _all_datas += d
        _all_binaries += b
        _all_hidden += h
    except Exception:
        pass

# ── Analysis ──────────────────────────────────────────────────────────────

a = Analysis(
    ['client.py'],
    pathex=[],
    binaries=_all_binaries,
    datas=_all_datas,
    hiddenimports=_all_hidden + [
        'PyQt5.sip',
        'numpy',
        'aiortc',
        'websockets',
        'websockets.legacy',
        'websockets.legacy.client',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'matplotlib', 'scipy', 'pandas',
        'pytest', 'IPython', 'notebook', 'sphinx',
    ],
    noarchive=False,
)

# ── Bundle ────────────────────────────────────────────────────────────────

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='VoiceChat',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # windowed (no black console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,              # set to 'icon.ico' if you have one
)
