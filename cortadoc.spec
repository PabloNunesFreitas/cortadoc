# -*- mode: python ; coding: utf-8 -*-
# Build: pyinstaller cortadoc.spec
# Requer que scripts/baixar_modelos.py tenha rodado antes (modelos no ckpt/).

# Modelo ONNX embutido junto do app (baixado por scripts/baixar_modelos.py).
datas = [("cortadoc/models/*.onnx", "models")]

hiddenimports = ["onnxruntime", "cv2", "numpy", "img2pdf", "fitz"]

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=["torch", "matplotlib", "tkinter", "PyQt5", "PyQt6"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="CortaDoc",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)
