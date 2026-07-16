"""Baixa o modelo ONNX do DocAligner para cortadoc/models/.

Rodado no CI antes do PyInstaller (e uma vez no dev). O app nunca precisa de
internet: o modelo é embutido no executável.

Modelo fastvit_sa24 (heatmap regression) — DocsaidLab/DocAligner, Apache 2.0.
"""

import subprocess
import sys
from pathlib import Path

FILE_ID = "14vUH77v6yGg7zFctUgcT6BzV5Iisg4Dl"
DESTINO = Path(__file__).parent.parent / "cortadoc" / "models" / \
    "fastvit_sa24_h_e_bifpn_256_fp32.onnx"


def main():
    if DESTINO.exists() and DESTINO.stat().st_size > 50_000_000:
        print(f"Modelo já existe: {DESTINO} ({DESTINO.stat().st_size / 1e6:.1f} MB)")
        return
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
    subprocess.run([sys.executable, "-m", "gdown", FILE_ID, "-O", str(DESTINO)],
                   check=True)
    tamanho = DESTINO.stat().st_size
    if tamanho < 50_000_000:
        raise SystemExit(f"ERRO: download suspeito ({tamanho} bytes)")
    print(f"OK: {DESTINO} ({tamanho / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
