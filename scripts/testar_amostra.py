"""Teste em lote da detecção: processa arquivos e salva cortes + overlays.

Uso: python3 scripts/testar_amostra.py <arquivos...> [--out PASTA]
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from cortadoc.core.detector import detectar_cantos
from cortadoc.core.warp import retificar
from cortadoc.core.limpeza import limpar_fundo
from cortadoc.core.io_utils import carregar_paginas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("arquivos", nargs="+")
    p.add_argument("--out", default="teste_saida")
    p.add_argument("--folga", type=int, default=40)
    args = p.parse_args()

    out = Path(args.out)
    (out / "cortes").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    for arq in args.arquivos:
        arq = Path(arq)
        try:
            paginas = carregar_paginas(arq)
        except Exception as e:
            print(f"{arq.name}: ERRO {e}")
            continue
        for i, img in enumerate(paginas):
            suf = f"_p{i+1}" if len(paginas) > 1 else ""
            nome = f"{arq.stem}{suf}"
            cantos, metodo = detectar_cantos(img)
            if cantos is None:
                print(f"{nome}: NENHUMA detecção")
                continue
            vis = img.copy()
            cv2.polylines(vis, [cantos.astype(int)], True, (0, 255, 0), 8)
            cv2.imwrite(str(out / "overlays" / f"{nome}.jpg"), vis)
            corte = limpar_fundo(retificar(img, cantos, folga_px=args.folga))
            cv2.imwrite(str(out / "cortes" / f"{nome}.png"), corte)
            print(f"{nome}: {metodo}  {img.shape[1]}x{img.shape[0]} -> "
                  f"{corte.shape[1]}x{corte.shape[0]}")


if __name__ == "__main__":
    main()
