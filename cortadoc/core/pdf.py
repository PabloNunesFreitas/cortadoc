"""Exportação de imagens cortadas para PDF (via img2pdf, sem reencodar JPEG)."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np

try:
    import img2pdf
    _TEM_IMG2PDF = True
except ImportError:
    _TEM_IMG2PDF = False


def _para_jpeg_bytes(img_bgr: np.ndarray, qualidade: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
    if not ok:
        raise RuntimeError("Falha ao codificar JPEG")
    return buf.tobytes()


def exportar_pdf(imagens: list[np.ndarray], destino: Path, qualidade: int = 92) -> Path:
    """Gera um PDF com uma página por imagem (ordem da lista).

    Usa img2pdf quando disponível (JPEG embutido sem reencodar → menor e mais
    rápido). Fallback: Pillow.
    """
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    jpegs = [_para_jpeg_bytes(img, qualidade) for img in imagens]

    if _TEM_IMG2PDF:
        destino.write_bytes(img2pdf.convert(jpegs))
        return destino

    from PIL import Image
    paginas = [Image.open(io.BytesIO(j)).convert("RGB") for j in jpegs]
    paginas[0].save(destino, save_all=True, append_images=paginas[1:])
    return destino
