"""Leitura de arquivos de entrada: imagens (JPG/PNG/...) e PDFs (rasterizados)."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("cortadoc.io")

EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DPI_PDF = 300
MAX_LADO_PX = 6000  # imagens maiores são reduzidas (evita estouro de memória)


def _normalizar_bgr(arr: np.ndarray, n_canais: int) -> np.ndarray:
    """Converte pixmap de N canais (1=cinza, 3=RGB, 4=RGBA) para BGR."""
    if n_canais == 1:
        return cv2.cvtColor(arr[:, :, 0] if arr.ndim == 3 else arr, cv2.COLOR_GRAY2BGR)
    if n_canais == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if n_canais == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    raise ValueError(f"Pixmap com {n_canais} canais não suportado")


def _limitar_tamanho(img: np.ndarray, origem: str) -> np.ndarray:
    h, w = img.shape[:2]
    maior = max(h, w)
    if maior <= MAX_LADO_PX:
        return img
    escala = MAX_LADO_PX / maior
    novo = cv2.resize(img, (int(w * escala), int(h * escala)),
                      interpolation=cv2.INTER_AREA)
    log.info("%s: reduzida de %dx%d para %dx%d", origem, w, h,
             novo.shape[1], novo.shape[0])
    return novo


def carregar_paginas(caminho: str | Path) -> list[np.ndarray]:
    """Devolve lista de imagens BGR do arquivo (1 por página se for PDF).

    Levanta ValueError com mensagem clara se o arquivo for ilegível.
    """
    caminho = Path(caminho)
    ext = caminho.suffix.lower()
    log.debug("Carregando %s (ext=%s, %.1f KB)", caminho.name, ext,
              caminho.stat().st_size / 1024 if caminho.exists() else -1)

    if not caminho.exists():
        raise ValueError(f"Arquivo não existe: {caminho}")
    if caminho.stat().st_size == 0:
        raise ValueError(f"Arquivo vazio (0 bytes): {caminho.name}")

    if ext == ".pdf":
        import fitz
        try:
            doc = fitz.open(str(caminho))
        except Exception as e:
            raise ValueError(f"PDF ilegível ({caminho.name}): {e}") from e
        if doc.needs_pass:
            doc.close()
            raise ValueError(f"PDF protegido por senha: {caminho.name}")
        zoom = DPI_PDF / 72
        mat = fitz.Matrix(zoom, zoom)
        paginas = []
        for num, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8)
            arr = arr.reshape(pix.height, pix.width, pix.n) if pix.n > 1 \
                else arr.reshape(pix.height, pix.width)
            bgr = _normalizar_bgr(arr, pix.n)
            paginas.append(_limitar_tamanho(bgr, f"{caminho.name} p{num + 1}"))
        doc.close()
        if not paginas:
            raise ValueError(f"PDF sem páginas: {caminho.name}")
        log.debug("%s: %d página(s) rasterizada(s) a %d dpi",
                  caminho.name, len(paginas), DPI_PDF)
        return paginas

    if ext in EXTENSOES_IMAGEM:
        # np.fromfile + imdecode funciona com acentos/unicode no Windows,
        # onde cv2.imread falha com caminhos não-ASCII.
        dados = np.fromfile(str(caminho), dtype=np.uint8)
        img = cv2.imdecode(dados, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(
                f"Imagem corrompida ou formato inválido: {caminho.name}")
        return [_limitar_tamanho(img, caminho.name)]

    raise ValueError(f"Extensão não suportada: {ext} ({caminho.name})")
