"""Detecção dos 4 cantos do documento na foto.

Estratégia em camadas:
1. DocAligner (rede neural, heatmap regression) — robusto a sombra, dobra e
   fundo parecido com o papel.
2. Fallback OpenCV (máscara HSV + minAreaRect) se o modelo não estiver
   disponível ou não achar nada.

Os cantos retornados são sempre np.ndarray (4, 2) float32 na ordem
[top-left, top-right, bottom-right, bottom-left].
"""

from __future__ import annotations

import logging
import time

import numpy as np
import cv2

log = logging.getLogger("cortadoc.detector")

def _ordenar_cantos(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype("float32")
    soma = pts.sum(axis=1)
    dif = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(soma)],
        pts[np.argmin(dif)],
        pts[np.argmax(soma)],
        pts[np.argmax(dif)],
    ], dtype="float32")


def detectar_cantos_neural(img_bgr: np.ndarray) -> np.ndarray | None:
    """Detecta os cantos com o modelo ONNX do DocAligner. None se falhar."""
    try:
        from cortadoc.core.neural import detectar_cantos_onnx
        t0 = time.perf_counter()
        pts = detectar_cantos_onnx(img_bgr)
        dt = (time.perf_counter() - t0) * 1000
        if pts is None or pts.shape != (4, 2):
            log.debug("neural: sem detecção (%.0f ms)", dt)
            return None
        area = cv2.contourArea(pts.astype(np.int32))
        h, w = img_bgr.shape[:2]
        if area < 0.05 * h * w:
            log.debug("neural: polígono pequeno demais (%.1f%% da imagem) — descartado",
                      100 * area / (h * w))
            return None
        log.debug("neural: ok em %.0f ms, área=%.1f%% da imagem",
                  dt, 100 * area / (h * w))
        return _ordenar_cantos(pts)
    except Exception:
        log.exception("neural: erro inesperado na inferência")
        return None


def detectar_cantos(img_bgr: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Detecta cantos do documento. Retorna (cantos, metodo).

    metodo: 'neural' ou 'nenhum'.

    Política deliberada: SÓ a rede neural decide. Sem detecção confiável, a
    página é mantida inteira — heurísticas OpenCV foram testadas e cortavam
    conteúdo de documentos digitais (fallback removido de propósito).
    O usuário sempre pode ajustar os cantos manualmente no editor.
    """
    pts = detectar_cantos_neural(img_bgr)
    if pts is not None:
        return pts, "neural"
    log.debug("nenhuma detecção — página será mantida inteira")
    return None, "nenhum"
