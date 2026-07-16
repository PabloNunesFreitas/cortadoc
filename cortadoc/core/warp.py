"""Retificação de perspectiva: dado os 4 cantos, endireita e recorta o papel."""

from __future__ import annotations

import numpy as np
import cv2


def _tamanho_destino(cantos: np.ndarray) -> tuple[int, int]:
    tl, tr, br, bl = cantos
    largura = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    altura = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    return int(round(largura)), int(round(altura))


def expandir_cantos(cantos: np.ndarray, folga_px: float, img_shape) -> np.ndarray:
    """Afasta cada canto do centróide em folga_px, clampando na imagem.

    Garante que texto encostado na borda do papel não seja cortado."""
    h, w = img_shape[:2]
    centro = cantos.mean(axis=0)
    novos = []
    for p in cantos:
        v = p - centro
        norma = np.linalg.norm(v)
        if norma < 1e-6:
            novos.append(p)
            continue
        novo = p + v / norma * folga_px
        novos.append([np.clip(novo[0], 0, w - 1), np.clip(novo[1], 0, h - 1)])
    return np.array(novos, dtype="float32")


def retificar(img_bgr: np.ndarray, cantos: np.ndarray, folga_px: float = 40,
              rotacionar_retrato: bool = False) -> np.ndarray:
    """Aplica warp de perspectiva nos 4 cantos e devolve o papel endireitado."""
    cantos = expandir_cantos(cantos, folga_px, img_bgr.shape)
    largura, altura = _tamanho_destino(cantos)
    destino = np.array([
        [0, 0],
        [largura - 1, 0],
        [largura - 1, altura - 1],
        [0, altura - 1],
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(cantos, destino)
    warp = cv2.warpPerspective(img_bgr, M, (largura, altura))
    if rotacionar_retrato and warp.shape[1] > warp.shape[0] * 1.3:
        warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
    return warp
