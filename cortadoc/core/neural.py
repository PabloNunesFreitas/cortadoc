"""Inferência standalone do modelo DocAligner (fastvit_sa24, heatmap regression).

Reimplementa o pre/pós-processamento do pacote docaligner-docsaid usando só
onnxruntime + OpenCV + numpy, sem as dependências pesadas do capybara.

Modelo: https://github.com/DocsaidLab/DocAligner (Apache 2.0)
Entrada: imagem BGR qualquer. Saída: 4 cantos do documento em pixels, ou None.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("cortadoc.neural")

TAMANHO_INFER = (256, 256)
LIMIAR_HEATMAP = 0.3

_sessao = None


def _caminho_modelo() -> Path:
    # No exe congelado (PyInstaller), os dados ficam em sys._MEIPASS.
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent.parent))
    return base / "models" / "fastvit_sa24_h_e_bifpn_256_fp32.onnx"


_avisou_modelo_ausente = False


def modelo_disponivel() -> bool:
    global _avisou_modelo_ausente
    existe = _caminho_modelo().exists()
    if not existe and not _avisou_modelo_ausente:
        _avisou_modelo_ausente = True
        log.error("MODELO AUSENTE em %s — detecção neural desativada; "
                  "rode scripts/baixar_modelos.py (ou o .exe foi empacotado "
                  "sem o modelo)", _caminho_modelo())
    return existe


def _sessao_onnx():
    global _sessao
    if _sessao is None:
        import onnxruntime as ort
        caminho = _caminho_modelo()
        t0 = time.perf_counter()
        _sessao = ort.InferenceSession(
            str(caminho), providers=["CPUExecutionProvider"])
        log.info("Modelo carregado em %.1fs: %s (%.1f MB, onnxruntime %s)",
                 time.perf_counter() - t0, caminho.name,
                 caminho.stat().st_size / 1e6, ort.__version__)
    return _sessao


def _preprocessar(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr, TAMANHO_INFER, interpolation=cv2.INTER_LINEAR)
    tensor = np.transpose(img, (2, 0, 1)).astype("float32")[None] / 255.0
    return tensor


def _centro_maior_blob(mask: np.ndarray) -> list[float] | None:
    """Centroide do maior componente conexo da máscara binária."""
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    maior = max(contornos, key=cv2.contourArea)
    M = cv2.moments(maior)
    if M["m00"] < 1e-6:
        x, y, w, h = cv2.boundingRect(maior)
        return [x + w / 2.0, y + h / 2.0]
    return [M["m10"] / M["m00"], M["m01"] / M["m00"]]


def detectar_cantos_onnx(img_bgr: np.ndarray) -> np.ndarray | None:
    """Roda o modelo e devolve os 4 cantos (4,2) float32 em pixels da imagem
    original, na ordem prevista pelo modelo (TL, TR, BR, BL). None se falhar."""
    if not modelo_disponivel():
        return None

    h, w = img_bgr.shape[:2]
    sessao = _sessao_onnx()
    nome_entrada = sessao.get_inputs()[0].name
    preds = sessao.run(None, {nome_entrada: _preprocessar(img_bgr)})[0]  # (1, 4, H, W)

    cantos = []
    for heatmap in preds[0]:
        # Redimensiona o heatmap para o tamanho da imagem original.
        hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        hm[hm < LIMIAR_HEATMAP] = 0
        mask = (hm > 0).astype(np.uint8) * 255
        ponto = _centro_maior_blob(mask)
        if ponto is None:
            return None
        cantos.append(ponto)

    if len(cantos) != 4:
        return None
    return np.array(cantos, dtype="float32")
