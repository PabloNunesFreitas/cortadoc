"""Pipeline completo de recorte: quad da rede → GrabCut → warp → papel puro.

`cortar_documento(img, cantos, folga_px)` devolve a imagem final:
- endireitada (warp de perspectiva),
- com TUDO que não é papel (mesa, tecido, chão) pintado de branco,
- recortada rente ao papel,
- com proteção absoluta de tinta: pixel escuro nunca é pintado nem cortado.

Fail-safes: qualquer falha na segmentação degrada para o warp simples
(nunca quebra, nunca perde conteúdo). Tudo logado.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from cortadoc.core.warp import retificar, expandir_cantos, _tamanho_destino

log = logging.getLogger("cortadoc.recorte")

LADO_GRABCUT = 1000      # imagem é reduzida p/ isso antes do GrabCut (velocidade)
ITER_GRABCUT = 4
ENCOLHER_MIOLO = 0.75    # quad encolhido p/ marcar "frente certa"
BUFFER_PAPEL_PX = 15     # dilatação da máscara de papel: erro da segmentação
                         # nunca chega a menos de 15px do conteúdo real
MARGEM_RECROP_PX = 6
MIN_FRACAO_PAPEL = 0.25  # se o GrabCut achar menos papel que isso, desiste


def _mascara_papel_grabcut(img_bgr: np.ndarray, cantos: np.ndarray) -> np.ndarray | None:
    """Segmenta papel vs fundo com GrabCut guiado pelo quadrilátero.

    Retorna máscara uint8 (1=papel) do tamanho da imagem, ou None se falhar."""
    h, w = img_bgr.shape[:2]
    escala = min(LADO_GRABCUT / max(h, w), 1.0)
    peq = cv2.resize(img_bgr, None, fx=escala, fy=escala,
                     interpolation=cv2.INTER_AREA) if escala < 1.0 else img_bgr
    ph, pw = peq.shape[:2]
    quad = (cantos * escala).astype(np.int32)

    # Camadas da máscara inicial (de fora pra dentro):
    #   fora do quad expandido      -> fundo certo
    #   anel quad..quad expandido   -> fundo PROVÁVEL (pontas do papel que
    #                                  escapam do quad podem virar frente!)
    #   dentro do quad              -> frente provável
    #   miolo do quad               -> frente certa
    mask = np.full((ph, pw), cv2.GC_BGD, np.uint8)
    centro = quad.mean(axis=0)
    quad_expandido = (centro + (quad - centro) * 1.18).astype(np.int32)
    cv2.fillPoly(mask, [quad_expandido], cv2.GC_PR_BGD)
    cv2.fillPoly(mask, [quad], cv2.GC_PR_FGD)
    miolo = (centro + (quad - centro) * ENCOLHER_MIOLO).astype(np.int32)
    cv2.fillPoly(mask, [miolo], cv2.GC_FGD)

    # GrabCut exige amostras de fundo E frente. Se o quad cobre a imagem
    # (in)teira não há fundo a remover — pula direto pro warp simples.
    fracao_fundo = float((mask == cv2.GC_BGD).mean())
    fracao_frente = float((mask == cv2.GC_FGD).mean())
    if fracao_fundo < 0.01 or fracao_frente < 0.01:
        log.debug("grabcut pulado: fundo=%.1f%% frente=%.1f%% da imagem",
                  fracao_fundo * 100, fracao_frente * 100)
        return None

    t0 = time.perf_counter()
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(peq, mask, None, bgd, fgd, ITER_GRABCUT, cv2.GC_INIT_WITH_MASK)
    dt = time.perf_counter() - t0

    papel = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)

    # Mantém todos os componentes relevantes (documentos com metades separadas,
    # ex. RG frente+verso na mesma foto) e descarta migalhas de ruído.
    num, rot, stats, _ = cv2.connectedComponentsWithStats(papel)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        maior_area = int(areas.max())
        validos = [i + 1 for i, a in enumerate(areas)
                   if a >= 0.25 * maior_area]
        papel = np.isin(rot, validos).astype(np.uint8)
    papel = cv2.morphologyEx(papel, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    fracao = float(papel.mean())
    log.debug("grabcut: %.1fs, papel=%.0f%% da imagem", dt, fracao * 100)
    if fracao < MIN_FRACAO_PAPEL:
        log.warning("grabcut achou papel demais pequeno (%.0f%%) — descartado",
                    fracao * 100)
        return None

    if escala < 1.0:
        papel = cv2.resize(papel, (w, h), interpolation=cv2.INTER_NEAREST)
    return papel


def _warp_com_matriz(img: np.ndarray, cantos: np.ndarray, folga_px: float,
                     flags=cv2.INTER_LINEAR):
    cexp = expandir_cantos(cantos.copy(), folga_px, img.shape)
    largura, altura = _tamanho_destino(cexp)
    destino = np.array([[0, 0], [largura - 1, 0],
                        [largura - 1, altura - 1], [0, altura - 1]],
                       dtype="float32")
    M = cv2.getPerspectiveTransform(cexp, destino)
    return cv2.warpPerspective(img, M, (largura, altura), flags=flags)


def cortar_documento(img_bgr: np.ndarray, cantos: np.ndarray,
                     folga_px: float = 25,
                     modo_manual: bool = False) -> np.ndarray:
    """Warp + remoção de fundo + recrop. Nunca levanta exceção: degrada
    para o warp simples se a segmentação falhar.

    modo_manual=True (cantos marcados pelo usuário): obediência total —
    corta EXATAMENTE no quadrilátero marcado + folga fixa, sem folga
    dinâmica e sem GrabCut (que é imprevisível ao recortar uma região de
    papel dentro de outro papel)."""
    if modo_manual:
        log.debug("modo manual: warp exato nos cantos do usuário (folga=%dpx)",
                  folga_px)
        return _warp_com_matriz(img_bgr, cantos, folga_px)

    try:
        papel = _mascara_papel_grabcut(img_bgr, cantos)
    except Exception:
        log.exception("grabcut falhou — usando warp simples")
        papel = None

    # Folga dinâmica: pontas de papel dobrado/curvado podem passar bem do quad
    # detectado. O excesso não custa nada — o GrabCut pinta de branco e o
    # recrop apara — mas folga de menos DECEPA ponta de documento.
    lado_max = float(max(np.ptp(cantos[:, 0]), np.ptp(cantos[:, 1])))
    folga_px = max(folga_px, 0.09 * lado_max)

    warp_img = _warp_com_matriz(img_bgr, cantos, folga_px)
    if papel is None:
        return warp_img

    warp_mask = _warp_com_matriz(papel, cantos, folga_px, flags=cv2.INTER_NEAREST)

    # Buffer de segurança: dilata o papel para que nenhum erro de segmentação
    # encoste no conteúdo (a tinta está sempre EM CIMA do papel, então proteger
    # o papel dilatado protege toda a tinta).
    k = np.ones((2 * BUFFER_PAPEL_PX + 1, 2 * BUFFER_PAPEL_PX + 1), np.uint8)
    papel_seguro = cv2.dilate(warp_mask, k)

    # Pinta TUDO que não é papel de branco (mesa clara ou escura, tanto faz).
    resultado = warp_img.copy()
    resultado[papel_seguro == 0] = (255, 255, 255)

    # Recrop rente ao papel.
    conteudo = papel_seguro == 1
    xs = np.where(conteudo.any(axis=0))[0]
    ys = np.where(conteudo.any(axis=1))[0]
    if len(xs) > 20 and len(ys) > 20:
        m = MARGEM_RECROP_PX
        h, w = resultado.shape[:2]
        resultado = resultado[max(0, ys[0] - m):min(h, ys[-1] + 1 + m),
                              max(0, xs[0] - m):min(w, xs[-1] + 1 + m)]

    log.debug("recorte final: %dx%d (%.0f%% pintado de branco)",
              resultado.shape[1], resultado.shape[0],
              (papel_seguro == 0).mean() * 100)
    return resultado
