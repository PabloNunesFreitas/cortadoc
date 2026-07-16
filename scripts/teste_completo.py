"""Suite de teste completa do CortaDoc (sem GUI).

1. Casos sintéticos extremos (corrompido, vazio, alpha, cinza, gigante, etc.)
2. Todos os documentos reais passados como argumento
3. Exportação de PDF (grupos, nomes hostis, colisões)

Uso: python3 scripts/teste_completo.py [arquivos reais...]
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from cortadoc.core.log import configurar
from cortadoc.core.io_utils import carregar_paginas
from cortadoc.core.detector import detectar_cantos
from cortadoc.core.warp import retificar
from cortadoc.core.recorte import cortar_documento
from cortadoc.core.pdf import exportar_pdf

log = configurar()

PASSOU, FALHOU = [], []


def caso(nome):
    def deco(fn):
        def wrapper():
            try:
                fn()
                PASSOU.append(nome)
                print(f"  ✓ {nome}")
            except Exception as e:
                FALHOU.append((nome, e, traceback.format_exc()))
                print(f"  ✗ {nome}: {e}")
        return wrapper
    return deco


def esperar_erro(fn, *args):
    """Executa fn esperando ValueError (erro limpo, não crash)."""
    try:
        fn(*args)
    except ValueError:
        return
    except Exception as e:
        raise AssertionError(f"esperava ValueError, veio {type(e).__name__}: {e}")
    raise AssertionError("esperava ValueError, não veio nenhum erro")


TMP = Path(tempfile.mkdtemp(prefix="cortadoc_teste_"))


# ------------------------------------------------------------------ sintéticos
@caso("arquivo inexistente -> erro limpo")
def t_inexistente():
    esperar_erro(carregar_paginas, TMP / "nao_existe.jpg")


@caso("arquivo vazio -> erro limpo")
def t_vazio():
    p = TMP / "vazio.jpg"
    p.write_bytes(b"")
    esperar_erro(carregar_paginas, p)


@caso("texto renomeado p/ .jpg -> erro limpo")
def t_fake_jpg():
    p = TMP / "fake.jpg"
    p.write_text("isto não é uma imagem")
    esperar_erro(carregar_paginas, p)


@caso("PDF corrompido -> erro limpo")
def t_pdf_corrompido():
    p = TMP / "quebrado.pdf"
    p.write_bytes(b"%PDF-1.4 lixo lixo lixo")
    esperar_erro(carregar_paginas, p)


@caso("extensão desconhecida -> erro limpo")
def t_ext_ruim():
    p = TMP / "doc.xyz"
    p.write_bytes(b"abc")
    esperar_erro(carregar_paginas, p)


@caso("nome com acento/espaço carrega ok")
def t_unicode():
    p = TMP / "certidão de nascimento (cópia) º.png"
    cv2.imwrite(str(p), np.full((100, 100, 3), 200, np.uint8))
    # imwrite pode falhar com unicode; usa o caminho robusto do app p/ criar
    if not p.exists():
        ok, buf = cv2.imencode(".png", np.full((100, 100, 3), 200, np.uint8))
        buf.tofile(str(p))
    pags = carregar_paginas(p)
    assert len(pags) == 1 and pags[0].shape == (100, 100, 3)


@caso("PNG com transparência carrega ok")
def t_alpha():
    p = TMP / "alpha.png"
    rgba = np.zeros((80, 80, 4), np.uint8)
    rgba[..., 3] = 128
    ok, buf = cv2.imencode(".png", rgba)
    buf.tofile(str(p))
    pags = carregar_paginas(p)
    assert pags[0].shape == (80, 80, 3)


@caso("imagem gigante é reduzida")
def t_gigante():
    p = TMP / "gigante.jpg"
    ok, buf = cv2.imencode(".jpg", np.full((9000, 7000, 3), 180, np.uint8))
    buf.tofile(str(p))
    pags = carregar_paginas(p)
    assert max(pags[0].shape[:2]) <= 6000, pags[0].shape


@caso("imagem minúscula (10x10) não crasha detector")
def t_minuscula():
    img = np.full((10, 10, 3), 255, np.uint8)
    cantos, metodo = detectar_cantos(img)  # qualquer resultado serve, sem crash


@caso("imagem toda preta não crasha")
def t_preta():
    cantos, metodo = detectar_cantos(np.zeros((500, 400, 3), np.uint8))


@caso("imagem toda branca não crasha")
def t_branca():
    cantos, metodo = detectar_cantos(np.full((500, 400, 3), 255, np.uint8))


def _foto_documento_sintetica():
    """Foto realista: papel com texto, rotacionado, sobre mesa texturizada."""
    rng = np.random.default_rng(42)
    img = np.zeros((1200, 900, 3), np.uint8)
    for c, base in enumerate((40, 60, 90)):  # madeira azulada com ruído
        img[:, :, c] = base + rng.normal(0, 8, (1200, 900)).clip(-20, 20)
    papel = np.full((700, 500, 3), 240, np.uint8)
    grad = np.linspace(0, -25, 700).astype(np.int16)[:, None]
    papel = (papel.astype(np.int16) + grad[..., None]).clip(0, 255).astype(np.uint8)
    for y in range(60, 660, 35):
        cv2.line(papel, (50, y), (450, y), (60, 60, 60), 3)
    cv2.putText(papel, "CERTIDAO", (90, 45), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (20, 20, 20), 3)
    M = cv2.getRotationMatrix2D((250, 350), 8, 1.0)
    ph, pw = papel.shape[:2]
    rot = cv2.warpAffine(papel, M, (pw, ph), borderValue=(0, 0, 0))
    mask = cv2.warpAffine(np.full((ph, pw), 255, np.uint8), M, (pw, ph))
    roi = img[250:250 + ph, 200:200 + pw]
    roi[mask > 128] = rot[mask > 128]
    return img


@caso("foto sintética realista é detectada e cortada")
def t_sintetico():
    img = _foto_documento_sintetica()
    cantos, metodo = detectar_cantos(img)
    assert metodo == "neural", f"esperava neural, veio {metodo}"
    corte = cortar_documento(img, cantos, folga_px=20)
    assert corte.shape[0] < 1200 and corte.shape[1] < 900
    assert corte.mean() > 150, f"corte não parece papel (média {corte.mean():.0f})"
    # fundo (mesa) não pode sobrar nos cantos do resultado
    for y, x in [(3, 3), (3, -3), (-3, 3), (-3, -3)]:
        assert corte[y, x].min() >= 150, f"canto ({y},{x}) ainda é mesa: {corte[y, x]}"


@caso("recorte remove fundo mas preserva o texto")
def t_recorte_fundo():
    img = np.full((1000, 800, 3), 120, np.uint8)               # tecido cinza
    cv2.rectangle(img, (150, 200), (650, 800), (250, 250, 250), -1)  # papel
    cv2.putText(img, "TEXTO", (200, 500), cv2.FONT_HERSHEY_SIMPLEX,
                2, (20, 20, 20), 5)
    cantos = np.array([[130, 180], [670, 180], [670, 820], [130, 820]],
                      dtype="float32")
    # baseline justa: warp puro (a reamostragem já muda a contagem de pixels);
    # a pintura de fundo não pode remover NENHUM pixel de tinta além disso.
    baseline = (retificar(img, cantos, folga_px=25)[:, :, 0] < 60).sum()
    corte = cortar_documento(img, cantos, folga_px=25)
    depois_texto = (corte[:, :, 0] < 60).sum()
    assert depois_texto >= baseline, \
        f"pintura apagou tinta: warp={baseline} -> corte={depois_texto} px escuros"
    # cantos do resultado devem ser papel/branco, não tecido cinza-médio
    for y, x in [(3, 3), (3, -3), (-3, 3), (-3, -3)]:
        px = corte[y, x]
        assert px.min() >= 180, f"canto ({y},{x}) ainda é fundo: {px}"


@caso("modo manual obedece exatamente os cantos marcados")
def t_manual_obedece():
    # página branca cheia de texto; usuário marca uma região de 400x300
    img = np.full((2000, 1500, 3), 245, np.uint8)
    for y in range(100, 1900, 60):
        cv2.putText(img, "LINHA DE TEXTO QUALQUER", (80, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 30, 30), 2)
    cantos = np.array([[500, 600], [900, 600], [900, 900], [500, 900]],
                      dtype="float32")
    corte = cortar_documento(img, cantos, folga_px=25, modo_manual=True)
    h, w = corte.shape[:2]
    # folga expande cada canto 25px na direção diagonal (a partir do centro),
    # o que dá ~20px/lado no eixo X e ~15px/lado no eixo Y neste retângulo
    assert abs(w - 440) <= 8 and abs(h - 330) <= 8, \
        f"manual desobedeceu: esperado ~440x330, veio {w}x{h}"


@caso("recorte com cantos degenerados não crasha")
def t_recorte_degenerado():
    img = np.full((400, 300, 3), 128, np.uint8)
    cantos = np.array([[10, 10], [200, 10], [200, 10.5], [10, 10.2]],
                      dtype="float32")
    try:
        cortar_documento(img, cantos, folga_px=10)
    except Exception:
        pass  # erro tratado é aceitável; segfault não (chegou aqui = ok)


@caso("warp com cantos degenerados (linha) não crasha")
def t_cantos_degenerados():
    img = np.full((400, 300, 3), 128, np.uint8)
    cantos = np.array([[10, 10], [200, 10], [200, 10.5], [10, 10.2]], dtype="float32")
    try:
        retificar(img, cantos, folga_px=10)
    except Exception:
        pass  # pode falhar, mas não pode ser segfault (chegou aqui = ok)


@caso("PDF multipágina exporta na ordem")
def t_pdf_grupo():
    a = np.full((300, 200, 3), 250, np.uint8)
    b = np.full((200, 300, 3), 100, np.uint8)
    destino = TMP / "saída teste" / "grupo çé.pdf"
    out = exportar_pdf([a, b], destino)
    import fitz
    d = fitz.open(str(out))
    assert len(d) == 2
    p0 = d[0].rect  # página 1 retrato, página 2 paisagem
    p1 = d[1].rect
    d.close()
    assert p0.height > p0.width and p1.width > p1.height, "ordem/orientação errada"


@caso("exportar lista vazia -> erro limpo")
def t_pdf_vazio():
    try:
        exportar_pdf([], TMP / "vazio.pdf")
        raise AssertionError("esperava erro com lista vazia")
    except (ValueError, IndexError):
        pass


@caso("nomes hostis p/ Windows são sanitizados")
def t_nomes():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from cortadoc.app import nome_arquivo_seguro, caminho_sem_colisao
    assert nome_arquivo_seguro('rg: frente/verso?*') == "rg_ frente_verso__"
    assert nome_arquivo_seguro("CON") == "CON_"
    assert nome_arquivo_seguro("...") == "documento"
    assert len(nome_arquivo_seguro("x" * 500)) == 150
    # colisão
    (TMP / "col.pdf").write_bytes(b"x")
    assert caminho_sem_colisao(TMP, "col").name == "col (2).pdf"


# --------------------------------------------------------------- arquivos reais
def testar_reais(arquivos: list[str], pasta_out: Path):
    (pasta_out / "cortes").mkdir(parents=True, exist_ok=True)
    (pasta_out / "overlays").mkdir(parents=True, exist_ok=True)
    print(f"\n=== Documentos reais ({len(arquivos)}) ===")
    resumo = []
    for arq in arquivos:
        arq = Path(arq)
        try:
            paginas = carregar_paginas(arq)
        except ValueError as e:
            resumo.append((arq.name, "ERRO-LIMPO", str(e)))
            print(f"  {arq.name}: erro limpo ({e})")
            continue
        for i, img in enumerate(paginas):
            suf = f"_p{i+1}" if len(paginas) > 1 else ""
            nome = f"{arq.stem}{suf}"
            try:
                cantos, metodo = detectar_cantos(img)
                if cantos is not None:
                    vis = img.copy()
                    cv2.polylines(vis, [cantos.astype(int)], True, (0, 255, 0), 8)
                    ok, buf = cv2.imencode(".jpg", vis)
                    buf.tofile(str(pasta_out / "overlays" / f"{nome}.jpg"))
                    corte = cortar_documento(img, cantos, folga_px=25)
                    ok, buf = cv2.imencode(".png", corte)
                    buf.tofile(str(pasta_out / "cortes" / f"{nome}.png"))
                resumo.append((nome, metodo,
                               f"{img.shape[1]}x{img.shape[0]}"))
                print(f"  {nome}: {metodo}")
            except Exception as e:
                resumo.append((nome, "CRASH", str(e)))
                print(f"  ✗ {nome}: CRASH {e}")
                traceback.print_exc()
    return resumo


def main():
    print("=== Casos sintéticos ===")
    for nome, fn in list(globals().items()):
        if nome.startswith("t_") and callable(fn):
            fn()

    reais = sys.argv[1:]
    resumo_reais = []
    if reais:
        resumo_reais = testar_reais(
            reais, Path(__file__).parent.parent / "teste_saida")

    print(f"\n=== RESULTADO ===")
    print(f"Sintéticos: {len(PASSOU)} ok, {len(FALHOU)} falhas")
    for nome, e, tb in FALHOU:
        print(f"\nFALHA: {nome}\n{tb}")
    crashes = [r for r in resumo_reais if r[1] == "CRASH"]
    if resumo_reais:
        neurais = sum(1 for r in resumo_reais if r[1] == "neural")
        print(f"Reais: {len(resumo_reais)} páginas, {neurais} neural, "
              f"{len(crashes)} crashes")
    sys.exit(1 if (FALHOU or crashes) else 0)


if __name__ == "__main__":
    main()
