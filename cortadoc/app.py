"""CortaDoc — corta documentos em fotos e agrupa em PDFs.

Fluxo:
1. Importar fotos/PDFs.
2. Corte automático (DocAligner → fallback OpenCV), com editor manual de cantos.
3. Galeria de revisão: selecionar itens e "Agrupar em PDF" (grupo = 1 PDF
   multipágina; sem grupo = 1 PDF individual).
4. Gerar PDFs.
"""

from __future__ import annotations

import logging
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QSize, QPointF
from PySide6.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QAction, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QListWidgetItem, QFileDialog, QProgressBar,
    QDialog, QDialogButtonBox, QMessageBox, QAbstractItemView, QSpinBox,
)

from cortadoc.core.detector import detectar_cantos
from cortadoc.core.recorte import cortar_documento
from cortadoc.core.pdf import exportar_pdf
from cortadoc.core.io_utils import carregar_paginas, EXTENSOES_IMAGEM
from cortadoc.core.log import configurar as configurar_log, caminho_log

log = logging.getLogger("cortadoc.app")

FOLGA_PADRAO_PX = 25
CORES_GRUPO = ["#e5484d", "#30a46c", "#0090ff", "#f76b15", "#8e4ec6",
               "#e93d82", "#12a594", "#ffb224", "#6e56cf", "#86662d"]


@dataclass
class DocItem:
    origem: Path
    pagina: int
    img_original: np.ndarray
    cantos: np.ndarray | None = None
    metodo: str = "nenhum"
    img_cortada: np.ndarray | None = None
    grupo: int | None = None
    rotacao: int = 0  # 0/90/180/270 aplicados sobre img_cortada

    @property
    def rotulo(self) -> str:
        suf = f" (p{self.pagina + 1})" if self.pagina else ""
        return f"{self.origem.stem}{suf}"

    def imagem_final(self) -> np.ndarray:
        img = self.img_cortada if self.img_cortada is not None else self.img_original
        for _ in range((self.rotacao // 90) % 4):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img


_INVALIDOS_WINDOWS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVADOS_WINDOWS = {"CON", "PRN", "AUX", "NUL",
                       *(f"COM{i}" for i in range(1, 10)),
                       *(f"LPT{i}" for i in range(1, 10))}


def nome_arquivo_seguro(nome: str) -> str:
    """Sanitiza um nome de arquivo para funcionar em Windows/Linux/macOS."""
    nome = _INVALIDOS_WINDOWS.sub("_", nome).strip(" .")
    if not nome:
        nome = "documento"
    if nome.upper() in _RESERVADOS_WINDOWS:
        nome = f"{nome}_"
    return nome[:150]  # margem folgada sob o limite de 255 do NTFS/ext4


def caminho_sem_colisao(pasta: Path, nome_base: str) -> Path:
    """Devolve pasta/nome_base.pdf, acrescentando (2), (3)… se já existir."""
    caminho = pasta / f"{nome_base}.pdf"
    n = 2
    while caminho.exists():
        caminho = pasta / f"{nome_base} ({n}).pdf"
        n += 1
    return caminho


def bgr_para_qpixmap(img_bgr: np.ndarray, max_lado: int | None = None) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    pix = QPixmap.fromImage(qimg)
    if max_lado and max(w, h) > max_lado:
        pix = pix.scaled(max_lado, max_lado, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pix


class ProcessadorThread(QThread):
    progresso = Signal(int, int, str)   # atual, total, nome
    item_pronto = Signal(object)        # DocItem
    arquivo_falhou = Signal(str, str)   # nome, motivo
    terminou = Signal()

    def __init__(self, caminhos: list[Path], folga_px: int):
        super().__init__()
        self.caminhos = caminhos
        self.folga_px = folga_px

    def run(self):
        # Nunca deixar uma exceção derrubar a thread: cada arquivo é isolado,
        # falhas viram sinal para a UI e traceback completo no log.
        try:
            self._processar_tudo()
        except Exception:
            log.critical("Falha inesperada na thread de processamento:\n%s",
                         traceback.format_exc())
        finally:
            self.terminou.emit()

    def _processar_tudo(self):
        total = len(self.caminhos)
        log.info("Iniciando lote: %d arquivo(s), folga=%dpx", total, self.folga_px)
        for i, caminho in enumerate(self.caminhos, 1):
            self.progresso.emit(i, total, caminho.name)
            try:
                paginas = carregar_paginas(caminho)
            except Exception as e:
                log.error("Arquivo pulado (%s): %s", caminho.name, e)
                self.arquivo_falhou.emit(caminho.name, str(e))
                continue
            for idx, img in enumerate(paginas):
                try:
                    cantos, metodo = detectar_cantos(img)
                    item = DocItem(origem=caminho, pagina=idx, img_original=img,
                                   cantos=cantos, metodo=metodo)
                    if cantos is not None:
                        item.img_cortada = cortar_documento(
                            img, cantos, folga_px=self.folga_px)
                    else:
                        item.img_cortada = img.copy()
                    log.info("%s p%d: método=%s %dx%d -> %dx%d",
                             caminho.name, idx + 1, metodo,
                             img.shape[1], img.shape[0],
                             item.img_cortada.shape[1], item.img_cortada.shape[0])
                    self.item_pronto.emit(item)
                except Exception as e:
                    log.error("Página falhou (%s p%d): %s\n%s", caminho.name,
                              idx + 1, e, traceback.format_exc())
                    self.arquivo_falhou.emit(
                        f"{caminho.name} (pág. {idx + 1})", str(e))
        log.info("Lote concluído")


class EditorCantos(QDialog):
    """Editor manual: arraste os 4 círculos para os cantos do papel."""

    RAIO = 14

    def __init__(self, item: DocItem, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Ajustar cantos — {item.rotulo}")
        self.item = item

        h, w = item.img_original.shape[:2]
        self.escala = min(900 / w, 700 / h, 1.0)
        self.pix = bgr_para_qpixmap(item.img_original, max_lado=int(max(w, h) * self.escala))

        if item.cantos is not None:
            self.cantos = item.cantos.copy() * self.escala
        else:
            m = 0.1
            self.cantos = np.array([
                [w * m, h * m], [w * (1 - m), h * m],
                [w * (1 - m), h * (1 - m)], [w * m, h * (1 - m)],
            ], dtype="float32") * self.escala

        self._arrastando = None

        self.canvas = QLabel()
        self.canvas.setFixedSize(self.pix.size())
        self.canvas.setMouseTracking(True)
        self.canvas.mousePressEvent = self._press
        self.canvas.mouseMoveEvent = self._move
        self.canvas.mouseReleaseEvent = self._release

        botoes = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botoes.accepted.connect(self.accept)
        botoes.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Arraste os círculos até os 4 cantos do papel:"))
        lay.addWidget(self.canvas)
        lay.addWidget(botoes)
        self._desenhar()

    def _desenhar(self):
        canvas = QPixmap(self.pix)
        p = QPainter(canvas)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#00e676"), 3)
        p.setPen(pen)
        pts = [QPointF(float(x), float(y)) for x, y in self.cantos]
        for a, b in zip(pts, pts[1:] + pts[:1]):
            p.drawLine(a, b)
        for pt in pts:
            p.setBrush(QColor(0, 230, 118, 160))
            p.drawEllipse(pt, self.RAIO, self.RAIO)
        p.end()
        self.canvas.setPixmap(canvas)

    def _canto_perto(self, pos) -> int | None:
        for i, (x, y) in enumerate(self.cantos):
            if (pos.x() - x) ** 2 + (pos.y() - y) ** 2 <= (self.RAIO * 2) ** 2:
                return i
        return None

    def _press(self, ev):
        self._arrastando = self._canto_perto(ev.position())

    def _move(self, ev):
        if self._arrastando is not None:
            pos = ev.position()
            x = min(max(pos.x(), 0), self.pix.width() - 1)
            y = min(max(pos.y(), 0), self.pix.height() - 1)
            self.cantos[self._arrastando] = [x, y]
            self._desenhar()

    def _release(self, ev):
        self._arrastando = None

    def cantos_reais(self) -> np.ndarray:
        return (self.cantos / self.escala).astype("float32")


class JanelaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CortaDoc")
        self.resize(1100, 750)
        self.itens: list[DocItem] = []
        self._proximo_grupo = 1
        self._thread = None

        central = QWidget()
        self.setCentralWidget(central)
        raiz = QVBoxLayout(central)

        # --- barra superior -------------------------------------------------
        barra = QHBoxLayout()
        self.btn_importar = QPushButton("📂 Importar fotos/PDFs")
        self.btn_importar.clicked.connect(self.importar)
        barra.addWidget(self.btn_importar)

        barra.addWidget(QLabel("Folga (px):"))
        self.spin_folga = QSpinBox()
        self.spin_folga.setRange(0, 300)
        self.spin_folga.setValue(FOLGA_PADRAO_PX)
        self.spin_folga.setToolTip(
            "Folga de segurança ao redor do papel para nunca cortar texto")
        barra.addWidget(self.spin_folga)
        barra.addStretch()

        self.btn_editar = QPushButton("✏️ Ajustar cantos")
        self.btn_editar.clicked.connect(self.editar_cantos)
        self.btn_girar = QPushButton("↻ Girar 90°")
        self.btn_girar.clicked.connect(self.girar)
        self.btn_agrupar = QPushButton("🔗 Agrupar em PDF")
        self.btn_agrupar.clicked.connect(self.agrupar)
        self.btn_desagrupar = QPushButton("✂ Desagrupar")
        self.btn_desagrupar.clicked.connect(self.desagrupar)
        self.btn_gerar = QPushButton("📄 Gerar PDFs")
        self.btn_gerar.setStyleSheet("font-weight: bold;")
        self.btn_gerar.clicked.connect(self.gerar_pdfs)
        for b in (self.btn_editar, self.btn_girar, self.btn_agrupar,
                  self.btn_desagrupar, self.btn_gerar):
            barra.addWidget(b)
        raiz.addLayout(barra)

        # --- galeria ---------------------------------------------------------
        self.lista = QListWidget()
        self.lista.setViewMode(QListWidget.IconMode)
        self.lista.setIconSize(QSize(200, 200))
        self.lista.setResizeMode(QListWidget.Adjust)
        self.lista.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lista.setSpacing(12)
        self.lista.itemDoubleClicked.connect(lambda _: self.editar_cantos())
        raiz.addWidget(self.lista, stretch=1)

        # --- status ----------------------------------------------------------
        rodape = QHBoxLayout()
        self.progresso = QProgressBar()
        self.progresso.setVisible(False)
        self.status = QLabel("Importe fotos para começar. Dica: selecione várias "
                             "miniaturas (Ctrl+clique) e use “Agrupar em PDF”.")
        rodape.addWidget(self.status, stretch=1)
        rodape.addWidget(self.progresso)
        raiz.addLayout(rodape)

    # ------------------------------------------------------------------ fluxo
    def importar(self):
        exts = " ".join(f"*{e}" for e in sorted(EXTENSOES_IMAGEM)) + " *.pdf"
        arquivos, _ = QFileDialog.getOpenFileNames(
            self, "Escolher fotos ou PDFs", str(Path.home()),
            f"Documentos ({exts})")
        if not arquivos:
            return
        caminhos = [Path(a) for a in arquivos]
        self.btn_importar.setEnabled(False)
        self.progresso.setVisible(True)
        self.progresso.setMaximum(len(caminhos))
        self._falhas: list[tuple[str, str]] = []
        self._thread = ProcessadorThread(caminhos, self.spin_folga.value())
        self._thread.progresso.connect(self._on_progresso)
        self._thread.item_pronto.connect(self._on_item)
        self._thread.arquivo_falhou.connect(
            lambda nome, motivo: self._falhas.append((nome, motivo)))
        self._thread.terminou.connect(self._on_fim)
        self._thread.start()

    def _on_progresso(self, atual, total, nome):
        self.progresso.setValue(atual)
        self.status.setText(f"Cortando {atual}/{total}: {nome}")

    def _on_item(self, item: DocItem):
        self.itens.append(item)
        self._adicionar_thumb(item)

    def _on_fim(self):
        self.btn_importar.setEnabled(True)
        self.progresso.setVisible(False)
        neurais = sum(1 for i in self.itens if i.metodo == "neural")
        inteiras = sum(1 for i in self.itens if i.metodo == "nenhum")
        msg = f"{len(self.itens)} páginas processadas ({neurais} cortadas pela IA)."
        if inteiras:
            msg += (f" {inteiras} mantidas inteiras (já eram só o documento) — "
                    "se alguma for foto, dê 2 cliques nela e ajuste os cantos.")
        self.status.setText(msg)
        if self._falhas:
            detalhes = "\n".join(f"• {n}: {m}" for n, m in self._falhas[:10])
            QMessageBox.warning(
                self, "Alguns arquivos falharam",
                f"{len(self._falhas)} arquivo(s) não puderam ser processados:\n\n"
                f"{detalhes}\n\nLog completo em:\n{caminho_log()}")

    # ------------------------------------------------------------------ thumbs
    def _adicionar_thumb(self, item: DocItem):
        li = QListWidgetItem(QIcon(bgr_para_qpixmap(item.imagem_final(), 200)),
                             self._titulo_item(item))
        li.setData(Qt.UserRole, len(self.itens) - 1)
        li.setSizeHint(QSize(220, 250))
        self.lista.addItem(li)

    def _titulo_item(self, item: DocItem) -> str:
        base = item.rotulo
        if item.grupo is not None:
            base = f"[PDF {item.grupo}] " + base
        if item.metodo == "nenhum":
            base = "▣ " + base  # mantida inteira (sem corte automático)
        return base

    def _atualizar_item_visual(self, linha: int):
        item = self.itens[linha]
        li = self.lista.item(linha)
        li.setIcon(QIcon(bgr_para_qpixmap(item.imagem_final(), 200)))
        li.setText(self._titulo_item(item))
        if item.grupo is not None:
            cor = QColor(CORES_GRUPO[(item.grupo - 1) % len(CORES_GRUPO)])
            cor.setAlpha(60)
            li.setBackground(cor)
        else:
            li.setBackground(QColor(0, 0, 0, 0))

    def _selecionados(self) -> list[int]:
        return sorted(li.data(Qt.UserRole) for li in self.lista.selectedItems())

    # ------------------------------------------------------------------ ações
    def editar_cantos(self):
        sel = self._selecionados()
        if len(sel) != 1:
            self.status.setText("Selecione exatamente 1 item para ajustar os cantos.")
            return
        idx = sel[0]
        item = self.itens[idx]
        dlg = EditorCantos(item, self)
        if dlg.exec() == QDialog.Accepted:
            item.cantos = dlg.cantos_reais()
            item.metodo = "manual"
            item.img_cortada = cortar_documento(
                item.img_original, item.cantos,
                folga_px=self.spin_folga.value(), modo_manual=True)
            self._atualizar_item_visual(idx)
            self.status.setText(f"Corte de “{item.rotulo}” atualizado.")

    def girar(self):
        sel = self._selecionados()
        if not sel:
            self.status.setText("Selecione ao menos 1 item para girar.")
            return
        for idx in sel:
            self.itens[idx].rotacao = (self.itens[idx].rotacao + 90) % 360
            self._atualizar_item_visual(idx)

    def agrupar(self):
        sel = self._selecionados()
        if not sel:
            self.status.setText("Selecione 1 ou mais itens para marcar como PDF "
                                "(Ctrl+clique para juntar vários num só).")
            return
        grupo = self._proximo_grupo
        self._proximo_grupo += 1
        for idx in sel:
            self.itens[idx].grupo = grupo
            self._atualizar_item_visual(idx)
        if len(sel) == 1:
            self.status.setText(f"PDF {grupo} criado com 1 página.")
        else:
            self.status.setText(f"PDF {grupo} criado com {len(sel)} páginas "
                                f"(ordem = ordem da galeria).")

    def desagrupar(self):
        sel = self._selecionados()
        if not sel:
            return
        for idx in sel:
            self.itens[idx].grupo = None
            self._atualizar_item_visual(idx)
        self.status.setText("Itens removidos dos grupos.")

    def gerar_pdfs(self):
        if not self.itens:
            return
        destino = QFileDialog.getExistingDirectory(
            self, "Pasta de saída dos PDFs", str(Path.home()))
        if not destino:
            return
        destino = Path(destino)

        grupos: dict[int, list[DocItem]] = {}
        individuais: list[DocItem] = []
        for item in self.itens:
            if item.grupo is None:
                individuais.append(item)
            else:
                grupos.setdefault(item.grupo, []).append(item)

        gerados, falhas = [], []
        for numero, membros in sorted(grupos.items()):
            caminho = caminho_sem_colisao(destino, f"documento_{numero:02d}")
            try:
                exportar_pdf([m.imagem_final() for m in membros], caminho)
                gerados.append(caminho.name)
                log.info("PDF grupo %d: %s (%d págs)", numero, caminho.name,
                         len(membros))
            except Exception as e:
                log.error("Falha no PDF do grupo %d: %s\n%s", numero, e,
                          traceback.format_exc())
                falhas.append((f"grupo {numero}", str(e)))
        for item in individuais:
            caminho = caminho_sem_colisao(destino,
                                          nome_arquivo_seguro(item.rotulo))
            try:
                exportar_pdf([item.imagem_final()], caminho)
                gerados.append(caminho.name)
                log.info("PDF individual: %s", caminho.name)
            except Exception as e:
                log.error("Falha no PDF de %s: %s\n%s", item.rotulo, e,
                          traceback.format_exc())
                falhas.append((item.rotulo, str(e)))

        if falhas:
            detalhes = "\n".join(f"• {n}: {m}" for n, m in falhas[:10])
            QMessageBox.critical(
                self, "Erro ao gerar alguns PDFs",
                f"{len(gerados)} PDFs ok, {len(falhas)} falharam:\n\n{detalhes}"
                f"\n\nLog completo em:\n{caminho_log()}")
            if not gerados:
                return

        QMessageBox.information(
            self, "PDFs gerados",
            f"{len(gerados)} PDFs criados em:\n{destino}\n\n" + "\n".join(gerados[:15]))
        self.status.setText(f"{len(gerados)} PDFs gerados em {destino}")


def main():
    configurar_log()
    app = QApplication(sys.argv)
    app.setApplicationName("CortaDoc")
    janela = JanelaPrincipal()
    janela.show()
    log.info("Janela principal exibida")
    codigo = app.exec()
    log.info("Encerrando (código %d)", codigo)
    sys.exit(codigo)


if __name__ == "__main__":
    main()
