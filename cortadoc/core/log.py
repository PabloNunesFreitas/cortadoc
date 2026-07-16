"""Logging do CortaDoc.

- Arquivo de log com rotação em ~/CortaDoc/logs/ (Windows: pasta do usuário).
- Todo crash não tratado (inclusive dentro de threads Qt) vai para o log com
  traceback completo.
- `caminho_log()` devolve o arquivo atual para mostrar ao usuário em erros.
"""

from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import platform
import sys
import threading
import traceback
from pathlib import Path

_LOG_DIR = Path.home() / "CortaDoc" / "logs"
_ARQUIVO = _LOG_DIR / "cortadoc.log"
_configurado = False


def caminho_log() -> Path:
    return _ARQUIVO


def configurar() -> logging.Logger:
    """Configura logging global (idempotente). Retorna o logger raiz do app."""
    global _configurado
    log = logging.getLogger("cortadoc")
    if _configurado:
        return log
    _configurado = True

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    arquivo = logging.handlers.RotatingFileHandler(
        _ARQUIVO, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    arquivo.setFormatter(fmt)
    arquivo.setLevel(logging.DEBUG)
    log.addHandler(arquivo)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.setLevel(logging.INFO)
    log.addHandler(console)

    # Crashes de segfault/estouro de pilha também vão para um arquivo.
    try:
        _fault_file = open(_LOG_DIR / "crash_nativo.log", "a", encoding="utf-8")
        faulthandler.enable(file=_fault_file)
    except OSError:
        pass

    # Exceções não tratadas na thread principal.
    def _excepthook(tipo, valor, tb):
        log.critical("EXCEÇÃO NÃO TRATADA:\n%s",
                      "".join(traceback.format_exception(tipo, valor, tb)))
        sys.__excepthook__(tipo, valor, tb)

    sys.excepthook = _excepthook

    # Exceções não tratadas em threads (ex.: QThread.run).
    def _threadhook(args):
        log.critical("EXCEÇÃO EM THREAD %s:\n%s", args.thread.name if args.thread else "?",
                      "".join(traceback.format_exception(
                          args.exc_type, args.exc_value, args.exc_traceback)))

    threading.excepthook = _threadhook

    log.info("=" * 60)
    log.info("CortaDoc iniciando — Python %s — %s %s",
             sys.version.split()[0], platform.system(), platform.release())
    log.info("Log em: %s", _ARQUIVO)
    return log
