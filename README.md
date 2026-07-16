# CortaDoc

Corta automaticamente o papel em fotos de documentos (sem cortar nenhuma
escrita) e agrupa as páginas em PDFs.

## Como funciona

1. **Importar** fotos (JPG/PNG/…) ou PDFs escaneados.
2. Cada página passa pela detecção de cantos com o modelo **DocAligner**
   (fastvit_sa24, heatmap regression — rodado direto em ONNX Runtime, sem as
   dependências pesadas do pacote original). Robusto a sombra, dobra e fundo
   parecido com o papel. Depois um **GrabCut** guiado pelo quadrilátero
   remove tudo que não é papel (mesa, tecido, chão vira branco), com buffer
   de segurança de 15px — tinta jamais é pintada ou cortada.
   Se a IA não detectar (ex.: scan/PDF digital que já é só o documento), a
   página é mantida inteira — heurísticas OpenCV foram REMOVIDAS de propósito
   após cortarem conteúdo em documentos digitais nos testes. O **editor
   manual de cantos** (arrasta 4 círculos) cobre qualquer caso restante.

### Logs

Tudo é logado em `~/CortaDoc/logs/cortadoc.log` (rotação automática, 3
backups): cada arquivo processado, método usado, tempos, e traceback completo
de qualquer erro — inclusive crashes de thread e exceções não tratadas.
Falhas por arquivo aparecem na interface com o caminho do log.
3. O papel é endireitado (warp de perspectiva) com **folga de segurança**
   configurável para nunca cortar texto.
4. Na galeria: selecione miniaturas com Ctrl+clique e **"Agrupar em PDF"** —
   cada grupo vira um PDF multipágina; o que ficar sem grupo vira PDF
   individual.
5. **Gerar PDFs** exporta tudo para a pasta escolhida (via img2pdf, JPEG
   embutido sem reencodar).

## Rodar do código

```bash
pip install -r requirements.txt
python scripts/baixar_modelos.py   # baixa o modelo ONNX (1x só, ~80MB)
python main.py
```

## Build do .exe (Windows)

O workflow `.github/workflows/build-windows.yml` gera o `CortaDoc.exe` num
runner Windows do GitHub Actions:

- automático ao criar uma tag `v*`;
- ou manual pela aba Actions → "Build Windows exe" → Run workflow.

O `.exe` sai como artifact `CortaDoc-windows`.

## Estrutura

```
cortadoc/
├── app.py            # GUI (PySide6)
└── core/
    ├── detector.py   # DocAligner + fallback OpenCV
    ├── warp.py       # retificação de perspectiva + folga
    ├── pdf.py        # exportação img2pdf
    └── io_utils.py   # leitura de imagens e rasterização de PDFs
```
