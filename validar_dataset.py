"""Valida os videos usados no treino."""

import os
import re
from collections import defaultdict

import cv2


PASTA_DATA = os.path.join("videos", "data")
EXTENSOES_VIDEO = {".mp4", ".avi", ".mov", ".mkv"}
PADRAO_NOME = re.compile(r"^(.+)_Articulador(\d*)$", re.IGNORECASE)

MIN_VIDEOS_AVISO = 10
MIN_VIDEOS_IDEAL = 20


def extrair_label(nome_arquivo):
    sem_ext = os.path.splitext(nome_arquivo)[0]
    match = PADRAO_NOME.match(sem_ext)
    if not match:
        return None

    label = match.group(1).strip()
    if not label:
        return None

    return label[:1].upper() + label[1:]


def contar_frames(caminho):
    cap = cv2.VideoCapture(caminho)
    aberto = cap.isOpened()
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if aberto else 0
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if aberto else 0
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if aberto else 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if aberto else 0.0
    cap.release()
    return aberto, frames, largura, altura, fps


def status_quantidade(qtd):
    if qtd < MIN_VIDEOS_AVISO:
        return "POUCOS"
    if qtd < MIN_VIDEOS_IDEAL:
        return "OK"
    return "BOM"


def main():
    print("=" * 60)
    print("  VALIDACAO DO DATASET")
    print("=" * 60)

    if not os.path.exists(PASTA_DATA):
        print(f"[ERRO] Pasta nao encontrada: {PASTA_DATA}")
        return 1

    arquivos = sorted(os.listdir(PASTA_DATA))
    videos_por_label = defaultdict(list)
    fora_do_padrao = []
    extensao_ignorada = []
    invalidos = []

    for nome in arquivos:
        caminho = os.path.join(PASTA_DATA, nome)
        if not os.path.isfile(caminho):
            continue

        ext = os.path.splitext(nome)[1].lower()
        if ext not in EXTENSOES_VIDEO:
            extensao_ignorada.append(nome)
            continue

        label = extrair_label(nome)
        if label is None:
            fora_do_padrao.append(nome)
            continue

        aberto, frames, largura, altura, fps = contar_frames(caminho)
        if (not aberto) or frames <= 0:
            invalidos.append((nome, frames))
            continue

        videos_por_label[label].append({
            "nome": nome,
            "frames": frames,
            "largura": largura,
            "altura": altura,
            "fps": fps,
        })

    total_validos = sum(len(v) for v in videos_por_label.values())

    print(f"\nPasta analisada : {PASTA_DATA}")
    print(f"Videos validos  : {total_validos}")
    print(f"Sinais validos  : {len(videos_por_label)}")

    if videos_por_label:
        print("\nVideos por sinal:")
        for label in sorted(videos_por_label):
            itens = videos_por_label[label]
            frames = [item["frames"] for item in itens]
            menor = min(frames)
            maior = max(frames)
            media = sum(frames) / len(frames)
            status = status_quantidade(len(itens))
            print(
                f"  {status:<6} {label:<15} {len(itens):>3} videos | "
                f"frames min/med/max: {menor}/{media:.0f}/{maior}"
            )

    problemas = False

    if fora_do_padrao:
        problemas = True
        print("\n[AVISO] Videos fora do padrao Nome_ArticuladorN.ext:")
        for nome in fora_do_padrao:
            print(f"  - {nome}")

    if invalidos:
        problemas = True
        print("\n[ERRO] Videos que nao abriram ou estao sem frames:")
        for nome, frames in invalidos:
            print(f"  - {nome} ({frames} frames)")

    sinais_poucos = {
        label: len(itens)
        for label, itens in videos_por_label.items()
        if len(itens) < MIN_VIDEOS_AVISO
    }
    if sinais_poucos:
        print(f"\n[AVISO] Sinais com menos de {MIN_VIDEOS_AVISO} videos:")
        for label in sorted(sinais_poucos):
            print(f"  - {label}: {sinais_poucos[label]}")

    if extensao_ignorada:
        print("\nArquivos ignorados por nao serem video:")
        for nome in extensao_ignorada[:10]:
            print(f"  - {nome}")
        if len(extensao_ignorada) > 10:
            print(f"  ... e mais {len(extensao_ignorada) - 10}")

    print("\nResultado:")
    if problemas:
        print("  Corrija os avisos/erros antes de rodar o preprocessamento final.")
        return 1

    print("  Dataset pronto para rodar etapa2_preprocessamento.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
