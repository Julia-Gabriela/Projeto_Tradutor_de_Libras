"""
=============================================================
ETAPA 2 — PRÉ-PROCESSAMENTO COM MEDIAPIPE HOLISTIC
Extrai pose + mão esquerda + mão direita + alguns pontos do rosto.
Gera landmarks.csv compatível com etapa3_treinamento.py e app.py.
=============================================================

RODAR:
  python etapa2_preprocessamento.py
"""

import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from tqdm import tqdm

# =========================
# CONFIGURAÇÕES
# =========================
PASTA_DATA = os.path.join("videos", "data")
ARQUIVO_SAIDA = "landmarks.csv"
MAX_FRAMES = 30

# Deixe [] para processar todos os sinais encontrados.
# Ou coloque os nomes exatamente como no arquivo antes de "_Articulador".
SINAIS_FILTRO = ["Oi", "Água", "Ajudar", "Obrigado", "Casa"]

# pose: 33 * 4 = 132
# left hand: 21 * 3 = 63
# right hand: 21 * 3 = 63
# face selecionado: 10 * 3 = 30
# total = 288
N_FEATURES = 288

FACE_INDICES = [1, 33, 61, 199, 263, 291, 13, 14, 10, 152]

mp_holistic = mp.solutions.holistic


def extrair_label(nome_arquivo):
    sem_ext = os.path.splitext(nome_arquivo)[0]
    partes = sem_ext.rsplit("_Articulador", 1)
    return partes[0].strip()


def zeros(n):
    return np.zeros(n, dtype=np.float32)


def extrair_features_frame(frame, holistic):
    if frame is None:
        return zeros(N_FEATURES)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = holistic.process(rgb)

    # POSE: 33 pontos x,y,z,visibility
    pose = []
    if result.pose_landmarks:
        for lm in result.pose_landmarks.landmark[:33]:
            pose.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        pose = [0.0] * (33 * 4)

    # MÃO ESQUERDA: 21 pontos x,y,z
    left_hand = []
    if result.left_hand_landmarks:
        for lm in result.left_hand_landmarks.landmark:
            left_hand.extend([lm.x, lm.y, lm.z])
    else:
        left_hand = [0.0] * (21 * 3)

    # MÃO DIREITA: 21 pontos x,y,z
    right_hand = []
    if result.right_hand_landmarks:
        for lm in result.right_hand_landmarks.landmark:
            right_hand.extend([lm.x, lm.y, lm.z])
    else:
        right_hand = [0.0] * (21 * 3)

    # ROSTO: apenas alguns pontos principais
    face = []
    if result.face_landmarks:
        for idx in FACE_INDICES:
            lm = result.face_landmarks.landmark[idx]
            face.extend([lm.x, lm.y, lm.z])
    else:
        face = [0.0] * (len(FACE_INDICES) * 3)

    features = np.array(pose + left_hand + right_hand + face, dtype=np.float32)

    if features.shape[0] != N_FEATURES:
        print("[AVISO] Tamanho inesperado:", features.shape[0])

    return features


def processar_video(caminho, holistic):
    cap = cv2.VideoCapture(caminho)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, MAX_FRAMES, dtype=int)
    sequencia = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if ok:
            features = extrair_features_frame(frame, holistic)
        else:
            features = zeros(N_FEATURES)

        sequencia.append(features)

    cap.release()
    return sequencia


def gerar_colunas():
    colunas = []

    for f in range(MAX_FRAMES):
        # pose
        for p in range(33):
            colunas += [
                f"f{f}_pose{p}_x",
                f"f{f}_pose{p}_y",
                f"f{f}_pose{p}_z",
                f"f{f}_pose{p}_v",
            ]

        # left hand
        for p in range(21):
            colunas += [
                f"f{f}_left_hand{p}_x",
                f"f{f}_left_hand{p}_y",
                f"f{f}_left_hand{p}_z",
            ]

        # right hand
        for p in range(21):
            colunas += [
                f"f{f}_right_hand{p}_x",
                f"f{f}_right_hand{p}_y",
                f"f{f}_right_hand{p}_z",
            ]

        # face
        for p in FACE_INDICES:
            colunas += [
                f"f{f}_face{p}_x",
                f"f{f}_face{p}_y",
                f"f{f}_face{p}_z",
            ]

    colunas.append("label")
    return colunas


def main():
    if not os.path.exists(PASTA_DATA):
        print(f"[ERRO] Pasta não encontrada: {PASTA_DATA}")
        return

    todos_videos = [
        f for f in os.listdir(PASTA_DATA)
        if f.lower().endswith(".mp4") and "_articulador" in f.lower()
    ]

    if SINAIS_FILTRO:
        filtro_lower = [s.lower() for s in SINAIS_FILTRO]
        todos_videos = [
            f for f in todos_videos
            if extrair_label(f).lower() in filtro_lower
        ]
        print(f"Filtro ativo: {SINAIS_FILTRO}")

    sinais_unicos = sorted(set(extrair_label(f) for f in todos_videos))

    print("\n" + "=" * 60)
    print("  ETAPA 2 — PRÉ-PROCESSAMENTO HOLISTIC")
    print("=" * 60)
    print(f"  Vídeos encontrados : {len(todos_videos)}")
    print(f"  Sinais únicos      : {len(sinais_unicos)}")
    print(f"  Sinais             : {sinais_unicos}")
    print(f"  Features/frame     : {N_FEATURES}")
    print()

    if len(todos_videos) == 0:
        print("[AVISO] Nenhum vídeo encontrado.")
        return

    linhas = []
    ok_count = 0
    err_count = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as holistic:

        for nome in tqdm(todos_videos, desc="Processando vídeos"):
            label = extrair_label(nome)
            caminho = os.path.join(PASTA_DATA, nome)

            seq = processar_video(caminho, holistic)

            if seq is None:
                err_count += 1
                continue

            linha = np.concatenate(seq).tolist()
            linha.append(label)
            linhas.append(linha)
            ok_count += 1

    if not linhas:
        print("[ERRO] Nenhum vídeo processado.")
        return

    colunas = gerar_colunas()
    df = pd.DataFrame(linhas, columns=colunas)
    df.to_csv(ARQUIVO_SAIDA, index=False)

    print("\n" + "=" * 60)
    print("  CONCLUÍDO!")
    print("=" * 60)
    print(f"  Processados : {ok_count} | Erros: {err_count}")
    print(f"  Amostras    : {len(df)}")
    print(f"  Sinais      : {df['label'].nunique()}")
    print(f"  CSV salvo em: {ARQUIVO_SAIDA}")
    print()
    print("  Próximo passo:")
    print("  python etapa3_treinamento.py")


if __name__ == "__main__":
    main()
