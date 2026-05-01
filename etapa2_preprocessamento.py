"""
=============================================================
ETAPA 2 — PRÉ-PROCESSAMENTO COM MEDIAPIPE HANDS
Extrai pose + mão esquerda + mão direita + alguns pontos do rosto.
Gera landmarks.csv compatível com etapa3_treinamento.py e app.py.
=============================================================

RODAR:
  python etapa2_preprocessamento.py
"""

import os
import re
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
MAX_FRAMES = 20
LARGURA_PROCESSAMENTO = 480
USAR_APENAS_VIDEOS_BASE = True

# Deixe [] para processar todos os sinais encontrados.
# Ou coloque os nomes exatamente como no arquivo antes de "_Articulador".
SINAIS_FILTRO = ["Oi", "Sim", "Não", "Obrigado", "Casa"]

# pose: 33 * 4 = 132
# left hand: 21 * 3 = 63
# right hand: 21 * 3 = 63
# face selecionado: 10 * 3 = 30
# total = 288
N_FEATURES = 288

FACE_INDICES = [1, 33, 61, 199, 263, 291, 13, 14, 10, 152]

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh


def extrair_label(nome_arquivo):
    sem_ext = os.path.splitext(nome_arquivo)[0]
    partes = sem_ext.rsplit("_Articulador", 1)
    return partes[0].strip()


def grupo_video(nome_arquivo):
    sem_ext = os.path.splitext(nome_arquivo)[0]
    anterior = None

    while anterior != sem_ext:
        anterior = sem_ext
        sem_ext = re.sub(r"\s+-\s+Copia(?:\s+\(\d+\))?$", "", sem_ext, flags=re.IGNORECASE)

    return sem_ext.strip().lower()


def zeros(n):
    return np.zeros(n, dtype=np.float32)


def normalizar_features(features):
    features = features.astype(np.float32).copy()

    pose = features[:132].reshape(33, 4)
    left_hand = features[132:195].reshape(21, 3)
    right_hand = features[195:258].reshape(21, 3)
    face = features[258:288].reshape(len(FACE_INDICES), 3)

    pontos = []
    if np.any(pose[:, :3] != 0):
        pontos.append(pose[:, :3])
    if np.any(left_hand != 0):
        pontos.append(left_hand)
    if np.any(right_hand != 0):
        pontos.append(right_hand)
    if np.any(face != 0):
        pontos.append(face)

    if not pontos:
        return features

    ombro_esq = pose[11, :3]
    ombro_dir = pose[12, :3]
    if np.any(ombro_esq != 0) and np.any(ombro_dir != 0):
        centro = (ombro_esq + ombro_dir) / 2.0
        escala = float(np.linalg.norm(ombro_esq[:2] - ombro_dir[:2]))
    else:
        todos = np.vstack(pontos)
        validos = todos[np.any(todos != 0, axis=1)]
        centro = np.mean(validos, axis=0)
        largura = float(np.max(validos[:, 0]) - np.min(validos[:, 0]))
        altura = float(np.max(validos[:, 1]) - np.min(validos[:, 1]))
        escala = max(largura, altura)

    escala = max(escala, 1e-3)

    def normalizar_bloco(bloco):
        mask = np.any(bloco != 0, axis=1)
        bloco[mask] = (bloco[mask] - centro) / escala

    mask_pose = np.any(pose[:, :3] != 0, axis=1)
    pose[mask_pose, :3] = (pose[mask_pose, :3] - centro) / escala
    normalizar_bloco(left_hand)
    normalizar_bloco(right_hand)
    normalizar_bloco(face)

    return features


def extrair_features_frame(frame, hands_detector, pose_detector, face_detector):
    if frame is None:
        return zeros(N_FEATURES)

    if LARGURA_PROCESSAMENTO and frame.shape[1] > LARGURA_PROCESSAMENTO:
        escala = LARGURA_PROCESSAMENTO / frame.shape[1]
        nova_altura = int(frame.shape[0] * escala)
        frame = cv2.resize(frame, (LARGURA_PROCESSAMENTO, nova_altura), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    hands_result = hands_detector.process(rgb)
    pose_result = pose_detector.process(rgb)
    face_result = face_detector.process(rgb)

    pose = []
    if pose_result.pose_landmarks:
        for lm in pose_result.pose_landmarks.landmark[:33]:
            pose.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        pose = [0.0] * (33 * 4)

    left_hand = [0.0] * (21 * 3)
    right_hand = [0.0] * (21 * 3)
    face = []
    if face_result.multi_face_landmarks:
        face_landmarks = face_result.multi_face_landmarks[0]
        for idx in FACE_INDICES:
            lm = face_landmarks.landmark[idx]
            face.extend([lm.x, lm.y, lm.z])
    else:
        face = [0.0] * (len(FACE_INDICES) * 3)

    maos_detectadas = []
    if hands_result.multi_hand_landmarks:
        for hand_landmarks in hands_result.multi_hand_landmarks[:2]:
            coords = []
            xs = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
                xs.append(lm.x)

            maos_detectadas.append((float(np.mean(xs)), coords))

        maos_detectadas.sort(key=lambda item: item[0])

        if len(maos_detectadas) >= 1:
            left_hand = maos_detectadas[0][1]
        if len(maos_detectadas) >= 2:
            right_hand = maos_detectadas[1][1]

    features = normalizar_features(np.array(pose + left_hand + right_hand + face, dtype=np.float32))

    if features.shape[0] != N_FEATURES:
        print("[AVISO] Tamanho inesperado:", features.shape[0])

    return features


def processar_video(caminho, hands_detector, pose_detector, face_detector):
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
            features = extrair_features_frame(frame, hands_detector, pose_detector, face_detector)
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

    if USAR_APENAS_VIDEOS_BASE:
        por_grupo = {}
        for nome in sorted(todos_videos):
            por_grupo.setdefault(grupo_video(nome), nome)
        todos_videos = list(por_grupo.values())
        print("Usando apenas vídeos-base; cópias foram ignoradas.")

    if SINAIS_FILTRO:
        filtro_lower = [s.lower() for s in SINAIS_FILTRO]
        todos_videos = [
            f for f in todos_videos
            if extrair_label(f).lower() in filtro_lower
        ]
        print(f"Filtro ativo: {SINAIS_FILTRO}")

    sinais_unicos = sorted(set(extrair_label(f) for f in todos_videos))

    print("\n" + "=" * 60)
    print("  ETAPA 2 — PRÉ-PROCESSAMENTO HANDS")
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

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    ) as hands_detector, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    ) as pose_detector, mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    ) as face_detector:

        for nome in tqdm(todos_videos, desc="Processando vídeos"):
            label = extrair_label(nome)
            caminho = os.path.join(PASTA_DATA, nome)

            seq = processar_video(caminho, hands_detector, pose_detector, face_detector)

            if seq is None:
                err_count += 1
                continue

            linha = np.concatenate(seq).tolist()
            linha.append(nome)
            linha.append(label)
            linhas.append(linha)
            ok_count += 1

    if not linhas:
        print("[ERRO] Nenhum vídeo processado.")
        return

    colunas = gerar_colunas()
    colunas.insert(-1, "source_video")
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
