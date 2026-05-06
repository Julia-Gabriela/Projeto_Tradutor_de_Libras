"""
=============================================================
ETAPA 2 — PRÉ-PROCESSAMENTO COM MEDIAPIPE HANDS (MELHORADO)
Extrai pose + mão esquerda + mão direita + pontos do rosto.
Gera landmarks.csv compatível com etapa3 e app.py.

MELHORIAS:
  - Suporte a múltiplos vídeos por sinal (melhora muito o treino!)
  - Detecção de qualidade do frame (descarta frames ruins)
  - Normalização mais robusta usando distância ombro-quadril
  - Janela deslizante: gera múltiplas amostras por vídeo longo
  - Relatório detalhado de qualidade por sinal
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
USAR_APENAS_VIDEOS_BASE = False  # MUDANÇA: False = usa TODOS os vídeos (mais dados!)

# Qualidade mínima: descarta frames onde as mãos não são detectadas
MIN_FRAMES_COM_MAOS = 6  # de 20 frames, pelo menos 8 devem ter mãos

# Janela deslizante para vídeos longos (gera mais amostras)
USAR_JANELA_DESLIZANTE = True
PASSO_JANELA = 15  # a cada 5 frames, gera uma nova amostra

SINAIS_FILTRO = []

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
    label = partes[0].strip()
    aliases = {
        "oi": "Oi",
        "casa": "Casa",
        "banheiro": "Banheiro",
        "livro": "Livro",
        "nome": "Nome",
        "obrigado": "Obrigado",
        "desconhecido": "Desconhecido",
    }
    return aliases.get(label.lower(), label[:1].upper() + label[1:])


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
    """
    Normalização melhorada: usa distância ombro-quadril como escala
    para ser mais robusta a variações de distância da câmera.
    """
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
    quadril_esq = pose[23, :3]
    quadril_dir = pose[24, :3]

    tem_ombros = np.any(ombro_esq != 0) and np.any(ombro_dir != 0)
    tem_quadril = np.any(quadril_esq != 0) and np.any(quadril_dir != 0)

    if tem_ombros and tem_quadril:
        # Escala = distância ombro-quadril (mais estável que só ombros)
        centro_ombros = (ombro_esq + ombro_dir) / 2.0
        centro_quadril = (quadril_esq + quadril_dir) / 2.0
        centro = (centro_ombros + centro_quadril) / 2.0
        escala = float(np.linalg.norm(centro_ombros[:2] - centro_quadril[:2]))
        # Garante escala mínima usando largura dos ombros como fallback
        escala_ombros = float(np.linalg.norm(ombro_esq[:2] - ombro_dir[:2]))
        escala = max(escala, escala_ombros * 0.5)
    elif tem_ombros:
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
        return zeros(N_FEATURES), False

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
    maos_detectadas_count = 0

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
        maos_detectadas_count = len(maos_detectadas)

        if len(maos_detectadas) >= 1:
            left_hand = maos_detectadas[0][1]
        if len(maos_detectadas) >= 2:
            right_hand = maos_detectadas[1][1]

    tem_maos = maos_detectadas_count > 0
    features = normalizar_features(np.array(pose + left_hand + right_hand + face, dtype=np.float32))

    return features, tem_maos


def processar_video_completo(caminho, hands_detector, pose_detector, face_detector):
    """Lê todos os frames do vídeo e retorna lista de (features, tem_maos)."""
    cap = cv2.VideoCapture(caminho)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    todos_frames = []
    for i in range(total_frames):
        ok, frame = cap.read()
        if ok:
            features, tem_maos = extrair_features_frame(frame, hands_detector, pose_detector, face_detector)
            todos_frames.append((features, tem_maos))
        else:
            todos_frames.append((zeros(N_FEATURES), False))

    cap.release()
    return todos_frames


def amostrar_sequencia(todos_frames, inicio, fim, n_frames=MAX_FRAMES):
    """Amostra n_frames frames uniformemente entre inicio e fim."""
    segmento = todos_frames[inicio:fim]
    if len(segmento) == 0:
        return None

    indices = np.linspace(0, len(segmento) - 1, n_frames, dtype=int)
    sequencia = [segmento[i][0] for i in indices]
    frames_com_maos = sum(1 for i in indices if segmento[i][1])
    return sequencia, frames_com_maos


def processar_video(caminho, hands_detector, pose_detector, face_detector):
    """
    Processa um vídeo e retorna uma ou mais amostras.
    Se USAR_JANELA_DESLIZANTE, gera múltiplas amostras por vídeo longo.
    """
    todos_frames = processar_video_completo(caminho, hands_detector, pose_detector, face_detector)
    if todos_frames is None or len(todos_frames) == 0:
        return []

    total = len(todos_frames)
    amostras = []

    if not USAR_JANELA_DESLIZANTE or total <= MAX_FRAMES:
        # Modo simples: uma única amostra
        resultado = amostrar_sequencia(todos_frames, 0, total)
        if resultado:
            seq, n_maos = resultado
            if n_maos >= MIN_FRAMES_COM_MAOS:
                amostras.append(seq)
    else:
        # Janela deslizante: gera múltiplas amostras
        for inicio in range(0, total - MAX_FRAMES + 1, PASSO_JANELA):
            fim = inicio + MAX_FRAMES
            resultado = amostrar_sequencia(todos_frames, inicio, fim)
            if resultado:
                seq, n_maos = resultado
                if n_maos >= MIN_FRAMES_COM_MAOS:
                    amostras.append(seq)

        # Garante pelo menos uma amostra (do vídeo inteiro)
        if not amostras:
            resultado = amostrar_sequencia(todos_frames, 0, total)
            if resultado:
                seq, n_maos = resultado
                amostras.append(seq)

    return amostras


def gerar_colunas():
    colunas = []
    for f in range(MAX_FRAMES):
        for p in range(33):
            colunas += [f"f{f}_pose{p}_x", f"f{f}_pose{p}_y", f"f{f}_pose{p}_z", f"f{f}_pose{p}_v"]
        for p in range(21):
            colunas += [f"f{f}_left_hand{p}_x", f"f{f}_left_hand{p}_y", f"f{f}_left_hand{p}_z"]
        for p in range(21):
            colunas += [f"f{f}_right_hand{p}_x", f"f{f}_right_hand{p}_y", f"f{f}_right_hand{p}_z"]
        for p in FACE_INDICES:
            colunas += [f"f{f}_face{p}_x", f"f{f}_face{p}_y", f"f{f}_face{p}_z"]
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
        print("Usando apenas vídeos-base.")
    else:
        print(f"Usando TODOS os {len(todos_videos)} vídeos encontrados (incluindo cópias).")

    if SINAIS_FILTRO:
        filtro_lower = [s.lower() for s in SINAIS_FILTRO]
        todos_videos = [f for f in todos_videos if extrair_label(f).lower() in filtro_lower]
        print(f"Filtro ativo: {SINAIS_FILTRO}")

    sinais_unicos = sorted(set(extrair_label(f) for f in todos_videos))

    print("\n" + "=" * 60)
    print("  ETAPA 2 — PRÉ-PROCESSAMENTO MELHORADO")
    print("=" * 60)
    print(f"  Vídeos encontrados  : {len(todos_videos)}")
    print(f"  Sinais únicos       : {len(sinais_unicos)}")
    print(f"  Sinais              : {sinais_unicos}")
    print(f"  Features/frame      : {N_FEATURES}")
    print(f"  Janela deslizante   : {'SIM (mais amostras!)' if USAR_JANELA_DESLIZANTE else 'NÃO'}")
    print()

    if len(todos_videos) == 0:
        print("[AVISO] Nenhum vídeo encontrado.")
        return

    linhas = []
    ok_count = 0
    err_count = 0
    descartados = 0
    amostras_por_sinal = {}

    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as hands_detector, mp_pose.Pose(
        static_image_mode=False, model_complexity=1, smooth_landmarks=True,
        enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as pose_detector, mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as face_detector:

        for nome in tqdm(todos_videos, desc="Processando vídeos"):
            label = extrair_label(nome)
            caminho = os.path.join(PASTA_DATA, nome)

            amostras = processar_video(caminho, hands_detector, pose_detector, face_detector)

            if not amostras:
                err_count += 1
                descartados += 1
                continue

            for seq in amostras:
                linha = np.concatenate(seq).tolist()
                linha.append(nome)
                linha.append(label)
                linhas.append(linha)

            amostras_por_sinal[label] = amostras_por_sinal.get(label, 0) + len(amostras)
            ok_count += 1

    if not linhas:
        print("[ERRO] Nenhum vídeo processado com sucesso.")
        return

    colunas = gerar_colunas()
    colunas.insert(-1, "source_video")
    df = pd.DataFrame(linhas, columns=colunas)

    # Equilibra limitando ao máximo por sinal
    MAX_AMOSTRAS_POR_SINAL = 150
    df = df.groupby('label').apply(
        lambda x: x.sample(min(len(x), MAX_AMOSTRAS_POR_SINAL), random_state=42)
    ).reset_index(drop=True)

    print(f"\n  Após balanceamento:")
    print(df['label'].value_counts().to_string())

    df.to_csv(ARQUIVO_SAIDA, index=False)

    print("\n" + "=" * 60)
    print("  CONCLUÍDO!")
    print("=" * 60)
    print(f"  Vídeos processados : {ok_count} | Erros/descartados: {err_count}")
    print(f"  Total de amostras  : {len(df)}")
    print(f"  Sinais únicos      : {df['label'].nunique()}")
    print(f"\n  Amostras por sinal:")
    for sinal, n in sorted(amostras_por_sinal.items()):
        status = "OK" if n >= 5 else "POUCOS DADOS"
        print(f"    {status}  {sinal}: {n} amostras")

    sinal_minimo = min(amostras_por_sinal.values()) if amostras_por_sinal else 0
    if sinal_minimo < 5:
        print("\n  ⚠ AVISO: Alguns sinais têm menos de 5 amostras.")
        print("    Adicione mais vídeos para esses sinais para melhorar a acurácia.")

    print(f"\n  CSV salvo em: {ARQUIVO_SAIDA}")
    print("\n  Próximo passo:")
    print("  python etapa3_treinamento.py")


if __name__ == "__main__":
    main()
