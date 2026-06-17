"""Extrai landmarks dos videos e gera data/landmarks.csv."""

import os
import re
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from tqdm import tqdm

PASTA_DATA = os.path.join("videos", "data")
DATA_DIR = "data"
ARQUIVO_SAIDA = os.path.join(DATA_DIR, "landmarks.csv")
ARQUIVO_QUALIDADE = os.path.join(DATA_DIR, "qualidade_preprocessamento.csv")
MAX_FRAMES = 20
LARGURA_PROCESSAMENTO = 480
USAR_APENAS_VIDEOS_BASE = False

MIN_FRAMES_COM_MAOS_PADRAO = 8
MIN_MOVIMENTO_MAOS_PADRAO = 0.0012

# Ajustes pontuais para sinais com pouca mao detectada ou movimento menor.
QUALIDADE_POR_SINAL = {
    "Desconhecido": {"min_maos": 4, "min_movimento": 0.0},
    "Nome": {"min_maos": 6, "min_movimento": 0.0},
    "Oi": {"min_maos": 8, "min_movimento": 0.0004},
    "Banheiro": {"min_maos": 8, "min_movimento": 0.0006},
    "Obrigado": {"min_maos": 6, "min_movimento": 0.0008},
}

USAR_MELHOR_JANELA_FALLBACK = True

USAR_JANELA_DESLIZANTE = True
PASSO_JANELA = 15

SINAIS_FILTRO = []

# pose + mao esquerda + mao direita + rosto selecionado.
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
        centro_ombros = (ombro_esq + ombro_dir) / 2.0
        centro_quadril = (quadril_esq + quadril_dir) / 2.0
        centro = (centro_ombros + centro_quadril) / 2.0
        escala = float(np.linalg.norm(centro_ombros[:2] - centro_quadril[:2]))
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
    segmento = todos_frames[inicio:fim]
    if len(segmento) == 0:
        return None

    indices = np.linspace(0, len(segmento) - 1, n_frames, dtype=int)
    sequencia = [segmento[i][0] for i in indices]
    frames_com_maos = sum(1 for i in indices if segmento[i][1])
    movimento_maos = calcular_movimento_maos(sequencia)
    return sequencia, frames_com_maos, movimento_maos


def calcular_movimento_maos(sequencia):
    if len(sequencia) < 2:
        return 0.0
    movimentos = []
    for i in range(1, len(sequencia)):
        atual = sequencia[i][132:258]
        anterior = sequencia[i - 1][132:258]
        if np.any(atual != 0) or np.any(anterior != 0):
            movimentos.append(float(np.mean(np.abs(atual - anterior))))
    return float(np.percentile(movimentos, 75)) if movimentos else 0.0


def obter_config_qualidade(label):
    config = QUALIDADE_POR_SINAL.get(label, {})
    return {
        "min_maos": config.get("min_maos", MIN_FRAMES_COM_MAOS_PADRAO),
        "min_movimento": config.get("min_movimento", MIN_MOVIMENTO_MAOS_PADRAO),
    }


def avaliar_janela(label, n_maos, movimento, fallback=False):
    config = obter_config_qualidade(label)
    min_maos = config["min_maos"]
    min_movimento = config["min_movimento"]

    if fallback:
        min_maos = max(3, min_maos - 2)
        min_movimento = min_movimento * 0.5

    if n_maos < min_maos:
        return False, "poucas_maos"
    if movimento < min_movimento:
        return False, "baixo_movimento"
    return True, "ok_fallback" if fallback else "ok"


def pontuar_janela(label, n_maos, movimento):
    config = obter_config_qualidade(label)
    score_maos = n_maos / max(config["min_maos"], 1)
    score_movimento = 1.0
    if config["min_movimento"] > 0:
        score_movimento = movimento / config["min_movimento"]
    return min(score_maos, 1.5) + min(score_movimento, 1.5)


def processar_video(caminho, label, hands_detector, pose_detector, face_detector):
    todos_frames = processar_video_completo(caminho, hands_detector, pose_detector, face_detector)
    if todos_frames is None or len(todos_frames) == 0:
        return [], [], 0

    total = len(todos_frames)
    amostras = []
    metricas = []
    candidatas = []

    if not USAR_JANELA_DESLIZANTE or total <= MAX_FRAMES:
        resultado = amostrar_sequencia(todos_frames, 0, total)
        if resultado:
            seq, n_maos, movimento = resultado
            aceita, motivo = avaliar_janela(label, n_maos, movimento)
            metricas.append((0, total, n_maos, movimento, aceita, motivo))
            candidatas.append((pontuar_janela(label, n_maos, movimento), seq, 0, total, n_maos, movimento))
            if aceita:
                amostras.append(seq)
    else:
        for inicio in range(0, total - MAX_FRAMES + 1, PASSO_JANELA):
            fim = inicio + MAX_FRAMES
            resultado = amostrar_sequencia(todos_frames, inicio, fim)
            if resultado:
                seq, n_maos, movimento = resultado
                aceita, motivo = avaliar_janela(label, n_maos, movimento)
                metricas.append((inicio, fim, n_maos, movimento, aceita, motivo))
                candidatas.append((pontuar_janela(label, n_maos, movimento), seq, inicio, fim, n_maos, movimento))
                if aceita:
                    amostras.append(seq)

        if not amostras:
            resultado = amostrar_sequencia(todos_frames, 0, total)
            if resultado:
                seq, n_maos, movimento = resultado
                aceita, motivo = avaliar_janela(label, n_maos, movimento)
                metricas.append((0, total, n_maos, movimento, aceita, motivo))
                candidatas.append((pontuar_janela(label, n_maos, movimento), seq, 0, total, n_maos, movimento))
                if aceita:
                    amostras.append(seq)

        if not amostras and USAR_MELHOR_JANELA_FALLBACK and candidatas:
            _, seq, inicio, fim, n_maos, movimento = max(candidatas, key=lambda item: item[0])
            aceita, motivo = avaliar_janela(label, n_maos, movimento, fallback=True)
            if aceita:
                amostras.append(seq)
                metricas.append((inicio, fim, n_maos, movimento, True, motivo))

    return amostras, metricas, total


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
        print(f"[ERRO] Pasta nao encontrada: {PASTA_DATA}")
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
        print("Usando apenas videos-base.")
    else:
        print(f"Usando TODOS os {len(todos_videos)} videos encontrados (incluindo copias).")

    if SINAIS_FILTRO:
        filtro_lower = [s.lower() for s in SINAIS_FILTRO]
        todos_videos = [f for f in todos_videos if extrair_label(f).lower() in filtro_lower]
        print(f"Filtro ativo: {SINAIS_FILTRO}")

    sinais_unicos = sorted(set(extrair_label(f) for f in todos_videos))

    print("\n" + "=" * 60)
    print("  ETAPA 2 - PRE-PROCESSAMENTO")
    print("=" * 60)
    print(f"  Videos encontrados  : {len(todos_videos)}")
    print(f"  Sinais unicos       : {len(sinais_unicos)}")
    print(f"  Sinais              : {sinais_unicos}")
    print(f"  Features/frame      : {N_FEATURES}")
    print(f"  Janela deslizante   : {'SIM (mais amostras!)' if USAR_JANELA_DESLIZANTE else 'NAO'}")
    print()

    if len(todos_videos) == 0:
        print("[AVISO] Nenhum video encontrado.")
        return

    linhas = []
    ok_count = 0
    err_count = 0
    descartados = 0
    amostras_por_sinal = {}
    qualidade = []

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

        for nome in tqdm(todos_videos, desc="Processando videos"):
            label = extrair_label(nome)
            caminho = os.path.join(PASTA_DATA, nome)

            amostras, metricas, total_frames = processar_video(
                caminho, label, hands_detector, pose_detector, face_detector
            )

            if not amostras:
                err_count += 1
                descartados += 1

            for idx_metrica, (inicio, fim, n_maos, movimento, aceita, motivo) in enumerate(metricas):
                qualidade.append({
                    "source_video": nome,
                    "label": label,
                    "total_frames_video": total_frames,
                    "janela_inicio": inicio,
                    "janela_fim": fim,
                    "frames_com_maos": n_maos,
                    "ratio_maos": round(n_maos / MAX_FRAMES, 4),
                    "movimento_maos": round(movimento, 6),
                    "aceita": aceita,
                    "motivo": motivo,
                    "indice_amostra": idx_metrica,
                })

            if not amostras:
                continue

            for seq in amostras:
                linha = np.concatenate(seq).tolist()
                linha.append(nome)
                linha.append(label)
                linhas.append(linha)

            amostras_por_sinal[label] = amostras_por_sinal.get(label, 0) + len(amostras)
            ok_count += 1

    if not linhas:
        print("[ERRO] Nenhum video processado com sucesso.")
        return

    colunas = gerar_colunas()
    colunas.insert(-1, "source_video")
    df = pd.DataFrame(linhas, columns=colunas)

    MAX_AMOSTRAS_POR_SINAL = 150
    df = df.groupby('label').apply(
        lambda x: x.sample(min(len(x), MAX_AMOSTRAS_POR_SINAL), random_state=42)
    ).reset_index(drop=True)

    print(f"\n  Apos balanceamento:")
    print(df['label'].value_counts().to_string())

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(ARQUIVO_SAIDA, index=False)
    pd.DataFrame(qualidade).to_csv(ARQUIVO_QUALIDADE, index=False)

    print("\n" + "=" * 60)
    print("  CONCLUIDO!")
    print("=" * 60)
    print(f"  Videos processados : {ok_count} | Erros/descartados: {err_count}")
    print(f"  Total de amostras  : {len(df)}")
    print(f"  Sinais unicos      : {df['label'].nunique()}")
    print(f"\n  Amostras por sinal:")
    for sinal, n in sorted(amostras_por_sinal.items()):
        status = "OK" if n >= 5 else "POUCOS DADOS"
        print(f"    {status}  {sinal}: {n} amostras")

    sinal_minimo = min(amostras_por_sinal.values()) if amostras_por_sinal else 0
    if sinal_minimo < 5:
        print("\n   AVISO: Alguns sinais tem menos de 5 amostras.")
        print("    Adicione mais videos para esses sinais para melhorar a acuracia.")

    print(f"\n  CSV salvo em: {ARQUIVO_SAIDA}")
    print(f"  Qualidade salva em: {ARQUIVO_QUALIDADE}")
    print("\n  Proximo passo:")
    print("  python etapa3_treinamento.py")


if __name__ == "__main__":
    main()