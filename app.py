"""
=============================================================
APP — Tradutor de Libras com IA (MELHORADO)
MediaPipe Hands + Flask + LSTM com Atenção

MELHORIAS:
  - Buffer com reset inteligente (não mistura sinais)
  - Sistema de votação por maioria ponderada
  - Debounce: evita repetir o mesmo sinal imediatamente
  - Thresholds calibrados por classe (mais justos)
  - UI com histórico de sinais traduzidos
  - Indicador visual de confiança em tempo real
=============================================================

RODAR:
  python app.py

ACESSAR:
  http://localhost:5000
"""

import os
import pickle
import base64
import threading
import unicodedata
import csv
import time
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from collections import deque

app = Flask(
    __name__,
    template_folder=os.path.join("front", "templates"),
    static_folder=os.path.join("front", "static"),
)

# =========================
# CONFIGURAÇÕES
# =========================
MAX_FRAMES = 20
N_FEATURES = 288
LARGURA_PROCESSAMENTO = 416
FACE_INDICES = [1, 33, 61, 199, 263, 291, 13, 14, 10, 152]
DATA_DIR = "data"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "modelo_libras.keras")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
THRESHOLDS_PATH = os.path.join(MODELS_DIR, "thresholds.pkl")

# --- Parâmetros de detecção ---
CONFIANCA_MINIMA_GLOBAL = 0.75   # Limiar padrão (sobrescrito pelos thresholds por classe)
MARGEM_MINIMA = 0.35           # Diferença mínima entre 1º e 2º lugar
VOTOS_NECESSARIOS = 2            # De 3 predições, quantas precisam concordar
JANELA_PREDICOES = 3             # Tamanho da janela de votação
DEBOUNCE_FRAMES = 10             # Frames para esperar antes de aceitar o mesmo sinal de novo
MOVIMENTO_MINIMO = 0.0006       # Movimento mínimo para considerar que há sinal
FRAMES_SEM_MAOS_TOLERANCIA = 4  # Evita resetar tudo quando o MediaPipe pisca por poucos frames
MIN_FRAMES_VALIDOS_PREDICAO = 8
FEEDBACK_CSV = os.path.join(DATA_DIR, "feedback_landmarks.csv")
DEBUG_PREDICOES = False

# Overrides manuais opcionais. Mantenha vazio para usar a calibragem do treino.
THRESHOLDS_MANUAIS = {}
MARGENS_MANUAIS = {}
VOTOS_MANUAIS = {}
LABELS_REJEICAO = {"Desconhecido", "Desconhecida", "Neutro", "Fundo", "Nada", "Outro"}

# Estado global
frame_buffer = deque(maxlen=MAX_FRAMES)
historico_predicoes = deque(maxlen=JANELA_PREDICOES)
ultimo_features = None
debounce_counter = 0
frames_sem_maos = 0
ultima_seq_feedback = None
ultimo_label_feedback = None
ultima_conf_feedback = 0.0

# Locks
hands_lock = threading.Lock()
pose_face_lock = threading.Lock()

# Cache de pose/rosto (processados a cada N frames)
visual_frame_count = 0
PROCESSAR_POSE_ROSTO_A_CADA = 3
ultimo_pose_features = [0.0] * (33 * 4)
ultimo_face_features = [0.0] * (len(FACE_INDICES) * 3)
ultimo_visual_pose = []
ultimo_visual_face = []

# =========================
# CARREGAR MODELO
# =========================
print("Carregando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

labels = le.classes_.tolist()
print(f"Modelo carregado! Sinais: {labels}")

# Carrega thresholds calibrados (se existir)
thresholds_por_classe = {}
if os.path.exists(THRESHOLDS_PATH):
    with open(THRESHOLDS_PATH, "rb") as f:
        thresholds_por_classe = pickle.load(f)
    print(f"Thresholds calibrados carregados: {thresholds_por_classe}")
else:
    print("thresholds.pkl não encontrado. Usando limiar global.")

def chave_label(label):
    return unicodedata.normalize("NFKD", str(label)).encode("ascii", "ignore").decode("ascii")


def eh_label_rejeicao(label):
    return chave_label(label) in LABELS_REJEICAO


def get_threshold(label):
    return THRESHOLDS_MANUAIS.get(chave_label(label), thresholds_por_classe.get(label, CONFIANCA_MINIMA_GLOBAL))


def get_margem_minima(label):
    return MARGENS_MANUAIS.get(chave_label(label), MARGEM_MINIMA)


def get_votos_necessarios(label):
    return VOTOS_MANUAIS.get(chave_label(label), VOTOS_NECESSARIOS)

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

hands_detector = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, model_complexity=1,
    min_detection_confidence=0.55, min_tracking_confidence=0.55
)
pose_detector = mp_pose.Pose(
    static_image_mode=False, model_complexity=1, smooth_landmarks=True,
    enable_segmentation=False, min_detection_confidence=0.55, min_tracking_confidence=0.55
)
face_detector = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=False,
    min_detection_confidence=0.55, min_tracking_confidence=0.55
)



# =========================
# FUNÇÕES DE PROCESSAMENTO
# =========================

def normalizar_features(features):
    features = features.astype(np.float32).copy()
    pose = features[:132].reshape(33, 4)
    left_hand = features[132:195].reshape(21, 3)
    right_hand = features[195:258].reshape(21, 3)
    face = features[258:288].reshape(len(FACE_INDICES), 3)

    pontos = []
    if np.any(pose[:, :3] != 0): pontos.append(pose[:, :3])
    if np.any(left_hand != 0): pontos.append(left_hand)
    if np.any(right_hand != 0): pontos.append(right_hand)
    if np.any(face != 0): pontos.append(face)

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


def extrair_features_e_visual(frame):
    global visual_frame_count, ultimo_pose_features, ultimo_face_features
    global ultimo_visual_pose, ultimo_visual_face

    if LARGURA_PROCESSAMENTO and frame.shape[1] > LARGURA_PROCESSAMENTO:
        escala = LARGURA_PROCESSAMENTO / frame.shape[1]
        nova_altura = int(frame.shape[0] * escala)
        frame = cv2.resize(frame, (LARGURA_PROCESSAMENTO, nova_altura), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    try:
        with hands_lock:
            result = hands_detector.process(rgb)
    except Exception as e:
        print("[ERRO MEDIAPIPE]", e)
        return np.zeros(N_FEATURES, dtype=np.float32), {"hands": [], "face": [], "pose": []}, {"hands": 0, "face": False, "pose": False}

    pose = ultimo_pose_features.copy()
    left_hand = [0.0] * (21 * 3)
    right_hand = [0.0] * (21 * 3)
    face = ultimo_face_features.copy()
    visual = {"hands": [], "face": ultimo_visual_face, "pose": ultimo_visual_pose}

    maos_detectadas = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks[:2]:
            coords = []
            pontos_2d = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
                pontos_2d.append([lm.x, lm.y])
            centro_x = float(np.mean([p[0] for p in pontos_2d]))
            maos_detectadas.append((centro_x, coords, pontos_2d))

        maos_detectadas.sort(key=lambda item: item[0])
        if len(maos_detectadas) >= 1:
            left_hand = maos_detectadas[0][1]
        if len(maos_detectadas) >= 2:
            right_hand = maos_detectadas[1][1]
        visual["hands"] = [mao[2] for mao in maos_detectadas]

    visual_frame_count += 1
    if visual_frame_count % PROCESSAR_POSE_ROSTO_A_CADA == 0:
        try:
            with pose_face_lock:
                pose_result = pose_detector.process(rgb)
                face_result = face_detector.process(rgb)

            if pose_result.pose_landmarks:
                pose = []
                for lm in pose_result.pose_landmarks.landmark[:33]:
                    pose.extend([lm.x, lm.y, lm.z, lm.visibility])
                ultimo_pose_features = pose.copy()
                pose_indices = [0, 11, 12, 13, 14, 15, 16]
                ultimo_visual_pose = [
                    [pose_result.pose_landmarks.landmark[i].x,
                     pose_result.pose_landmarks.landmark[i].y]
                    for i in pose_indices
                ]
            else:
                pose = [0.0] * (33 * 4)
                ultimo_pose_features = pose.copy()
                ultimo_visual_pose = []

            if face_result.multi_face_landmarks:
                face_landmarks = face_result.multi_face_landmarks[0]
                face = []
                for idx in FACE_INDICES:
                    lm = face_landmarks.landmark[idx]
                    face.extend([lm.x, lm.y, lm.z])
                ultimo_face_features = face.copy()
                ultimo_visual_face = [[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y] for i in FACE_INDICES]
            else:
                face = [0.0] * (len(FACE_INDICES) * 3)
                ultimo_face_features = face.copy()
                ultimo_visual_face = []

            visual["pose"] = ultimo_visual_pose
            visual["face"] = ultimo_visual_face
        except Exception as e:
            print("[ERRO VISUAL POSE/ROSTO]", e)

    features = normalizar_features(np.array(pose + left_hand + right_hand + face, dtype=np.float32))
    debug = {"hands": len(visual["hands"]), "face": bool(visual["face"]), "pose": bool(visual["pose"])}
    return features, visual, debug


def calcular_movimento(features_atual, features_anterior):
    if features_anterior is None:
        return 999.0
    maos_atual = features_atual[132:258]
    maos_anterior = features_anterior[132:258]
    pose_atual = features_atual[0:132]
    pose_anterior = features_anterior[0:132]
    return float(np.mean(np.abs(maos_atual - maos_anterior)) * 0.9 +
             np.mean(np.abs(pose_atual - pose_anterior)) * 0.1)


def calcular_movimento_sequencia(buffer):
    if len(buffer) < 2:
        return 0.0
    frames = list(buffer)
    movimentos = [
        calcular_movimento(frames[i], frames[i - 1])
        for i in range(1, len(frames))
    ]
    return float(np.percentile(movimentos, 75))


def votar_predicao(historico_pred):
    """
    Votação ponderada: predicões mais recentes têm mais peso.
    Retorna (label, confiança_media, votos) ou None.
    """
    if len(historico_pred) < JANELA_PREDICOES:
        return None

    # Peso crescente para predicções mais recentes
    pesos = np.linspace(0.5, 1.0, len(historico_pred))
    probs_hist = np.array([p[0] for p in historico_pred], dtype=np.float32)
    probs_media = np.average(probs_hist, axis=0, weights=pesos)

    idx_vencedor = int(np.argmax(probs_media))
    ordem = np.argsort(probs_media)[::-1]
    conf_media = float(probs_media[idx_vencedor])
    idx_segundo = int(ordem[1]) if len(ordem) > 1 else idx_vencedor
    segunda_conf = float(probs_media[idx_segundo]) if len(ordem) > 1 else 0.0
    margem_media = conf_media - segunda_conf

    top_indices = [int(np.argmax(p[0])) for p in historico_pred]
    votos_vencedor = sum(1 for idx in top_indices if idx == idx_vencedor)

    return labels[idx_vencedor], conf_media, votos_vencedor, margem_media, labels[idx_segundo], segunda_conf


def gerar_colunas_feedback():
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
    colunas.append("source_video")
    colunas.append("label")
    return colunas


def salvar_feedback(label):
    if ultima_seq_feedback is None:
        return False, "Nenhuma sequencia recente para salvar."

    seq = np.array(ultima_seq_feedback, dtype=np.float32).reshape(MAX_FRAMES, N_FEATURES)
    linha = seq.reshape(-1).astype(float).tolist()
    linha.append(f"feedback_{int(time.time() * 1000)}")
    linha.append(label)

    os.makedirs(os.path.dirname(FEEDBACK_CSV), exist_ok=True)
    escrever_cabecalho = not os.path.exists(FEEDBACK_CSV) or os.path.getsize(FEEDBACK_CSV) == 0
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if escrever_cabecalho:
            writer.writerow(gerar_colunas_feedback())
        writer.writerow(linha)

    return True, "Exemplo salvo."


# =========================
# ROTAS
# =========================

def resetar_estado_inferencia():
    global ultimo_features, debounce_counter, frames_sem_maos
    global ultima_seq_feedback, ultimo_label_feedback, ultima_conf_feedback
    frame_buffer.clear()
    historico_predicoes.clear()
    ultimo_features = None
    debounce_counter = 0
    frames_sem_maos = 0
    ultima_seq_feedback = None
    ultimo_label_feedback = None
    ultima_conf_feedback = 0.0


@app.route("/")
def index():
    return render_template("index.html", labels=labels)


@app.route("/reset", methods=["POST"])
def reset():
    resetar_estado_inferencia()
    return jsonify({"ok": True})


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json() or {}
    label = str(data.get("label", "")).strip()

    if not label or label == "-":
        return jsonify({"ok": False, "error": "Escolha um sinal valido."}), 400
    if label not in labels:
        return jsonify({"ok": False, "error": "Sinal fora da lista do modelo."}), 400

    ok, mensagem = salvar_feedback(label)
    if not ok:
        return jsonify({"ok": False, "error": mensagem}), 400

    return jsonify({
        "ok": True,
        "label": label,
        "predito": ultimo_label_feedback,
        "confidence": ultima_conf_feedback,
        "message": mensagem
    })


@app.route("/predict", methods=["POST"])
def predict():
    global ultimo_features, debounce_counter, frames_sem_maos
    global ultima_seq_feedback, ultimo_label_feedback, ultima_conf_feedback

    data = request.get_json(silent=True) or {}
    b64 = data.get("frame", "")

    try:
        img_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"label": None, "confidence": 0.0, "visual": {}, "debug": {}, "waiting": "Erro ao ler frame."})

    if frame is None:
        return jsonify({"label": None, "confidence": 0.0, "visual": {}, "debug": {}, "waiting": "Frame inválido."})

    features, visual, debug = extrair_features_e_visual(frame)

    # Sem mãos → reseta tudo
    if debug["hands"] <= 0:
        frames_sem_maos += 1
        debounce_counter = max(0, debounce_counter - 1)
        if frames_sem_maos <= FRAMES_SEM_MAOS_TOLERANCIA and len(frame_buffer) > 0:
            return jsonify({"label": None, "confidence": 0.0, "visual": visual, "debug": debug,
                            "waiting": "Mantendo leitura..."})

        frame_buffer.clear()
        historico_predicoes.clear()
        ultimo_features = None
        return jsonify({"label": None, "confidence": 0.0, "visual": visual, "debug": debug,
                        "waiting": "Mostre as mãos para a câmera."})

    frames_sem_maos = 0

    movimento = calcular_movimento(features, ultimo_features)
    ultimo_features = features.copy()

    # Se houver muito movimento (troca de sinal), reseta o buffer
    if movimento > 0.03 and len(frame_buffer) > 5:
        frame_buffer.clear()
        historico_predicoes.clear()

    frame_buffer.append(features)
    movimento_seq = calcular_movimento_sequencia(frame_buffer)

    frames_validos = sum(
        1 for f in frame_buffer
        if np.sum(np.abs(f[132:258])) > 0
    )

    if frames_validos < MIN_FRAMES_VALIDOS_PREDICAO:
        return jsonify({
            "label": None,
            "confidence": 0.0,
            "visual": visual,
            "debug": debug,
            "waiting": "Ajuste as mãos na câmera..."
        })

    # Aguarda buffer cheio
    if len(frame_buffer) < MAX_FRAMES:
        return jsonify({"label": None, "confidence": 0.0, "visual": visual, "debug": debug,
                        "waiting": f"Lendo sinal... {len(frame_buffer)}/{MAX_FRAMES}"})

    # Faz predição
    X = np.array(list(frame_buffer), dtype=np.float32).reshape(1, MAX_FRAMES, N_FEATURES)
    probs = model.predict(X, verbose=0)[0]

    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label_previsto = labels[idx]
    ultima_seq_feedback = X.reshape(MAX_FRAMES, N_FEATURES).copy()
    ultimo_label_feedback = label_previsto
    ultima_conf_feedback = conf
    ordem_probs = np.argsort(probs)[::-1]
    segunda_conf = float(probs[ordem_probs[1]]) if len(ordem_probs) > 1 else 0.0
    margem = conf - segunda_conf

    historico_predicoes.append((probs, conf, margem))

    if DEBUG_PREDICOES:
        print(
            f"Pred: {label_previsto} | Conf: {conf:.3f} | Margem: {margem:.3f} | "
            f"Mov: {movimento:.4f} | Seq: {movimento_seq:.4f}"
        )

    # Debounce: desconta contador a cada frame
    if debounce_counter > 0:
        debounce_counter -= 1

    # Votação ponderada
    resultado_voto = votar_predicao(list(historico_predicoes))
    label_final = None
    conf_voto = conf  # fallback seguro
    
    if resultado_voto:
        label_voto, conf_voto, votos, margem_voto, _, _ = resultado_voto
        threshold = get_threshold(label_voto)

        margem_ajustada = get_margem_minima(label_voto)
        votos_necessarios = get_votos_necessarios(label_voto)

        aceitar = (
            votos >= votos_necessarios and
            conf_voto >= threshold and
            margem_voto >= margem_ajustada and
            movimento_seq > MOVIMENTO_MINIMO * 2 and
            debounce_counter == 0
        )

        if aceitar:
            label_final = None if eh_label_rejeicao(label_voto) else label_voto
            debounce_counter = DEBOUNCE_FRAMES
            # Reseta buffer parcialmente para permitir próximo sinal
            for _ in range(MAX_FRAMES // 2):
                if frame_buffer:
                    frame_buffer.popleft()
            historico_predicoes.clear()

    waiting_msg = (
        f"Analisando: {label_previsto} ({conf * 100:.0f}%) | "
        f"margem {margem * 100:.0f}% | mov {movimento_seq:.4f}"
    )
    if resultado_voto and not label_final:
        if eh_label_rejeicao(label_voto):
            waiting_msg = "Gesto fora do vocabulário."
        else:
            waiting_msg += f" | voto {label_voto} {conf_voto * 100:.0f}%"
    if debounce_counter > 0 and not label_final:
        waiting_msg = f"Aguardando próximo sinal... ({debounce_counter})"

    return jsonify({
        "label": label_final,
        "confidence": conf_voto if resultado_voto else conf,
        "live_label": label_previsto,
        "live_conf": conf,
        "visual": visual,
        "debug": debug,
        "waiting": waiting_msg
    })


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Servidor iniciado!")
    print("  Acesse: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=False, port=5000, threaded=False)
