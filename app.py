"""
=============================================================
APP — Tradutor de Libras com IA
MediaPipe Hands + Flask + LSTM
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
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# =========================
# CONFIGURAÇÕES
# =========================
MAX_FRAMES = 20
N_FEATURES = 288
LARGURA_PROCESSAMENTO = 480

FACE_INDICES = [1, 33, 61, 199, 263, 291, 13, 14, 10, 152]

frame_buffer = []
ultimas_predicoes = []
ultimo_features = None

REPETICOES_ESTAVEIS = 3
CONFIANCA_MINIMA = 0.75
MARGEM_MINIMA = 0.12
MOVIMENTO_MINIMO = 0.0004

# =========================
# CARREGAR MODELO
# =========================
print("Carregando modelo...")

model = tf.keras.models.load_model("modelo_libras.keras")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

labels = le.classes_.tolist()
print(f"Modelo carregado! Sinais: {labels}")

# =========================
# MEDIAPIPE HANDS
# =========================
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45
)

pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45
)

face_detector = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45
)

hands_lock = threading.Lock()
pose_face_lock = threading.Lock()
visual_frame_count = 0
ultimo_visual_pose = []
ultimo_visual_face = []
PROCESSAR_POSE_ROSTO_A_CADA = 2
ultimo_pose_features = [0.0] * (33 * 4)
ultimo_face_features = [0.0] * (len(FACE_INDICES) * 3)

# =========================
# HTML
# =========================
HTML = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Tradutor de Libras</title>

<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --bg:#0a0a0f;
  --surface:#12121a;
  --border:#1e1e2e;
  --accent:#00e5ff;
  --accent2:#7c3aed;
  --green:#00ff88;
  --text:#e8e8f0;
  --muted:#77779a;
  --danger:#ff4466;
}

* {
  margin:0;
  padding:0;
  box-sizing:border-box;
}

body {
  background:var(--bg);
  color:var(--text);
  font-family:'DM Sans',sans-serif;
  min-height:100vh;
}

body::before {
  content:'';
  position:fixed;
  inset:0;
  pointer-events:none;
  z-index:0;
  background-image:
    linear-gradient(rgba(0,229,255,.03) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,229,255,.03) 1px,transparent 1px);
  background-size:40px 40px;
}

.wrap {
  position:relative;
  z-index:1;
  max-width:1100px;
  margin:0 auto;
  padding:2rem 1.5rem;
}

header {
  display:flex;
  align-items:center;
  gap:1rem;
  margin-bottom:2rem;
}

.logo {
  width:44px;
  height:44px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  border-radius:12px;
  display:grid;
  place-items:center;
  font-size:1.3rem;
}

h1 {
  font-family:'Syne',sans-serif;
  font-size:1.6rem;
  font-weight:800;
}

h1 span {
  color:var(--accent);
}

.badge {
  margin-left:auto;
  background:rgba(0,229,255,.1);
  border:1px solid rgba(0,229,255,.25);
  color:var(--accent);
  font-size:.7rem;
  font-weight:500;
  padding:.3rem .7rem;
  border-radius:999px;
  text-transform:uppercase;
  letter-spacing:.05em;
}

.grid {
  display:grid;
  grid-template-columns:1fr 380px;
  gap:1.5rem;
}

.card {
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:16px;
  padding:1.25rem;
}

.card-title {
  font-family:'Syne',sans-serif;
  font-size:.75rem;
  font-weight:700;
  letter-spacing:.12em;
  text-transform:uppercase;
  color:var(--muted);
  margin-bottom:1rem;
  display:flex;
  align-items:center;
  gap:.5rem;
}

.dot {
  width:6px;
  height:6px;
  border-radius:50%;
  background:var(--accent);
}

.camera-box {
  position:relative;
  aspect-ratio:4/3;
  background:#070710;
  border-radius:12px;
  overflow:hidden;
}

#video {
  width:100%;
  height:100%;
  object-fit:cover;
  transform:scaleX(-1);
  border-radius:12px;
}

#overlay {
  position:absolute;
  inset:0;
  width:100%;
  height:100%;
  transform:scaleX(-1);
  pointer-events:none;
}

.cam-status {
  position:absolute;
  top:.75rem;
  left:.75rem;
  background:rgba(0,0,0,.7);
  backdrop-filter:blur(6px);
  border:1px solid var(--border);
  border-radius:999px;
  padding:.3rem .8rem;
  font-size:.72rem;
  font-weight:500;
  display:flex;
  align-items:center;
  gap:.4rem;
}

.pulse {
  width:7px;
  height:7px;
  border-radius:50%;
  background:var(--muted);
}

.active .pulse {
  background:var(--green);
  animation:blink 1s infinite;
}

@keyframes blink {
  0%,100% { opacity:1 }
  50% { opacity:.3 }
}

#btnCamera {
  margin-top:1rem;
  width:100%;
  padding:.9rem;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#fff;
  border:none;
  border-radius:10px;
  font-family:'Syne',sans-serif;
  font-size:.95rem;
  font-weight:700;
  cursor:pointer;
  letter-spacing:.04em;
}

.right {
  display:flex;
  flex-direction:column;
  gap:1.25rem;
}

.result-card {
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:16px;
  padding:1.5rem;
  text-align:center;
}

.result-label {
  font-size:.7rem;
  font-weight:700;
  letter-spacing:.12em;
  text-transform:uppercase;
  color:var(--muted);
  margin-bottom:.75rem;
}

#resultado {
  font-family:'Syne',sans-serif;
  font-size:2.5rem;
  font-weight:800;
  color:var(--accent);
  min-height:3.2rem;
}

#confianca,
#statusDetect {
  font-size:.82rem;
  color:var(--muted);
  margin-top:.4rem;
}

.conf-bar-wrap {
  margin-top:.75rem;
  height:4px;
  background:var(--border);
  border-radius:999px;
  overflow:hidden;
}

#confBar {
  height:100%;
  width:0%;
  background:linear-gradient(90deg,var(--accent2),var(--accent));
  transition:width .4s ease;
}

.frase-box {
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:16px;
  padding:1.25rem;
}

#frase {
  font-size:1rem;
  min-height:3rem;
  line-height:1.6;
  word-break:break-word;
}

.word {
  display:inline-block;
  background:rgba(0,229,255,.08);
  border:1px solid rgba(0,229,255,.2);
  border-radius:6px;
  padding:.1rem .4rem;
  margin:.1rem .15rem;
  font-size:.9rem;
}

.actions {
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:.6rem;
}

.btn-action {
  padding:.7rem;
  background:transparent;
  border:1px solid var(--border);
  border-radius:10px;
  color:var(--text);
  font-family:'DM Sans',sans-serif;
  font-size:.85rem;
  cursor:pointer;
}

.btn-action:hover {
  border-color:var(--accent);
  color:var(--accent);
}

.btn-action.danger:hover {
  border-color:var(--danger);
  color:var(--danger);
}

.info-box {
  font-size:.78rem;
  color:var(--muted);
  line-height:1.6;
  padding:.8rem;
  background:rgba(124,58,237,.06);
  border:1px solid rgba(124,58,237,.15);
  border-radius:10px;
}
</style>
</head>

<body>
<div class="wrap">

<header>
  <div class="logo">🤟</div>
  <h1>Tradutor de <span>Libras</span></h1>
  <div class="badge">Hands + LSTM</div>
</header>

<div class="grid">

  <div>
    <div class="card">
      <div class="card-title"><span class="dot"></span>Câmera</div>

      <div class="camera-box">
        <video id="video" autoplay muted playsinline></video>
        <canvas id="overlay"></canvas>

        <div class="cam-status" id="camStatus">
          <div class="pulse"></div>
          <span id="camTxt">Câmera desligada</span>
        </div>
      </div>

      <button id="btnCamera">▶ Iniciar Câmera</button>
    </div>

    <div class="card" style="margin-top:1.25rem;">
      <div class="card-title"><span class="dot"></span>Como usar</div>

      <div class="info-box">
        1. Clique em <b>Iniciar Câmera</b><br>
        2. Faça o sinal com boa iluminação<br>
        3. Os pontos de <b>mãos, rosto e pose</b> aparecem na tela<br>
        4. O sistema só reconhece quando o sinal fica estável<br><br>
        <b>Sinais disponíveis:</b> {{ labels|join(', ') }}
      </div>
    </div>
  </div>

  <div class="right">
    <div class="result-card">
      <div class="result-label">Sinal Detectado</div>
      <div id="resultado" style="color:var(--border)">—</div>
      <div id="confianca">Aguardando sinal...</div>
      <div id="statusDetect"></div>
      <div class="conf-bar-wrap"><div id="confBar"></div></div>
    </div>

    <div class="frase-box">
      <div class="card-title"><span class="dot"></span>Frase</div>
      <div id="frase" style="color:var(--muted);font-style:italic;font-size:.85rem;">
        Os sinais reconhecidos aparecerão aqui...
      </div>
    </div>

    <div class="actions">
      <button class="btn-action" onclick="falar()">🔊 Falar Frase</button>
      <button class="btn-action" onclick="adicionar()">➕ Adicionar</button>
      <button class="btn-action danger" onclick="limpar()">🗑 Limpar</button>
      <button class="btn-action" onclick="copiar()">📋 Copiar</button>
    </div>
  </div>

</div>
</div>

<script>
let cameraOn = false;
let stream = null;
let interval = null;
let processandoFrame = false;

let palavraAtual = '';
let fraseWords = [];

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');

const btnCam = document.getElementById('btnCamera');
const elRes = document.getElementById('resultado');
const elConf = document.getElementById('confianca');
const elBar = document.getElementById('confBar');
const elFrase = document.getElementById('frase');
const camStatus = document.getElementById('camStatus');
const camTxt = document.getElementById('camTxt');
const statusDetect = document.getElementById('statusDetect');

btnCam.addEventListener('click', async () => {
  if (!cameraOn) {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        }
      });

      video.srcObject = stream;

      video.onloadedmetadata = () => {
        overlay.width = video.videoWidth || 1280;
        overlay.height = video.videoHeight || 720;
      };

      cameraOn = true;
      btnCam.textContent = '⏹ Parar Câmera';
      camStatus.classList.add('active');
      camTxt.textContent = 'Câmera ativa';

      interval = setInterval(enviarFrame, 70);

    } catch (e) {
      resetCameraUi();

      const msg = e.name === 'NotReadableError'
        ? 'A câmera está em uso por outro app/aba. Feche Zoom, Teams, câmera do Windows ou outra aba usando a webcam e tente novamente.'
        : 'Erro ao acessar a câmera: ' + e.message;

      alert(msg);
      console.error(e);
    }

  } else {
    resetCameraUi();
  }
});

async function enviarFrame() {
  if (!cameraOn || !video.videoWidth || processandoFrame) return;

  processandoFrame = true;

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;

  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

  const base64 = canvas.toDataURL('image/jpeg', 0.55).split(',')[1];

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: base64 })
    });

    const data = await resp.json();

    desenharTudo(data.visual || {});

    statusDetect.textContent =
      `Mãos: ${data.debug?.hands || 0} | Rosto: ${data.debug?.face ? 'sim' : 'não'} | Pose: ${data.debug?.pose ? 'sim' : 'não'}`;

    if (data.label) {
      palavraAtual = data.label;
      elRes.textContent = data.label.toUpperCase();
      elRes.style.color = 'var(--accent)';

      elConf.textContent = `Confiança: ${(data.confidence * 100).toFixed(1)}%`;
      elBar.style.width = `${(data.confidence * 100).toFixed(1)}%`;

    } else {
      palavraAtual = '';
      elRes.textContent = '—';
      elRes.style.color = 'var(--border)';

      if (data.waiting) {
        elConf.textContent = data.waiting;
      } else {
        elConf.textContent = 'Aguardando um sinal estável...';
      }

      elBar.style.width = '0%';
    }

  } catch (e) {
    console.error(e);

  } finally {
    processandoFrame = false;
  }
}

function resetCameraUi() {
  clearInterval(interval);
  interval = null;

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
  }

  stream = null;
  video.srcObject = null;
  cameraOn = false;
  processandoFrame = false;

  ctx.clearRect(0, 0, overlay.width, overlay.height);

  btnCam.textContent = '▶ Iniciar Câmera';
  camStatus.classList.remove('active');
  camTxt.textContent = 'Câmera desligada';

  elRes.textContent = '—';
  elRes.style.color = 'var(--border)';
  elConf.textContent = 'Aguardando sinal...';
  elBar.style.width = '0%';
}

function desenharTudo(visual) {
  overlay.width = video.videoWidth || 1280;
  overlay.height = video.videoHeight || 720;

  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (visual.pose) desenharPontos(visual.pose, '#7c3aed', 3);
  if (visual.face) desenharPontos(visual.face, '#ffdd55', 3);

  if (visual.hands) {
    visual.hands.forEach(hand => desenharMao(hand));
  }
}

function desenharPontos(points, color, radius) {
  ctx.fillStyle = color;

  points.forEach(pt => {
    const x = pt[0] * overlay.width;
    const y = pt[1] * overlay.height;

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fill();
  });
}

function desenharMao(landmarks) {
  if (!landmarks || landmarks.length === 0) return;

  const conexoes = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17]
  ];

  ctx.lineWidth = 3;
  ctx.strokeStyle = '#00e5ff';
  ctx.fillStyle = '#00ff88';

  conexoes.forEach(([a, b]) => {
    if (!landmarks[a] || !landmarks[b]) return;

    const x1 = landmarks[a][0] * overlay.width;
    const y1 = landmarks[a][1] * overlay.height;
    const x2 = landmarks[b][0] * overlay.width;
    const y2 = landmarks[b][1] * overlay.height;

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  });

  landmarks.forEach(pt => {
    const x = pt[0] * overlay.width;
    const y = pt[1] * overlay.height;

    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fill();
  });
}

function adicionar() {
  if (!palavraAtual) return;

  fraseWords.push(palavraAtual.toLowerCase());
  renderFrase();
}

function falar() {
  const texto = fraseWords.join(' ');
  if (!texto) return;

  const u = new SpeechSynthesisUtterance(texto);
  u.lang = 'pt-BR';
  u.rate = 0.9;

  const voz = speechSynthesis.getVoices().find(v => v.lang.startsWith('pt'));
  if (voz) u.voice = voz;

  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

function limpar() {
  fraseWords = [];
  palavraAtual = '';

  renderFrase();

  elRes.textContent = '—';
  elRes.style.color = 'var(--border)';
  elConf.textContent = 'Aguardando sinal...';
  elBar.style.width = '0%';
}

function copiar() {
  navigator.clipboard.writeText(fraseWords.join(' '));
}

function renderFrase() {
  if (!fraseWords.length) {
    elFrase.innerHTML =
      '<span style="color:var(--muted);font-style:italic;font-size:.85rem;">Os sinais reconhecidos aparecerão aqui...</span>';
    return;
  }

  elFrase.innerHTML = fraseWords
    .map(w => `<span class="word">${w}</span>`)
    .join(' ');
}

speechSynthesis.onvoiceschanged = () => speechSynthesis.getVoices();
</script>

</body>
</html>
"""

# =========================
# FUNÇÕES DE LANDMARKS
# =========================
def extrair_features_e_visual(frame):
    global visual_frame_count, ultimo_visual_pose, ultimo_visual_face
    global ultimo_pose_features, ultimo_face_features

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

        visual = {"hands": [], "face": [], "pose": []}
        debug = {"hands": 0, "face": False, "pose": False}

        return np.zeros(N_FEATURES, dtype=np.float32), visual, debug

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

                ultimo_visual_face = [
                    [face_landmarks.landmark[i].x,
                     face_landmarks.landmark[i].y]
                    for i in FACE_INDICES
                ]
            else:
                face = [0.0] * (len(FACE_INDICES) * 3)
                ultimo_face_features = face.copy()
                ultimo_visual_face = []

            visual["pose"] = ultimo_visual_pose
            visual["face"] = ultimo_visual_face
        except Exception as e:
            print("[ERRO VISUAL POSE/ROSTO]", e)

    features = normalizar_features(np.array(pose + left_hand + right_hand + face, dtype=np.float32))

    debug = {
        "hands": len(visual["hands"]),
        "face": bool(visual["face"]),
        "pose": bool(visual["pose"])
    }

    return features, visual, debug


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


def calcular_movimento_maos(features_atual, features_anterior):
    if features_anterior is None:
        return 999.0

    # POSE fica de 0 até 132
    # Vamos pegar principalmente braços:
    # ombros, cotovelos e punhos ficam dentro da pose
    pose_atual = features_atual[0:132]
    pose_anterior = features_anterior[0:132]

    # Mãos ficam de 132 até 258
    maos_atual = features_atual[132:258]
    maos_anterior = features_anterior[132:258]

    movimento_pose = float(np.mean(np.abs(pose_atual - pose_anterior)))
    movimento_maos = float(np.mean(np.abs(maos_atual - maos_anterior)))

    # Dá mais peso para mãos, mas agora considera braço também
    movimento_total = (movimento_maos * 0.7) + (movimento_pose * 0.3)

    return movimento_total
# =========================
# ROTAS
# =========================
@app.route("/")
def index():
    return render_template_string(HTML, labels=labels)


@app.route("/predict", methods=["POST"])
def predict():
    global frame_buffer, ultimas_predicoes, ultimo_features

    data = request.get_json()
    b64 = data.get("frame", "")

    try:
        img_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({
            "label": None,
            "confidence": 0.0,
            "visual": {},
            "debug": {"hands": 0, "face": False, "pose": False},
            "waiting": "Erro ao ler frame."
        })

    if frame is None:
        return jsonify({
            "label": None,
            "confidence": 0.0,
            "visual": {},
            "debug": {"hands": 0, "face": False, "pose": False},
            "waiting": "Frame inválido."
        })

    features, visual, debug = extrair_features_e_visual(frame)

    if debug["hands"] <= 0:
        frame_buffer.clear()
        ultimas_predicoes.clear()
        ultimo_features = None

        return jsonify({
            "label": None,
            "confidence": 0.0,
            "visual": visual,
            "debug": debug,
            "waiting": "Mostre as mãos para a câmera."
        })

    movimento = calcular_movimento_maos(features, ultimo_features)
    ultimo_features = features.copy()

    frame_buffer.append(features)

    if len(frame_buffer) > MAX_FRAMES:
        frame_buffer.pop(0)

    if len(frame_buffer) < MAX_FRAMES:
        return jsonify({
            "label": None,
            "confidence": 0.0,
            "visual": visual,
            "debug": debug,
            "waiting": f"Lendo mãos... {len(frame_buffer)}/{MAX_FRAMES}"
        })

    X = np.array(frame_buffer, dtype=np.float32).reshape(
        1,
        MAX_FRAMES,
        N_FEATURES
    )

    probs = model.predict(X, verbose=0)[0]

    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label_previsto = labels[idx]
    ordem_probs = np.argsort(probs)[::-1]
    segunda_conf = float(probs[ordem_probs[1]]) if len(ordem_probs) > 1 else 0.0
    margem = conf - segunda_conf

    print(
        "Predição:",
        label_previsto,
        "| Confiança:",
        round(conf, 4),
        "| Margem:",
        round(margem, 4),
        "| Movimento:",
        round(movimento, 4),
        "| Debug:",
        debug
    )

    ultimas_predicoes.append((label_previsto, conf, margem))

    if len(ultimas_predicoes) > REPETICOES_ESTAVEIS:
        ultimas_predicoes.pop(0)

    label_final = None

    if len(ultimas_predicoes) == REPETICOES_ESTAVEIS:
        labels_recentes = [p[0] for p in ultimas_predicoes]
        confiancas_recentes = [p[1] for p in ultimas_predicoes]
        margens_recentes = [p[2] for p in ultimas_predicoes]

        mesmo_label = labels_recentes.count(labels_recentes[0]) == len(labels_recentes)
        confianca_media = float(np.mean(confiancas_recentes))
        margem_media = float(np.mean(margens_recentes))

        if (
            mesmo_label
            and confianca_media >= CONFIANCA_MINIMA
            and margem_media >= MARGEM_MINIMA
        ):
            label_final = labels_recentes[0]

    return jsonify({
        "label": label_final,
        "confidence": conf if label_final else 0.0,
        "visual": visual,
        "debug": debug,
        "waiting": (
            f"Aguardando sinal confiavel... "
            f"topo {label_previsto}: {conf * 100:.1f}% | margem {margem * 100:.1f}%"
        )
    })


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Servidor iniciado!")
    print("  Acesse: http://localhost:5000")
    print("=" * 50 + "\n")

    app.run(debug=False, port=5000, threaded=False)
