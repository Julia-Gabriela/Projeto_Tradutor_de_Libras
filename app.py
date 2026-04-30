"""
=============================================================
APP — Tradutor de Libras com IA
MediaPipe Holistic + Flask + LSTM
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
MAX_FRAMES = 30
N_FEATURES = 288

FACE_INDICES = [1, 33, 61, 199, 263, 291, 13, 14, 10, 152]

frame_buffer = []
ultimas_predicoes = []
ultimo_features = None

REPETICOES_ESTAVEIS = 3
CONFIANCA_MINIMA = 0.55
MOVIMENTO_MINIMO = 0.002

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
# MEDIAPIPE HOLISTIC
# =========================
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

holistic_lock = threading.Lock()

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
  <div class="badge">Holistic + LSTM</div>
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
          width: { ideal: 1280 },
          height: { ideal: 720 },
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

      interval = setInterval(enviarFrame, 200);

    } catch (e) {
      alert('Erro ao acessar a câmera: ' + e.message);
      console.error(e);
    }

  } else {
    clearInterval(interval);

    if (stream) {
      stream.getTracks().forEach(t => t.stop());
    }

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
});

async function enviarFrame() {
  if (!cameraOn || !video.videoWidth || processandoFrame) return;

  processandoFrame = true;

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;

  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

  const base64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];

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
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    try:
        with holistic_lock:
            result = holistic.process(rgb)
    except Exception as e:
        print("[ERRO MEDIAPIPE]", e)

        visual = {"hands": [], "face": [], "pose": []}
        debug = {"hands": 0, "face": False, "pose": False}

        return np.zeros(N_FEATURES, dtype=np.float32), visual, debug

    pose = []

    if result.pose_landmarks:
        for lm in result.pose_landmarks.landmark[:33]:
            pose.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        pose = [0.0] * (33 * 4)

    left_hand = []

    if result.left_hand_landmarks:
        for lm in result.left_hand_landmarks.landmark:
            left_hand.extend([lm.x, lm.y, lm.z])
    else:
        left_hand = [0.0] * (21 * 3)

    right_hand = []

    if result.right_hand_landmarks:
        for lm in result.right_hand_landmarks.landmark:
            right_hand.extend([lm.x, lm.y, lm.z])
    else:
        right_hand = [0.0] * (21 * 3)

    face = []

    if result.face_landmarks:
        for idx in FACE_INDICES:
            lm = result.face_landmarks.landmark[idx]
            face.extend([lm.x, lm.y, lm.z])
    else:
        face = [0.0] * (len(FACE_INDICES) * 3)

    features = np.array(pose + left_hand + right_hand + face, dtype=np.float32)

    visual = {"hands": [], "face": [], "pose": []}

    if result.left_hand_landmarks:
        visual["hands"].append([
            [lm.x, lm.y] for lm in result.left_hand_landmarks.landmark
        ])

    if result.right_hand_landmarks:
        visual["hands"].append([
            [lm.x, lm.y] for lm in result.right_hand_landmarks.landmark
        ])

    if result.face_landmarks:
        visual["face"] = [
            [result.face_landmarks.landmark[i].x,
             result.face_landmarks.landmark[i].y]
            for i in FACE_INDICES
        ]

    if result.pose_landmarks:
        pose_indices = [0, 11, 12, 13, 14, 15, 16]
        visual["pose"] = [
            [result.pose_landmarks.landmark[i].x,
             result.pose_landmarks.landmark[i].y]
            for i in pose_indices
        ]

    debug = {
        "hands": len(visual["hands"]),
        "face": bool(result.face_landmarks),
        "pose": bool(result.pose_landmarks)
    }

    return features, visual, debug


def calcular_movimento_maos(features_atual, features_anterior):
    if features_anterior is None:
        return 999.0

    maos_atual = features_atual[132:258]
    maos_anterior = features_anterior[132:258]

    if np.sum(np.abs(maos_atual)) == 0:
        return 0.0

    return float(np.mean(np.abs(maos_atual - maos_anterior)))


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

    if movimento < MOVIMENTO_MINIMO:
        frame_buffer.clear()
        ultimas_predicoes.clear()

        return jsonify({
            "label": None,
            "confidence": 0.0,
            "visual": visual,
            "debug": debug,
            "waiting": "Aguardando movimento do sinal..."
        })

    frame_buffer.append(features)

    if len(frame_buffer) > MAX_FRAMES:
        frame_buffer.pop(0)

    if len(frame_buffer) < MAX_FRAMES:
        return jsonify({
            "label": None,
            "confidence": 0.0,
            "visual": visual,
            "debug": debug,
            "waiting": f"Coletando movimento... {len(frame_buffer)}/{MAX_FRAMES}"
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

    print(
        "Predição:",
        label_previsto,
        "| Confiança:",
        round(conf, 4),
        "| Movimento:",
        round(movimento, 4),
        "| Debug:",
        debug
    )

    ultimas_predicoes.append((label_previsto, conf))

    if len(ultimas_predicoes) > REPETICOES_ESTAVEIS:
        ultimas_predicoes.pop(0)

    label_final = None

    if len(ultimas_predicoes) == REPETICOES_ESTAVEIS:
        labels_recentes = [p[0] for p in ultimas_predicoes]
        confiancas_recentes = [p[1] for p in ultimas_predicoes]

        mesmo_label = labels_recentes.count(labels_recentes[0]) == len(labels_recentes)
        confianca_media = float(np.mean(confiancas_recentes))

        if mesmo_label and confianca_media >= CONFIANCA_MINIMA:
            label_final = labels_recentes[0]

    return jsonify({
        "label": label_final,
        "confidence": conf,
        "visual": visual,
        "debug": debug,
        "waiting": "Aguardando estabilidade..."
    })


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Servidor iniciado!")
    print("  Acesse: http://localhost:5000")
    print("=" * 50 + "\n")

    app.run(debug=False, port=5000, threaded=False)