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
from flask import Flask, render_template_string, request, jsonify
from collections import deque

app = Flask(__name__)

# =========================
# CONFIGURAÇÕES
# =========================
MAX_FRAMES = 20
N_FEATURES = 288
LARGURA_PROCESSAMENTO = 416
FACE_INDICES = [1, 33, 61, 199, 263, 291, 13, 14, 10, 152]

# --- Parâmetros de detecção ---
CONFIANCA_MINIMA_GLOBAL = 0.75   # Limiar padrão (sobrescrito pelos thresholds por classe)
MARGEM_MINIMA = 0.35           # Diferença mínima entre 1º e 2º lugar
VOTOS_NECESSARIOS = 2            # De 3 predições, quantas precisam concordar
JANELA_PREDICOES = 3             # Tamanho da janela de votação
DEBOUNCE_FRAMES = 10             # Frames para esperar antes de aceitar o mesmo sinal de novo
MOVIMENTO_MINIMO = 0.0006       # Movimento mínimo para considerar que há sinal
FRAMES_SEM_MAOS_TOLERANCIA = 4  # Evita resetar tudo quando o MediaPipe pisca por poucos frames
MIN_FRAMES_VALIDOS_PREDICAO = 8
FEEDBACK_CSV = "feedback_landmarks.csv"

# Ajustes práticos para a demo: sinais que estão gerando falso positivo
# precisam ser mais difíceis de aceitar do que sinais que o modelo quase não libera.
THRESHOLDS_DEMO = {
    "Sim": 0.90,
    "Não": 0.68,
    "Nao": 0.68,
    "Obrigado": 0.78,
    "Oi": 0.67,
    "Casa": 0.90,
}
MARGENS_DEMO = {
    "Sim": 0.58,
    "Não": 0.48,
    "Nao": 0.48,
    "Obrigado": 0.28,
    "Oi": 0.35,
    "Casa": 0.35,
}
VOTOS_DEMO = {
    "Sim": 5,
    "Não": 4,
    "Nao": 4,
    "Obrigado": 4,
    "Oi": 4,
    "Casa": 4,
}
LABELS_REJEICAO = {"Desconhecido", "Desconhecida", "Neutro", "Fundo", "Nada", "Outro"}

# Recalibrado depois do treino com Banheiro/Casa/Desconhecido/Livro/Nome/Obrigado/Oi.
THRESHOLDS_DEMO = {
    "Banheiro": 0.75,
    "Casa": 0.92,
    "Desconhecido": 0.72,
    "Livro": 0.75,
    "Nome": 0.79,
    "Obrigado": 0.55,
    "Oi": 0.81,
}
MARGENS_DEMO = {
    "Banheiro": 0.35,
    "Casa": 0.35,
    "Desconhecido": 0.25,
    "Livro": 0.35,
    "Nome": 0.35,
    "Obrigado": 0.28,
    "Oi": 0.35,
}
VOTOS_DEMO = {
    "Banheiro": 2,
    "Casa": 2,
    "Desconhecido": 2,
    "Livro": 2,
    "Nome": 2,
    "Obrigado": 2,
    "Oi": 2,
}

# Estado global
frame_buffer = deque(maxlen=MAX_FRAMES)
historico_predicoes = deque(maxlen=JANELA_PREDICOES)
ultimo_features = None
debounce_counter = 0
ultimo_label_aceito = None
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
model = tf.keras.models.load_model("modelo_libras.keras")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

labels = le.classes_.tolist()
print(f"Modelo carregado! Sinais: {labels}")

# Carrega thresholds calibrados (se existir)
thresholds_por_classe = {}
if os.path.exists("thresholds.pkl"):
    with open("thresholds.pkl", "rb") as f:
        thresholds_por_classe = pickle.load(f)
    print(f"Thresholds calibrados carregados: {thresholds_por_classe}")
else:
    print("thresholds.pkl não encontrado. Usando limiar global.")

def chave_label(label):
    return unicodedata.normalize("NFKD", str(label)).encode("ascii", "ignore").decode("ascii")


def eh_label_rejeicao(label):
    return chave_label(label) in LABELS_REJEICAO


def get_threshold(label):
    return THRESHOLDS_DEMO.get(chave_label(label), thresholds_por_classe.get(label, CONFIANCA_MINIMA_GLOBAL))

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
  --warn:#f59e0b;
}

* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:'DM Sans',sans-serif; min-height:100vh; }
body::before {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image: linear-gradient(rgba(0,229,255,.03) 1px,transparent 1px),
                    linear-gradient(90deg,rgba(0,229,255,.03) 1px,transparent 1px);
  background-size:40px 40px;
}
.wrap { position:relative; z-index:1; max-width:1100px; margin:0 auto; padding:2rem 1.5rem; }

header { display:flex; align-items:center; gap:1rem; margin-bottom:2rem; }
.logo {
  width:44px; height:44px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  border-radius:12px; display:grid; place-items:center; font-size:1.3rem;
}
h1 { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; }
h1 span { color:var(--accent); }
.badge {
  margin-left:auto; background:rgba(0,229,255,.1);
  border:1px solid rgba(0,229,255,.25); color:var(--accent);
  font-size:.7rem; font-weight:500; padding:.3rem .7rem;
  border-radius:999px; text-transform:uppercase; letter-spacing:.05em;
}

.grid { display:grid; grid-template-columns:1fr 380px; gap:1.5rem; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:1.25rem; }
.card-title {
  font-family:'Syne',sans-serif; font-size:.75rem; font-weight:700;
  letter-spacing:.12em; text-transform:uppercase; color:var(--muted);
  margin-bottom:1rem; display:flex; align-items:center; gap:.5rem;
}
.dot { width:6px; height:6px; border-radius:50%; background:var(--accent); }

.camera-box { position:relative; aspect-ratio:4/3; background:#070710; border-radius:12px; overflow:hidden; }
#video { width:100%; height:100%; object-fit:cover; transform:scaleX(-1); border-radius:12px; }
#overlay { position:absolute; inset:0; width:100%; height:100%; transform:scaleX(-1); pointer-events:none; }

.cam-status {
  position:absolute; top:.75rem; left:.75rem;
  background:rgba(0,0,0,.7); backdrop-filter:blur(6px);
  border:1px solid var(--border); border-radius:999px;
  padding:.3rem .8rem; font-size:.72rem; font-weight:500;
  display:flex; align-items:center; gap:.4rem;
}
.pulse { width:7px; height:7px; border-radius:50%; background:var(--muted); }
.active .pulse { background:var(--green); animation:blink 1s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* Barra de confiança em tempo real */
.conf-bar-wrap {
  position:absolute; bottom:.75rem; left:.75rem; right:.75rem;
  background:rgba(0,0,0,.6); backdrop-filter:blur(4px);
  border:1px solid var(--border); border-radius:8px;
  padding:.4rem .6rem; display:none;
}
.conf-bar-wrap.visible { display:block; }
.conf-label { font-size:.65rem; color:var(--muted); margin-bottom:.25rem; display:flex; justify-content:space-between; }
.conf-bar { height:4px; background:rgba(255,255,255,.1); border-radius:2px; overflow:hidden; }
.conf-fill { height:100%; border-radius:2px; transition:width .2s, background .2s; }

#btnCamera {
  margin-top:1rem; width:100%; padding:.9rem;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#fff; border:none; border-radius:10px;
  font-family:'Syne',sans-serif; font-size:.95rem; font-weight:700;
  cursor:pointer; letter-spacing:.04em;
}

.right { display:flex; flex-direction:column; gap:1.25rem; }

.result-card {
  background:var(--surface); border:1px solid var(--border);
  border-radius:16px; padding:1.5rem; text-align:center;
}
.sinal-label {
  font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800;
  color:var(--accent); min-height:3.5rem; display:flex;
  align-items:center; justify-content:center;
  transition: all .3s ease;
}
.sinal-label.flash { animation: flash-in .4s ease; }
@keyframes flash-in {
  0%   { transform:scale(1.3); color:#fff; }
  100% { transform:scale(1);   color:var(--accent); }
}
.waiting-text { font-size:.8rem; color:var(--muted); margin-top:.5rem; min-height:1.2rem; }

/* Debug / sensores */
.debug-row { display:flex; gap:.5rem; margin-top:.75rem; flex-wrap:wrap; justify-content:center; }
.dbg-chip {
  font-size:.65rem; padding:.2rem .6rem; border-radius:999px;
  background:rgba(255,255,255,.05); border:1px solid var(--border);
  color:var(--muted); transition:.2s;
}
.dbg-chip.on { background:rgba(0,255,136,.1); border-color:var(--green); color:var(--green); }

/* Histórico */
.historico-card { background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:1.25rem; flex:1; }
.historico-lista { list-style:none; display:flex; flex-direction:column; gap:.5rem; max-height:220px; overflow-y:auto; }
.historico-lista li {
  background:rgba(0,229,255,.05); border:1px solid rgba(0,229,255,.1);
  border-radius:8px; padding:.4rem .75rem;
  display:flex; justify-content:space-between; align-items:center;
  animation: slide-in .3s ease;
}
@keyframes slide-in { from { opacity:0; transform:translateX(-10px); } to { opacity:1; transform:none; } }
.hist-sinal { font-family:'Syne',sans-serif; font-weight:700; color:var(--accent); }
.hist-conf { font-size:.7rem; color:var(--muted); }
.historico-vazio { font-size:.8rem; color:var(--muted); text-align:center; padding:1rem; }

/* Botão limpar histórico */
.btn-limpar {
  margin-top:.75rem; width:100%; padding:.5rem;
  background:transparent; color:var(--muted);
  border:1px solid var(--border); border-radius:8px;
  font-size:.75rem; cursor:pointer; transition:.2s;
}
.btn-limpar:hover { border-color:var(--danger); color:var(--danger); }

/* Sinais disponíveis */
.sinais-grid { display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.5rem; }
.sinal-chip {
  font-size:.72rem; padding:.3rem .65rem; border-radius:999px;
  background:rgba(124,58,237,.1); border:1px solid rgba(124,58,237,.25);
  color:#a78bfa;
}

/* Frase construída */
.frase-wrap { margin-top:1rem; padding:.75rem; background:rgba(0,229,255,.04); border-radius:10px; border:1px solid rgba(0,229,255,.12); min-height:50px; }
.frase-label { font-size:.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-bottom:.3rem; }
.frase-texto { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:var(--text); word-break:break-word; }
.btn-row { display:flex; gap:.5rem; margin-top:.5rem; }
.btn-mini {
  flex:1; padding:.4rem; background:transparent;
  border:1px solid var(--border); border-radius:8px;
  font-size:.72rem; cursor:pointer; color:var(--muted); transition:.2s;
}
.btn-mini:hover { border-color:var(--accent); color:var(--accent); }
.btn-mini.danger:hover { border-color:var(--danger); color:var(--danger); }

@media(max-width:740px) {
  .grid { grid-template-columns:1fr; }
}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo">🤟</div>
    <h1>Tradutor de <span>Libras</span></h1>
    <span class="badge">IA v2</span>
  </header>

  <div class="grid">
    <!-- CÂMERA -->
    <div>
      <div class="card-title"><span class="dot"></span>Câmera ao Vivo</div>
      <div class="camera-box">
        <video id="video" autoplay playsinline muted></video>
        <canvas id="overlay"></canvas>
        <div class="cam-status" id="camStatus">
          <div class="pulse"></div>
          <span id="statusText">Câmera desligada</span>
        </div>
        <!-- Barra de confiança em tempo real -->
        <div class="conf-bar-wrap" id="confBarWrap">
          <div class="conf-label">
            <span id="confLabelSinal">—</span>
            <span id="confLabelPct">0%</span>
          </div>
          <div class="conf-bar"><div class="conf-fill" id="confFill" style="width:0%;background:var(--muted)"></div></div>
        </div>
      </div>
      <button id="btnCamera" onclick="toggleCamera()">▶ Iniciar Câmera</button>
    </div>

    <!-- PAINEL DIREITO -->
    <div class="right">
      <!-- Resultado atual -->
      <div class="result-card">
        <div class="card-title"><span class="dot"></span>Sinal Detectado</div>
        <div class="sinal-label" id="sinaLabel">—</div>
        <div class="waiting-text" id="waitingText">Inicie a câmera para começar</div>
        <div class="btn-row">
          <button class="btn-mini" onclick="enviarFeedbackAcerto()">Acertou</button>
          <select class="btn-mini" id="feedbackLabel">
            {% for l in labels %}
            <option value="{{ l }}">{{ l }}</option>
            {% endfor %}
          </select>
          <button class="btn-mini" onclick="enviarFeedbackCorrecao()">Corrigir</button>
        </div>
        <div class="waiting-text" id="feedbackStatus"></div>
        <div class="debug-row">
          <span class="dbg-chip" id="chipMaos">✋ Mãos: 0</span>
          <span class="dbg-chip" id="chipRosto">😐 Rosto</span>
          <span class="dbg-chip" id="chipPose">🧍 Pose</span>
        </div>
      </div>

      <!-- Frase construída -->
      <div class="result-card">
        <div class="card-title"><span class="dot"></span>Frase</div>
        <div class="frase-wrap">
          <div class="frase-label">Palavras detectadas</div>
          <div class="frase-texto" id="fraseTexto">—</div>
        </div>
        <div class="btn-row">
          <button class="btn-mini" onclick="removerUltima()">← Apagar última</button>
          <button class="btn-mini danger" onclick="limparFrase()">✕ Limpar tudo</button>
        </div>
      </div>

      <!-- Histórico -->
      <div class="historico-card">
        <div class="card-title"><span class="dot"></span>Histórico</div>
        <ul class="historico-lista" id="historicoLista">
          <li class="historico-vazio">Nenhum sinal detectado ainda.</li>
        </ul>
        <button class="btn-limpar" onclick="limparHistorico()">Limpar histórico</button>
      </div>

      <!-- Sinais disponíveis -->
      <div class="card">
        <div class="card-title"><span class="dot"></span>Sinais Disponíveis</div>
        <div class="sinais-grid">
          {% for l in labels %}
          <span class="sinal-chip">{{ l }}</span>
          {% endfor %}
        </div>
      </div>
    </div>
  </div>
</div>

<script>
let cameraAtiva = false;
let stream = null;
let intervalo = null;
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
let historico = [];
let frase = [];
let primeiroHistorico = true;
let vozAutomatica = true;
let framesSemSinal = 0;
let enviandoFrame = false;
let ultimoLabelFeedback = null;
let ultimaConfFeedback = 0;
const FRAMES_PARA_LIMPAR_SINAL = 6;

async function toggleCamera() {
  if (!cameraAtiva) {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } });
      video.srcObject = stream;
      await video.play();
      await resetarBackend();
      cameraAtiva = true;
      document.getElementById('btnCamera').textContent = '⏹ Parar Câmera';
      document.getElementById('camStatus').classList.add('active');
      document.getElementById('statusText').textContent = 'Ao vivo';
      intervalo = setInterval(enviarFrame, 80);
    } catch (e) {
      alert('Não foi possível acessar a câmera: ' + e.message);
    }
  } else {
    pararCamera();
  }
}

function pararCamera() {
  clearInterval(intervalo);
  if (stream) stream.getTracks().forEach(t => t.stop());
  cameraAtiva = false;
  document.getElementById('btnCamera').textContent = '▶ Iniciar Câmera';
  document.getElementById('camStatus').classList.remove('active');
  document.getElementById('statusText').textContent = 'Câmera desligada';
  document.getElementById('confBarWrap').classList.remove('visible');
  limparSinalAtual();
  ultimoLabelFeedback = null;
  ultimaConfFeedback = 0;
  document.getElementById('feedbackStatus').textContent = '';
  resetarBackend();
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function limparSinalAtual() {
  const el = document.getElementById('sinaLabel');
  el.textContent = '-';
  el.classList.remove('flash');
  framesSemSinal = 0;
}

async function resetarBackend() {
  try {
    await fetch('/reset', { method: 'POST' });
  } catch (e) {
    console.warn('Nao foi possivel resetar o backend:', e);
  }
}

function capturarFrame() {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 320;
  canvas.height = video.videoHeight || 240;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
}

async function enviarFrame() {
  if (!cameraAtiva || video.readyState < 2 || enviandoFrame) return;
  enviandoFrame = true;
  overlay.width = video.clientWidth;
  overlay.height = video.clientHeight;

  const b64 = capturarFrame();
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: b64 })
    });
    const data = await res.json();
    processarResposta(data);
  } catch (e) {
    console.error('Erro ao enviar frame:', e);
  } finally {
    enviandoFrame = false;
  }
}

function processarResposta(data) {
  // Debug chips
  const maos = data.debug?.hands || 0;
  const rosto = data.debug?.face || false;
  const pose = data.debug?.pose || false;

  const chipMaos = document.getElementById('chipMaos');
  chipMaos.textContent = `✋ Mãos: ${maos}`;
  chipMaos.className = 'dbg-chip' + (maos > 0 ? ' on' : '');

  const chipRosto = document.getElementById('chipRosto');
  chipRosto.className = 'dbg-chip' + (rosto ? ' on' : '');

  const chipPose = document.getElementById('chipPose');
  chipPose.className = 'dbg-chip' + (pose ? ' on' : '');

  // Barra de confiança em tempo real
  if (data.live_conf !== undefined && data.live_label) {
    const confBarWrap = document.getElementById('confBarWrap');
    confBarWrap.classList.add('visible');
    const pct = Math.round(data.live_conf * 100);
    document.getElementById('confLabelSinal').textContent = data.live_label;
    document.getElementById('confLabelPct').textContent = pct + '%';
    const fill = document.getElementById('confFill');
    fill.style.width = pct + '%';
    fill.style.background = pct >= 80 ? 'var(--green)' : pct >= 60 ? 'var(--warn)' : 'var(--muted)';
  }

  // Waiting text
  document.getElementById('waitingText').textContent = data.waiting || '';

  // Sinal detectado
  if (data.label) {
    const el = document.getElementById('sinaLabel');
    el.textContent = data.label;
    el.classList.remove('flash');
    void el.offsetWidth;
    el.classList.add('flash');

    adicionarHistorico(data.label, data.confidence);
    adicionarFrase(data.label);
    ultimoLabelFeedback = data.label;
    ultimaConfFeedback = data.confidence || 0;
    document.getElementById('feedbackStatus').textContent =
      `Ultimo para feedback: ${data.label} (${Math.round(ultimaConfFeedback * 100)}%)`;
    framesSemSinal = 0;
  } else {
    framesSemSinal += 1;
    const waiting = data.waiting || '';
    const deveLimparAgora =
      maos <= 0 ||
      waiting.includes('vocabul') ||
      waiting.includes('Mostre as') ||
      waiting.includes('Ajuste as');

    if (deveLimparAgora || framesSemSinal >= FRAMES_PARA_LIMPAR_SINAL) {
      limparSinalAtual();
    }
  }

  // Desenha landmarks
  desenharLandmarks(data.visual || {});
}

async function enviarFeedback(label) {
  const status = document.getElementById('feedbackStatus');
  if (!label || label === '—' || label === '-') {
    status.textContent = 'Nenhum sinal recente para salvar.';
    return;
  }

  status.textContent = 'Salvando exemplo...';
  try {
    const res = await fetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label })
    });
    const data = await res.json();
    status.textContent = data.ok
      ? `Exemplo salvo como ${data.label}.`
      : (data.error || 'Nao foi possivel salvar.');
  } catch (e) {
    status.textContent = 'Erro ao salvar feedback.';
  }
}

function enviarFeedbackAcerto() {
  const labelAtual = document.getElementById('sinaLabel').textContent.trim();
  const labelFeedback = labelAtual && labelAtual !== '-' ? labelAtual : ultimoLabelFeedback;
  enviarFeedback(labelFeedback);
}

function enviarFeedbackCorrecao() {
  const labelCorreto = document.getElementById('feedbackLabel').value;
  enviarFeedback(labelCorreto);
}

function adicionarHistorico(label, conf) {
  if (primeiroHistorico) {
    document.getElementById('historicoLista').innerHTML = '';
    primeiroHistorico = false;
  }
  historico.unshift({ label, conf });

  const lista = document.getElementById('historicoLista');
  const li = document.createElement('li');
  li.innerHTML = `
    <span class="hist-sinal">${label}</span>
    <span class="hist-conf">${(conf * 100).toFixed(0)}%</span>
  `;
  lista.insertBefore(li, lista.firstChild);

  // Limita histórico a 20 itens
  while (lista.children.length > 20) lista.removeChild(lista.lastChild);
}

function adicionarFrase(label) {
  frase.push(label);
  atualizarFrase();
  if (vozAutomatica) falarTexto(label);
}

function atualizarFrase() {
  const el = document.getElementById('fraseTexto');
  el.textContent = frase.length > 0 ? frase.join(' ') : '—';
}

function removerUltima() {
  frase.pop();
  atualizarFrase();
}

function falarTexto(texto) {
  if (!('speechSynthesis' in window) || !texto) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(texto);
  utterance.lang = 'pt-BR';
  utterance.rate = 0.95;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
}

function falarFrase() {
  falarTexto(frase.join(' '));
}

function alternarVoz() {
  vozAutomatica = !vozAutomatica;
}

function limparFrase() {
  frase = [];
  atualizarFrase();
}

function limparHistorico() {
  historico = [];
  document.getElementById('historicoLista').innerHTML =
    '<li class="historico-vazio">Nenhum sinal detectado ainda.</li>';
  primeiroHistorico = true;
}

function desenharLandmarks(visual) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const W = overlay.width, H = overlay.height;

  // Mãos
  if (visual.hands) {
    visual.hands.forEach((mao, idx) => {
      ctx.strokeStyle = idx === 0 ? '#00e5ff' : '#7c3aed';
      ctx.lineWidth = 1.5;
      mao.forEach(p => {
        ctx.beginPath();
        ctx.arc(p[0] * W, p[1] * H, 3, 0, Math.PI * 2);
        ctx.fillStyle = idx === 0 ? '#00e5ff' : '#7c3aed';
        ctx.fill();
      });
    });
  }

  // Pose
  if (visual.pose && visual.pose.length > 0) {
    ctx.fillStyle = 'rgba(0,255,136,0.6)';
    visual.pose.forEach(p => {
      ctx.beginPath();
      ctx.arc(p[0] * W, p[1] * H, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  // Rosto
  if (visual.face && visual.face.length > 0) {
    ctx.fillStyle = 'rgba(245,158,11,0.5)';
    visual.face.forEach(p => {
      ctx.beginPath();
      ctx.arc(p[0] * W, p[1] * H, 2, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}
</script>
</body>
</html>
"""


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
    global ultimo_features, debounce_counter, ultimo_label_aceito, frames_sem_maos
    frame_buffer.clear()
    historico_predicoes.clear()
    ultimo_features = None
    debounce_counter = 0
    ultimo_label_aceito = None
    frames_sem_maos = 0


@app.route("/")
def index():
    return render_template_string(HTML, labels=labels)


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
    global ultimo_features, debounce_counter, ultimo_label_aceito, frames_sem_maos
    global ultima_seq_feedback, ultimo_label_feedback, ultima_conf_feedback

    data = request.get_json()
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

    # Calcula movimento
    movimento = calcular_movimento(features, ultimo_features)
    ultimo_features = features.copy()

    # Se houver muito movimento (troca de sinal), reseta o buffer
    if movimento > 0.03 and len(frame_buffer) > 5:
        frame_buffer.clear()
        historico_predicoes.clear()

    frame_buffer.append(features)
    movimento_seq = calcular_movimento_sequencia(frame_buffer)

# Verifica se há frames suficientes com mãos válidas
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

    print(f"Pred: {label_previsto} | Conf: {conf:.3f} | Margem: {margem:.3f} | Mov: {movimento:.4f} | Seq: {movimento_seq:.4f}")

    # Debounce: desconta contador a cada frame
    if debounce_counter > 0:
        debounce_counter -= 1

    # Votação ponderada
    resultado_voto = votar_predicao(list(historico_predicoes))
    label_final = None
    conf_voto = conf  # fallback seguro
    
    if resultado_voto:
        label_voto, conf_voto, votos, margem_voto, segundo_label, segunda_conf_voto = resultado_voto
        threshold = get_threshold(label_voto)

        # Regras especiais para sinais parecidos
        margem_ajustada = MARGEM_MINIMA

        if label_voto in ["Sim", "Não"]:
            margem_ajustada = 0.45  # mais rigor pra evitar confusão

        label_chave = chave_label(label_voto)
        margem_ajustada = MARGENS_DEMO.get(label_chave, margem_ajustada)
        votos_necessarios = VOTOS_DEMO.get(label_chave, VOTOS_NECESSARIOS)
        conflito_sim_obrigado = (
            label_voto == "Sim" and
            segundo_label == "Obrigado" and
            segunda_conf_voto >= 0.25
        )

        aceitar = (
            votos >= votos_necessarios and
            conf_voto >= threshold and
            margem_voto >= margem_ajustada and
            movimento_seq > MOVIMENTO_MINIMO * 2 and
            debounce_counter == 0 and
            not conflito_sim_obrigado
        )

        if aceitar:
            label_final = None if eh_label_rejeicao(label_voto) else label_voto
            debounce_counter = DEBOUNCE_FRAMES
            ultimo_label_aceito = label_voto
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
