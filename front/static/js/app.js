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
let ultimoFrameRecebidoEm = null;
let fpsSuavizado = null;
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
  document.getElementById('perfPanel').classList.remove('visible');
  ultimoFrameRecebidoEm = null;
  fpsSuavizado = null;
  document.getElementById('fpsValue').textContent = '--';
  document.getElementById('latencyValue').textContent = '-- ms';
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
  const inicioReq = performance.now();
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: b64 })
    });
    const data = await res.json();
    data.client_latency_ms = performance.now() - inicioReq;
    processarResposta(data);
  } catch (e) {
    console.error('Erro ao enviar frame:', e);
  } finally {
    enviandoFrame = false;
  }
}

function processarResposta(data) {
  atualizarPerformance(data.client_latency_ms);

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

    const fraseAtualizada = adicionarFrase(data.label);
    if (fraseAtualizada) adicionarHistorico(data.label, data.confidence);
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

function atualizarPerformance(latenciaMs) {
  const agora = performance.now();
  if (ultimoFrameRecebidoEm !== null) {
    const delta = Math.max(agora - ultimoFrameRecebidoEm, 1);
    const fpsAtual = 1000 / delta;
    fpsSuavizado = fpsSuavizado === null ? fpsAtual : (fpsSuavizado * 0.8) + (fpsAtual * 0.2);
  }
  ultimoFrameRecebidoEm = agora;

  document.getElementById('perfPanel').classList.add('visible');
  document.getElementById('fpsValue').textContent = fpsSuavizado === null ? '--' : fpsSuavizado.toFixed(1);
  document.getElementById('latencyValue').textContent = `${Math.round(latenciaMs)} ms`;
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
  const ultimoLabel = frase.length > 0 ? frase[frase.length - 1] : null;
  if (ultimoLabel === label) {
    atualizarEstadoVoz('Sinal repetido ignorado na frase');
    return false;
  }

  frase.push(label);
  atualizarFrase();
  if (vozAutomatica) falarTexto(label);
  return true;
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
  const texto = frase.join(' ').trim();
  if (!texto) {
    const status = document.getElementById('voiceStatus');
    status.textContent = 'Nenhuma palavra na frase para falar.';
    status.style.color = 'var(--warn)';
    return;
  }
  falarTexto(texto);
  atualizarEstadoVoz('Falando frase completa');
}

function alternarVoz() {
  vozAutomatica = !vozAutomatica;
  atualizarEstadoVoz();
}

function pararVoz() {
  if ('speechSynthesis' in window) window.speechSynthesis.cancel();
  atualizarEstadoVoz('Voz interrompida');
}

function atualizarEstadoVoz(mensagemTemporaria) {
  const btn = document.getElementById('btnVozAuto');
  const status = document.getElementById('voiceStatus');

  btn.textContent = vozAutomatica ? 'Voz auto: ligada' : 'Voz auto: desligada';
  btn.classList.toggle('active', vozAutomatica);
  btn.setAttribute('aria-pressed', vozAutomatica ? 'true' : 'false');

  status.textContent = mensagemTemporaria || (vozAutomatica ? 'Voz automática ligada' : 'Voz automática desligada');
  status.style.color = vozAutomatica ? 'var(--green)' : 'var(--muted)';
}

function limparFrase() {
  frase = [];
  atualizarFrase();
  pararVoz();
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

atualizarEstadoVoz();
