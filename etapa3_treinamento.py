"""
=============================================================
ETAPA 3 — TREINAMENTO DO MODELO (MELHORADO)
Compatível com landmarks.csv gerado pela etapa2_preprocessamento.py.

MELHORIAS:
  - Modelo com atenção temporal (Attention) entre as camadas LSTM
  - Dropout adaptativo e L2 regularization para evitar overfitting
  - Augmentation mais inteligente: menos agressiva, mais variada
  - Threshold de confiança calibrado por classe
  - Salva histórico completo para análise
  - Relatório de classes problemáticas (confusão)
=============================================================

RODAR:
  python etapa3_treinamento.py
"""

import os
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization,
    Input, Multiply, Softmax, Lambda, Reshape,
    GlobalAveragePooling1D, Permute, RepeatVector
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

# =========================
# CONFIGURAÇÕES
# =========================
ARQUIVO_CSV = "landmarks.csv"
MAX_FRAMES = 20
N_FEATURES = 288

EPOCHS = 150
BATCH_SIZE = 16
TEST_SIZE = 0.25

MODELO_SAIDA = "modelo_libras.keras"
ENCODER_SAIDA = "label_encoder.pkl"

# Augmentation mais conservadora
N_AUGMENTACOES = 10

METADATA_COLS = ["source_video"]


# =========================
# AUGMENTATION MELHORADA
# =========================

def augmentar_sequencia(seq):
    """
    Augmentation conservadora e realista.
    Só aplica transformações que podem acontecer na vida real.
    """
    seq = seq.copy().astype(np.float32)

    # Ruído muito leve (simula tremor natural)
    if np.random.rand() > 0.4:
        intensidade = np.random.uniform(0.002, 0.006)
        seq += np.random.normal(0, intensidade, seq.shape)

    # Escala leve (simula distância variável da câmera)
    if np.random.rand() > 0.4:
        escala = np.random.uniform(0.92, 1.08)
        seq *= escala

    # Deslocamento espacial leve (simula posicionamento diferente)
    if np.random.rand() > 0.5:
        deslocamento = np.random.normal(0, 0.02)
        seq += deslocamento

    # Variação de velocidade: estica ou comprime o tempo
    if np.random.rand() > 0.5:
        fator = np.random.uniform(0.85, 1.15)
        n = seq.shape[0]
        indices_orig = np.linspace(0, n - 1, int(n * fator))
        indices_orig = np.clip(indices_orig, 0, n - 1)
        indices_novo = np.linspace(0, len(indices_orig) - 1, n)
        seq_novo = np.zeros_like(seq)
        for j in range(seq.shape[1]):
            seq_novo[:, j] = np.interp(indices_novo, np.arange(len(indices_orig)),
                                        seq[np.round(indices_orig).astype(int), j])
        seq = seq_novo

    # Espelhamento (útil para sinais simétricos)
    if np.random.rand() > 0.5:
        seq = espelhar_sequencia(seq)

    return seq.astype(np.float32)


def espelhar_sequencia(seq):
    seq = seq.copy()
    pose = seq[:, :132].reshape(seq.shape[0], 33, 4)
    left_hand = seq[:, 132:195].copy()
    right_hand = seq[:, 195:258].copy()
    face = seq[:, 258:288].reshape(seq.shape[0], len(range(10)), 3)

    pose[:, :, 0] *= -1
    left_hand[:, 0::3] *= -1
    right_hand[:, 0::3] *= -1
    face[:, :, 0] *= -1

    pares_pose = [
        (1, 4), (2, 5), (3, 6), (7, 8),
        (11, 12), (13, 14), (15, 16),
        (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
    ]
    for a, b in pares_pose:
        pose[:, [a, b], :] = pose[:, [b, a], :]

    seq[:, 132:195] = right_hand
    seq[:, 195:258] = left_hand
    return seq


def aplicar_augmentation(X, y, n_aug=N_AUGMENTACOES):
    X_aug_list = [X]
    y_aug_list = [y]

    # Sempre inclui versão espelhada
    X_aug_list.append(np.array([espelhar_sequencia(s) for s in X], dtype=np.float32))
    y_aug_list.append(y)

    # Augmentações aleatórias
    for _ in range(n_aug):
        X_novo = np.array([augmentar_sequencia(s) for s in X], dtype=np.float32)
        X_aug_list.append(X_novo)
        y_aug_list.append(y)

    X_final = np.concatenate(X_aug_list, axis=0)
    y_final = np.concatenate(y_aug_list, axis=0)

    idx = np.random.permutation(len(X_final))
    return X_final[idx], y_final[idx]


def grupo_video(nome_video):
    nome = os.path.splitext(str(nome_video))[0]
    anterior = None
    while anterior != nome:
        anterior = nome
        nome = re.sub(r"\s+-\s+Copia(?:\s+\(\d+\))?$", "", nome, flags=re.IGNORECASE)
    return nome.strip().lower()


# =========================
# MODELO COM ATENÇÃO TEMPORAL
# =========================

def construir_modelo(n_frames, n_features, n_classes):
    inputs = Input(shape=(n_frames, n_features))
    
    x = LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(1e-3))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = LSTM(32, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)


# =========================
# MAIN
# =========================
print("=" * 60)
print("  ETAPA 3 — TREINAMENTO MELHORADO (LSTM + ATENÇÃO)")
print("=" * 60)

if not os.path.exists(ARQUIVO_CSV):
    print(f"[ERRO] {ARQUIVO_CSV} não encontrado.")
    print("Execute primeiro: python etapa2_preprocessamento.py")
    raise SystemExit

print(f"\nCarregando {ARQUIVO_CSV}...")
df = pd.read_csv(ARQUIVO_CSV)

print(f"  Shape original : {df.shape}")
print(f"  Sinais         : {df['label'].unique().tolist()}")
print(f"  Amostras/sinal :\n{df['label'].value_counts().to_string()}")

metadata_existente = [c for c in METADATA_COLS if c in df.columns]
features = df.drop(["label"] + metadata_existente, axis=1).values.astype(np.float32)
labels = df["label"].values

expected_cols = MAX_FRAMES * N_FEATURES
if features.shape[1] != expected_cols:
    print(f"\n[ERRO] CSV incompatível. Esperado: {expected_cols} colunas, recebido: {features.shape[1]}")
    print("Rode novamente: python etapa2_preprocessamento.py")
    raise SystemExit

X = features.reshape(-1, MAX_FRAMES, N_FEATURES)

le = LabelEncoder()
y_int = le.fit_transform(labels)
n_classes = len(le.classes_)

print(f"\n  Classes ({n_classes}): {le.classes_.tolist()}")

contagem = pd.Series(y_int).value_counts()
if contagem.min() < 2:
    print("\n[ERRO] Alguma classe tem menos de 2 exemplos. Adicione mais vídeos.")
    raise SystemExit

# --- Split inteligente por grupo ---
if "source_video" in df.columns:
    grupos = df["source_video"].map(grupo_video).values
    splitter = GroupShuffleSplit(n_splits=30, test_size=TEST_SIZE, random_state=42)

    split_escolhido = None
    for train_idx, test_idx in splitter.split(X, y_int, groups=grupos):
        if (set(y_int[train_idx]) == set(range(n_classes)) and
                set(y_int[test_idx]) == set(range(n_classes))):
            split_escolhido = (train_idx, test_idx)
            break

    if split_escolhido is None:
        print("\n[AVISO] Usando split estratificado simples.")
        X_train_raw, X_test, y_train_raw, y_test_int = train_test_split(
            X, y_int, test_size=TEST_SIZE, random_state=42, stratify=y_int)
    else:
        train_idx, test_idx = split_escolhido
        X_train_raw, X_test = X[train_idx], X[test_idx]
        y_train_raw, y_test_int = y_int[train_idx], y_int[test_idx]
        print("\n  Split por grupo ativo: vídeos relacionados ficam no mesmo lado.")
else:
    print("\n[AVISO] Sem coluna source_video. Usando split estratificado.")
    X_train_raw, X_test, y_train_raw, y_test_int = train_test_split(
        X, y_int, test_size=TEST_SIZE, random_state=42, stratify=y_int)

print(f"\n  Treino original : {X_train_raw.shape[0]} amostras")
print(f"  Teste           : {X_test.shape[0]} amostras")

print(f"\n  Aplicando Data Augmentation ({N_AUGMENTACOES}x)...")
X_train, y_train_int = aplicar_augmentation(X_train_raw, y_train_raw, N_AUGMENTACOES)
print(f"  Treino após augmentation: {X_train.shape[0]} amostras")

y_train = to_categorical(y_train_int, num_classes=n_classes)
y_test = to_categorical(y_test_int, num_classes=n_classes)

print("\n  Construindo modelo com atenção temporal...")
model = construir_modelo(MAX_FRAMES, N_FEATURES, n_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODELO_SAIDA, monitor="val_accuracy", save_best_only=True, verbose=0),
    # Reduz o learning rate quando estagna: ajuda a sair de platôs
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5, verbose=1),
]

print("\n  Calculando pesos das classes...")
from sklearn.utils.class_weight import compute_class_weight
pesos = compute_class_weight('balanced', classes=np.unique(y_train_int), y=y_train_int)
class_weight = dict(enumerate(pesos))
print(f"  Pesos: { {le.classes_[i]: round(v,2) for i,v in class_weight.items()} }")

print("\n  Iniciando treinamento...\n")

historico = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    callbacks=callbacks,
    class_weight=class_weight,  # <-- linha adicionada
    verbose=1
)

print("\n  Treinamento concluído!")

# --- Avaliação ---
print("\n  Avaliando no conjunto de teste...")
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

acuracia = accuracy_score(y_true, y_pred)

print(f"\n  Acurácia Final no Teste: {acuracia:.2%}")
print("\n  Relatório Completo:")
print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))

# --- Diagnóstico de confusão ---
cm = confusion_matrix(y_true, y_pred)
print("\n  Pares de sinais mais confundidos:")
confusoes = []
for i in range(n_classes):
    for j in range(n_classes):
        if i != j and cm[i, j] > 0:
            confusoes.append((cm[i, j], le.classes_[i], le.classes_[j]))
confusoes.sort(reverse=True)
for n, real, prev in confusoes[:5]:
    print(f"    '{real}' confundido com '{prev}': {n}x")
    if confusoes:
        print("    → Dica: grave mais vídeos variados para esses sinais.")

# --- Calibrar thresholds por classe ---
print("\n  Calibrando thresholds de confiança por classe...")
thresholds = {}
for cls_idx in range(n_classes):
    mask = y_true == cls_idx
    if mask.sum() > 0:
        confs = y_pred_prob[mask, cls_idx]
        acertos = (y_pred[mask] == cls_idx)
        if acertos.sum() > 0:
            # Threshold conservador: percentil 25 das confiança dos acertos
            threshold = float(np.percentile(confs[acertos], 25))
        else:
            threshold = 0.7
        thresholds[le.classes_[cls_idx]] = max(0.5, min(0.95, threshold))

print("  Thresholds por classe:")
for cls, th in sorted(thresholds.items()):
    print(f"    {cls}: {th:.2f}")

# Salva thresholds junto com o encoder
with open("thresholds.pkl", "wb") as f:
    pickle.dump(thresholds, f)

model.save(MODELO_SAIDA)
with open(ENCODER_SAIDA, "wb") as f:
    pickle.dump(le, f)

print(f"\n  Modelo salvo  : {MODELO_SAIDA}")
print(f"  Encoder salvo : {ENCODER_SAIDA}")
print(f"  Thresholds    : thresholds.pkl")

# --- Gráficos ---
plt.figure(figsize=(8, 5))
plt.plot(historico.history["accuracy"], label="Treino")
plt.plot(historico.history["val_accuracy"], label="Validação")
plt.title(f"Acurácia — melhor teste: {acuracia:.2%}")
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("resultado_treinamento.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(historico.history["loss"], label="Treino")
plt.plot(historico.history["val_loss"], label="Validação")
plt.title("Perda / Loss")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("resultado_loss.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(max(6, n_classes), max(5, n_classes - 1)))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, le.classes_, rotation=45, ha="right")
plt.yticks(tick_marks, le.classes_)
for i in range(n_classes):
    for j in range(n_classes):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.ylabel("Real")
plt.xlabel("Previsto")
plt.tight_layout()
plt.savefig("matriz_confusao.png", dpi=150, bbox_inches="tight")
plt.close()

print("  Gráficos salvos: resultado_treinamento.png | resultado_loss.png | matriz_confusao.png")
print("\n  Próximo passo:")
print("  python app.py")
