"""
=============================================================
ETAPA 3 — TREINAMENTO DO MODELO COM HOLISTIC + LSTM
Compatível com landmarks.csv gerado pela etapa2_preprocessamento.py.
=============================================================

RODAR:
  python etapa3_treinamento.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# =========================
# CONFIGURAÇÕES
# =========================
ARQUIVO_CSV = "landmarks.csv"
MAX_FRAMES = 30
N_FEATURES = 288

EPOCHS = 150
BATCH_SIZE = 16
TEST_SIZE = 0.34

MODELO_SAIDA = "modelo_libras.keras"
ENCODER_SAIDA = "label_encoder.pkl"

N_AUGMENTACOES = 12


def augmentar_sequencia(seq):
    """
    Data augmentation leve.
    Mantém o formato (30, 288).
    """
    seq = seq.copy()

    # Ruído leve
    if np.random.rand() > 0.3:
        seq += np.random.normal(0, 0.005, seq.shape)

    # Escala geral leve
    if np.random.rand() > 0.4:
        escala = np.random.uniform(0.95, 1.05)
        seq *= escala

    # Deslocamento x/y apenas nas colunas que parecem x/y
    # Como os dados são mistos, fazemos um deslocamento leve por segurança.
    if np.random.rand() > 0.4:
        deslocamento = np.random.normal(0, 0.003, seq.shape)
        seq += deslocamento

    return seq.astype(np.float32)


def aplicar_augmentation(X, y, n_aug=N_AUGMENTACOES):
    X_aug_list = [X]
    y_aug_list = [y]

    for _ in range(n_aug):
        X_novo = np.array([augmentar_sequencia(seq) for seq in X], dtype=np.float32)
        X_aug_list.append(X_novo)
        y_aug_list.append(y)

    X_final = np.concatenate(X_aug_list, axis=0)
    y_final = np.concatenate(y_aug_list, axis=0)

    idx = np.random.permutation(len(X_final))
    return X_final[idx], y_final[idx]


print("=" * 60)
print("  ETAPA 3 — TREINAMENTO HOLISTIC + LSTM")
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

features = df.drop("label", axis=1).values.astype(np.float32)
labels = df["label"].values

expected_cols = MAX_FRAMES * N_FEATURES
if features.shape[1] != expected_cols:
    print("\n[ERRO] O CSV não bate com esta versão do treinamento.")
    print(f"Esperado: {expected_cols} colunas de features")
    print(f"Recebido: {features.shape[1]} colunas de features")
    print("Rode novamente: python etapa2_preprocessamento.py")
    raise SystemExit

X = features.reshape(-1, MAX_FRAMES, N_FEATURES)

le = LabelEncoder()
y_int = le.fit_transform(labels)
n_classes = len(le.classes_)

print(f"\n  Classes ({n_classes}): {le.classes_.tolist()}")

# Verifica se há exemplos suficientes por classe
contagem = pd.Series(y_int).value_counts()
if contagem.min() < 2:
    print("\n[ERRO] Alguma classe tem menos de 2 exemplos.")
    print("Adicione mais vídeos por sinal antes de treinar.")
    raise SystemExit

X_train_raw, X_test, y_train_raw, y_test_int = train_test_split(
    X,
    y_int,
    test_size=TEST_SIZE,
    random_state=42,
    stratify=y_int
)

print(f"\n  Treino original : {X_train_raw.shape[0]} amostras")
print(f"  Teste           : {X_test.shape[0]} amostras")

print(f"\n  Aplicando Data Augmentation ({N_AUGMENTACOES}x)...")
X_train, y_train_int = aplicar_augmentation(X_train_raw, y_train_raw, N_AUGMENTACOES)
print(f"  Treino após augmentation: {X_train.shape[0]} amostras")

y_train = to_categorical(y_train_int, num_classes=n_classes)
y_test = to_categorical(y_test_int, num_classes=n_classes)

print("\n  Construindo modelo...")

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(MAX_FRAMES, N_FEATURES)),
    BatchNormalization(),
    Dropout(0.35),

    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.35),

    Dense(96, activation="relu"),
    Dropout(0.3),

    Dense(n_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        MODELO_SAIDA,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=0
    )
]

print("\n  Iniciando treinamento...\n")

historico = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

print("\n  Treinamento concluído!")

print("\n  Avaliando no conjunto de teste...")
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

acuracia = accuracy_score(y_true, y_pred)

print(f"\n  Acurácia Final no Teste: {acuracia:.2%}")
print("\n  Relatório Completo:")
print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))

# Gráficos
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
plt.show()

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
plt.show()

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(max(6, n_classes), max(5, n_classes - 1)))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.colorbar()

tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, le.classes_, rotation=45, ha="right")
plt.yticks(tick_marks, le.classes_)

for i in range(n_classes):
    for j in range(n_classes):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black"
        )

plt.ylabel("Real")
plt.xlabel("Previsto")
plt.tight_layout()
plt.savefig("matriz_confusao.png", dpi=150, bbox_inches="tight")
plt.show()

with open(ENCODER_SAIDA, "wb") as f:
    pickle.dump(le, f)

print(f"\n  Modelo salvo  : {MODELO_SAIDA}")
print(f"  Encoder salvo : {ENCODER_SAIDA}")
print("  Gráficos salvos: resultado_treinamento.png | resultado_loss.png | matriz_confusao.png")
print("\n  Próximo passo:")
print("  python app.py")
