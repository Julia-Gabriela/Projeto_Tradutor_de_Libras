"""Treina o classificador usando data/landmarks.csv."""

import os
import pickle
import re
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Conv1D,
    Input, Multiply, Softmax, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

DATA_DIR = "data"
MODELS_DIR = "models"
REPORTS_DIR = "reports"

ARQUIVO_CSV = os.path.join(DATA_DIR, "landmarks.csv")
FEEDBACK_CSV = os.path.join(DATA_DIR, "feedback_landmarks.csv")
QUALIDADE_CSV = os.path.join(DATA_DIR, "qualidade_preprocessamento.csv")
MAX_FRAMES = 20
N_FEATURES = 288

EPOCHS = 150
BATCH_SIZE = 16
TEST_SIZE = 0.25

MODELO_SAIDA = os.path.join(MODELS_DIR, "modelo_libras.keras")
ENCODER_SAIDA = os.path.join(MODELS_DIR, "label_encoder.pkl")

N_AUGMENTACOES = 8
USAR_ESPELHAMENTO = False

METADATA_COLS = ["source_video"]


def mascara_coordenadas(seq):
    mask = np.zeros(seq.shape, dtype=bool)
    pose = seq[:, :132].reshape(seq.shape[0], 33, 4)
    left_hand = seq[:, 132:195].reshape(seq.shape[0], 21, 3)
    right_hand = seq[:, 195:258].reshape(seq.shape[0], 21, 3)
    face = seq[:, 258:288].reshape(seq.shape[0], len(range(10)), 3)

    mask_pose = mask[:, :132].reshape(seq.shape[0], 33, 4)
    mask_left = mask[:, 132:195].reshape(seq.shape[0], 21, 3)
    mask_right = mask[:, 195:258].reshape(seq.shape[0], 21, 3)
    mask_face = mask[:, 258:288].reshape(seq.shape[0], len(range(10)), 3)

    mask_pose[:, :, :3] = np.any(pose[:, :, :3] != 0, axis=2, keepdims=True)
    mask_left[:, :, :] = np.any(left_hand != 0, axis=2, keepdims=True)
    mask_right[:, :, :] = np.any(right_hand != 0, axis=2, keepdims=True)
    mask_face[:, :, :] = np.any(face != 0, axis=2, keepdims=True)
    return mask


def deslocar_eixos(seq, dx=0.0, dy=0.0, dz=0.0):
    seq = seq.copy()
    deslocamentos = np.array([dx, dy, dz], dtype=np.float32)
    pose = seq[:, :132].reshape(seq.shape[0], 33, 4)
    blocos = (
        pose[:, :, :3],
        seq[:, 132:195].reshape(seq.shape[0], 21, 3),
        seq[:, 195:258].reshape(seq.shape[0], 21, 3),
        seq[:, 258:288].reshape(seq.shape[0], len(range(10)), 3),
    )
    for bloco in blocos:
        mask = np.any(bloco != 0, axis=2)
        bloco[mask] += deslocamentos
    return seq


def aplicar_dropout_landmarks(seq):
    seq = seq.copy()
    for inicio, fim in ((132, 195), (195, 258)):
        if np.random.rand() < 0.18:
            bloco = seq[:, inicio:fim].reshape(seq.shape[0], 21, 3)
            frames = np.random.choice(seq.shape[0], size=np.random.randint(1, 4), replace=False)
            pontos = np.random.choice(21, size=np.random.randint(2, 7), replace=False)
            bloco[np.ix_(frames, pontos)] = 0.0

    if np.random.rand() < 0.12:
        face = seq[:, 258:288].reshape(seq.shape[0], len(range(10)), 3)
        frames = np.random.choice(seq.shape[0], size=np.random.randint(1, 4), replace=False)
        face[frames, :, :] = 0.0
    return seq

def augmentar_sequencia(seq):
    seq = seq.copy().astype(np.float32)
    coord_mask = mascara_coordenadas(seq)

    if np.random.rand() > 0.4:
        intensidade = np.random.uniform(0.002, 0.006)
        ruido = np.random.normal(0, intensidade, seq.shape).astype(np.float32)
        seq[coord_mask] += ruido[coord_mask]

    if np.random.rand() > 0.4:
        escala = np.random.uniform(0.92, 1.08)
        seq[coord_mask] *= escala

    if np.random.rand() > 0.5:
        dx, dy, dz = np.random.normal(0, [0.025, 0.025, 0.01])
        seq = deslocar_eixos(seq, dx, dy, dz)

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

    if USAR_ESPELHAMENTO and np.random.rand() > 0.65:
        seq = espelhar_sequencia(seq)

    seq = aplicar_dropout_landmarks(seq)

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


def auditar_qualidade_preprocessamento():
    if not os.path.exists(QUALIDADE_CSV):
        print(f"\n  [INFO] {QUALIDADE_CSV} nao encontrado. Rode etapa2 para gerar auditoria.")
        return None

    df_q = pd.read_csv(QUALIDADE_CSV)
    if df_q.empty:
        print(f"\n  [AVISO] {QUALIDADE_CSV} esta vazio.")
        return None

    resumo = (
        df_q.groupby("label")
        .agg(
            janelas=("label", "size"),
            aceitas=("aceita", "sum"),
            media_ratio_maos=("ratio_maos", "mean"),
            media_movimento_maos=("movimento_maos", "mean"),
        )
        .reset_index()
    )
    resumo["taxa_aceitacao"] = resumo["aceitas"] / resumo["janelas"].clip(lower=1)
    resumo = resumo.sort_values(["taxa_aceitacao", "media_movimento_maos"])

    print("\n  Auditoria do preprocessamento:")
    for _, row in resumo.iterrows():
        alertas = []
        if row["taxa_aceitacao"] < 0.65:
            alertas.append("muitas janelas descartadas")
        if row["media_ratio_maos"] < 0.70:
            alertas.append("pouca mao detectada")
        if row["media_movimento_maos"] < 0.004:
            alertas.append("baixo movimento")
        aviso = f" ({'; '.join(alertas)})" if alertas else ""
        print(
            f"    {row['label']}: {int(row['aceitas'])}/{int(row['janelas'])} aceitas | "
            f"maos {row['media_ratio_maos']:.2f} | mov {row['media_movimento_maos']:.4f}{aviso}"
        )

    os.makedirs(REPORTS_DIR, exist_ok=True)
    saida = os.path.join(REPORTS_DIR, "auditoria_preprocessamento.csv")
    resumo.to_csv(saida, index=False)
    print(f"  Auditoria resumida salva em: {saida}")
    return resumo


def construir_modelo(n_frames, n_features, n_classes):
    inputs = Input(shape=(n_frames, n_features))

    x = Conv1D(
        96,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(8e-4),
    )(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Bidirectional(
        LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(8e-4))
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)
    x = Bidirectional(
        LSTM(48, return_sequences=True, kernel_regularizer=regularizers.l2(8e-4))
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.30)(x)
    scores = Dense(1, activation="tanh", name="attention_scores")(x)
    pesos = Softmax(axis=1, name="attention_weights")(scores)
    contexto = Multiply(name="attention_apply")([x, pesos])
    x = GlobalAveragePooling1D(name="attention_context")(contexto)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(8e-4))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)


print("=" * 60)
print("  ETAPA 3 - TREINAMENTO (LSTM + ATENCAO)")
print("=" * 60)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

if not os.path.exists(ARQUIVO_CSV):
    print(f"[ERRO] {ARQUIVO_CSV} nao encontrado.")
    print("Execute primeiro: python etapa2_preprocessamento.py")
    raise SystemExit

print(f"\nCarregando {ARQUIVO_CSV}...")
df = pd.read_csv(ARQUIVO_CSV)
resumo_qualidade = auditar_qualidade_preprocessamento()

if os.path.exists(FEEDBACK_CSV):
    print(f"Carregando feedback coletado em {FEEDBACK_CSV}...")
    df_feedback = pd.read_csv(FEEDBACK_CSV)
    colunas_faltando = [c for c in df.columns if c not in df_feedback.columns]
    if colunas_faltando:
        print(f"  [AVISO] Feedback ignorado: {len(colunas_faltando)} colunas faltando.")
    else:
        df_feedback = df_feedback.reindex(columns=df.columns)
        df = pd.concat([df, df_feedback], ignore_index=True)
        print(f"  Feedback adicionado: {len(df_feedback)} amostras")

print(f"  Shape original : {df.shape}")
print(f"  Sinais         : {df['label'].unique().tolist()}")
print(f"  Amostras/sinal :\n{df['label'].value_counts().to_string()}")

metadata_existente = [c for c in METADATA_COLS if c in df.columns]
features = df.drop(["label"] + metadata_existente, axis=1).values.astype(np.float32)
labels = df["label"].values

expected_cols = MAX_FRAMES * N_FEATURES
if features.shape[1] != expected_cols:
    print(f"\n[ERRO] CSV incompativel. Esperado: {expected_cols} colunas, recebido: {features.shape[1]}")
    print("Rode novamente: python etapa2_preprocessamento.py")
    raise SystemExit

X = features.reshape(-1, MAX_FRAMES, N_FEATURES)

le = LabelEncoder()
y_int = le.fit_transform(labels)
n_classes = len(le.classes_)

print(f"\n  Classes ({n_classes}): {le.classes_.tolist()}")

contagem = pd.Series(y_int).value_counts()
if contagem.min() < 2:
    print("\n[ERRO] Alguma classe tem menos de 2 exemplos. Adicione mais videos.")
    raise SystemExit

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
        print("\n  Split por grupo ativo: videos relacionados ficam no mesmo lado.")
else:
    print("\n[AVISO] Sem coluna source_video. Usando split estratificado.")
    X_train_raw, X_test, y_train_raw, y_test_int = train_test_split(
        X, y_int, test_size=TEST_SIZE, random_state=42, stratify=y_int)

print(f"\n  Treino original : {X_train_raw.shape[0]} amostras")
print(f"  Teste           : {X_test.shape[0]} amostras")

val_stratify = y_train_raw if pd.Series(y_train_raw).value_counts().min() >= 2 else None
X_train_base, X_val, y_train_base, y_val_int = train_test_split(
    X_train_raw,
    y_train_raw,
    test_size=0.20,
    random_state=42,
    stratify=val_stratify,
)
print(f"  Validacao real : {X_val.shape[0]} amostras (sem augmentation)")

print(f"\n  Aplicando Data Augmentation ({N_AUGMENTACOES}x)...")
print(f"  Espelhamento automatico: {'SIM' if USAR_ESPELHAMENTO else 'NAO'}")
X_train, y_train_int = aplicar_augmentation(X_train_base, y_train_base, N_AUGMENTACOES)
print(f"  Treino apos augmentation: {X_train.shape[0]} amostras")

y_train = to_categorical(y_train_int, num_classes=n_classes)
y_val = to_categorical(y_val_int, num_classes=n_classes)
y_test = to_categorical(y_test_int, num_classes=n_classes)

print("\n  Construindo modelo com atencao temporal...")
model = construir_modelo(MAX_FRAMES, N_FEATURES, n_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.04),
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODELO_SAIDA, monitor="val_accuracy", save_best_only=True, verbose=0),
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
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=2
)

print("\n  Treinamento concluido!")

print("\n  Avaliando no conjunto de teste...")
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

acuracia = accuracy_score(y_true, y_pred)
acuracia_balanceada = balanced_accuracy_score(y_true, y_pred)
relatorio_dict = classification_report(
    y_true, y_pred, target_names=le.classes_, zero_division=0, output_dict=True
)
relatorio_texto = classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0)

print(f"\n  Acuracia Final no Teste: {acuracia:.2%}")
print(f"  Acuracia Balanceada   : {acuracia_balanceada:.2%}")
print("\n  Relatorio Completo:")
print(relatorio_texto)

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
    print("    -> Dica: grave mais videos variados para esses sinais.")

print("\n  Calibrando thresholds de confianca por classe...")
thresholds = {}
for cls_idx in range(n_classes):
    y_bin = (y_true == cls_idx)
    probs_cls = y_pred_prob[:, cls_idx]
    melhor_threshold = 0.75
    melhor_score = -1.0

    for threshold in np.linspace(0.45, 0.95, 51):
        pred_bin = probs_cls >= threshold
        tp = np.sum(pred_bin & y_bin)
        fp = np.sum(pred_bin & ~y_bin)
        fn = np.sum(~pred_bin & y_bin)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        precision_floor = 0.60 if le.classes_[cls_idx] not in ["Sim", "Nao"] else 0.70

        if precision >= precision_floor and f1 > melhor_score:
            melhor_score = f1
            melhor_threshold = float(threshold)

    thresholds[le.classes_[cls_idx]] = max(0.55, min(0.95, melhor_threshold))

print("  Thresholds por classe:")
for cls, th in sorted(thresholds.items()):
    print(f"    {cls}: {th:.2f}")

thresholds_saida = os.path.join(MODELS_DIR, "thresholds.pkl")
with open(thresholds_saida, "wb") as f:
    pickle.dump(thresholds, f)

model.save(MODELO_SAIDA)
with open(ENCODER_SAIDA, "wb") as f:
    pickle.dump(le, f)

metricas_saida = os.path.join(REPORTS_DIR, "metricas_teste.json")
relatorio_saida = os.path.join(REPORTS_DIR, "relatorio_classificacao.txt")
confusoes_saida = os.path.join(REPORTS_DIR, "confusoes.csv")

metricas = {
    "accuracy": float(acuracia),
    "balanced_accuracy": float(acuracia_balanceada),
    "classes": le.classes_.tolist(),
    "samples": {
        "train_original": int(X_train_raw.shape[0]),
        "validation": int(X_val.shape[0]),
        "test": int(X_test.shape[0]),
        "train_after_augmentation": int(X_train.shape[0]),
    },
    "augmentation": {
        "n_augmentacoes": int(N_AUGMENTACOES),
        "usar_espelhamento": bool(USAR_ESPELHAMENTO),
    },
    "thresholds": thresholds,
    "classification_report": relatorio_dict,
}

with open(metricas_saida, "w", encoding="utf-8") as f:
    json.dump(metricas, f, ensure_ascii=False, indent=2)

with open(relatorio_saida, "w", encoding="utf-8") as f:
    f.write(f"Acuracia final no teste: {acuracia:.2%}\n")
    f.write(f"Acuracia balanceada: {acuracia_balanceada:.2%}\n\n")
    f.write(relatorio_texto)

pd.DataFrame(
    [{"quantidade": n, "real": real, "previsto": prev} for n, real, prev in confusoes]
).to_csv(confusoes_saida, index=False)

print(f"\n  Modelo salvo  : {MODELO_SAIDA}")
print(f"  Encoder salvo : {ENCODER_SAIDA}")
print(f"  Thresholds    : {thresholds_saida}")
print(f"  Metricas      : {metricas_saida}")
print(f"  Relatorio     : {relatorio_saida}")

plt.figure(figsize=(8, 5))
plt.plot(historico.history["accuracy"], label="Treino")
plt.plot(historico.history["val_accuracy"], label="Validacao")
plt.title(f"Acuracia - melhor teste: {acuracia:.2%}")
plt.xlabel("Epoca")
plt.ylabel("Acuracia")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "resultado_treinamento.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(historico.history["loss"], label="Treino")
plt.plot(historico.history["val_loss"], label="Validacao")
plt.title("Perda / Loss")
plt.xlabel("Epoca")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "resultado_loss.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(max(6, n_classes), max(5, n_classes - 1)))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Matriz de Confusao")
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
plt.savefig(os.path.join(REPORTS_DIR, "matriz_confusao.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"  Graficos salvos em: {REPORTS_DIR}")
print("\n  Proximo passo:")
print("  python app.py")
