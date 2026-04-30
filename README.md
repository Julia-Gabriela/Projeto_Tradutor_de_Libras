# 🤟 Tradutor de Libras — Guia Passo a Passo

## Estrutura de arquivos

```
seu_projeto/
│
├── videos/                        
│   ├── ola/
│   │   ├── video1.mp4
│   │   └── video2.mp4
│   ├── agua/
│   │   ├── video1.mp4
│   │   └── video2.mp4
│   └── ajuda/
│       └── video1.mp4
│
├── etapa2_preprocessamento.py     ← roda primeiro
├── etapa3_treinamento.py          ← roda depois
│
├── landmarks.csv                  ← gerado pela etapa 2
├── modelo_libras.keras            ← gerado pela etapa 3
├── label_encoder.pkl              ← gerado pela etapa 3
├── resultado_treinamento.png      ← gerado pela etapa 3
└── matriz_confusao.png            ← gerado pela etapa 3
```

---

## 1. Instalar dependências

```bash
pip install mediapipe opencv-python pandas numpy tqdm tensorflow scikit-learn matplotlib
```

---

## 2. Baixar os vídeos de Libras

Acesse o site do V-Librasil e baixe vídeos de algumas palavras:
🔗 https://libras.cin.ufpe.br/

Organize as pastas como mostrado acima.
Pelo menos 10 vídeos por sinal é o recomendado.

---

## 3. Rodar a Etapa 2 (Pré-processamento)

```bash
python etapa2_preprocessamento.py
```

O que acontece:
- Lê cada vídeo da pasta `videos/`
- Usa o MediaPipe para detectar a mão em cada frame
- Extrai 21 pontos (x, y, z) = 63 valores por frame
- Pega 30 frames por vídeo de forma uniforme
- Salva tudo em `landmarks.csv`

---

## 4. Rodar a Etapa 3 (Treinamento)

```bash
python etapa3_treinamento.py
```

O que acontece:
- Lê o `landmarks.csv`
- Organiza os dados como sequências temporais (30 frames × 63 landmarks)
- Treina uma rede LSTM com 2 camadas
- Para automaticamente quando não melhora mais (EarlyStopping)
- Mostra acurácia, relatório de precisão/recall
- Salva gráficos de acurácia e matriz de confusão
- Salva o modelo treinado em `modelo_libras.keras`

---

## 5. O que esperar de resultado

| Quantidade de vídeos por sinal | Acurácia esperada |
|-------------------------------|-------------------|
| 5–10 vídeos                   | ~70–80%           |
| 20–30 vídeos                  | ~85–92%           |
| 50+ vídeos                    | ~93–97%           |

---

## Dúvidas frequentes

**O MediaPipe não detectou a mão no vídeo?**
→ O script preenche com zeros automaticamente. Isso pode baixar a acurácia.
→ Solução: usar vídeos bem iluminados e com a mão visível.

**Erro "landmarks.csv not found" na etapa 3?**
→ Rode a etapa 2 primeiro.

**A acurácia ficou muito baixa?**
→ Tente adicionar mais vídeos por sinal ou reduzir o número de sinais (comece com 3-5).
