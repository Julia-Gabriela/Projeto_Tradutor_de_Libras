# 🤟 Tradutor de Libras com IA

Este projeto tem como objetivo reconhecer sinais da Língua Brasileira de Sinais (Libras) a partir de vídeos, utilizando técnicas de visão computacional e aprendizado de máquina.

A proposta é transformar movimentos das mãos em dados numéricos e treinar um modelo capaz de identificar automaticamente qual sinal está sendo realizado.

---

## 🧠 Como o projeto funciona

O sistema é dividido em duas etapas principais:

### 🔹 1. Extração de dados (pré-processamento)

Os vídeos são analisados frame a frame utilizando o MediaPipe, uma biblioteca de visão computacional.

Para cada frame:
- A mão é detectada  
- São extraídos 21 pontos (landmarks) da mão  
- Cada ponto possui coordenadas (x, y, z)  

Isso gera:
- 63 valores por frame  
- 30 frames por vídeo  
- Total de 1890 valores por vídeo  

Esses dados são organizados em formato tabular e armazenados no arquivo:

landmarks.csv

---

### 🔹 2. Treinamento do modelo

Os dados extraídos são utilizados para treinar uma rede neural do tipo LSTM (Long Short-Term Memory).

Esse tipo de rede é ideal para:
- Processar sequências  
- Entender movimento ao longo do tempo  

O modelo aprende padrões de movimento específicos de cada sinal.

Ao final do treinamento, são gerados:

- modelo_libras.keras → modelo treinado  
- label_encoder.pkl → mapeamento dos sinais  
- resultado_treinamento.png → gráfico de desempenho  
- matriz_confusao.png → análise dos erros do modelo  

---

## 📂 Organização dos dados

Os vídeos devem ser organizados em pastas, onde cada pasta representa um sinal:

videos/
├── ola/
├── agua/
├── ajuda/

Cada pasta contém vídeos diferentes do mesmo sinal.

Isso permite que o modelo aprenda variações do movimento.

---

## 🎯 Objetivo do modelo

O modelo é treinado para:

- Identificar padrões de movimento das mãos  
- Diferenciar sinais semelhantes  
- Generalizar para novos vídeos  

---

## 📊 Desempenho esperado

A qualidade do modelo depende diretamente da quantidade e qualidade dos dados:

| Vídeos por sinal | Desempenho esperado |
|-----------------|--------------------|
| 5–10            | médio              |
| 20–30           | bom                |
| 50+             | alto               |

---

## ⚠️ Limitações

- Sensível à iluminação e qualidade do vídeo  
- Pode confundir sinais parecidos  
- Depende da consistência dos dados de entrada  

---

## 🚀 Possíveis melhorias

- Adicionar mais sinais  
- Utilizar mãos + rosto + corpo (Holistic)  
- Melhorar o dataset  
- Implementar reconhecimento em tempo real  
- Otimizar o modelo para maior precisão  

---

## 📌 Resumo

O projeto transforma vídeos em dados numéricos e utiliza uma rede neural para aprender padrões de movimento, permitindo o reconhecimento automático de sinais em Libras.