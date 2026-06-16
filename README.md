<div align="center">

# Tradutor de Libras com IA

**Um prototipo web para reconhecer sinais de Libras em tempo real pela camera.**

![Python](https://img.shields.io/badge/Python-3.11-ff4fbd?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web-9d57ff?style=for-the-badge&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-ff6db7?style=for-the-badge&logo=tensorflow&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Landmarks-7ee8ff?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Camera-7b5cff?style=for-the-badge&logo=opencv&logoColor=white)

</div>

---

## Sobre o projeto

O **Tradutor de Libras com IA** e uma aplicacao desenvolvida para reconhecer sinais de Libras a partir da webcam. O sistema extrai pontos das maos, pose corporal e rosto com **MediaPipe**, organiza esses dados em sequencias temporais e usa um modelo neural treinado com videos para classificar o sinal realizado.

A interface foi pensada para demonstracao em tempo real, com visual escuro em tons de rosa e roxo, camera ao vivo, palavra detectada, frase montada, historico e lista de sinais disponiveis.

---

## Destaques

| Recurso | Descricao |
| --- | --- |
| Camera em tempo real | Captura frames da webcam direto no navegador. |
| Landmarks com MediaPipe | Extrai maos, pose e pontos selecionados do rosto. |
| Modelo temporal | Classifica sequencias de 20 frames. |
| Interface web | Front em HTML, CSS e JavaScript com backend Flask. |
| Classe `Desconhecido` | Ajuda a identificar gestos fora do vocabulario treinado. |
| Relatorios | Gera metricas, matriz de confusao e auditoria do preprocessamento. |

---

## Stack

| Area | Tecnologias |
| --- | --- |
| Backend | Python, Flask |
| IA | TensorFlow/Keras, Scikit-learn |
| Visao computacional | MediaPipe, OpenCV |
| Dados | NumPy, Pandas |
| Frontend | HTML, CSS, JavaScript |

---

## Sinais reconhecidos

| Sinal | Videos |
| --- | ---: |
| Aceitar | 20 |
| Banheiro | 20 |
| Bebida | 22 |
| Calmo | 20 |
| Casa | 20 |
| Desconhecido | 120 |
| Livro | 20 |
| Nome | 20 |
| Obrigado | 20 |
| Oi | 20 |
| Sortudo | 20 |

---

## Resultado atual

Ultima avaliacao do modelo:

| Metrica | Valor |
| --- | ---: |
| Acuracia no teste | **67.03%** |
| Acuracia balanceada | **67.79%** |
| Amostras de teste | **543** |

O modelo apresentou melhor desempenho em classes como `Banheiro`, `Bebida`, `Calmo`, `Casa`, `Obrigado` e `Sortudo`. A classe `Desconhecido` foi reforcada para reduzir leituras indevidas em gestos que nao pertencem ao vocabulario.

> Este e um prototipo academico. A qualidade da leitura depende da iluminacao, enquadramento, velocidade do sinal e variedade dos videos de treinamento.

---

## Como funciona

```text
Videos de treino
      |
      v
Validacao do dataset
      |
      v
Extracao de landmarks com MediaPipe
      |
      v
Sequencias temporais de 20 frames
      |
      v
Treinamento do modelo neural
      |
      v
Aplicacao Flask com camera em tempo real
```

---

## Estrutura

```text
.
|-- app.py
|-- etapa2_preprocessamento.py
|-- etapa3_treinamento.py
|-- validar_dataset.py
|-- requirements.txt
|-- front/
|   |-- static/
|   |   |-- css/
|   |   `-- js/
|   `-- templates/
|-- videos/
|   `-- data/
|-- data/
|-- models/
`-- reports/
```

---

## Como executar

Crie o ambiente virtual:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Valide os videos:

```powershell
python validar_dataset.py
```

Rode o preprocessamento e o treinamento:

```powershell
python etapa2_preprocessamento.py
python etapa3_treinamento.py
```

Inicie o sistema:

```powershell
python app.py
```

Acesse:

```text
http://localhost:5000
```

---

## Padrao dos videos

Os arquivos devem ficar em `videos/data` e seguir este formato:

```text
NomeDoSinal_Articulador1.mp4
NomeDoSinal_Articulador2.mp4
NomeDoSinal_Articulador3.mp4
```

Exemplos:

```text
Oi_Articulador1.mp4
Casa_Articulador2.mp4
Obrigado_Articulador3.mp4
```

---

## Arquivos gerados

| Pasta | Arquivos |
| --- | --- |
| `data/` | `landmarks.csv`, `qualidade_preprocessamento.csv` |
| `models/` | `modelo_libras.keras`, `label_encoder.pkl`, `thresholds.pkl` |
| `reports/` | metricas, matriz de confusao, graficos de treino e auditoria |

---

## Status

Versao funcional para demonstracao, com fluxo completo de:

- validacao do dataset;
- preprocessamento dos videos;
- treinamento do modelo;
- avaliacao com metricas;
- reconhecimento em tempo real pela camera.

<div align="center">

**Feito com Python, IA e um tantinho de brilho rosa.**

</div>
