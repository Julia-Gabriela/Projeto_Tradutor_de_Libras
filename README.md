<div align="center">

# Tradutor de Libras com IA

**Prototipo web para reconhecer sinais de Libras em tempo real pela camera e sugerir frases em portugues.**

![Python](https://img.shields.io/badge/Python-3.11-ff4fbd?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web-9d57ff?style=for-the-badge&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-ff6db7?style=for-the-badge&logo=tensorflow&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Landmarks-7ee8ff?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Camera-7b5cff?style=for-the-badge&logo=opencv&logoColor=white)

</div>

---

## Sobre o projeto

O **Tradutor de Libras com IA** e uma aplicacao web que reconhece sinais de Libras a partir da webcam. O sistema extrai landmarks das maos, pose corporal e pontos selecionados do rosto com **MediaPipe**, organiza esses dados em sequencias temporais de 20 frames e usa um modelo neural treinado com **TensorFlow/Keras** para classificar o sinal realizado.

Alem da classificacao isolada do sinal, o projeto tambem possui uma camada local simples de **RAG baseado em JSON**. Essa camada nao melhora a acuracia do modelo de visao; ela atua como pos-processamento textual, consultando exemplos de sequencias conhecidas antes de sugerir a frase final em portugues.

---

## Destaques

| Recurso | Descricao |
| --- | --- |
| Camera em tempo real | Captura frames da webcam pelo navegador e envia ao backend Flask. |
| Landmarks com MediaPipe | Extrai maos, pose e pontos selecionados do rosto. |
| Modelo temporal | Classifica sequencias de 20 frames com TensorFlow/Keras. |
| Classe `Desconhecido` | Ajuda a rejeitar gestos fora do vocabulario treinado. |
| RAG local simples | Consulta `base_libras_rag.json` para sugerir frases a partir da sequencia de sinais. |
| Fallback por regras | Usa `frase_builder.py` quando o RAG nao encontra exemplo confiavel. |
| Interface web | Mostra camera, sinal detectado, confianca, historico, sinais detectados, frase sugerida, origem e explicacao. |
| Relatorios | Gera metricas, matriz de confusao, graficos de treino e auditoria do preprocessamento. |

---

## Stack

| Area | Tecnologias |
| --- | --- |
| Backend | Python, Flask |
| IA | TensorFlow/Keras, Scikit-learn |
| Visao computacional | MediaPipe, OpenCV |
| Dados | NumPy, Pandas |
| Frontend | HTML, CSS, JavaScript |
| Pos-processamento textual | RAG local em JSON e regras locais |

---

## Sinais reconhecidos

O modelo atual trabalha apenas com as classes abaixo. A base RAG e a montagem de frases tambem usam somente esses sinais.

| Sinal | Videos |
| --- | ---: |
| Aceitar | 20 |
| Banheiro | 20 |
| Bebida | 22 |
| Calmo | 20 |
| Casa | 20 |
| Desconhecido | 63 |
| Eu | 27 |
| Livro | 20 |
| Nome | 20 |
| Obrigado | 20 |
| Oi | 20 |
| Quero | 28 |
| Sortudo | 20 |

---

## Resultado atual

Ultima avaliacao salva em `reports/metricas_teste.json`:

| Metrica | Valor |
| --- | ---: |
| Acuracia no teste | **66.47%** |
| Acuracia balanceada | **69.19%** |
| Amostras de teste | **498** |
| Amostras de treino apos aumento | **11781** |

O modelo ainda apresenta confusoes em alguns sinais visualmente proximos ou sensiveis ao enquadramento, principalmente envolvendo `Oi`, `Bebida`, `Nome`, `Casa` e `Desconhecido`. Por isso, o app possui alguns filtros de pos-processamento em tempo real, como:

- rejeicao de `Bebida` quando a mao nao esta perto da regiao da boca;
- reforco pontual para `Oi` quando a inferencia espelhada ajuda no uso da mao esquerda;
- filtros de margem/confianca para reduzir aceitacoes indevidas;
- classe `Desconhecido` para gestos fora do vocabulario.

> Este e um prototipo academico. A qualidade da leitura depende da iluminacao, enquadramento, velocidade do sinal, mao usada, distancia da camera e variedade dos videos de treinamento.

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
      |
      v
Sequencia de sinais detectados
      |
      v
RAG local em JSON + fallback por regras
      |
      v
Frase sugerida em portugues
```

---

## RAG local

A camada RAG fica nos arquivos:

| Arquivo | Funcao |
| --- | --- |
| `rag_libras.py` | Carrega a base, normaliza sinais, busca correspondencias e gera a frase final. |
| `base_libras_rag.json` | Guarda exemplos de sequencias de sinais e traducoes em portugues. |
| `frase_builder.py` | Continua existindo como fallback por regras. |

Fluxo da funcao `gerar_frase_com_rag(sinais)`:

1. Remove sinais `DESCONHECIDO` da frase, exceto quando a sequencia contem apenas desconhecido.
2. Tenta encontrar correspondencia exata em `base_libras_rag.json`.
3. Se nao encontrar, tenta uma busca aproximada simples.
4. Se ainda nao encontrar, usa `frase_builder.py`.
5. Se nenhuma regra montar frase, exibe a sequencia de sinais.

Exemplo:

```python
gerar_frase_com_rag(["EU", "QUERO", "BEBIDA"])
```

Resultado esperado:

```json
{
  "frase": "Eu quero uma bebida.",
  "fonte": "rag_exato",
  "explicacao": "Sequencia formada por sujeito, verbo de desejo e objeto."
}
```

---

## Estrutura

```text
.
|-- app.py
|-- base_libras_rag.json
|-- rag_libras.py
|-- frase_builder.py
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

Crie e ative o ambiente virtual:

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

Acesse no navegador:

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
Bebida_Articulador1.mp4
Quero_Articulador1.mp4
Eu_Articulador1.mp4
```

---

## Arquivos gerados

| Pasta | Arquivos |
| --- | --- |
| `data/` | `landmarks.csv`, `qualidade_preprocessamento.csv` |
| `models/` | `modelo_libras.keras`, `label_encoder.pkl`, `thresholds.pkl` |
| `reports/` | metricas, matriz de confusao, graficos de treino, relatorio de classificacao e auditoria |

---

## Observacoes importantes

- A camada RAG e uma demonstracao local simples, sem API externa, LangChain ou banco vetorial.
- O RAG nao altera a acuracia do modelo de visao; ele apenas melhora a interpretacao textual da sequencia ja reconhecida.
- A interface exibe a origem da frase: `RAG exato`, `RAG aproximado`, `Fallback por regras`, `Sequencia de sinais` ou `Sinal desconhecido`.
- Ao adicionar novos videos, rode novamente validacao, preprocessamento e treinamento.
- As regras de pos-processamento do `app.py` servem para reduzir confusoes em tempo real, mas nao substituem uma base de treino mais variada.

---

## Status

Versao funcional para demonstracao, com fluxo completo de:

- validacao do dataset;
- preprocessamento dos videos;
- treinamento do modelo;
- avaliacao com metricas;
- reconhecimento em tempo real pela camera;
- montagem de sequencia de sinais;
- geracao de frase com RAG local e fallback por regras.

<div align="center">

**Feito com Python, IA e um tantinho de brilho rosa.**

</div>
