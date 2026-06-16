# Tradutor de Libras com IA

Aplicacao web para reconhecer sinais de Libras em tempo real pela camera. O projeto usa visao computacional para extrair pontos do corpo, maos e rosto, e uma rede neural temporal para classificar o gesto realizado.

O objetivo e demonstrar um prototipo funcional de traducao de sinais para palavras, com interface simples, historico de leituras e tratamento para gestos fora do vocabulario treinado.

## Visao geral

- Reconhecimento em tempo real pela webcam.
- Extracao de landmarks com MediaPipe.
- Classificacao temporal com rede neural treinada em videos.
- Interface web feita com Flask, HTML, CSS e JavaScript.
- Classe `Desconhecido` para reduzir classificacoes indevidas.
- Relatorios de treino e avaliacao gerados automaticamente.

## Tecnologias usadas

- Python 3.11
- Flask
- TensorFlow/Keras
- MediaPipe
- OpenCV
- NumPy, Pandas e Scikit-learn
- HTML, CSS e JavaScript

## Sinais reconhecidos

O modelo atual trabalha com as seguintes classes:

| Classe | Videos |
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

## Como funciona

1. Os videos sao armazenados em `videos/data`.
2. O script de preprocessamento extrai landmarks de pose, maos e rosto com MediaPipe.
3. As sequencias extraidas sao transformadas em janelas temporais de 20 frames.
4. O modelo neural aprende a classificar cada sequencia.
5. A aplicacao Flask usa a camera em tempo real e aplica o mesmo fluxo de extracao.
6. A interface mostra a palavra reconhecida, historico e sinais disponiveis.

## Resultado atual

Ultima avaliacao registrada:

- Acuracia no teste: **67.03%**
- Acuracia balanceada: **67.79%**
- Amostras de teste: **543**

Algumas classes tiveram melhor desempenho, como `Banheiro`, `Bebida`, `Calmo`, `Casa`, `Obrigado` e `Sortudo`. A classe `Desconhecido` foi reforcada para ajudar o sistema a rejeitar gestos que nao fazem parte do vocabulario.

Como o dataset ainda e pequeno para algumas classes, o modelo pode confundir sinais parecidos ou variar conforme iluminacao, enquadramento e velocidade do gesto.

## Estrutura do projeto

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

## Como executar

Crie e ative o ambiente virtual:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Valide a base de videos:

```powershell
python validar_dataset.py
```

Rode o preprocessamento e o treinamento:

```powershell
python etapa2_preprocessamento.py
python etapa3_treinamento.py
```

Inicie a aplicacao:

```powershell
python app.py
```

Acesse no navegador:

```text
http://localhost:5000
```

## Padrao dos videos

Os videos devem ficar em `videos/data` e seguir o formato:

```text
NomeDoSinal_Articulador1.mp4
NomeDoSinal_Articulador2.mp4
NomeDoSinal_Articulador3.mp4
```

Exemplo:

```text
Oi_Articulador1.mp4
Casa_Articulador2.mp4
Obrigado_Articulador3.mp4
```

## Arquivos gerados

Durante o preprocessamento:

- `data/landmarks.csv`
- `data/qualidade_preprocessamento.csv`

Durante o treinamento:

- `models/modelo_libras.keras`
- `models/label_encoder.pkl`
- `models/thresholds.pkl`
- `reports/metricas_teste.json`
- `reports/relatorio_classificacao.txt`
- `reports/confusoes.csv`
- `reports/matriz_confusao.png`
- `reports/resultado_treinamento.png`
- `reports/resultado_loss.png`

## Limitacoes

Este projeto e um prototipo academico. O desempenho depende diretamente da variedade e qualidade dos videos usados no treinamento. Para melhorar a generalizacao, o ideal e aumentar o dataset com mais pessoas, iluminacoes, distancias da camera e exemplos de gestos desconhecidos.

## Status

Projeto em versao funcional para demonstracao, com pipeline completo de validacao, preprocessamento, treinamento, avaliacao e uso em tempo real.
