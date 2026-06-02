# Tradutor de Libras com IA

Projeto para reconhecer sinais de Libras a partir de videos e da camera, usando MediaPipe para extrair landmarks e uma rede neural temporal para classificar os sinais.

## Como funciona

O projeto tem tres partes principais:

1. **Pre-processamento**
   - Le os videos em `videos/data`.
   - Usa MediaPipe para extrair pose, mao esquerda, mao direita e alguns pontos do rosto.
   - Gera sequencias com `20` frames.
   - Salva os dados em `data/landmarks.csv`.

2. **Treinamento**
   - Le `data/landmarks.csv`.
   - Treina um modelo com Conv1D, LSTM bidirecional e atencao temporal.
   - Salva o modelo, o encoder das classes, thresholds e relatorios.

3. **Aplicacao**
   - Roda uma interface Flask.
   - Captura frames da camera.
   - Aplica o mesmo pre-processamento usado no treino.
   - Mostra o sinal reconhecido e a confianca.

## Estrutura principal

```text
.
├── app.py
├── etapa2_preprocessamento.py
├── etapa3_treinamento.py
├── validar_dataset.py
├── requirements.txt
├── videos/
│   └── data/
├── data/
├── models/
├── reports/
└── front/
```

## Padrao dos videos

Os videos devem ficar em:

```text
videos/data/
```

O nome precisa seguir este formato:

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
```

Sempre que um novo sinal for adicionado, basta colocar os videos nesse padrao. O codigo reconhece novas classes automaticamente.

## Instalacao

Use Python 3.11.

No PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se o PowerShell bloquear a ativacao do ambiente virtual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\activate
```

## Validar a base de videos

Antes de rodar o pre-processamento, confira se os videos estao corretos:

```powershell
python validar_dataset.py
```

Esse script verifica:

- arquivos fora do padrao `NomeDoSinal_ArticuladorN.mp4`;
- videos que nao abrem ou estao sem frames;
- quantidade de videos por sinal;
- sinais com poucos videos.

## Rodar o projeto

Depois de organizar os videos:

```powershell
python validar_dataset.py
python etapa2_preprocessamento.py
python etapa3_treinamento.py
python app.py
```

Abra no navegador:

```text
http://localhost:5000
```

## Arquivos gerados

O pre-processamento gera:

```text
data/landmarks.csv
data/qualidade_preprocessamento.csv
```

O treinamento gera:

```text
models/modelo_libras.keras
models/label_encoder.pkl
models/thresholds.pkl
reports/metricas_teste.json
reports/relatorio_classificacao.txt
reports/confusoes.csv
reports/resultado_treinamento.png
reports/resultado_loss.png
reports/matriz_confusao.png
reports/auditoria_preprocessamento.csv
```

## Quando adicionar novos videos

Sempre que novos videos entrarem na pasta `videos/data`, rode novamente:

```powershell
python validar_dataset.py
python etapa2_preprocessamento.py
python etapa3_treinamento.py
```

Depois disso, rode a aplicacao:

```powershell
python app.py
```

## Recomendacao de quantidade

A qualidade do modelo depende bastante da quantidade e da variedade dos videos.

| Videos por sinal | Expectativa |
| --- | --- |
| menos de 10 | fraco/instavel |
| 10 a 20 | razoavel |
| 20 a 30 | bom |
| 50 ou mais | melhor generalizacao |

Tente variar pessoa, iluminacao, distancia da camera e velocidade do sinal.

## Observacoes

- O modelo atual usa `MAX_FRAMES = 20` e `N_FEATURES = 288`.
- O pre-processamento extrai pose, duas maos e pontos selecionados do rosto.
- O sinal `Desconhecido` ajuda o sistema a rejeitar movimentos que nao pertencem as classes principais.
- Se a acuracia ficar baixa, olhe primeiro os arquivos em `reports/`, principalmente `relatorio_classificacao.txt`, `confusoes.csv` e `auditoria_preprocessamento.csv`.
