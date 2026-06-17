"""RAG local simples para sugerir frases a partir de sinais reconhecidos.

Esta camada nao altera a acuracia do modelo de visao. Ela apenas consulta uma
base JSON local com exemplos de sequencias validas e usa o frase_builder como
fallback quando nao encontra uma correspondencia confiavel.
"""

import json
import os
import unicodedata

from frase_builder import montar_frase


BASE_RAG_PATH = os.path.join(os.path.dirname(__file__), "base_libras_rag.json")

SINAIS_PERMITIDOS = {
    "ACEITAR",
    "BANHEIRO",
    "BEBIDA",
    "CALMO",
    "CASA",
    "DESCONHECIDO",
    "EU",
    "LIVRO",
    "NOME",
    "OBRIGADO",
    "OI",
    "QUERO",
    "SORTUDO",
}

SINAIS_REJEICAO = {"DESCONHECIDO"}
MIN_SCORE_APROXIMADO = 0.74
MIN_SCORE_APROXIMADO_COM_SINAL_FALTANDO = 0.90


def normalizar_sinal(sinal):
    texto = " ".join(str(sinal or "").strip().split())
    sem_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return sem_acentos.upper()


def normalizar_sinais(sinais):
    return [normalizar_sinal(sinal) for sinal in sinais if normalizar_sinal(sinal)]


def limpar_sinais_para_frase(sinais):
    sinais_norm = [
        sinal for sinal in normalizar_sinais(sinais)
        if sinal in SINAIS_PERMITIDOS
    ]
    sem_rejeicao = [sinal for sinal in sinais_norm if sinal not in SINAIS_REJEICAO]
    return sem_rejeicao if sem_rejeicao else sinais_norm


def carregar_base_rag(caminho=BASE_RAG_PATH):
    if not os.path.exists(caminho):
        return []

    try:
        with open(caminho, "r", encoding="utf-8") as arquivo:
            dados = json.load(arquivo)
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(dados, list):
        return []

    exemplos = []
    for item in dados:
        if not isinstance(item, dict):
            continue
        padrao = limpar_sinais_para_frase(item.get("padrao_sinais", []))
        traducao = str(item.get("traducao_pt", "")).strip()
        if not padrao or not traducao:
            continue
        exemplo = dict(item)
        exemplo["padrao_sinais"] = padrao
        exemplos.append(exemplo)
    return exemplos


def buscar_correspondencia_exata(sinais, base=None):
    sinais_limpos = limpar_sinais_para_frase(sinais)
    if not sinais_limpos:
        return None

    exemplos = carregar_base_rag() if base is None else base
    for exemplo in exemplos:
        if exemplo.get("padrao_sinais") == sinais_limpos:
            return exemplo
    return None


def contem_em_ordem(seq_detectada, seq_exemplo):
    pos = 0
    for sinal in seq_detectada:
        if pos < len(seq_exemplo) and sinal == seq_exemplo[pos]:
            pos += 1
    return pos == len(seq_exemplo)


def maior_subsequencia_comum(a, b):
    tabela = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, sinal_a in enumerate(a, start=1):
        for j, sinal_b in enumerate(b, start=1):
            if sinal_a == sinal_b:
                tabela[i][j] = tabela[i - 1][j - 1] + 1
            else:
                tabela[i][j] = max(tabela[i - 1][j], tabela[i][j - 1])
    return tabela[-1][-1]


def pontuar_aproximacao(sinais, padrao):
    if not sinais or not padrao:
        return 0.0

    intersecao = len(set(sinais) & set(padrao))
    uniao = len(set(sinais) | set(padrao))
    score_intersecao = intersecao / max(uniao, 1)

    lcs = maior_subsequencia_comum(sinais, padrao)
    score_ordem = lcs / max(len(padrao), 1)

    score_tamanho = min(len(sinais), len(padrao)) / max(len(sinais), len(padrao), 1)
    bonus_mesmo_inicio = 0.12 if sinais[0] == padrao[0] else 0.0
    bonus_subsequencia = 0.12 if contem_em_ordem(sinais, padrao) or contem_em_ordem(padrao, sinais) else 0.0

    return (0.42 * score_intersecao) + (0.34 * score_ordem) + (0.12 * score_tamanho) + bonus_mesmo_inicio + bonus_subsequencia


def buscar_correspondencia_aproximada(sinais, base=None):
    sinais_limpos = limpar_sinais_para_frase(sinais)
    if not sinais_limpos:
        return None

    exemplos = carregar_base_rag() if base is None else base
    melhor = None
    melhor_score = 0.0

    for exemplo in exemplos:
        padrao = exemplo.get("padrao_sinais", [])
        score = pontuar_aproximacao(sinais_limpos, padrao)
        threshold = (
            MIN_SCORE_APROXIMADO_COM_SINAL_FALTANDO
            if len(sinais_limpos) < len(padrao)
            else MIN_SCORE_APROXIMADO
        )
        if score < threshold:
            continue
        if score > melhor_score:
            melhor = exemplo
            melhor_score = score

    if melhor:
        resultado = dict(melhor)
        resultado["score_aproximacao"] = round(melhor_score, 3)
        return resultado
    return None


def _fallback_por_regras(sinais_limpos):
    frase = montar_frase(sinais_limpos)
    if frase:
        return {
            "frase": frase,
            "fonte": "fallback_regras",
            "exemplo_recuperado": None,
            "explicacao": "Frase gerada pelas regras locais do frase_builder.py.",
        }

    texto = " | ".join(sinais_limpos)
    return {
        "frase": texto,
        "fonte": "fallback_sinais",
        "exemplo_recuperado": None,
        "explicacao": "Nao foi possivel montar uma frase; exibindo a sequencia de sinais.",
    }


def gerar_frase_com_rag(sinais):
    sinais_limpos = limpar_sinais_para_frase(sinais)

    if not sinais_limpos:
        return {
            "frase": "",
            "fonte": "sem_sinais",
            "exemplo_recuperado": None,
            "explicacao": "",
        }

    if all(sinal in SINAIS_REJEICAO for sinal in sinais_limpos):
        return {
            "frase": "Sinal nao reconhecido.",
            "fonte": "sinal_desconhecido",
            "exemplo_recuperado": None,
            "explicacao": "A sequencia contem apenas sinal desconhecido.",
        }

    base = carregar_base_rag()
    exemplo_exato = buscar_correspondencia_exata(sinais_limpos, base)
    if exemplo_exato:
        return {
            "frase": exemplo_exato["traducao_pt"],
            "fonte": "rag_exato",
            "exemplo_recuperado": exemplo_exato,
            "explicacao": exemplo_exato.get("explicacao", ""),
        }

    exemplo_aproximado = buscar_correspondencia_aproximada(sinais_limpos, base)
    if exemplo_aproximado:
        return {
            "frase": exemplo_aproximado["traducao_pt"],
            "fonte": "rag_aproximado",
            "exemplo_recuperado": exemplo_aproximado,
            "explicacao": exemplo_aproximado.get("explicacao", ""),
        }

    return _fallback_por_regras(sinais_limpos)
