"""Montagem simples de frases a partir de sinais isolados.

A ideia aqui nao e traduzir Libras continua. Esta camada so aplica uma
gramatica pequena sobre a sequencia de sinais ja confirmados pelo modelo.
Para ampliar o prototipo, cadastre novos sinais nos dicionarios abaixo.
"""

import unicodedata


def normalizar_sinal(sinal):
    texto = str(sinal or "").strip()
    sem_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return sem_acentos.upper()


def formatar_sinal(sinal):
    return str(sinal or "").strip().upper()


FRASES_FIXAS = {
    ("OI",): "Oi.",
    ("OBRIGADO",): "Obrigado.",
    ("EU", "CASA"): "Eu estou em casa.",
}

SUJEITOS = {
    "EU": "Eu",
    "VOCE": "Você",
    "ELE": "Ele",
    "ELA": "Ela",
    "NOS": "Nós",
    "VOCES": "Vocês",
}

SUJEITO_PADRAO = "EU"

ESTAR = {
    "EU": "estou",
    "VOCE": "está",
    "ELE": "está",
    "ELA": "está",
    "NOS": "estamos",
    "VOCES": "estão",
}

SER = {
    "EU": "sou",
    "VOCE": "é",
    "ELE": "é",
    "ELA": "é",
    "NOS": "somos",
    "VOCES": "são",
}

VERBOS = {
    "ACEITAR": {
        "EU": "aceito",
        "VOCE": "aceita",
        "ELE": "aceita",
        "ELA": "aceita",
        "NOS": "aceitamos",
        "VOCES": "aceitam",
    },
    "GOSTAR": {
        "EU": "gosto de",
        "VOCE": "gosta de",
        "ELE": "gosta de",
        "ELA": "gosta de",
        "NOS": "gostamos de",
        "VOCES": "gostam de",
    },
    "IR": {
        "EU": "vou para",
        "VOCE": "vai para",
        "ELE": "vai para",
        "ELA": "vai para",
        "NOS": "vamos para",
        "VOCES": "vão para",
    },
    "PRECISAR": {
        "EU": "preciso",
        "VOCE": "precisa",
        "ELE": "precisa",
        "ELA": "precisa",
        "NOS": "precisamos",
        "VOCES": "precisam",
    },
    "QUERER": {
        "EU": "quero",
        "VOCE": "quer",
        "ELE": "quer",
        "ELA": "quer",
        "NOS": "queremos",
        "VOCES": "querem",
    },
    "TER": {
        "EU": "tenho",
        "VOCE": "tem",
        "ELE": "tem",
        "ELA": "tem",
        "NOS": "temos",
        "VOCES": "tem",
    },
}

OBJETOS = {
    "AGUA": "água",
    "BANHEIRO": "banheiro",
    "BEBIDA": "uma bebida",
    "CAFE": "café",
    "CASA": "casa",
    "LIVRO": "livro",
    "NOME": "nome",
}

COMPLEMENTOS_POR_VERBO = {
    ("GOSTAR", "AGUA"): "água",
    ("GOSTAR", "BEBIDA"): "bebida",
    ("GOSTAR", "CAFE"): "café",
    ("GOSTAR", "LIVRO"): "livro",
    ("IR", "BANHEIRO"): "o banheiro",
    ("IR", "CASA"): "casa",
    ("PRECISAR", "AGUA"): "de água",
    ("PRECISAR", "BANHEIRO"): "ir ao banheiro",
    ("PRECISAR", "BEBIDA"): "de uma bebida",
    ("PRECISAR", "CAFE"): "de café",
    ("PRECISAR", "LIVRO"): "de um livro",
    ("QUERER", "AGUA"): "água",
    ("QUERER", "BANHEIRO"): "ir ao banheiro",
    ("QUERER", "BEBIDA"): "uma bebida",
    ("QUERER", "CAFE"): "café",
    ("QUERER", "LIVRO"): "um livro",
}

ESTADOS = {
    "CALMO": ("ESTAR", "calmo"),
    "SORTUDO": ("SER", "sortudo"),
}

LOCAIS = {
    "BANHEIRO": "no banheiro",
    "CASA": "em casa",
}

PALAVRAS_PT = {
    "ACEITAR": "aceitar",
    "AGUA": "água",
    "BANHEIRO": "banheiro",
    "BEBIDA": "bebida",
    "CAFE": "café",
    "CALMO": "calmo",
    "CASA": "casa",
    "EU": "eu",
    "GOSTAR": "gostar",
    "IR": "ir",
    "LIVRO": "livro",
    "NOME": "nome",
    "OBRIGADO": "obrigado",
    "OI": "oi",
    "PRECISAR": "precisar",
    "QUERER": "querer",
    "SORTUDO": "sortudo",
    "TER": "ter",
    "VOCE": "você",
}


def pontuar(frase):
    frase = " ".join(str(frase or "").split())
    if not frase:
        return ""
    return frase[:1].upper() + frase[1:] + "."


def verbo_para_sujeito(verbo, sujeito):
    conjugacoes = VERBOS.get(verbo, {})
    return conjugacoes.get(sujeito) or conjugacoes.get(SUJEITO_PADRAO) or PALAVRAS_PT.get(verbo, verbo.lower())


def auxiliar_estado(tipo, sujeito):
    conjugacoes = SER if tipo == "SER" else ESTAR
    return conjugacoes.get(sujeito) or conjugacoes[SUJEITO_PADRAO]


def montar_complemento(verbo, complementos):
    partes = []
    for sinal in complementos:
        partes.append(COMPLEMENTOS_POR_VERBO.get((verbo, sinal), OBJETOS.get(sinal, PALAVRAS_PT.get(sinal, sinal.lower()))))
    return " ".join(partes)


def montar_frase(sinais):
    """Monta uma frase sugerida usando regras reaproveitaveis por padrao."""
    sinais_normalizados = [normalizar_sinal(sinal) for sinal in sinais if str(sinal or "").strip()]
    if not sinais_normalizados:
        return ""

    chave = tuple(sinais_normalizados)
    if chave in FRASES_FIXAS:
        return FRASES_FIXAS[chave]

    if len(sinais_normalizados) == 1:
        sinal = sinais_normalizados[0]
        return pontuar(PALAVRAS_PT.get(sinal, sinal.lower()))

    primeiro = sinais_normalizados[0]

    if primeiro in SUJEITOS:
        sujeito = primeiro
        restantes = sinais_normalizados[1:]

        if len(restantes) == 1 and restantes[0] in ESTADOS:
            tipo_estado, texto_estado = ESTADOS[restantes[0]]
            return pontuar(f"{SUJEITOS[sujeito]} {auxiliar_estado(tipo_estado, sujeito)} {texto_estado}")

        if len(restantes) == 1 and restantes[0] in LOCAIS:
            return pontuar(f"{SUJEITOS[sujeito]} {auxiliar_estado('ESTAR', sujeito)} {LOCAIS[restantes[0]]}")

        if restantes and restantes[0] in VERBOS:
            verbo = restantes[0]
            complemento = montar_complemento(verbo, restantes[1:])
            return pontuar(f"{SUJEITOS[sujeito]} {verbo_para_sujeito(verbo, sujeito)} {complemento}")

    if primeiro in VERBOS:
        verbo = primeiro
        complemento = montar_complemento(verbo, sinais_normalizados[1:])
        return pontuar(f"{verbo_para_sujeito(verbo, SUJEITO_PADRAO)} {complemento}")

    palavras = [PALAVRAS_PT.get(sinal, sinal.lower()) for sinal in sinais_normalizados]
    return pontuar(" ".join(palavras))
