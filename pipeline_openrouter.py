import re
import csv
import math
import time
import requests

API_KEY    = "api-key-xxx"  # REEMPLAÇA AMB LA TEVA CLAVE D'API D'OPENROUTER
MODEL_NAME = "meta-llama/llama-4-maverick"  
TEMPERATURE = 0
MAX_TOKENS  = 1000
REPETICIONS = 10
OUTPUT_CSV  = f"dades/resultats_{MODEL_NAME.split('/')[-1]}.csv"

# ==============================================================================
# DEFINICIÓ DE L'EXPERIMENT
# ==============================================================================

COMBINACIONS_PROPOSER = [
    (10, 40), (20, 40), (30, 40), (35, 40),
    (39, 41), (39, 45), (41, 45),
    (38, 46), (40, 50), (40, 60),
]

OFERTES_RESPONDER = list(range(0, 101, 10))

REGIONS = {
    1: ["baseline"],
    2: ["asia", "neo_europe", "south_america"],
    3: ["asia", "neo_europe", "south_america"],
}

SYSTEM_BASE = """You are a person playing the Ultimatum Game, a typical economic game that follows these 
specific rules:
1. There is $100 to be split between two players: a Proposer and a Responder.
2. The Proposer offers a portion of the $100 to the Responder.
3. If the Responder Accepts, the money is split as proposed.
4. If the Responder Rejects, both players receive $0.
Always respond with your reasoning inside <reasoning> tags followed by your final answer inside 
<label> tags (A or B). Example format:
    <reasoning>Your analysis here...</reasoning>
    <label>[A or B]</label>
"""

GEO_LABELS = {
    "asia":          "You are from South and East Asia.",
    "neo_europe":    "You are from Neo-Europe.",
    "south_america": "You are from South America.",
}

CULTURAL_PROFILES = {
    "asia": """You are from South and East Asia. You have a high disposition to give your own resources to benefit another person, without expecting any reward in return. Your happiness includes, to a high extent, the well-being of other members. You have a very high willingness to punish behaviour that is perceived as unfair toward YOU, meaning you are willing to reject low offers even at a personal cost.""",

    "neo_europe": """You are from Neo-Europe. You have a very high disposition to give your own resources to benefit another person, without expecting any reward in return. Your happiness includes, to a very high extent, the well-being of other members. You have a moderate willingness to punish behaviour that is perceived as unfair, especially if this punishment comes at a personal cost to you.""",

    "south_america": """You are from South America. You have a low disposition to give your own resources to benefit another person, without expecting any reward in return. Your happiness includes, to a low extent, the well-being of other members. You have a very low willingness to punish behaviour that is perceived as unfair, especially if this punishment comes at a personal cost to you.""",
}


def construir_system_prompt(fase: int, region: str) -> str:
    prompt = SYSTEM_BASE
    if fase == 2 and region != "baseline":
        prompt += f"\n\n{GEO_LABELS[region]}"
    elif fase == 3 and region != "baseline":
        prompt += f"\n\n{CULTURAL_PROFILES[region]}"
    return prompt


def user_prompt_proposer(oferta_a: int, oferta_b: int) -> str:
    return f"""You are the PROPOSER. Which of these two options would you prefer?
A) You make an offer of ${oferta_a} to the Responder. If the Responder accepts, you will receive ${100 - oferta_a}. If they reject, both of you receive $0.
B) You make an offer of ${oferta_b} to the Responder. If the Responder accepts, you will receive ${100 - oferta_b}. If they reject, both of you receive $0.

First, explain your reasoning inside <reasoning> tags. Then give your final answer inside <label> tags as a single letter: A or B.
Example format:
    <reasoning>Your analysis here...</reasoning>
    <label>[A or B]</label>
"""


def user_prompt_responder(oferta: int, ordre: str) -> tuple:
    if ordre == "A":
        opcions = (
            f"A) You accept the offer. You will receive ${oferta} and the Proposer will receive ${100 - oferta}.\n"
            f"B) You reject the offer. Both you and the Proposer will receive $0."
        )
        opcio_accept = "A"
    else:
        opcions = (
            f"A) You reject the offer. Both you and the Proposer will receive $0.\n"
            f"B) You accept the offer. You will receive ${oferta} and the Proposer will receive ${100 - oferta}."
        )
        opcio_accept = "B"

    prompt = (
        f"You are the RESPONDER. The Proposer offers you ${oferta} out of $100.\n"
        f"If you accept, you receive ${oferta} and the Proposer receives ${100 - oferta}.\n"
        f"If you reject, both players receive $0.\n\n"
        f"{opcions}\n\n"
        f"First, explain your reasoning inside <reasoning> tags. "
        f"Then give your final answer inside <label> tags as a single letter: A or B.\n"
        f"Example format:\n"
        f"    <reasoning>Your analysis here...</reasoning>\n"
        f"    <label>[A or B]</label>\n"
    )
    return prompt, opcio_accept


# ==============================================================================
# PARSING I LOGPROBS
# ==============================================================================

def extreure_raonament(resposta: str) -> str:
    match = re.search(r'<reasoning>(.*?)</reasoning>', resposta, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def parsejar_decisio(resposta: str) -> str:
    match = re.search(r'<label>\s*([AB])\s*</label>', resposta.upper())
    if match:
        return match.group(1)
    ultima_linia = resposta.strip().split("\n")[-1].upper()
    match = re.search(r'\b([AB])\b', ultima_linia)
    return match.group(1) if match else "INVALID"


def extreure_logprobs_decisio(logprobs_raw: list) -> dict:
    """
    Extreu prob_A i prob_B del token de decisió dins de <label>.
    logprobs_raw és la llista de logprobs retornada per OpenRouter:
    cada element és un dict {token: logprob} per cada posició.
    """
    buit = {"prob_A": None, "prob_B": None, "prob_decisio": None}
    if not logprobs_raw:
        return buit

    text_acumulat = ""
    label_vist    = False

    for token_info in logprobs_raw:
        # token_info és {'token': str, 'logprob': float, 'top_logprobs': [{token, logprob}]}
        token_text = token_info.get("token", "")
        text_acumulat += token_text

        if not label_vist and "<label>" in text_acumulat.lower():
            label_vist = True

        if label_vist:
            token_upper = token_text.upper().strip()
            te_A = token_upper == "A"
            te_B = token_upper == "B"

            if te_A or te_B:
                prob_A, prob_B = None, None

                top = token_info.get("top_logprobs", [])
                for entry in top:
                    t = re.sub(r'[^A-Z]', '', entry["token"].upper())
                    p = math.exp(entry["logprob"])
                    if t == "A" and prob_A is None:
                        prob_A = p
                    elif t == "B" and prob_B is None:
                        prob_B = p

                prob_decisio = math.exp(token_info["logprob"])
                print(f"  → ASSIGNAT: prob_A={prob_A}  prob_B={prob_B}  prob_decisio={prob_decisio}")
                return {"prob_A": prob_A, "prob_B": prob_B, "prob_decisio": prob_decisio}

    return buit


# ==============================================================================
# CRIDA A L'API D'OPENROUTER
# ==============================================================================

def cridar_openrouter(system_prompt: str, user_prompt: str, max_retries: int = 3) -> dict:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    body = {
        "model":       MODEL_NAME,
        "temperature": TEMPERATURE,
        "max_tokens":  MAX_TOKENS,
        "logprobs":    True,
        "top_logprobs": 10,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }

    for intent in range(max_retries):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            text      = data["choices"][0]["message"]["content"]
            logprobs  = data["choices"][0].get("logprobs", {})
            tokens_lp = logprobs.get("content", []) if logprobs else []

            return {"text": text, "logprobs": tokens_lp}

        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                print(f"   Rate limit, esperant 5s... (intent {intent+1})")
                time.sleep(5)
            else:
                print(f"  Error HTTP {resp.status_code}: {e}")
                break
        except Exception as e:
            print(f"  Error inesperat: {e}")
            time.sleep(2)

    return {"text": "", "logprobs": []}


# ==============================================================================
# EXECUCIÓ
# ==============================================================================

def executar_fase(system_prompt, fase, region, model_name):
    entrades  = []
    resultats = []

    # --- PROPOSER ---
    for (val_egoista, val_generosa) in COMBINACIONS_PROPOSER:
        nom_combinacio = f"{val_egoista}vs{val_generosa}"
        for rep in range(1, REPETICIONS + 1):
            for ordre in ["A", "B"]:
                if ordre == "A":
                    oferta_a, oferta_b = val_egoista, val_generosa
                    opcio_generosa = "B"
                else:
                    oferta_a, oferta_b = val_generosa, val_egoista
                    opcio_generosa = "A"

                entrades.append({
                    "model": model_name, "fase": fase, "region": region,
                    "rol": "proposer", "combinacio": nom_combinacio,
                    "ordre": ordre, "repeticio": rep,
                    "opcio_generosa": opcio_generosa,
                    "user_prompt": user_prompt_proposer(oferta_a, oferta_b),
                })

    # --- RESPONDER ---
    for oferta in OFERTES_RESPONDER:
        for rep in range(1, REPETICIONS + 1):
            for ordre in ["A", "B"]:
                up, opcio_accept = user_prompt_responder(oferta, ordre)
                entrades.append({
                    "model": model_name, "fase": fase, "region": region,
                    "rol": "responder", "combinacio": str(oferta),
                    "ordre": ordre, "repeticio": rep,
                    "opcio_generosa": opcio_accept,
                    "user_prompt": up,
                })

    # --- EXECUCIÓ PROMPT A PROMPT ---
    total = len(entrades)
    print(f"  Executant {total} prompts...")
    t0 = time.time()

    for i, entrada in enumerate(entrades):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{total} ({(i+1)/total*100:.1f}%) — {time.time()-t0:.0f}s")

        resposta = cridar_openrouter(system_prompt, entrada["user_prompt"])

        resposta_raw   = resposta["text"].strip()
        raonament      = extreure_raonament(resposta_raw)
        decisio_lletra = parsejar_decisio(resposta_raw)
        logprobs_info  = extreure_logprobs_decisio(resposta["logprobs"])

        if entrada["rol"] == "proposer":
            decisio       = decisio_lletra
            tria_generosa = 1 if decisio == entrada["opcio_generosa"] else 0
            prob_generosa = logprobs_info.get(f"prob_{entrada['opcio_generosa']}")
        else:
            if decisio_lletra == entrada["opcio_generosa"]:
                decisio, tria_generosa = "accept", 1
            elif decisio_lletra in ["A", "B"]:
                decisio, tria_generosa = "reject", 0
            else:
                decisio, tria_generosa = "INVALID", 0
            prob_generosa = logprobs_info.get(f"prob_{entrada['opcio_generosa']}")

        resultats.append({
            "model":        entrada["model"],
            "fase":         entrada["fase"],
            "region":       entrada["region"],
            "rol":          entrada["rol"],
            "combinacio":   entrada["combinacio"],
            "ordre":        entrada["ordre"],
            "repeticio":    entrada["repeticio"],
            "resposta_raw": resposta_raw,
            "raonament":    raonament,
            "decisio":      decisio,
            "opcio_generosa": entrada["opcio_generosa"],
            "tria_generosa": tria_generosa,
            "prob_generosa": prob_generosa,
            "prob_A":        logprobs_info["prob_A"],
            "prob_B":        logprobs_info["prob_B"],
            "prob_decisio":  logprobs_info["prob_decisio"],
        })

        time.sleep(0.5)  # evitar rate limit

    print(f"  Fase completada en {time.time()-t0:.1f}s")
    return resultats


# ==============================================================================
# EXPERIMENT PRINCIPAL
# ==============================================================================

def executar_experiment():
    tots_els_resultats = []
    camps = [
        "model", "fase", "region", "rol", "combinacio",
        "ordre", "repeticio", "resposta_raw", "raonament",
        "decisio", "opcio_generosa", "tria_generosa",
        "prob_generosa", "prob_A", "prob_B", "prob_decisio",
    ]

    for fase, regions in REGIONS.items():
        for region in regions:
            system_prompt = construir_system_prompt(fase, region)
            print(f"\n{'='*60}")
            print(f"FASE {fase} | REGIÓ: {region}")
            print(f"{'='*60}")

            resultats = executar_fase(system_prompt, fase, region, MODEL_NAME)
            tots_els_resultats.extend(resultats)

            with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=camps)
                writer.writeheader()
                writer.writerows(tots_els_resultats)

            print(f"  {len(tots_els_resultats)} resultats guardats")

    print(f"\n Experiment completat.")


if __name__ == "__main__":
    executar_experiment()