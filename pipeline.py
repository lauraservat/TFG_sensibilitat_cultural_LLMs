import re
import csv
import math
from vllm import LLM, SamplingParams
import time

MODEL_NAME  = "Qwen3.5-27B"  
MODEL_PATH  = "/gpfs/projects/bsc100/models/Qwen/Qwen3.5-27B"
NUM_GPUS    = 4
TEMPERATURE = 0
MAX_TOKENS  = 1000
REPETICIONS = 10
OUTPUT_CSV  = f"dades/resultats_{MODEL_NAME}.csv"

llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=NUM_GPUS,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        enforce_eager=True,
    )
params = SamplingParams(
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    logprobs=10,
)

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

SYSTEM_BASE = """You are playing the Ultimatum Game, a typical economic game that follows these specific rules:
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
    if fase == 1 or region == "baseline":
        prefix = "You are a person"
    elif fase == 2:
        prefix = f"{GEO_LABELS[region]} You are"
    else:
        prefix = f"{CULTURAL_PROFILES[region]} You are"
    return f"{prefix} {SYSTEM_BASE}"


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
            f"A) You accept the offer. You will receive ${oferta} and the  Proposer will receive ${100 - oferta}.\n"
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


def extreure_logprobs_decisio(output) -> dict:
    buit = {"prob_A": None, "prob_B": None, "prob_decisio": None}
    if not output.outputs[0].logprobs:
        return buit
    logprobs = output.outputs[0].logprobs
    tokens_generats = []
    for pos, token_dict in enumerate(logprobs):
        token_triat = max(token_dict.items(), key=lambda x: x[1].logprob)
        token_text  = token_triat[1].decoded_token
        tokens_generats.append((pos, token_text, token_dict))
    text_acumulat = ""
    label_vist = False
    
    for pos, token_text, token_dict in tokens_generats:
        text_acumulat += token_text
        if not label_vist and "<label>" in text_acumulat.lower():
            label_vist = True
        if label_vist:
            token_upper = token_text.upper()
            te_A = "A" in token_upper and "B" not in token_upper
            te_B = "B" in token_upper and "A" not in token_upper
            if te_A or te_B:
                prob_A, prob_B = None, None
                for token_id, logprob_info in token_dict.items():
                    nomes_lletres = re.sub(r'[^A-Z]', '', logprob_info.decoded_token.upper())
                    prob = math.exp(logprob_info.logprob)
                    if nomes_lletres == "A":
                        if prob_A is None:
                            prob_A = prob
                    elif nomes_lletres == "B":
                        if prob_B is None:
                            prob_B = prob
                token_triat_decisio = max(
                    token_dict.items(), key=lambda x: x[1].logprob
                )
                prob_decisio = math.exp(token_triat_decisio[1].logprob)
                print(f"  → ASSIGNAT: prob_A={prob_A}  prob_B={prob_B}  prob_decisio={prob_decisio}")
                return {"prob_A": prob_A, "prob_B": prob_B, "prob_decisio": prob_decisio}
    return buit

# ==============================================================================
# EXECUCIÓ EN BATCH
# ==============================================================================

def executar_fase_en_batch(llm, params, system_prompt, fase, region, model_name):
    entrades = []
    prompts  = []

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

                user_prompt = user_prompt_proposer(oferta_a, oferta_b)
                prompt_complet = llm.get_tokenizer().apply_chat_template(
                    [{"role": "system", "content": system_prompt},
                     {"role": "user",   "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt_complet)
                entrades.append({
                    "model": model_name, "fase": fase, "region": region,
                    "rol": "proposer", "combinacio": nom_combinacio,
                    "ordre": ordre, "repeticio": rep,
                    "opcio_generosa": opcio_generosa,
                })

    # --- RESPONDER ---
    for oferta in OFERTES_RESPONDER:
        for rep in range(1, REPETICIONS + 1):
            for ordre in ["A", "B"]:
                user_prompt, opcio_accept = user_prompt_responder(oferta, ordre)
                prompt_complet = llm.get_tokenizer().apply_chat_template(
                    [{"role": "system", "content": system_prompt},
                     {"role": "user",   "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt_complet)
                entrades.append({
                    "model": model_name, "fase": fase, "region": region,
                    "rol": "responder", "combinacio": str(oferta),
                    "ordre": ordre, "repeticio": rep,
                    "opcio_generosa": opcio_accept,
                })

    # --- EXECUCIÓ ---
    print(f"  Executant batch de {len(prompts)} prompts...")
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    print(f"  Batch completat en {time.time() - t0:.1f}s")

    # --- PROCESSAR RESULTATS ---
    resultats = []
    for entrada, output in zip(entrades, outputs):
        resposta_raw   = output.outputs[0].text.strip()
        raonament      = extreure_raonament(resposta_raw)
        decisio_lletra = parsejar_decisio(resposta_raw)
        logprobs_info  = extreure_logprobs_decisio(output)

        if entrada["rol"] == "proposer":
            decisio       = decisio_lletra
            tria_generosa = 1 if decisio == entrada["opcio_generosa"] else 0
            opcio_gen     = entrada["opcio_generosa"]
            prob_generosa = logprobs_info.get(f"prob_{opcio_gen}")
        else:
            if decisio_lletra == entrada["opcio_generosa"]:
                decisio, tria_generosa = "accept", 1
            elif decisio_lletra in ["A", "B"]:
                decisio, tria_generosa = "reject", 0
            else:
                decisio, tria_generosa = "INVALID", 0
            opcio_acc     = entrada["opcio_generosa"]
            prob_generosa = logprobs_info.get(f"prob_{opcio_acc}")

        resultats.append({
            **entrada,
            "resposta_raw":  resposta_raw,
            "raonament":     raonament,
            "decisio":       decisio,
            "tria_generosa": tria_generosa,
            "prob_generosa": prob_generosa,
            "prob_A":        logprobs_info["prob_A"],
            "prob_B":        logprobs_info["prob_B"],
            "prob_decisio":  logprobs_info["prob_decisio"],
        })

    return resultats


# ==============================================================================
# EXECUCIÓ DE L'EXPERIMENT
# ==============================================================================

def executar_experiment():
    tots_els_resultats = []


    for fase, regions in REGIONS.items():
        for region in regions:
            system_prompt = construir_system_prompt(fase, region)
            print(f"\n{'='*60}")
            print(f"FASE {fase} | REGIÓ: {region}")
            print(f"{'='*60}")

            resultats = executar_fase_en_batch(
                llm, params, system_prompt, fase, region, MODEL_NAME
            )
            tots_els_resultats.extend(resultats)

            camps = [
                "model", "fase", "region", "rol", "combinacio",
                "ordre", "repeticio", "resposta_raw", "raonament",
                "decisio", "opcio_generosa", "tria_generosa",
                "prob_generosa", "prob_A", "prob_B", "prob_decisio",
            ]
            with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=camps)
                writer.writeheader()
                writer.writerows(tots_els_resultats)

            print(f"  {len(tots_els_resultats)} resultats guardats")

    print(f"\n Experiment completat.")


if __name__ == "__main__":
    
    executar_experiment()