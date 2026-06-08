"""
analysis.py
-----------
Llegeix el CSV generat per pipeline.py i calcula totes les mètriques
definides a la metodologia:

ROL RESPONDER:
  - RMSE vs baseline (mètrica principal)
  - KL-divergència vs baseline (addicional, renormalitzada)
  - Delta mitjà (direcció del desplaçament)
  - Punt de Ruptura (PR)
  - Verificació de monotonia (per magnitud total de violacions)
  - Verificació de position bias

ROL PROPOSER:
  - RMSE vs baseline (mètrica principal)
  - KL-divergència vs baseline (addicional, renormalitzada)
  - Delta mitjà vs baseline
  - p_(e,g) per a les combinacions 5-8 (comparació directa entre regions)
  - Verificació de coherència a les combinacions de control (valors en brut)
  - Verificació de position bias

H1:
  - PR i oferta mitjana del baseline en absolut
  - RMSE del baseline vs cada perfil de la Fase 3

H3:
  - Correlació de Spearman entre paràmetres actius i RMSE/plasticitat
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
import os
import warnings
warnings.filterwarnings("ignore")


# ==============================================================================
# CONFIGURACIÓ
# ==============================================================================

MODEL_NAME = "claude-haiku-4.5"  
INPUT_CSV  = f"resultats/resultats_{MODEL_NAME}.csv"

OFERTES = list(range(0, 101, 10))

COMBINACIONS_CONTROL_INF = ["10vs40", "20vs40", "30vs40", "35vs40"]
COMBINACIONS_CRITIQUES   = ["39vs41", "39vs45", "41vs45", "38vs46"]
COMBINACIONS_CONTROL_SUP = ["40vs50", "40vs60"]
TOTES_COMBINACIONS       = (COMBINACIONS_CONTROL_INF +
                             COMBINACIONS_CRITIQUES +
                             COMBINACIONS_CONTROL_SUP)

CONDICIONS = [
    (1, "baseline"),
    (2, "asia"), (2, "neo_europe"), (2, "south_america"),
    (3, "asia"), (3, "neo_europe"), (3, "south_america"),
]

PARAMETRES_ACTIUS = {
    "Mistral-7B-Instruct-v0.3":    7,
    "Mistral-Small-24B-Instruct":  24,
    "Llama-3.1-8B-Instruct":       8,
    "Llama4-Scout":                17,
    "Llama4-Maverick":             17,
    "Qwen3-32B":                   32,
    "Qwen3-235B-A22B":             22,
    "Qwen3.5-27B":                 27,
    "Qwen3.5-122B-A10B":           10,
    "DeepSeek-V3.2":               37,
    "GPT-5-mini":                  None,
    "Gemini-2.5-Pro":              None,
    "Claude-Haiku-4.5":            None,
}

ANY_LLANCAMENT = {
    "Mistral-7B-Instruct-v0.3":    2023,
    "Mistral-Small-24B-Instruct":  2025,
    "Llama-3.1-8B-Instruct":       2024,
    "Llama4-Scout":                2025,
    "Llama4-Maverick":             2025,
    "Qwen3-32B":                   2025,
    "Qwen3-235B-A22B":             2025,
    "Qwen3.5-27B":                 2026,
    "Qwen3.5-122B-A10B":           2026,
    "DeepSeek-V3.2":               2025,
    "GPT-5-mini":                  2025,
    "Gemini-2.5-Pro":              2025,
    "Claude-Haiku-4.5":            2025,
}

PR_GLOBAL_OOSTERBEEK     = 20.0
OFERTA_GLOBAL_OOSTERBEEK = 40.0
LLINDAR_MAGNITUD_MONO    = 0.2

# Configuració visual
COLORS = {
    "baseline":      "black",
    "asia":          "red",
    "neo_europe":    "blue",
    "south_america": "green",
}
LABELS = {
    "baseline":      "Baseline",
    "asia":          "Asia",
    "neo_europe":    "Neo-Europe",
    "south_america": "South America",
}
GROUND_TRUTH_PR = {
    "baseline":      20.0,
    "asia":          23.0,
    "neo_europe":    20.0,
    "south_america": 17.0,
}


# ==============================================================================
# FUNCIONS AUXILIARS
# ==============================================================================

def rmse(p, q):
    """RMSE entre dues distribucions. Mètrica principal."""
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    return float(np.sqrt(np.mean((p - q) ** 2)))


def kl_divergence(p, q, epsilon=1e-9):
    """
    KL-divergència D_KL(P || Q) amb suavitzat i renormalització.
    Mètrica addicional — no és la mètrica principal perquè requereix
    renormalitzar vectors que no són distribucions de probabilitat estrictes.
    """
    p = np.array(p, dtype=float) + epsilon
    q = np.array(q, dtype=float) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def delta_mitja(p, q):
    """Diferència mitjana entre dues distribucions (direcció del desplaçament)."""
    return float(np.mean(np.array(p) - np.array(q)))


def sigmoide(x, pr, pendent):
    return 1 / (1 + np.exp(-pendent * (x - pr)))


def calcular_punt_ruptura(ofertes, taxes):
    for o, t in zip(ofertes, taxes):
        if not np.isnan(t) and t >= 0.5:
            return float(o)
    return None


def calcular_magnitud_violacions(taxes, epsilon=0.05):
    """
    Suma la magnitud de les violacions de monotonia.
    Una violació de 0.06 contribueix 0.01, una de 0.40 contribueix 0.35.
    """
    return sum(
        max(0, taxes[i] - taxes[i + 1] - epsilon)
        for i in range(len(taxes) - 1)
        if not np.isnan(taxes[i]) and not np.isnan(taxes[i + 1])
    )

def test_permutacions_correcte(df_rol, model, region, fase, baseline_region="baseline",
                                baseline_fase=1, combinacions=None, ofertes=None,
                                rol="responder", n_permutacions=1000):
    """
    Test de permutacions sobre observacions individuals.
    Compara RMSE d'una condició vs baseline.
    """
    if rol == "responder":
        keys = ["oferta"]
        vals_base = []
        vals_cond = []
        for o in ofertes:
            b = df_rol[
                (df_rol["model"] == model) &
                (df_rol["fase"] == baseline_fase) &
                (df_rol["region"] == baseline_region) &
                (df_rol["oferta"] == o)
            ]["tria_generosa"].values
            c = df_rol[
                (df_rol["model"] == model) &
                (df_rol["fase"] == fase) &
                (df_rol["region"] == region) &
                (df_rol["oferta"] == o)
            ]["tria_generosa"].values
            vals_base.append(b)
            vals_cond.append(c)

    else:  # proposer
        vals_base = []
        vals_cond = []
        for comb in combinacions:
            b = df_rol[
                (df_rol["model"] == model) &
                (df_rol["fase"] == baseline_fase) &
                (df_rol["region"] == baseline_region) &
                (df_rol["combinacio"] == comb)
            ]["tria_generosa"].values
            c = df_rol[
                (df_rol["model"] == model) &
                (df_rol["fase"] == fase) &
                (df_rol["region"] == region) &
                (df_rol["combinacio"] == comb)
            ]["tria_generosa"].values
            vals_base.append(b)
            vals_cond.append(c)

    # Freqüències reals
    freq_base = np.array([v.mean() for v in vals_base])
    freq_cond = np.array([v.mean() for v in vals_cond])

    # Comprova que no hi ha NaNs
    if np.any(np.isnan(freq_base)) or np.any(np.isnan(freq_cond)):
        return None

    rmse_real = rmse(freq_cond, freq_base)

    # Permutacions sobre observacions individuals
    deltas_perm = []
    for _ in range(n_permutacions):
        freq_perm_base = []
        freq_perm_cond = []
        for i in range(len(vals_base)):
            if len(vals_base[i]) == 0 or len(vals_cond[i]) == 0:
                freq_perm_base.append(np.nan)
                freq_perm_cond.append(np.nan)
                continue
            totes = np.concatenate([vals_base[i], vals_cond[i]])
            np.random.shuffle(totes)
            n = len(vals_base[i])
            freq_perm_base.append(totes[:n].mean())
            freq_perm_cond.append(totes[n:].mean())

        freq_perm_base = np.array(freq_perm_base)
        freq_perm_cond = np.array(freq_perm_cond)

        if np.any(np.isnan(freq_perm_base)) or np.any(np.isnan(freq_perm_cond)):
            continue
        deltas_perm.append(rmse(freq_perm_cond, freq_perm_base))

    if len(deltas_perm) == 0:
        return None

    p_valor = np.mean(np.array(deltas_perm) >= rmse_real)
    return round(p_valor, 4)


def calcular_pvalors_correctes(df_resp, df_prop, n_permutacions=1000):
    """
    Calcula p-valors per a cada (model, fase, regió) per als dos rols.
    """
    OFERTES = list(range(0, 101, 10))
    TOTES_COMBINACIONS = ["10vs40", "20vs40", "30vs40", "35vs40",
                          "39vs41", "39vs45", "41vs45", "38vs46",
                          "40vs50", "40vs60"]

    resultats_resp = []
    resultats_prop = []

    models = df_resp["model"].unique()

    for model in models:
        for fase in [2, 3]:
            for region in ["asia", "neo_europe", "south_america"]:

                # Responent
                p_resp = test_permutacions_correcte(
                    df_resp, model, region, fase,
                    combinacions=None, ofertes=OFERTES,
                    rol="responder", n_permutacions=n_permutacions
                )
                resultats_resp.append({
                    "model": model, "fase": fase,
                    "region": region, "p_valor": p_resp
                })

                # Proposant
                p_prop = test_permutacions_correcte(
                    df_prop, model, region, fase,
                    combinacions=TOTES_COMBINACIONS, ofertes=None,
                    rol="proposer", n_permutacions=n_permutacions
                )
                resultats_prop.append({
                    "model": model, "fase": fase,
                    "region": region, "p_valor": p_prop
                })

    df_pvalors_resp = pd.DataFrame(resultats_resp)
    df_pvalors_prop = pd.DataFrame(resultats_prop)

    print("\n P-VALORS — RESPONDER:")
    print(df_pvalors_resp.to_string(index=False))
    print("\n P-VALORS — PROPOSER:")
    print(df_pvalors_prop.to_string(index=False))

    return df_pvalors_resp, df_pvalors_prop

# ==============================================================================
# CÀRREGA I PREPROCESSAMENT
# ==============================================================================

def carregar_dades(path):
    df = pd.read_csv(path)
    df["model"] = df["model"].str.split("/").str[-1] 
    invalids = df[df["decisio"] == "INVALID"]
    if len(invalids) > 0:
        print(f" S'han eliminat {len(invalids)} respostes invàlides "
              f"({len(invalids)/len(df)*100:.1f}%)")
        df = df[df["decisio"] != "INVALID"].copy()

    df["prob_gen_robusta"] = df.apply(prob_generosa_robusta, axis=1)

    df_resp = df[df["rol"] == "responder"].copy()
    df_resp["oferta"] = pd.to_numeric(df_resp["combinacio"])  # ← aquí
    df_prop = df[df["rol"] == "proposer"].copy()

    return df, df_resp, df_prop

def prob_generosa_robusta(row):
    if row["opcio_generosa"] == "A":
        p_gen = row["prob_A"]
        p_alt = row["prob_B"]
    else:
        p_gen = row["prob_B"]
        p_alt = row["prob_A"]

    if pd.notna(p_gen) and pd.notna(p_alt):
        total = p_gen + p_alt
        return p_gen / total        # renormalitzem entre les dues opcions
    elif pd.notna(p_gen):
        return p_gen                # només tenim una, usem directament
    else:
        return 0.0                  # no estava al top-10, molt improbable

# ==============================================================================
# POSITION BIAS
# ==============================================================================

def analitzar_position_bias(df_resp, df_prop):
    resultats = []

    for (model, fase, region, oferta), grup in df_resp.groupby(
        ["model", "fase", "region", "oferta"]
    ):
        ordre_a = grup[grup["ordre"] == "A"]["tria_generosa"].mean()
        ordre_b = grup[grup["ordre"] == "B"]["tria_generosa"].mean()
        diff    = abs(ordre_a - ordre_b)
        resultats.append({
            "rol": "responder", "model": model, "fase": fase,
            "region": region, "combinacio": str(int(oferta)),
            "p_generosa_A": round(ordre_a, 3),
            "p_generosa_B": round(ordre_b, 3),
            "diff_posicio": round(diff, 3),
            "bias_detectat": diff > 0.3,
        })

    for (model, fase, region, combinacio), grup in df_prop.groupby(
        ["model", "fase", "region", "combinacio"]
    ):
        ordre_a = grup[grup["ordre"] == "A"]["tria_generosa"].mean()
        ordre_b = grup[grup["ordre"] == "B"]["tria_generosa"].mean()
        diff    = abs(ordre_a - ordre_b)
        resultats.append({
            "rol": "proposer", "model": model, "fase": fase,
            "region": region, "combinacio": combinacio,
            "p_generosa_A": round(ordre_a, 3),
            "p_generosa_B": round(ordre_b, 3),
            "diff_posicio": round(diff, 3),
            "bias_detectat": diff > 0.3,
        })

    df_bias = pd.DataFrame(resultats)
    n_bias  = df_bias["bias_detectat"].sum()
    print(f"\n POSITION BIAS: {n_bias} casos amb diff > 0.3 de {len(df_bias)} totals")
    return df_bias


# ==============================================================================
# RESPONDER — MONOTONIA
# ==============================================================================

def verificar_monotonia(df_resp):
    resultats = []

    for (model, fase, region), grup in df_resp.groupby(["model", "fase", "region"]):
        taxes    = grup.groupby("oferta")["tria_generosa"].mean().reindex(OFERTES).values
        magnitud = round(calcular_magnitud_violacions(taxes), 4)

        resultats.append({
            "model":               model,
            "fase":                fase,
            "region":              region,
            "magnitud_violacions": magnitud,
            "corba_valida":        magnitud <= LLINDAR_MAGNITUD_MONO,
        })

    df_mono = pd.DataFrame(resultats)
    n_inv   = (~df_mono["corba_valida"]).sum()
    print(f"\n MONOTONIA RESPONDER: {n_inv} corbes amb magnitud > "
          f"{LLINDAR_MAGNITUD_MONO} de {len(df_mono)} totals")
    print(df_mono[["fase", "region", "magnitud_violacions",
                   "corba_valida"]].to_string(index=False))
    return df_mono


# ==============================================================================
# RESPONDER — MÈTRIQUES PRINCIPALS + BASELINE ABSOLUT
# ==============================================================================

def metriques_responder(df_resp, df_mono):
    corbes = {}
    for (model, fase, region), grup in df_resp.groupby(["model", "fase", "region"]):
        taxes = grup.groupby("oferta")["tria_generosa"].mean().reindex(OFERTES)
        corbes[(model, fase, region)] = taxes

    resultats          = []
    resultats_baseline = []

    for model in set(k[0] for k in corbes):
        p_base = corbes.get((model, 1, "baseline"))
        if p_base is None:
            print(f"⚠️  No hi ha baseline per al model {model}")
            continue

        pr_base   = calcular_punt_ruptura(OFERTES, p_base)
        mono_base = df_mono[
            (df_mono["model"] == model) &
            (df_mono["fase"] == 1) &
            (df_mono["region"] == "baseline")
        ]["corba_valida"].values

        resultats_baseline.append({
            "model":                 model,
            "punt_ruptura_baseline": pr_base,
            "pr_ref_oosterbeek":     PR_GLOBAL_OOSTERBEEK,
            "diff_pr_oosterbeek":    round(pr_base - PR_GLOBAL_OOSTERBEEK, 2)
                                     if pr_base else None,
            "corba_valida":          len(mono_base) > 0 and mono_base[0],
        })

        for (fase, region) in [(f, r) for f, r in CONDICIONS
                               if not (f == 1 and r == "baseline")]:
            p_cond = corbes.get((model, fase, region))
            if p_cond is None:
                continue

            mono_cond = df_mono[
                (df_mono["model"] == model) &
                (df_mono["fase"] == fase) &
                (df_mono["region"] == region)
            ]["corba_valida"].values

            ofertes_comunes = p_base.index.intersection(p_cond.index)
            ofertes_comunes = ofertes_comunes[
                p_base[ofertes_comunes].notna() & p_cond[ofertes_comunes].notna()
            ]
            if len(ofertes_comunes) > 0:
                rmse_val = round(rmse(p_cond[ofertes_comunes].values, 
                                    p_base[ofertes_comunes].values), 4)
            else:
                rmse_val = None
            kl_val   = round(kl_divergence(p_cond, p_base), 4)
            dm_val   = round(delta_mitja(p_cond, p_base), 4)

            cond_valida = len(mono_cond) > 0 and mono_cond[0]
            if cond_valida:
                pr_val  = calcular_punt_ruptura(OFERTES, p_cond)
                dpr_val = round(pr_val - pr_base, 2) if pr_val and pr_base else None
            else:
                pr_val  = None
                dpr_val = None

            resultats.append({
                "model":             model,
                "fase":              fase,
                "region":            region,
                "rmse_vs_baseline":  rmse_val,
                "kl_vs_baseline":    kl_val,
                "delta_vs_baseline": dm_val,
                "punt_ruptura":      pr_val,
                "delta_pr":          dpr_val,
                "corba_valida":      cond_valida,
            })

    df_resp_metrics = pd.DataFrame(resultats)
    df_base_absolut = pd.DataFrame(resultats_baseline)

    print("\nBASELINE ABSOLUT — RESPONDER (referència Oosterbeek ~20$):")
    print(df_base_absolut.to_string(index=False))

    print("\nMÈTRIQUES RESPONDER:")
    print(df_resp_metrics.to_string(index=False))

    return df_resp_metrics, df_base_absolut


# ==============================================================================
# PROPOSER — COHERÈNCIA INTERNA (valors en brut)
# ==============================================================================

def verificar_coherencia_proposer(df_prop):
    resultats = []
    resultats_magnitud = []

    for (model, fase, region), grup in df_prop.groupby(["model", "fase", "region"]):
        controls = grup[grup["combinacio"].isin(
            COMBINACIONS_CONTROL_INF + COMBINACIONS_CONTROL_SUP
        )]

        # Valors en brut
        for comb, fila in controls.groupby("combinacio"):
            p_gen = fila["tria_generosa"].mean()
            tipus = "control_inf" if comb in COMBINACIONS_CONTROL_INF else "control_sup"
            resultats.append({
                "model":      model,
                "fase":       fase,
                "region":     region,
                "combinacio": comb,
                "tipus":      tipus,
                "p_generosa": round(p_gen, 3),
            })

        # Magnitud de violacions ordinals (només control inf)
        # Ordre esperat: 10vs40 >= 20vs40 >= 30vs40 >= 35vs40
        vals_inf = []
        for comb in COMBINACIONS_CONTROL_INF:
            fila = grup[grup["combinacio"] == comb]
            if len(fila) > 0:
                vals_inf.append(fila["tria_generosa"].mean())
            else:
                vals_inf.append(None)

        if all(v is not None for v in vals_inf):
            magnitud = sum(
                max(0, vals_inf[i+1] - vals_inf[i] - 0.05)
                for i in range(len(vals_inf) - 1)
            )
        else:
            magnitud = None

        resultats_magnitud.append({
            "model":                      model,
            "fase":                       fase,
            "region":                     region,
            "magnitud_violacions_ordinals": round(magnitud, 4) if magnitud else None,
        })

    df_coh      = pd.DataFrame(resultats)
    df_magnitud = pd.DataFrame(resultats_magnitud)

    print(f"\n COHERÈNCIA PROPOSER (controls — valors en brut):")
    print(df_coh[["fase", "region", "combinacio", "tipus",
                  "p_generosa"]].to_string(index=False))

    print(f"\n COHERÈNCIA ORDINAL PROPOSER (magnitud violacions control inf):")
    print(df_magnitud.to_string(index=False))

    return df_coh, df_magnitud


# ==============================================================================
# PROPOSER — MÈTRIQUES PRINCIPALS + BASELINE ABSOLUT
# ==============================================================================

def metriques_proposer(df_prop):
    proporcions = {}
    for (model, fase, region, combinacio), grup in df_prop.groupby(
        ["model", "fase", "region", "combinacio"]
    ):
        p_gen = grup.groupby("ordre")["tria_generosa"].mean().mean() 
        proporcions[(model, fase, region, combinacio)] = round(p_gen, 4)

    resultats      = []
    resultats_crit = []
    resultats_base = []

    for model in set(k[0] for k in proporcions):
        p_base_dict = {
            comb: proporcions.get((model, 1, "baseline", comb))
            for comb in TOTES_COMBINACIONS
        }

        p_ctrl_inf = {c: p_base_dict[c] for c in COMBINACIONS_CONTROL_INF
                      if p_base_dict.get(c) is not None}
        p_ctrl_sup = {c: p_base_dict[c] for c in COMBINACIONS_CONTROL_SUP
                      if p_base_dict.get(c) is not None}

        vals_inf_base = [p_base_dict[c] for c in COMBINACIONS_CONTROL_INF
                         if p_base_dict.get(c) is not None]
        vals_sup_base = [p_base_dict[c] for c in COMBINACIONS_CONTROL_SUP
                 if p_base_dict.get(c) is not None]
        p_inf_base = round(np.mean(vals_inf_base), 4) if vals_inf_base else None
        p_sup_base = round(1 - np.mean(vals_sup_base), 4) if vals_sup_base else None

        resultats_base.append({
            "model":             model,
            "ref_oosterbeek_40": OFERTA_GLOBAL_OOSTERBEEK,
            "p_inf_baseline":    p_inf_base,
            "p_sup_baseline":    p_sup_base,
            **{f"p_{c}": v for c, v in p_ctrl_inf.items()},
            **{f"p_{c}": v for c, v in p_ctrl_sup.items()},
        })

        for (fase, region) in [(f, r) for f, r in CONDICIONS
                               if not (f == 1 and r == "baseline")]:
            p_cond_dict = {
                comb: proporcions.get((model, fase, region, comb))
                for comb in TOTES_COMBINACIONS
            }

            combs_valides = [
                c for c in TOTES_COMBINACIONS
                if p_base_dict.get(c) is not None and p_cond_dict.get(c) is not None
            ]

            if combs_valides:
                p_base_vec = [p_base_dict[c] for c in combs_valides]
                p_cond_vec = [p_cond_dict[c] for c in combs_valides]
                rmse_val   = round(rmse(p_cond_vec, p_base_vec), 4)
                kl_val     = round(kl_divergence(p_cond_vec, p_base_vec), 4)
                dm_val     = round(delta_mitja(p_cond_vec, p_base_vec), 4)
            else:
                rmse_val = None
                kl_val   = None
                dm_val   = None

            # p_inf i p_sup — ABANS del resultats.append
            vals_inf_cond = [p_cond_dict[c] for c in COMBINACIONS_CONTROL_INF
                             if p_cond_dict.get(c) is not None]
            vals_sup_cond = [p_cond_dict[c] for c in COMBINACIONS_CONTROL_SUP
                 if p_cond_dict.get(c) is not None]
            
            p_inf_cond = round(np.mean(vals_inf_cond), 4) if vals_inf_cond else None
            p_sup_cond = round(1 - np.mean(vals_sup_cond), 4) if vals_sup_cond else None

            resultats.append({
                "model":             model,
                "fase":              fase,
                "region":            region,
                "rmse_vs_baseline":  rmse_val,
                "kl_vs_baseline":    kl_val,
                "delta_vs_baseline": dm_val,
                "p_inf":             p_inf_cond,
                "p_sup":             p_sup_cond,
            })

            for comb in COMBINACIONS_CRITIQUES:
                p_c = p_cond_dict.get(comb)
                resultats_crit.append({
                    "model":      model,
                    "fase":       fase,
                    "region":     region,
                    "combinacio": comb,
                    "p_generosa": p_c,
                })

    df_prop_metrics = pd.DataFrame(resultats)
    df_crit         = pd.DataFrame(resultats_crit)
    df_base_absolut = pd.DataFrame(resultats_base)

    print("\n BASELINE ABSOLUT — PROPOSER (referència Oosterbeek ~40$):")
    print(df_base_absolut.to_string(index=False))

    print("\n MÈTRIQUES PROPOSER (RMSE, KL, Delta, p_inf, p_sup):")
    print(df_prop_metrics.to_string(index=False))

    print("\n COMBINACIONS CRÍTIQUES (5-8) — p_(e,g) per regió:")
    df_crit_pivot = df_crit.pivot_table(
        index=["fase", "combinacio"],
        columns="region",
        values="p_generosa"
    )
    print(df_crit_pivot.to_string())

    return df_prop_metrics, df_crit, df_base_absolut



# ==============================================================================
# VISUALITZACIÓ — FUNCIONS AUXILIARS
# ==============================================================================

def calcular_taxes_resp(df_resp, model, fase, region):
    grup = df_resp[
        (df_resp["model"] == model) &
        (df_resp["fase"] == fase) &
        (df_resp["region"] == region)
    ]
    return grup.groupby("oferta")["tria_generosa"].mean().reindex(OFERTES).values


def calcular_vals_prop(df_prop, model, fase, region, combinacions):
    grup = df_prop[
        (df_prop["model"] == model) &
        (df_prop["fase"] == fase) &
        (df_prop["region"] == region)
    ]
    vals = []
    for comb in combinacions:
        p = grup[grup["combinacio"] == comb]["tria_generosa"].mean()
        vals.append(p if not np.isnan(p) else 0.0)
    return vals





# ==============================================================================
# CRITERI DE QUALITAT PER MODEL
# ==============================================================================

def verificar_qualitat_model(input_csv, df_bias, df_mono, model_name):
    """
    Aplica els tres criteris d'exclusió i genera un resum de qualitat.
    
    Criteris:
    1. Taxa de respostes invàlides <= 10%
    2. Position bias <= 30% de combinacions
    3. Baseline del Responder vàlid (magnitud <= 0.2)
    """
    # Criteri 1 — Taxa d'invàlids (rellegim el CSV original)
    df_raw       = pd.read_csv(input_csv)
    n_total      = len(df_raw)
    n_invalids   = (df_raw["decisio"] == "INVALID").sum()
    pct_invalids = round(n_invalids / n_total * 100, 1) if n_total > 0 else 0.0
    passa_invalids = pct_invalids <= 10.0

    # Criteri 2 — Position bias
    n_combinacions = len(df_bias)
    n_bias         = df_bias["bias_detectat"].sum()
    pct_bias       = round(n_bias / n_combinacions * 100, 1) \
                     if n_combinacions > 0 else 0.0
    passa_bias = pct_bias <= 30.0

    # Criteri 3 — Baseline del Responder vàlid
    baseline_mono  = df_mono[
        (df_mono["fase"] == 1) &
        (df_mono["region"] == "baseline")
    ]
    baseline_valid = bool(baseline_mono["corba_valida"].values[0]) \
                     if len(baseline_mono) > 0 else False

    # Resultat final
    if passa_invalids and passa_bias and baseline_valid:
        resultat = "Analitzable"
    elif not passa_invalids or not passa_bias:
        resultat = "Exclòs"
    else:
        resultat = "Parcial (Responder exclòs)"

    df_qualitat = pd.DataFrame([{
        "model":             model_name,
        "taxa_invalids_pct": pct_invalids,
        "passa_invalids":    passa_invalids,
        "position_bias_pct": pct_bias,
        "passa_bias":        passa_bias,
        "baseline_valid":    baseline_valid,
        "resultat":          resultat,
    }])

    print("\n CRITERI DE QUALITAT:")
    print(df_qualitat.to_string(index=False))

    return df_qualitat

# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main():
    print("=" * 60)
    print("ANÀLISI DE DADES — Experiment Ultimatum Game")
    print("=" * 60)

    df, df_resp, df_prop = carregar_dades(INPUT_CSV)
    print(f"\n Dades carregades: {len(df)} files")

    # Validacions prèvies
    df_bias = analitzar_position_bias(df_resp, df_prop)
    df_mono = verificar_monotonia(df_resp)
    df_coh, df_magnitud = verificar_coherencia_proposer(df_prop)

    # Criteri de qualitat
    df_qualitat = verificar_qualitat_model(INPUT_CSV, df_bias, df_mono, MODEL_NAME)

    # Mètriques principals
    df_resp_metrics, df_resp_base = metriques_responder(df_resp, df_mono)
    df_prop_metrics, df_crit, df_prop_base = metriques_proposer(df_prop)

    # Guardar resultats
    output_dir = f"resultats_{MODEL_NAME}"
    os.makedirs(output_dir, exist_ok=True)

    df_pvalors_resp, df_pvalors_prop = calcular_pvalors_correctes(
        df_resp, df_prop, n_permutacions=1000
    )
    df_pvalors_resp.to_csv(f"{output_dir}/resultats_pvalors_responder_{MODEL_NAME}.csv", index=False)
    df_pvalors_prop.to_csv(f"{output_dir}/resultats_pvalors_proposer_{MODEL_NAME}.csv", index=False)
    df_bias.to_csv(f"{output_dir}/resultats_position_bias_{MODEL_NAME}.csv", index=False)
    df_mono.to_csv(f"{output_dir}/resultats_monotonia_{MODEL_NAME}.csv", index=False)
    df_coh.to_csv(f"{output_dir}/resultats_coherencia_proposer_{MODEL_NAME}.csv", index=False)
    df_qualitat.to_csv(f"{output_dir}/resultats_qualitat_{MODEL_NAME}.csv", index=False)
    df_magnitud.to_csv(f"{output_dir}/resultats_coherencia_ordinal_proposer_{MODEL_NAME}.csv", index=False)
    df_resp_metrics.to_csv(f"{output_dir}/resultats_responder_{MODEL_NAME}.csv", index=False)
    df_resp_base.to_csv(f"{output_dir}/resultats_responder_baseline_{MODEL_NAME}.csv", index=False)
    df_prop_metrics.to_csv(f"{output_dir}/resultats_proposer_{MODEL_NAME}.csv", index=False)
    df_prop_base.to_csv(f"{output_dir}/resultats_proposer_baseline_{MODEL_NAME}.csv", index=False)
    df_crit.to_csv(f"{output_dir}/resultats_combinacions_critiques_{MODEL_NAME}.csv", index=False)

    print("\n Anàlisi completada. Fitxers generats:")
    for nom in [
        "position_bias", "monotonia",
        "coherencia_proposer", "coherencia_ordinal_proposer",
        "qualitat",
        "responder", "responder_baseline",
        "proposer", "proposer_baseline",
        "combinacions_critiques", "h3"
    ]:
        print(f"  — {output_dir}/resultats_{nom}_{MODEL_NAME}.csv")

  


if __name__ == "__main__":
    main()