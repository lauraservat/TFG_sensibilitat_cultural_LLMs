import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURACIÓ
# ==============================================================================

MODEL_NAME = "qwen3-235b-a22b"
INPUT_CSV  = f"resultats/resultats_{MODEL_NAME}.csv"
OUTPUT_DIR = f"grafics/figures_{MODEL_NAME}"

OFERTES = list(range(0, 101, 10))
COMBINACIONS_CONTROL_INF = ["10vs40", "20vs40", "30vs40", "35vs40"]
COMBINACIONS_CRITIQUES   = ["39vs41", "39vs45", "41vs45", "38vs46"]
COMBINACIONS_CONTROL_SUP = ["40vs50", "40vs60"]

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
PR_GLOBAL_OOSTERBEEK = 20  # Referència global de PR segons Oosterbeek et al. (2016)

# ==============================================================================
# CÀRREGA
# ==============================================================================

def carregar_dades(path):
    df = pd.read_csv(path)
    df["model"] = df["model"].str.split("/").str[-1]
    df = df[df["decisio"] != "INVALID"].copy()
    df_resp = df[df["rol"] == "responder"].copy()
    df_resp["oferta"] = pd.to_numeric(df_resp["combinacio"])
    df_prop = df[df["rol"] == "proposer"].copy()
    return df_resp, df_prop

# ==============================================================================
# FUNCIONS AUXILIARS
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
# GRÀFICS PER FASE
# ==============================================================================

def grafics_per_fase(df_resp, df_prop, model, output_dir):
    control_inf = ["10vs40", "20vs40", "30vs40", "35vs40"]
    control_sup = ["40vs50", "40vs60"]

    for fase in [1, 2, 3]:
        titol_fase = {1: "Baseline",
                      2: "Etiqueta geogràfica",
                      3: "Perfil psicomètric"}

        regions = ["baseline"] if fase == 1 else \
                  ["baseline", "asia", "neo_europe", "south_america"]

        # ── GRÀFIC 1: RESPONENT ──
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        fig.suptitle(f"{model} — Fase {fase} — {titol_fase[fase]} — Responent",
                     fontsize=11, fontweight="bold")
        ax.set_title("Taxes d'acceptació")
        ax.set_xlabel("Oferta ($)")
        ax.set_ylabel("Probabilitat d'acceptació")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)

        width = 4 if fase == 1 else 2
        n = len(regions)
        for i, region in enumerate(regions):
            reg_fase = 1 if region == "baseline" else fase
            taxes = calcular_taxes_resp(df_resp, model, reg_fase, region)
            offset = (i - (n - 1) / 2) * width
            ax.bar([o + offset for o in OFERTES], taxes, width=width,
                   color=COLORS[region], alpha=0.6, label=LABELS[region])
        ax.set_xticks(OFERTES)
        ax.set_xticklabels([f"${o}" for o in OFERTES], fontsize=8,
                           rotation=45, ha="right")
        ax.tick_params(axis='x', which='both', length=4)
        ax.set_xlim(-7, 107)
        for o in OFERTES:
            ax.axvline(o - 5, color="gray", linewidth=0.3, alpha=0.3)
        ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left',
                  borderaxespad=0)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        path = os.path.join(output_dir, f"fase{fase}_{model}_resp.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  — fase{fase}_{model}_resp.png")

        # ── GRÀFIC 2: PROPOSER CONTROL ──
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"{model} — Fase {fase} — {titol_fase[fase]} — Proposant control",
                     fontsize=11, fontweight="bold")

        ax = axes[0]
        ax.set_title("Control inferior\n(alternativa < 40$)")
        ax.set_xlabel("Combinació")
        ax.set_ylabel("p generosa (triar 40$)")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        x_pos = list(range(len(control_inf)))
        width = 0.8 / len(regions)
        for i, region in enumerate(regions):
            reg_fase = 1 if region == "baseline" else fase
            vals = calcular_vals_prop(df_prop, model, reg_fase, region, control_inf)
            offset = (i - (len(regions) - 1) / 2) * width
            ax.bar([x + offset for x in x_pos], vals,
                   width=width, color=COLORS[region], alpha=0.6, label=LABELS[region])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(control_inf, rotation=45, ha="right", fontsize=9)

        ax = axes[1]
        ax.set_title("Control superior\n(alternativa > 40$)")
        ax.set_xlabel("Combinació")
        ax.set_ylabel("p generosa (triar alternativa)")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        x_pos = list(range(len(control_sup)))
        for i, region in enumerate(regions):
            reg_fase = 1 if region == "baseline" else fase
            vals = calcular_vals_prop(df_prop, model, reg_fase, region, control_sup)
            offset = (i - (len(regions) - 1) / 2) * width
            ax.bar([x + offset for x in x_pos], vals,
                   width=width, color=COLORS[region], alpha=0.6, label=LABELS[region])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(control_sup, rotation=45, ha="right", fontsize=9)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=9, bbox_to_anchor=(1.01, 0.5),
                   loc='center left', borderaxespad=0)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        path = os.path.join(output_dir, f"fase{fase}_{model}_prop_control.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  — fase{fase}_{model}_prop_control.png")

        # ── GRÀFIC 3: PROPOSER CRÍTIQUES ──
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        fig.suptitle(f"{model} — Fase {fase} — {titol_fase[fase]} — Proposant crítiques",
                     fontsize=11, fontweight="bold")
        ax.set_title("Combinacions crítiques\n(frontera cultural)")
        ax.set_xlabel("Combinació")
        ax.set_ylabel("p generosa")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        x_pos = list(range(len(COMBINACIONS_CRITIQUES)))
        width = 0.8 / len(regions)
        for i, region in enumerate(regions):
            reg_fase = 1 if region == "baseline" else fase
            vals = calcular_vals_prop(df_prop, model, reg_fase, region,
                                      COMBINACIONS_CRITIQUES)
            offset = (i - (len(regions) - 1) / 2) * width
            ax.bar([x + offset for x in x_pos], vals,
                   width=width, color=COLORS[region], alpha=0.6, label=LABELS[region])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(COMBINACIONS_CRITIQUES, rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left',
                  borderaxespad=0)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        path = os.path.join(output_dir, f"fase{fase}_{model}_prop_critiques.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  — fase{fase}_{model}_prop_critiques.png")

# ==============================================================================
# GRÀFICS PER REGIÓ I FASE
# ==============================================================================

def grafics_per_regio_fase(df_resp, df_prop, model, output_dir):
    control_inf = ["10vs40", "20vs40", "30vs40", "35vs40"]
    control_sup = ["40vs50", "40vs60"]

    for fase in [2, 3]:
        for region in ["asia", "neo_europe", "south_america"]:

            # ── GRÀFIC 1: RESPONENT ──
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            fig.suptitle(
                f"{model} — Fase {fase} | {LABELS[region]} vs Baseline — Responent",
                fontsize=11, fontweight="bold"
            )
            ax.set_title("Taxes d'acceptació")
            ax.set_xlabel("Oferta ($)")
            ax.set_ylabel("Probabilitat d'acceptació")
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)

            taxes_base = calcular_taxes_resp(df_resp, model, 1, "baseline")
            taxes_cond = calcular_taxes_resp(df_resp, model, fase, region)
            width = 4
            ax.bar([o - width/2 for o in OFERTES], taxes_base, width=width,
                   color="black", alpha=0.6, label="Baseline")
            ax.bar([o + width/2 for o in OFERTES], taxes_cond, width=width,
                   color=COLORS[region], alpha=0.6, label=LABELS[region])
            ax.set_xticks(OFERTES)
            ax.set_xticklabels([f"${o}" for o in OFERTES], fontsize=8,
                               rotation=45, ha="right")
            ax.tick_params(axis='x', which='both', length=4)
            ax.set_xlim(-7, 107)
            for o in OFERTES:
                ax.axvline(o - 5, color="gray", linewidth=0.3, alpha=0.3)
            ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left',
                      borderaxespad=0)

            plt.tight_layout(rect=[0, 0, 0.85, 1])
            path = os.path.join(output_dir, f"fase{fase}_{region}_{model}_resp.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  — fase{fase}_{region}_{model}_resp.png")

            # ── GRÀFIC 2: PROPOSER CONTROL ──
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(
                f"{model} — Fase {fase} | {LABELS[region]} vs Baseline — Proposant control",
                fontsize=11, fontweight="bold"
            )

            ax = axes[0]
            ax.set_title("Control inferior\n(alternativa < 40$)")
            ax.set_xlabel("Combinació")
            ax.set_ylabel("p generosa (triar 40$)")
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
            x_pos = list(range(len(control_inf)))
            width = 0.35
            for i, (r, reg_fase) in enumerate([("baseline", 1), (region, fase)]):
                vals = calcular_vals_prop(df_prop, model, reg_fase, r, control_inf)
                color = "black" if r == "baseline" else COLORS[region]
                label = "Baseline" if r == "baseline" else LABELS[region]
                offset = -width/2 if i == 0 else width/2
                ax.bar([x + offset for x in x_pos], vals,
                       width=width, color=color, alpha=0.6, label=label)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(control_inf, rotation=45, ha="right", fontsize=9)

            ax = axes[1]
            ax.set_title("Control superior\n(alternativa > 40$)")
            ax.set_xlabel("Combinació")
            ax.set_ylabel("p generosa (triar alternativa)")
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
            x_pos = list(range(len(control_sup)))
            for i, (r, reg_fase) in enumerate([("baseline", 1), (region, fase)]):
                vals = calcular_vals_prop(df_prop, model, reg_fase, r, control_sup)
                color = "black" if r == "baseline" else COLORS[region]
                label = "Baseline" if r == "baseline" else LABELS[region]
                offset = -width/2 if i == 0 else width/2
                ax.bar([x + offset for x in x_pos], vals,
                       width=width, color=color, alpha=0.6, label=label)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(control_sup, rotation=45, ha="right", fontsize=9)

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, fontsize=9, bbox_to_anchor=(1.01, 0.5),
                       loc='center left', borderaxespad=0)

            plt.tight_layout(rect=[0, 0, 0.85, 1])
            path = os.path.join(output_dir, f"fase{fase}_{region}_{model}_prop_control.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  — fase{fase}_{region}_{model}_prop_control.png")

            # ── GRÀFIC 3: PROPOSER CRÍTIQUES ──
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            fig.suptitle(
                f"{model} — Fase {fase} | {LABELS[region]} vs Baseline — Proposant crítiques",
                fontsize=11, fontweight="bold"
            )
            ax.set_title("Combinacions crítiques\n(frontera cultural)")
            ax.set_xlabel("Combinació")
            ax.set_ylabel("p generosa")
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
            x_pos = list(range(len(COMBINACIONS_CRITIQUES)))
            width = 0.35
            for i, (r, reg_fase) in enumerate([("baseline", 1), (region, fase)]):
                vals = calcular_vals_prop(df_prop, model, reg_fase, r,
                                          COMBINACIONS_CRITIQUES)
                color = "black" if r == "baseline" else COLORS[region]
                label = "Baseline" if r == "baseline" else LABELS[region]
                offset = -width/2 if i == 0 else width/2
                ax.bar([x + offset for x in x_pos], vals,
                       width=width, color=color, alpha=0.6, label=label)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(COMBINACIONS_CRITIQUES, rotation=45, ha="right", fontsize=9)
            ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left',
                      borderaxespad=0)

            plt.tight_layout(rect=[0, 0, 0.85, 1])
            path = os.path.join(output_dir, f"fase{fase}_{region}_{model}_prop_critiques.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  — fase{fase}_{region}_{model}_prop_critiques.png")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_resp, df_prop = carregar_dades(INPUT_CSV)
    model = df_resp["model"].unique()[0]
    print(f"Generant gràfics per {model}...")
    grafics_per_fase(df_resp, df_prop, model, OUTPUT_DIR)
    grafics_per_regio_fase(df_resp, df_prop, model, OUTPUT_DIR)
    print("Done!")