# Sensibilitat Cultural en LLMs: Una Anàlisi Empírica Mitjançant el Joc de l'Ultimàtum

Codi, dades i pipeline d'anàlisi del Treball Final de Grau de Laura Servat Bach (Universitat de Barcelona, 2026), dirigit per la Dra. Maite López Sánchez i Chengheng Li Chen.

## Descripció

Aquest repositori conté tot el codi necessari per reproduir l'experiment descrit al TFG: l'avaluació de 10 LLMs mitjançant el Joc de l'Ultimàtum, sota tres fases de condicionament cultural progressiu (Baseline, etiqueta geogràfica i perfil cultural detallat) aplicades a tres regions culturals (Neo-Europe, South and East Asia i South America).

## Estructura del repositori

```
TFG_sensibilitat_cultural_LLMs/
├── pipeline.py                 # Pipeline per a models executats localment (vLLM / MareNostrum 5)
├── pipeline_openrouter.py      # Pipeline per a models accessibles via API (OpenRouter)
├── analysis.py                 # Càlcul de mètriques, test de permutacions i criteris d'exclusió
├── grafics.py                  # Generació de figures per als resultats
├── dades/                      # CSVs de resultats per a cada model
└── resultats/                  # Figures i taules generades per l'anàlisi
```

## Scripts

### `pipeline.py`
Executa l'experiment en models allotjats localment mitjançant [vLLM](https://github.com/vllm-project/vllm). Dissenyat per funcionar al clúster MareNostrum 5 (BSC). Genera un fitxer CSV amb totes les respostes del model, incloent la decisió, el raonament i els logprobs del token de decisió.

Per executar-lo cal modificar les variables de configuració a dalt del fitxer:
```python
MODEL_NAME = "nom-del-model"
MODEL_PATH = "/ruta/al/model"
NUM_GPUS   = 4
```

### `pipeline_openrouter.py`
Executa l'experiment en models accessibles via l'API d'[OpenRouter](https://openrouter.ai). Equivalent funcional a `pipeline.py` però sense dependència de vLLM. Cal substituir la clau d'API:
```python
API_KEY    = "la-teva-clau-openrouter"
MODEL_NAME = "proveïdor/nom-del-model"
```

### `analysis.py`
A partir dels CSVs de resultats, calcula totes les mètriques de l'estudi: taxes d'acceptació, RMSE, $\bar{\Delta}$, Punt de Ruptura, $\bar{p}_{\text{inf}}$, $\bar{p}_{\text{sup}}$, biaix de posició, monotonia i el test de permutacions per a la significança estadística.

### `grafics.py`
Genera les figures de l'anàlisi: gràfics de taxes d'acceptació del Responent i gràfics de combinacions del Proposant (control i crítiques) per a cada model, fase i regió.

## Dades

La carpeta `dades/` conté un fitxer CSV per a cada model avaluat amb el format `resultats_<nom-model>.csv`. Cada fila correspon a una interacció individual i inclou els camps següents:

| Camp | Descripció |
|------|------------|
| `model` | Nom del model |
| `fase` | Fase de condicionament (1, 2 o 3) |
| `region` | Regió cultural (baseline, asia, neo_europe, south_america) |
| `rol` | Rol del model (proposer o responder) |
| `combinacio` | Combinació d'ofertes (proposer) o oferta presentada (responder) |
| `ordre` | Ordre de presentació de les opcions (A o B) |
| `repeticio` | Número de repetició (1-10) |
| `decisio` | Decisió del model (accept/reject per al responder, A/B per al proposer) |
| `tria_generosa` | 1 si el model ha triat l'opció generosa, 0 si no |
| `prob_generosa` | Probabilitat assignada a l'opció generosa |
| `prob_A` | Probabilitat del token A |
| `prob_B` | Probabilitat del token B |
| `raonament` | Raonament explícit generat pel model |

## Instal·lació

```bash
pip install pandas numpy matplotlib scipy requests
```

Per als models locals (MareNostrum 5):
```bash
pip install vllm
```

## Reproducció de l'experiment

1. Per a models locals, configura `pipeline.py` i executa:
```bash
python pipeline.py
```

2. Per a models via API, configura `pipeline_openrouter.py` amb la teva clau d'API i executa:
```bash
python pipeline_openrouter.py
```

3. Un cop tens els CSVs a `dades/`, executa l'anàlisi:
```bash
python analysis.py
```

4. Per generar les figures:
```bash
python grafics.py
```

## Referència

Si fas servir aquest codi, si us plau cita el treball original:

> Servat Bach, L. (2026). *Sensibilitat Cultural en LLMs: Una Anàlisi Empírica Mitjançant el Joc de l'Ultimàtum*. Treball Final de Grau, Grau d'Informàtica, Universitat de Barcelona.