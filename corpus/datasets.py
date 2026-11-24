"""
exp70-multi — per-dataset + PER-TASK configuration (single source of truth).

Generalizes the exp70 BBBP PMC-literature pipeline to the MoleculeNet datasets
lipo / sider / hiv / tox21. Active dataset via EXP70_DATASET (import time).

TWO search routes:
  * MOLECULE route (dataset-agnostic): per-molecule, all verified aliases.
  * TASK/PROPERTY route (PER-TASK): each prediction task (the distinct values of the
    'task' column) gets its OWN stable task_id = "<dataset>:<task>", its OWN example-free
    Opus query-generation instruction, and its OWN BM25 prescreen topic keyword list.
    41 tasks total across the 4 datasets: lipo=1, hiv=1, tox21=12, sider=27.

Everything dataset/task-specific lives here; the molecule instruction / molecule judge
prompt are dataset-agnostic and stay in querygen.py / judge.py.
"""
import os

DATASET = os.environ.get("EXP70_DATASET", "lipo").lower()

DATA_ROOT = os.environ.get("MOLERAG_DATA_ROOT",
                           os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "moleculenet"))
REFERENCE_SEED = "seed_0"
SPLITS = ["train", "valid", "test"]

_INSTR_TMPL = (
    "Generate a broad set of Boolean queries for searching PMC (PubMed Central) literature "
    "about {bio}. The purpose is to collect a candidate literature corpus. Prioritize "
    "coverage. Determine the relevant terminology and search concepts independently. Generate "
    "as many distinct queries as necessary to express the coverage you intend. There is no "
    "predetermined query count. Do not require any particular molecule to appear. Do not "
    "impose publication-date, language, or article-type restrictions. Return executable PMC "
    "Boolean queries, a brief purpose for each query, and a concise description of its "
    "intended coverage. Do not assess article relevance or predict molecular properties."
)


def _task(task, bio, topics):
    return dict(task=task, bio=bio, topics=list(topics))


# --------------------------------------------------------------------------- LIPO
_LIPO_TASKS = [
    _task("lipo",
          "lipophilicity of drug-like molecules — the octanol-water partition "
          "coefficient (logP), the pH-dependent distribution coefficient (logD, e.g. "
          "logD7.4), octanol-water partitioning, and lipophilic character relevant to "
          "absorption, distribution, permeability and solubility",
          ["octanol water partition coefficient logP",
           "distribution coefficient logD pH 7.4",
           "lipophilicity measurement drug molecule",
           "shake flask logP determination",
           "chromatographic lipophilicity HPLC logP",
           "membrane partitioning lipophilic permeation",
           "pKa ionization effect on logD lipophilicity",
           "lipophilicity solubility ADME physicochemical property",
           "lipophilic efficiency ligand lipophilicity LLE"]),
]

# --------------------------------------------------------------------------- HIV
_HIV_TASKS = [
    _task("HIV_active",
          "inhibition of HIV-1 replication and anti-HIV antiviral activity of small "
          "molecules, including reverse transcriptase inhibitors (nucleoside and "
          "non-nucleoside), integrase strand-transfer inhibitors, protease inhibitors, "
          "entry / fusion / coreceptor (CCR5, CXCR4) inhibitors, capsid and maturation "
          "inhibitors, anti-HIV screening assays, EC50 / IC50 antiviral potency, and "
          "selectivity / cytotoxicity",
          ["HIV-1 replication inhibition antiviral",
           "anti-HIV activity small molecule inhibitor EC50",
           "antiretroviral drug HIV infection",
           "HIV reverse transcriptase inhibitor NNRTI NRTI",
           "HIV integrase strand transfer inhibitor",
           "HIV protease inhibitor",
           "HIV entry fusion inhibitor gp41 gp120",
           "HIV coreceptor CCR5 CXCR4 antagonist",
           "anti-HIV screening selectivity index cytotoxicity",
           "HIV drug resistance mutation antiviral"]),
]

# --------------------------------------------------------------------------- TOX21 (12)
_TOX21_TASKS = [
    _task("NR-AR", "androgen receptor (AR) agonism and antagonism, (anti)androgenic "
          "activity, and endocrine disruption of androgen signaling",
          ["androgen receptor agonist antagonist assay",
           "antiandrogen androgen receptor activity",
           "endocrine disruption androgen signaling"]),
    _task("NR-AR-LBD", "activation of the androgen receptor ligand-binding domain (AR-LBD) "
          "in reporter-gene assays and androgen-receptor binding",
          ["androgen receptor ligand binding domain reporter assay",
           "androgen receptor binding affinity",
           "AR LBD transactivation androgen"]),
    _task("NR-AhR", "aryl hydrocarbon receptor (AhR) agonism and activation, dioxin-like "
          "activity, and CYP1A induction",
          ["aryl hydrocarbon receptor AhR activation agonist",
           "dioxin like activity AhR",
           "CYP1A induction aryl hydrocarbon receptor"]),
    _task("NR-Aromatase", "aromatase (CYP19) inhibition and modulation of estrogen "
          "biosynthesis",
          ["aromatase CYP19 inhibition assay",
           "aromatase inhibitor estrogen biosynthesis",
           "steroidogenesis aromatase activity"]),
    _task("NR-ER", "estrogen receptor (ER) agonism and antagonism, estrogenic and "
          "antiestrogenic activity, and endocrine disruption of estrogen signaling",
          ["estrogen receptor agonist antagonist assay",
           "estrogenic activity endocrine disruption",
           "xenoestrogen estrogen receptor binding"]),
    _task("NR-ER-LBD", "activation of the estrogen receptor ligand-binding domain (ER-LBD) "
          "in reporter-gene assays and estrogen-receptor binding",
          ["estrogen receptor ligand binding domain reporter assay",
           "estrogen receptor binding affinity",
           "ER LBD transactivation estrogen"]),
    _task("NR-PPAR-gamma", "peroxisome proliferator-activated receptor gamma (PPAR-gamma) "
          "agonism and activation",
          ["PPAR gamma agonist activation assay",
           "peroxisome proliferator activated receptor gamma ligand",
           "PPARgamma transactivation adipogenesis"]),
    _task("SR-ARE", "activation of the antioxidant response element (ARE) / Nrf2 pathway "
          "and oxidative-stress response",
          ["antioxidant response element ARE Nrf2 activation",
           "oxidative stress response Nrf2 pathway",
           "ARE reporter gene oxidative stress"]),
    _task("SR-ATAD5", "ATAD5-reported genotoxicity and DNA-damage / genome-instability "
          "stress response",
          ["ATAD5 genotoxicity DNA damage reporter",
           "genotoxicity DNA damage response assay",
           "genome instability replication stress chemical"]),
    _task("SR-HSE", "activation of the heat-shock response element (HSE) / heat-shock "
          "factor and proteotoxic stress response",
          ["heat shock response element HSE activation",
           "heat shock factor HSF1 proteotoxic stress",
           "heat shock response chemical stress assay"]),
    _task("SR-MMP", "disruption of mitochondrial membrane potential (MMP) and "
          "mitochondrial toxicity",
          ["mitochondrial membrane potential disruption toxicity",
           "mitochondrial toxicity membrane potential assay",
           "mitochondrial dysfunction chemical MMP"]),
    _task("SR-p53", "activation of the p53 DNA-damage response pathway",
          ["p53 DNA damage response activation",
           "p53 pathway genotoxic stress reporter",
           "p53 activation DNA damage chemical"]),
]

# --------------------------------------------------------------------------- SIDER (27 SOC)
def _soc(task, organ, extra=""):
    bio = ("adverse drug reactions, drug side effects and drug-induced toxicity affecting "
           "the %s (MedDRA system organ class '%s')%s, including pharmacovigilance and "
           "post-marketing safety reports" % (organ, task.replace("_", " "),
                                              (" " + extra if extra else "")))
    kws = [("%s adverse drug reaction" % organ),
           ("drug induced %s disorder toxicity" % organ),
           ("%s adverse effect pharmacovigilance" % organ)]
    return _task(task, bio, kws)


_SIDER_TASKS = [
    _soc("Blood_and_lymphatic_system_disorders", "blood and lymphatic system",
         "anemia, thrombocytopenia, neutropenia, coagulopathy"),
    _soc("Cardiac_disorders", "heart / cardiac system",
         "arrhythmia, QT prolongation, heart failure, cardiotoxicity"),
    _soc("Congenital,_familial_and_genetic_disorders", "congenital, familial and genetic conditions",
         "teratogenicity, birth defects"),
    _soc("Ear_and_labyrinth_disorders", "ear and labyrinth",
         "ototoxicity, hearing loss, tinnitus, vertigo"),
    _soc("Endocrine_disorders", "endocrine system",
         "thyroid dysfunction, adrenal insufficiency, hormonal disruption"),
    _soc("Eye_disorders", "eye / ocular system",
         "ocular toxicity, retinopathy, visual disturbance"),
    _soc("Gastrointestinal_disorders", "gastrointestinal tract",
         "nausea, vomiting, diarrhea, GI bleeding"),
    _soc("General_disorders_and_administration_site_conditions",
         "general systemic and administration-site reactions", "fatigue, fever, injection site reaction"),
    _soc("Hepatobiliary_disorders", "liver and biliary system",
         "hepatotoxicity, drug induced liver injury, cholestasis"),
    _soc("Immune_system_disorders", "immune system",
         "hypersensitivity, anaphylaxis, autoimmune reaction"),
    _soc("Infections_and_infestations", "infections",
         "opportunistic infection, sepsis, drug associated infection"),
    _soc("Injury,_poisoning_and_procedural_complications", "injury, poisoning and procedural complications",
         "overdose, toxicity, procedural complication"),
    _soc("Investigations", "laboratory investigations and test abnormalities",
         "liver enzyme elevation, abnormal laboratory finding"),
    _soc("Metabolism_and_nutrition_disorders", "metabolism and nutrition",
         "hyperglycemia, hyperlipidemia, electrolyte disturbance, weight change"),
    _soc("Musculoskeletal_and_connective_tissue_disorders", "musculoskeletal and connective tissue",
         "myopathy, rhabdomyolysis, arthralgia"),
    _soc("Neoplasms_benign,_malignant_and_unspecified_(incl_cysts_and_polyps)",
         "neoplasms (benign and malignant)", "carcinogenicity, secondary malignancy"),
    _soc("Nervous_system_disorders", "nervous system",
         "neurotoxicity, seizure, headache, peripheral neuropathy"),
    _soc("Pregnancy,_puerperium_and_perinatal_conditions", "pregnancy, puerperium and perinatal period",
         "pregnancy complication, fetal toxicity"),
    _soc("Product_issues", "product quality and device issues", "product defect"),
    _soc("Psychiatric_disorders", "psychiatric / mental health",
         "depression, anxiety, insomnia, psychosis, suicidal ideation"),
    _soc("Renal_and_urinary_disorders", "kidney and urinary tract",
         "nephrotoxicity, acute kidney injury, renal impairment"),
    _soc("Reproductive_system_and_breast_disorders", "reproductive system and breast",
         "sexual dysfunction, menstrual disorder, gynecomastia"),
    _soc("Respiratory,_thoracic_and_mediastinal_disorders", "respiratory and thoracic system",
         "dyspnea, cough, pulmonary toxicity, bronchospasm"),
    _soc("Skin_and_subcutaneous_tissue_disorders", "skin and subcutaneous tissue",
         "rash, Stevens-Johnson syndrome, pruritus, photosensitivity"),
    _soc("Social_circumstances", "social circumstances", ""),
    _soc("Surgical_and_medical_procedures", "surgical and medical procedures", ""),
    _soc("Vascular_disorders", "blood vessels / vascular system",
         "hypertension, hypotension, thrombosis, vasculitis"),
]

_DEFS = {
    "lipo": dict(name="lipo", pretty="lipophilicity (logP / logD7.4, octanol-water partition)",
                 mol_digits=5, tasks=_LIPO_TASKS),
    "hiv": dict(name="hiv", pretty="inhibition of HIV-1 replication / anti-HIV antiviral activity",
                mol_digits=6, tasks=_HIV_TASKS),
    "tox21": dict(name="tox21",
                  pretty="Tox21 toxicity endpoints (nuclear-receptor signaling and "
                         "cellular stress-response pathways)",
                  mol_digits=5, tasks=_TOX21_TASKS),
    "sider": dict(name="sider",
                  pretty="adverse drug reactions / side effects / drug safety across the "
                         "MedDRA system organ classes",
                  mol_digits=5, tasks=_SIDER_TASKS),
}


def _active():
    if DATASET not in _DEFS:
        raise ValueError("unknown EXP70_DATASET=%r (known: %s)" % (DATASET, sorted(_DEFS)))
    return _DEFS[DATASET]


CFG = _active()


def mol_id(idx):
    return "%s-m%0*d" % (CFG["name"], CFG["mol_digits"], idx)


def csv_path(split):
    d = CFG["name"]
    return os.path.join(DATA_ROOT, REFERENCE_SEED, d, "%s_%s.csv" % (d, split))


def pretty():
    return CFG["pretty"]


def task_id(task):
    return "%s:%s" % (CFG["name"], task)


def tasks():
    """List of (task_id, task, bio, topics) for the active dataset."""
    return [(task_id(t["task"]), t["task"], t["bio"], t["topics"]) for t in CFG["tasks"]]


def property_instruction(bio):
    return _INSTR_TMPL.format(bio=bio)


def property_topics():
    """BM25 prescreen topic map: task_id -> [keyword query strings] (per-task)."""
    return {task_id(t["task"]): list(t["topics"]) for t in CFG["tasks"]}


# ---------------------------------------------------------------------------
# Judge PROPERTY relevance prompts (generalized; dataset-property level, spanning
# this dataset's tasks). Molecule prompts stay dataset-agnostic in judge.py.
# ---------------------------------------------------------------------------
def judge_property_system_min():
    p = CFG["pretty"]
    return (
        "You are a biomedical relevance filter building a candidate literature corpus about "
        "%s. You receive a JSON object 'passages' mapping passage_id -> passage text. For "
        "EACH passage decide, using ONLY that passage's own text, whether it is relevant to "
        "%s — i.e. it discusses this property/endpoint (or one of its component tasks), "
        "its measurement, prediction, mechanisms or modulation, or a specific compound's "
        "behavior with respect to it. "
        "Output ONLY a single JSON object mapping EACH passage_id to exactly one label "
        "string: \"relevant\", \"irrelevant\", or \"uncertain\". "
        "Example: {\"ab12cd\":\"relevant\",\"ef34gh\":\"irrelevant\"}. "
        "Include every supplied passage_id exactly once and no other keys. "
        "Do NOT output explanations, reasons, evidence, extra fields, or any <think> text. "
        "Output the JSON object and nothing else." % (p, p)
    )


def judge_property_system():
    p = CFG["pretty"]
    return (
        "You are a biomedical relevance filter building a candidate literature corpus about "
        "%s. For EACH supplied passage decide whether it is relevant to this property/endpoint "
        "(or one of its component tasks) - i.e. it discusses the property, its measurement or "
        "prediction, mechanisms or modulation, or a specific compound's behavior with respect "
        "to it. Judge each passage INDEPENDENTLY using ONLY that passage's own text. Return "
        "ONLY a single JSON object: {\"verdicts\":[{\"passage_id\":<str>,\"relevance\":"
        "\"relevant|not_relevant|uncertain\",\"needs_context\":<bool>,\"evidence_spans\":"
        "[<short quotes copied from THIS passage>],\"reason\":<str>}]}. Include exactly one "
        "verdict object per supplied passage_id and no others." % p
    )


# ---------------------------------------------------------------------------
# CONTROLLED ATTRIBUTE VOCABULARY (frozen, versioned) for the Qwen3-32B structured-
# evidence extraction (structuring steps 11/14). TWO LEVELS: attribute_path =
# "<attribute_group>.<attribute>" (the specific property IS the attribute). Every
# finding also keeps attribute_label_original (paper's verbatim wording). "other" is
# allowed as a catch-all group and as an attribute within any group. The evidence
# index is queryable by group (e.g. lipophilicity.*) or exact group.attribute.
# ---------------------------------------------------------------------------
ATTR_VOCAB_VERSION = "v2-2level"

_TOX21_ENDPOINTS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
                    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
_SIDER_SOC_ATTR = [
    "blood_and_lymphatic", "cardiac", "congenital", "ear", "endocrine", "eye",
    "gastrointestinal", "general", "hepatobiliary", "immune_system", "infections",
    "injury_poisoning", "investigations", "metabolism_nutrition", "musculoskeletal",
    "neoplasms", "nervous_system", "pregnancy", "product_issues", "psychiatric",
    "renal_and_urinary", "reproductive_breast", "respiratory_thoracic",
    "skin_and_subcutaneous", "social_circumstances", "surgical_medical_procedures",
    "vascular"]

_HIER = {
    "lipo": dict(
        compound={
            "physicochemical": ["TPSA", "molecular_weight", "HBD", "HBA", "rotatable_bonds",
                                 "aromatic_rings", "other"],
            "lipophilicity": ["logP", "clogP_computed", "logD7.4", "logD_other_pH",
                              "octanol_water_partition", "other"],
            "ionization": ["pKa", "pct_ionized_pH7.4", "charge_class", "other"],
            "solubility": ["aqueous_solubility", "logS", "other"],
            "permeability": ["caco2_papp", "pampa_papp", "mdck_papp", "other"],
            "transporters": ["pgp_efflux_ratio", "bcrp_substrate", "other"],
            "measurement": ["shake_flask", "chromatographic_HPLC", "computational", "other"],
            "other": ["other"]},
        task={
            "qsar_model_performance": ["rmse", "mae", "r2", "accuracy", "auc", "method", "other"],
            "predictive_feature": ["logd_vs_logp_ionization", "lipophilicity_predictive_feature", "other"],
            "measurement_method": ["shake_flask", "chromatographic_HPLC", "computational", "other"],
            "dataset_benchmark": ["dataset_benchmark", "other"],
            "other": ["other"]}),
    "hiv": dict(
        compound={
            "antiviral_activity": ["EC50", "IC50", "pct_inhibition", "classification", "other"],
            "mechanism": ["target_class", "rt_inhibitor", "integrase_inhibitor",
                          "protease_inhibitor", "entry_fusion_inhibitor", "other"],
            "selectivity": ["CC50", "cytotoxicity", "selectivity_index", "other"],
            "resistance": ["mutation", "cross_resistance", "other"],
            "admet": ["logP", "permeability", "metabolic_stability", "other"],
            "other": ["other"]},
        task={
            "qsar_model_performance": ["accuracy", "auc", "f1", "method", "other"],
            "antiviral_sar": ["sar", "other"],
            "potency_metric_assay": ["assay_protocol", "potency_metric", "other"],
            "hiv_activity_mechanism": ["mechanism", "target_class", "other"],
            "dataset_benchmark": ["dataset_benchmark", "other"],
            "other": ["other"]}),
    "tox21": dict(
        compound={
            "assay_activity": (list(_TOX21_ENDPOINTS)
                               + ["AC50_%s" % e for e in _TOX21_ENDPOINTS] + ["other"]),
            "mechanism": ["nuclear_receptor_agonism", "nuclear_receptor_antagonism",
                          "stress_response_activation", "mitochondrial_toxicity",
                          "endocrine_disruption", "other"],
            "structure": ["structural_alert", "toxicophore", "other"],
            "general": ["general_toxicity", "other"],
            "other": ["other"]},
        task={
            "assay_target_biology": ["target_biology", "other"],
            "toxicity_mechanism": ["mechanism", "other"],
            "structure": ["structural_alert", "toxicophore", "other"],
            "qsar_model_performance": ["accuracy", "auc", "method", "other"],
            "assay_protocol": ["protocol", "other"],
            "dataset_benchmark": ["dataset_benchmark", "other"],
            "other": ["other"]}),
    "sider": dict(
        compound={
            "adverse_reaction": list(_SIDER_SOC_ATTR) + ["other"],
            "severity": ["black_box_warning", "contraindication", "frequency", "other"],
            "mechanism": ["adverse_mechanism", "on_off_target", "other"],
            "biomarker": ["ALT_AST", "QT", "creatinine", "eosinophilia", "other"],
            "pharmacology": ["drug_class", "target_moa", "indication", "other"],
            "other": ["other"]},
        task={
            "adr_mechanism": ["mechanism", "other"],
            "safety_biomarker": ["ALT_AST", "QT", "creatinine", "other"],
            "structure_toxicity_relationship": ["sar", "other"],
            "qsar_model_performance": ["accuracy", "auc", "method", "other"],
            "dataset_benchmark": ["dataset_benchmark", "other"],
            "other": ["other"]}),
}


def _paths(hier):
    out = []
    for g, attrs in hier.items():
        for a in attrs:
            out.append("%s.%s" % (g, a))
    return out


def attribute_vocab():
    h = _HIER.get(CFG["name"], dict(compound={"other": ["other"]}, task={"other": ["other"]}))
    return dict(version=ATTR_VOCAB_VERSION, dataset=CFG["name"], levels=2,
                attribute_path_format="attribute_group.attribute",
                compound=h["compound"], task=h["task"],
                compound_paths=_paths(h["compound"]), task_paths=_paths(h["task"]))
