"""
Stage 0 — ingest a MoleculeNet scaffold-split dataset (generalized).

Reads seed_0/<dataset>/<dataset>_{train,valid,test}.csv ('smiles' column). The
multi-task datasets (sider, tox21) repeat each SMILES once per task; dedup is by
RDKit canonical isomeric SMILES, so those rows collapse to one unique structure
whose labels/tasks/splits are all retained. Query generation never sees labels.

Outputs (under RUN_DIR/data):
  input_molecules.jsonl   — one line per ORIGINAL row
  structures.jsonl        — one line per UNIQUE valid canonical structure
  ingest_summary.json     — counts, hashes, timings
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import datasets as DS
import csv

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, False
    try:
        can = Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None, None, False
    try:
        ik = Chem.MolToInchiKey(mol)
    except Exception:
        ik = None
    return can, ik, True


def read_split(split):
    rows = []
    path = DS.csv_path(split)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            rows.append(dict(split=split, row_index=i,
                             smiles=(r.get("smiles") or "").strip(),
                             label=r.get("label"), task=r.get("task"),
                             dataset=r.get("dataset"), task_type=r.get("task_type"),
                             metric=r.get("metric")))
    return rows


def file_hashes():
    h = {}
    for split in DS.SPLITS:
        path = DS.csv_path(split)
        h[split] = dict(path=path, sha256=C.sha256_file(path), bytes=os.path.getsize(path))
    return h


def run():
    C.ensure_dirs()
    start = C.now_iso()
    C.log("INFO", "ingest", "start dataset=%s" % DS.CFG["name"])

    all_rows = []
    for sp in DS.SPLITS:
        all_rows.extend(read_split(sp))

    input_records = []
    structures = {}        # canonical_smiles -> structure record
    invalid_rows = []
    for r in all_rows:
        can, ik, ok = canonicalize(r["smiles"])
        row_id = "%s:%s:%d" % (DS.REFERENCE_SEED, r["split"], r["row_index"])
        rec = dict(row_id=row_id, seed=DS.REFERENCE_SEED, split=r["split"],
                   row_index=r["row_index"], original_smiles=r["smiles"],
                   label=r["label"], task=r["task"], dataset=r["dataset"],
                   valid_structure=ok, canonical_smiles=can, inchikey=ik, mol_id=None)
        if ok:
            st = structures.get(can)
            if st is None:
                st = dict(canonical_smiles=can, inchikey=ik,
                          example_original_smiles=r["smiles"],
                          original_smiles_variants=set(), splits=set(),
                          tasks=set(), row_ids=[], labels=[])
                structures[can] = st
            st["original_smiles_variants"].add(r["smiles"])
            st["splits"].add(r["split"])
            if r["task"]:
                st["tasks"].add(r["task"])
            st["row_ids"].append(row_id)
            st["labels"].append(r["label"])
        else:
            invalid_rows.append(rec)
        input_records.append(rec)

    ordered = sorted(structures.keys())
    mol_id_of = {}
    for idx, can in enumerate(ordered):
        mid = DS.mol_id(idx)
        mol_id_of[can] = mid
        structures[can]["mol_id"] = mid
    for rec in input_records:
        if rec["valid_structure"] and rec["canonical_smiles"] in mol_id_of:
            rec["mol_id"] = mol_id_of[rec["canonical_smiles"]]

    with open(C.P("data", "input_molecules.jsonl"), "w") as f:
        for rec in input_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    struct_records = []
    for can in ordered:
        st = structures[can]
        srec = dict(mol_id=st["mol_id"], canonical_smiles=st["canonical_smiles"],
                    inchikey=st["inchikey"],
                    example_original_smiles=st["example_original_smiles"],
                    original_smiles_variants=sorted(st["original_smiles_variants"]),
                    splits=sorted(st["splits"]),
                    split=(sorted(st["splits"])[0] if len(st["splits"]) == 1
                           else "+".join(sorted(st["splits"]))),
                    multi_split=(len(st["splits"]) > 1),
                    tasks=sorted(st["tasks"]),
                    row_ids=st["row_ids"], labels=st["labels"],
                    n_source_rows=len(st["row_ids"]))
        struct_records.append(srec)
    with open(C.P("data", "structures.jsonl"), "w") as f:
        for srec in struct_records:
            f.write(json.dumps(srec, ensure_ascii=False) + "\n")

    split_counts = {}
    for sp in DS.SPLITS:
        rowsN = sum(1 for r in input_records if r["split"] == sp)
        validN = sum(1 for r in input_records if r["split"] == sp and r["valid_structure"])
        uniqN = len(set(r["canonical_smiles"] for r in input_records
                        if r["split"] == sp and r["valid_structure"]))
        split_counts[sp] = dict(rows=rowsN, valid_rows=validN, unique_structures=uniqN)

    summary = dict(
        run_started=start, finished=C.now_iso(),
        dataset=DS.CFG["name"], property=DS.pretty(),
        input_root=DS.DATA_ROOT, reference_seed=DS.REFERENCE_SEED,
        reference_files=file_hashes(),
        total_input_rows=len(input_records),
        valid_rows=sum(1 for r in input_records if r["valid_structure"]),
        invalid_rows=len(invalid_rows),
        invalid_row_ids=[r["row_id"] for r in invalid_rows],
        unique_valid_structures=len(ordered),
        per_split=split_counts,
        tasks=sorted({t for st in structures.values() for t in st["tasks"]}),
        note=("MoleculeNet %s scaffold split (seed_0). Multi-task rows (same SMILES "
              "across task labels) are deduped by RDKit canonical isomeric SMILES; "
              "labels/tasks/splits retained but EXCLUDED from query-generation inputs. "
              "Molecule identity for querying resolved offline from the structure."
              % DS.CFG["name"]),
    )
    C.write_json(C.P("data", "ingest_summary.json"), summary)
    C.log("INFO", "ingest",
          "dataset=%s rows=%d valid=%d invalid=%d unique_structures=%d"
          % (DS.CFG["name"], summary["total_input_rows"], summary["valid_rows"],
             summary["invalid_rows"], summary["unique_valid_structures"]))
    return summary


if __name__ == "__main__":
    run()
