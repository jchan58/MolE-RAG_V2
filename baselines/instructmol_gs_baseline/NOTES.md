# instructmol_gs_baseline

Baseline: InstructMol-GS (Vicuna-7B + GIN)
Status: **BLOCKED — no released checkpoint (STOP EARLY).** See RESULTS.md.

Recon (2026-10-11): repo github.com/IDEA-XL/InstructMol confirmed (HEAD 5b2aa484, 2024-12-02).
README '### InstructMol Weights' = '* TODO: coming soon'. No trained checkpoint (GS, G, or the
stage-1 projector) on HF/Drive/anywhere (checked IDEA-XL org, HF search, author accts hcaoaf/hcao,
project page). Only component weights released: vicuna-7b-v1.3 (open), MoleculeSTM GIN init
(molecule_model.pth), scibert; + training code. Nothing to run -> 0 SageMaker jobs, 0 billable.

Even with a checkpoint, InstructMol's MoleculeNet eval supports only bace/bbbp/hiv (3/9); its
regression eval = QM9 HOMO/LUMO, not MoleculeNet esol/freesolv/lipo.

Deliverables: RESULTS.md (blocker table), analysis/instructmol_gs_results.json (full recon +
prompt/scoring provenance), code/UPSTREAM_* (preserved upstream README/Evaluation/eval scripts).
