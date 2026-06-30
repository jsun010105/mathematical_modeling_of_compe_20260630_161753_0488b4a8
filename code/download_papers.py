#!/usr/bin/env python3
"""Download curated arXiv PDFs into papers/."""
import os, time, sys
import requests

PAPERS = [
    # (arxiv_id, filename_stub)
    ("2307.15771", "McGrath2023_hydra_effect_self_repair"),
    ("2402.15390", "RushingNanda2024_explorations_self_repair"),
    ("2211.00593", "Wang2022_IOI_circuit_backup_name_movers"),
    ("2406.11717", "Arditi2024_refusal_single_direction"),
    ("2502.17420", "Wollschlager2025_geometry_refusal_concept_cones"),
    ("2602.02132", "Joad2026_more_to_refusal_than_single_direction"),
    ("2511.08379", "Piras2025_SOM_directions_multidirectional_refusal"),
    ("2406.05946", "Qi2024_safety_more_than_few_tokens_deep"),
    ("2310.03693", "Qi2023_finetuning_compromises_safety"),
    ("1004.2515",  "WilliamsBeer2010_nonnegative_decomposition_multivariate_info"),
    ("2210.02996", "Proca2022_synergistic_information_neural_networks"),
    ("2408.05451", "Hanni2024_math_models_computation_superposition"),
    ("2512.13568", "Bereska2025_superposition_lossy_compression_adversarial"),
    ("2309.16042", "ZhangNanda2023_best_practices_activation_patching"),
    ("2301.04709", "Geiger2023_causal_abstraction_mech_interp"),
    ("0907.0510",  "WhitacreBender2009_degeneracy_robustness_evolvability"),
    ("2604.04385", "Frank2026_how_alignment_routes_policy_circuits"),
]

OUT = os.path.join(os.path.dirname(__file__), "..", "papers")
OUT = os.path.abspath(OUT)
HEADERS = {"User-Agent": "Mozilla/5.0 (research literature review)"}

def fetch(aid, stub):
    dest = os.path.join(OUT, f"{aid}_{stub}.pdf")
    if os.path.exists(dest) and os.path.getsize(dest) > 20000:
        return f"SKIP (exists) {dest}"
    url = f"https://arxiv.org/pdf/{aid}.pdf"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=90)
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                with open(dest, "wb") as f:
                    f.write(r.content)
                return f"OK   {len(r.content)//1024} KB  {dest}"
            else:
                last = f"status={r.status_code} head={r.content[:8]}"
        except Exception as e:
            last = str(e)
        time.sleep(3)
    return f"FAIL {aid}  ({last})"

if __name__ == "__main__":
    for aid, stub in PAPERS:
        print(fetch(aid, stub), flush=True)
        time.sleep(1.5)
