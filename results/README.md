# Results

Raw outputs behind the numbers in the project report (Chapter 6). Datasets and score arrays are not included.

| File | What it holds | Report section |
|---|---|---|
| `eval.json` | SOCOFing test EER, FAR/FRR at the validation threshold, linkability, CKKS precision | 6.2, 6.3, 6.8, 6.10 |
| `experiments.json` | harder protocols over 3 seeds, D<->sys with noise floor, MLP attacker, service-wide Procrustes attack | 6.4, 6.8, 6.9 |
| `optimization.json`, `optimization_search_history.json` | OPA vs random search, 3-seed retraining of the best settings | 6.6 |
| `fvc.json` | multi-session data: zero-shot and first fine-tuning (96 px) | 6.5 |
| `fvc_v2_summary.json` | 192 px recipe and the same recipe at 96 px | 6.5 |
| `nested_and_minutiae_summary.json` | nested recipe selection (16.8%) and the SourceAFIS baseline (5.8%) | 6.5.1, 6.5.2 |
| `attacks_v3.json` | per-user known-plaintext attack and collection-level linkage | 6.8.1, 6.9.1 |
| `mac_benchmark_and_enclave.json`, `benchmark_rop_cached.json` | per-stage latency on the MacBook, Secure Enclave tests | 6.11, 6.13 |
| `mac_followup_summary.json` | 192 px and Secure Enclave latency, forged-score test, test count | 6.11, 6.13 |
| `loadtest.json` | throughput and latency with 1 to 16 clients | 6.11 |
| `resolution_counts.json` | image sizes per database (why 192 px helps) | 6.5 |
