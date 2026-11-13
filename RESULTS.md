# Report tables and evidence

Run `python -B reproduce_results.py` from this directory. Inputs in `release/phase9c_v1/data/` are checked against the candidate artifact manifest. Each protocol has its own population.

| Table | Retained original evidence | Recomputed result |
|---|---|---|
| 1 — HP curriculum | `output/overnight_hp_pipeline/cache/hp_comparison_20381e74d95c8dbdd1a1/{per_seed.csv,paired_by_hp.csv}`, copied with the `hp_` prefix | Control→curriculum 826→865, 816→863, 820→850, each out of 960 on reused hallway validation. Paired mean +4.0278 pp, sample SD 0.8859 pp. |
| 2 — Independent card evaluation | `output/phase7h4_analysis_v1/verified/{per_arm.csv,paired_comparisons.csv,root_mean_sensitivity.csv,per_origin.csv}` and the H manifest's statistical key | Clears 10/2/8 and 4/4 per 100. Wilson intervals, six exact McNemar tests and Holm adjustment, and paired and joint-origin conditional bootstraps. The released vectors retain aligned binary outcomes without origin labels, action seeds or journals. The statistical key recreates the 10,000 analysis resamples. |
| 3 — Training-origin diversity | `output/phase8e7_offline_analysis_v1/` root and condition metrics, update-32 paired table and joint interval. Phase 9C derived the joint histogram from the retained wave receipt. | Sparse64 33/384, Sparse128 27/384. −1.5625 pp, exact empirical joint-origin conditional 95% interval [−4.9479,+1.8229] pp. Three exact McNemar tests with Holm adjustment. This is the opened development panel. |

The bundle supports recomputation from saved evidence. Reverification of complete historical case provenance or new execution of the experimental protocols requires the retained workspace inputs and separate authorization.

