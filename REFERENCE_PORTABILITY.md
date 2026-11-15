> **Public repository scope:** This document describes the complete
> locally validated reproducibility package. The current public GitHub
> repository contains the verified statistical reproduction materials
> and selected research files, but does not yet provide a standalone
> executable hybrid agent. Commands referencing `run_agent.py`,
> `sts2_env`, or `card_reference_cache.py` require the complete package
> and the corresponding authorized reference inputs.

# Card reference portability

**Verdict: PARTIAL.** The trained hybrid can use one locally generated static JSON cache in place of the 579 C# files at runtime. Two development runs matched the original reference-backed path. The cache still derives from game reference data, and its redistribution rights are unresolved. The candidate ships the generator and loader, with no generated cache.

## Actual dependencies and minimum inputs

| Input | Frozen usage | Minimum validated requirement |
|---|---|---|
| `docs/CARDS_REFERENCE.md` | `sts2_env/cards/factory.py` parses definitions for fallback card construction, identity aliases, rarity, target, tags, values and upgrades. Reward generation probes the registry, so a run can need definitions beyond its visible Ironclad offers. | Required in both modes. Bundled local copy SHA-256 `fa437fef041e737c1ad5d212821a315a8054d0b9a1926c2be1fc3c3e3af84e60`. Redistribution pending. |
| 579 top-level C# files | `reference_static_metadata.py` parses them in sorted order into a registry keyed by `CardId`. Consumers in `cards/base.py`, `cards/factory.py` and `run/run_state.py` use it for card properties and eligibility. Reward code in `run/rewards.py` uses eligible card pools, so changed metadata can change seeded offers. | For the original path, all 579 byte-matching files. For the portable path, the 579 corresponding typed metadata records in one cache. No smaller subset was qualified. |

The live simulator consumers use the base metadata registry. Separate upgraded and dynamic-variable source parser helpers have no live caller in the frozen simulator tree and are outside the cache. Markdown definitions and Python effect implementations remain in use. The C# registry includes IDs, costs, type/target and rarity, keywords/tags, generation and playability flags, upgrade limits, pool, multiplayer constraint and other static fields. The cache contains those typed fields only. It contains no C# text or effect bodies.

## Implementation and verification

`card_reference_cache.py` calls the existing parser after checking the original per-file inventory, serializes its typed result in registry order, and reconstructs the same dataclasses, enums and frozensets. All 579 ordered records round-tripped exactly. The release launcher binds the reconstructed registry in memory before gameplay. Frozen source files and model archives are unchanged. The C# reference-backed route remains available.

The retained C# corpus is 855,478 bytes across 579 files. The JSON is 332,754 bytes. The cache binds the Markdown hash, frozen parser hash `84d8fc8391cebb176b04049ec835e42032a31a6ccc31aad137681f4786569ea6`, and source inventory digest `61ecf331f9279f8cc828e50153dc0845421867992218f839cd8506ae3637433a`. The pinned canonical record digest is `4ea5c1a3258787d73f41b5c3999edefc70dcaef1dcb1f5873223c61095cddc82`. A modified cost with a recomputed embedded hash was rejected against the pinned digest. Individual source hashes are in `frozen_source.json`. These bind the retained v0.107.1 checkout; the historical Phase 7H seal does not independently bind the complete corpus.

## Behavioral comparison

Paired tests used fresh Python processes, the same original trained policy tensors and action seed 42, CPU inference, and the native 2,000-decision limit. In the cache process the C# parser directory was deliberately set to a nonexistent path. Ordered traces matched for the observed eligible reward offers and card identities, Card PPO legal masks, selected actions and policy metadata, native action masks, and Combat/Arm D controller decisions. Final launcher output also matched.

| Opened development origin | Card root | Matched endpoint | Native decisions / learned picks | Matching trace SHA-256 |
|---|---:|---|---:|---|
| 42 | 1 | Completed episode, death on floor 16 | 150 / 6 | `13aaeedf8b904e13f4e41cbb1c592c405e52a5a56b1d8e8aea84d97321230dd9` |
| 12104718 | 2 | Two-card cutoff on floor 3 | 48 / 2 | `cadddeab507a0f137ecc8c12cd0b4718168b3bb3f508b048f02ee412c676d7b4` |

No mismatch appeared in these paired tests. The earlier reference-free smoke offered Hemokinesis where the reference-backed run offered Howl From Beyond. Omitting the registry changes fallback metadata and can alter reward behavior. No per-flag cause was isolated. The comparison qualifies these development traces and the typed registry round-trip. It is not an exhaustive simulator parity claim or a Phase 7H rerun.

## Professor setup

Install Python dependencies per [REPRODUCIBILITY.md](REPRODUCIBILITY.md), then run from the candidate root in PowerShell. Obtain authorized access to the exact retained `MegaCrit.Sts2.Core.Models.Cards` directory identified in [UPSTREAM_ATTRIBUTION.md](UPSTREAM_ATTRIBUTION.md) and `frozen_source.json`. A current game installation alone does not supply the hash-pinned C# directory. Request the matching authorized reference inputs from the project submitter if they are unavailable.

```powershell
$refs = 'D:\authorized\decompiled\MegaCrit.Sts2.Core.Models.Cards'
python -B run_agent.py --verify-models
python -B card_reference_cache.py --card-reference $refs --output local-reference.json
python -B run_agent.py --card-reference-cache local-reference.json
python -B reproduce_results.py
```

Once built, the local cache can be used without the external C# directory. Keep it local unless redistribution has been cleared. For direct reference-backed execution, run `python -B run_agent.py --card-reference $refs`. If neither authorized input is available, checkpoint verification and statistical reproduction still work. Reference-free gameplay remains a diagnostic with different data.

No research source, checkpoint, or historical results were changed. No models were retrained, no sealed panel was rerun, and nothing was published.



