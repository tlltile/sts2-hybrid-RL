# Upstream attribution

The cached simulator origin is [zhiyue/sts2-rl-agent](https://github.com/zhiyue/sts2-rl-agent), revision `5d874e855ae1d0550d5c30416fadd7dbb0d133b7`. The cached tree has no verified upstream license grant. Slay the Spire 2 is the work of Mega Crit. The historical simulator reference is game v0.107.1.

`frozen_source.json` records SHA-256 for every copied source path and classifies it against the cached upstream tree as upstream, modified upstream, or local addition. The main source is copied byte-for-byte from the Phase 7H source snapshot. Changes from upstream include the native Act 1 and reward interfaces, simulator corrections and training configuration. Local additions include Card PPO and its encoder and decision boundary, Arm D control, experiment scripts, and the submission launch and analysis entry points. `docs/CARDS_REFERENCE.md` is a required supplemental input copied from the retained checkout. Its hash is recorded separately from the archived Phase 7H source.

Gymnasium, Stable-Baselines3, sb3-contrib, NumPy and PyTorch are external dependencies. This candidate does not vendor their distributions. Preserve their applicable notices if a dependency distribution is bundled later.

