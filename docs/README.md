# Reusable setup and working experience

Use `docs/` for knowledge that helps the next GPU session work faster. Save study-specific results,
configuration, logs and incident details under [report/<study-id>/](../report/README.md).

| Document | Purpose |
| --- | --- |
| [agent_quickstart.md](agent_quickstart.md) | **Start here to run the study quickly:** commands, gates, timings, failure table |
| [experience.md](experience.md) | Read first: recorded sessions, CUDA/EP fixes, KV checks, instrumentation and open failures |
| [gpu_setup_and_fixes.md](gpu_setup_and_fixes.md) | Frozen runtime, node setup and inherited measurement pitfalls |
| [setup_and_download_experience.md](setup_and_download_experience.md) | Practical setup/download workflow and avoiding repeated work |
| [model_architectures.md](model_architectures.md) | Checkpoint identities and attention architecture inspection leads |
| `import_provenance.json` | Hashes of the Frontier source material originally imported |

After a useful fix or setup/download improvement, add a short reusable lesson here: symptom/task,
what worked, exact version/config and commands, evidence link into `report/<study-id>/`, and limits.
Separate observed improvements from ideas that have not been timed. Keep per-model completion
times and performance results in the report. Update this index when adding a document.
