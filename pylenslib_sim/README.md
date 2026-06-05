# pylenslib_sim

Lightweight orchestrator for producing P_GGSL mock catalogs using the
existing `agent_sim` PyLenslib simulator.

Usage

1. Edit or copy `config_examples/pggsl_config.yaml` and adjust parameters.
2. Run:

```bash
source /home/alessio/lensing/.lens/bin/activate
python -m pylenslib_sim.main pylenslib_sim/config_examples/pggsl_config.yaml
```

Outputs are written into `output_dir` declared in the config (default `pylenslib_outputs`).
