# Custom PhaseNet weights

Place custom SeisBench PhaseNet weights here as a pair of files:

```
<weightname>.pt.v1      # torch state dict (SeisBench format)
<weightname>.json.v1    # SeisBench weight metadata (at minimum: {})
```

They are copied into `/root/.seisbench/models/v3/phasenet/` at Docker build
time and can then be selected at job submission with `--weight <weightname>`.

The default weights (`instance`, plus the set from the SeisBench model
repository baked in by the Dockerfile) remain available.
