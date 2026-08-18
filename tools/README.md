# Reproduction tools

`reproduce.py` is the implementation behind the stable root command:

```bash
python reproduce.py --check
```

It validates source/data provenance and checksums, orchestrates cached redraws,
and launches the optional matched-control or full workflows. Users should call
the root entry point so the implementation can move without changing the
documented interface.
