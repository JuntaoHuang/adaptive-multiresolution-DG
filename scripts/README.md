# Plot Scripts

## `plot_profile1d.py`

Run from the project root:

```bash
python3 scripts/plot_profile1d.py --help
```

Common options:

- `--final`: Single-file mode (similar to the old `plot_profile1d.py`)
- `--input-file`: Input file in `--final` mode, default is `profile1D_final.txt`
- `--time-interval`: Minimum time gap between snapshots in time-series mode
- `--tmin`, `--tmax`: Time range filters
- `--output`: Output image filename
- `--show`: Display the figure window after running

Examples:

```bash
# 1) Time-series mode, plot all snapshots
python3 scripts/plot_profile1d.py --show

# 2) Time-series mode, sparse sampling
python3 scripts/plot_profile1d.py --time-interval 0.02 --tmin 0.1 --tmax 0.2 --show

# 3) Final single-file mode
python3 scripts/plot_profile1d.py --final --input-file profile1D_final.txt --show
```

## One-Command Export (Recommended)

Use this script to export `profile1D_*.txt` and figures into a new folder under `results/`,
and save reproducibility info (`metadata.txt` + `reproduce.sh`):

```bash
bash scripts/export_profile_bundle.sh
```

Equivalent Make target:

```bash
make profile-export
```

Show export options:

```bash
make profile-export-help
```

Examples:

```bash
# Time-series export with defaults
bash scripts/export_profile_bundle.sh

# Time-series export with filters
bash scripts/export_profile_bundle.sh --time-interval 0.05 --tmin 0.1 --tmax 0.2

# Final-only export
bash scripts/export_profile_bundle.sh --final --input-file profile1D_final.txt

# Record solver command for reproducibility
bash scripts/export_profile_bundle.sh --run-command "./02_hyperbolic_06_burgers_shock_1d_filter -NM 7 -N0 7 -cfl 0.1 -tf 0.2 -nu 1.0 -p 100 -v 1"
```

You can also pass it through Make:

```bash
make profile-export RUN_CMD="./02_hyperbolic_06_burgers_shock_1d_filter -NM 7 -N0 7 -cfl 0.1 -tf 0.2 -nu 1.0 -p 100 -v 1"
```
