# Data

The three CSV files below are included in this repository and are exact copies of the inputs used in the experiments. They require no download or conversion.

| File | Dataset | Observations | Columns |
| --- | --- | --- | --- |
| `hk.csv` | Hong Kong pollutants and cardiovascular admissions | 1,032 | Five observed variables |
| `traffic_data_6_9.csv` | PeMSBay, morning interval | 2,520 | Timestamp and seven sensor variables |
| `traffic_data_9_12.csv` | PeMSBay, later interval | 2,520 | Timestamp and seven sensor variables |

Run the notebooks from the repository root after installing `requirements.txt`. Only the Causal Chambers wind-tunnel inputs require downloading. Experiment caches are generated locally and are excluded from version control.

## Hong Kong pollutants and cardiovascular admissions

`Fig6_CVDs and air polluations.ipynb` reads the included `data/hk.csv`. Its columns are `cardio,no2,so2,rspar,o3`; the notebook uses the first four columns and performs interpolation and normalization internally. The CSV supplies row indices rather than a separate date column.

Source: [Intervened Reservoir Computing repository, commit c43310cb9de275f8cbc2a548abe80dccb1539b31](https://github.com/Jintong-CNS/Intervened-Reservoir-Computing/blob/c43310cb9de275f8cbc2a548abe80dccb1539b31/data/hk.csv). The upstream paper is [Detecting dynamical causality via intervened reservoir computing](https://doi.org/10.1038/s42005-024-01730-6). Consult its dataset references for provenance and use terms.

## PeMSBay

`Fig5_traffic.ipynb` reads the two included traffic CSVs and performs interpolation and normalization internally.

The sensor columns, in order, are:

```text
400911,400863,402364,404753,400001,400965,402362
```

Both outputs cover 1 January through 11 March 2017 inclusive, at the original five-minute spacing:

- `traffic_data_6_9.csv`: 06:00 through 08:55 each day.
- `traffic_data_9_12.csv`: 09:00 through 11:55 each day.

Each CSV contains a `time` column followed by the seven sensor columns. These inputs are subsets of the PeMSBay benchmark described in the [DCRNN repository](https://github.com/liyaguang/DCRNN); the corresponding source archive is available in [Zenodo record 4263971](https://doi.org/10.5281/zenodo.4263971).

## Causal Chambers wind tunnel

`Tab1_wind-tunnel.ipynb` downloads the official `wt_walks_v1.zip` archive and standard wind-tunnel adjacency table when needed. To download those inputs separately, run from the repository root:

```bash
python scripts/prepare_data.py wind-tunnel
```

Files are stored in `data/CausalChambers/`. Both files are checked against the SHA-256 values used by this release. The adjacency download is pinned to package commit `9fb5d82e391bb91a89a64f2e1c9b6ae8e7aed6d6`.

The notebook reads the actuator-random-walk records directly from the ZIP archive. Records 1-10 are used for training, 11-13 for threshold calibration, and 14-16 for testing. No interpolation is applied; normalization is fitted on the pooled training records.

- Dataset: https://github.com/juangamella/causal-chamber/tree/main/datasets/wt_walks_v1
- Ground-truth package: https://github.com/juangamella/causal-chamber-package
- Paper: Gamella, J. L., Peters, J. and Buhlmann, P. *Causal chambers as a real-world physical testbed for AI methodology*. Nature Machine Intelligence 7, 107-118 (2025). https://doi.org/10.1038/s42256-024-00964-x

## Synthetic Lorenz systems

The Lorenz and appendix notebooks generate their observations using the supplied numerical integration code. No external data download is required.

## Data attribution

Dataset rights and use terms remain with their respective providers; the GPL-3.0 software license does not relicense the observations. Cite the corresponding data sources when using these files.
