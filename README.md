# ISTC

Code for *Detecting interventional spatiotemporal causality in dynamical systems* by Tianying Chen, Jintong Zhao, Siyang Leng and Wei Lin.

ISTC learns a reservoir-computing surrogate of dynamical observations and quantifies target responses to source-specific interventions through the dynamic deviation index (DDI).

## Installation

Use Python 3.10. From the repository root, create an environment and install the dependencies:

```bash
conda env create -f environment.yml
conda activate istc
python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (ISTC)"
python scripts/check_install.py
```

Alternatively, create a Python 3.10 virtual environment and run `python -m pip install -r requirements.txt`, followed by the kernel registration and installation check above. The installation check trains only small temporary models in memory; it does not reproduce the paper results.

## Instructions for use

View files `Fig*.ipynb` to inspect and reproduce the corresponding figure results in the article.

View file `Tab1_wind-tunnel.ipynb` to reproduce the wind-tunnel causal graph recovery results in Table 1.

View files `A.2.ipynb` to `A.7.ipynb` to reproduce the corresponding appendix experiments.

View file `ESN.ipynb` for the reservoir computing implementation and `Reconstruct.ipynb` for intervention-based DDI calculation.

View files `time_Lorenz_data_*.ipynb` for coupled Lorenz data generation.

View file `Plot_DDI.ipynb` for plotting functions and `time_Lorenz_ROC_comparison.py` for the PCMCI and lagged CCM comparisons used in Figure 3.

Run notebooks from the repository root and execute their cells in order. Start JupyterLab with `jupyter lab`, or use the command-line runner:

```bash
python scripts/run_notebook.py A.3.ipynb
```

The runner saves an executed copy under `result/executed/` and leaves the published notebook unchanged. Full experiments use the settings in each notebook, not the reduced installation-check settings.

## Experiments and execution order

| Notebook | Experiment | Input preparation |
| --- | --- | --- |
| `Fig2_Fig7_Lorenz_experiment.ipynb` | Lorenz causal topology and delay experiments | Generates synthetic observations |
| `Fig3_comparison.ipynb` | ISTC, PCMCI and lagged CCM ROC comparison | Run the Lorenz notebook first |
| `Fig5_traffic.ipynb` | PeMSBay traffic sensors | Uses the two traffic CSV files included in `data/` |
| `Fig6_CVDs and air polluations.ipynb` | Hong Kong pollutants and cardiovascular admissions | Uses `data/hk.csv`, included in this release |
| `Tab1_wind-tunnel.ipynb` | Wind-tunnel directed-edge recovery | Downloads official data when missing |
| `A.2.ipynb` | DDI and true-system intervention responses | Generates synthetic observations |
| `A.3.ipynb` | Coupling strength and DDI | Generates synthetic observations |
| `A.4.ipynb` | Dynamical and observational noise | Generates synthetic observations |
| `A.5.ipynb` | Training length, reservoir size and spectral radius | Generates synthetic observations |
| `A.6.ipynb` | Closely spaced causal delays | Generates synthetic observations |
| `A.7.ipynb` | Unknown-delay search and joint candidate evaluation | Generates synthetic observations |

All appendix experiments are independent of the Figure 2 output files. Figure 3 explicitly consumes Figure 2 results and must be run afterwards. See [data/README.md](data/README.md) for data sources, exact intervals and schemas.

## Outputs and computational requirements

The Hong Kong and traffic CSV inputs are included in `data/`. Only the wind-tunnel inputs require downloading, either through `Tab1_wind-tunnel.ipynb` or with `python scripts/prepare_data.py wind-tunnel`. No pretrained weights, computed DDI results or executed notebook outputs are included. The notebooks create missing output directories and compute results from scratch. Compatible results are reused where the corresponding notebook implements caching.

- Appendix caches are saved under `result/A.2/` through `result/A.7/`.
- Appendix PNGs are saved directly under `result/figure/`.
- Wind-tunnel weights, scores and tables are saved under `result/Wind_tunnel/`.
- Other figure experiments use their notebook-specific directories under `result/`.

The formal Lorenz and traffic experiments use dense recurrent matrices with thousands of reservoir units. CPU runtime and memory use are substantially greater than for the installation check. The full A.4 run can save approximately 142 GiB of model weights, so allow sufficient disk space or run selected conditions deliberately. Changing model counts, training lengths or reservoir sizes creates a different experimental setting.

Use the pinned numerical dependencies for reproduction. Floating-point libraries and platforms can affect results. Times New Roman is requested for figures; Matplotlib uses a fallback if the font is unavailable.

## DDI and evaluation conventions

The shared reconstruction routine normalizes the target response by the signed control-source mean. Histories whose absolute denominator is below the specified threshold are excluded; strictly positive history scores are averaged, with zero when none remain. Model-level matrices are averaged before selecting delay or edge maxima. Notebook-specific protocols, including the wind-tunnel history adapter and validation-calibrated thresholds, are specified in their Markdown cells and code.

The Lorenz plotting threshold is a post hoc choice for the reported benchmark. Wind-tunnel precision and recall use thresholds selected on validation records and held fixed on test records. These are different evaluation procedures.

## Citation and license

Citation metadata are provided in [CITATION.cff](CITATION.cff). The accompanying manuscript is listed as unpublished until final publication details are available; no DOI or GitHub repository URL is assumed.

This project's code is released under GPL-3.0-only; see [LICENSE](LICENSE). Retained third-party code keeps its original notices and license terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Dataset terms are not replaced by the software license.
