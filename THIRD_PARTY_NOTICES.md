# Third-party code and data

## Tigramite

`tigramite-master/tigramite/` contains the Python source files from the Tigramite copy used by the PCMCI comparison. Its documentation, tutorials, tests and Python bytecode are omitted. The source files are otherwise unchanged.

- Upstream: https://github.com/jakobrunge/tigramite
- Author and copyright notices: retained in the source files.
- License: GNU General Public License, version 3; see `tigramite-master/license.txt` and the upstream file headers for applicable terms.
- Use in this project: `time_Lorenz_ROC_comparison.py` imports the vendored implementation for the Figure 3 PCMCI baseline.

The upstream revision of the original local Tigramite copy was not recorded. This release therefore includes that exact source snapshot rather than claiming a particular upstream version. Its file checksums are listed in `tigramite-master/SOURCE_MANIFEST.json`.

## Datasets

No real-world observations, pretrained weights or computed experiment caches are distributed in this release. Dataset terms are separate from the software license. Public download links and preparation procedures are in `data/README.md`.

The Hong Kong download points to the previously published Intervened Reservoir Computing repository. The PeMSBay download points to a public dataset archive linked to the DCRNN benchmark. The wind-tunnel files are provided by the Causal Chambers authors. Cite the corresponding original work when using each dataset.
