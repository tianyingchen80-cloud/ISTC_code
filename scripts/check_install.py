import ast
import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import nbformat
import numpy as np
from IPython.core.inputtransformer2 import TransformerManager
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]


def load_helper(name, namespace):
    notebook = nbformat.read(ROOT / name, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            exec(compile(cell.source, name, 'exec'), namespace)


def main():
    transformer = TransformerManager()
    notebooks = sorted(ROOT.glob('*.ipynb'))
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                compile(transformer.transform_cell(cell.source), str(path), 'exec')
    for name in ['numpy', 'scipy', 'pandas', 'matplotlib', 'sklearn',
                 'threadpoolctl', 'IPython', 'nbclient', 'nbconvert', 'ipykernel', 'tables']:
        importlib.import_module(name)
    namespace = {'np': np}
    for name in ['time_Lorenz_data_6time.ipynb', 'ESN.ipynb', 'Reconstruct.ipynb']:
        load_helper(name, namespace)
    np.random.seed(99)
    coupling = [np.zeros((2, 2)) for unused in range(6)]
    coupling[3][0, 1] = 1.0
    values = namespace['gen_data_norm_6time'](2, *coupling, 600, 12, 2,
                                             sigma_dyn=1e-6, sigma_obs=1e-6)
    delays = np.arange(6) * 2
    embedded = np.concatenate([values[10 - delay:600 - delay] for delay in delays], axis=1)
    with threadpool_limits(limits=2):
        model = namespace['ESN'](48, embedded.shape[1], embedded.shape[1])
        model.train(embedded[:20], embedded[20:299], embedded[21:300])
        scores = namespace['cal_time_TDI_copy_detect'](
            model, embedded[300:], 2, 3, 6, 3, 6, 0, 1, 20, 5, np.array([25, 45, 65]), 3)
    assert values.shape == (600, 6) and np.isfinite(values).all()
    assert scores.shape == (2, 2) and np.isfinite(scores).all()
    assert np.all(scores >= 0) and np.all(np.diag(scores) == 0)
    sys.path.insert(0, str(ROOT))
    from time_Lorenz_ROC_comparison import run_pcmci_scores
    baseline = run_pcmci_scores(values[:120], n_x=2, tau_max=2,
                                 tigramite_path=str(ROOT / 'tigramite-master'))
    assert baseline.shape == (3, 2, 2) and np.isfinite(baseline).all()
    wind = nbformat.read(ROOT / 'Tab1_wind-tunnel.ipynb', as_version=4)
    definitions = []
    for cell in wind.cells:
        if cell.cell_type == 'code':
            definitions.extend(node for node in ast.parse(cell.source).body
                               if isinstance(node, ast.ClassDef) and node.name in {'HistoryESN', 'HistoryAdapter'})
    wind_namespace = {'np': np, 'NODES': 3, 'SETTINGS': {'first_anchor': 20}}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'wind-model', 'exec'), wind_namespace)
    rng = np.random.RandomState(99)
    training = rng.normal(size=(100, 3))
    config = {'n_units': 32, 'input_scaling': 0.5, 'spectral_radius': 0.9, 'leak': 0.3, 'ridge': 0.01}
    with threadpool_limits(limits=2):
        wind_model = wind_namespace['HistoryESN'](config, 99, 1).fit_records({1: training})
        adapter = wind_namespace['HistoryAdapter'](wind_model)
        initial = adapter.initial_states(training, np.array([30, 50]))
        predicted = np.stack([adapter.evolve(state, 5) for state in initial])
    assert np.isfinite(predicted).all()
    print(f'Validated {len(notebooks)} notebooks and installed dependencies.')
    print('Small Lorenz generation, ESN training, DDI, PCMCI and wind-model checks passed.')
    print('This installation check does not run or reproduce the full paper experiments.')


if __name__ == '__main__':
    main()
