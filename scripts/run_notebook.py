import argparse
import asyncio
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser(description='Execute an ISTC notebook from the repository root.')
    parser.add_argument('notebook', help='Notebook filename, for example A.3.ipynb')
    parser.add_argument('--output-dir', type=Path, help='Directory for the executed notebook copy.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / args.notebook
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=None, kernel_name='python3',
                            resources={'metadata': {'path': str(root)}})
    print(f'Running {path.name}', flush=True)
    client.execute()
    output_dir = args.output_dir if args.output_dir is not None else root / 'result' / 'executed'
    output = output_dir / path.name
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(f'Executed notebook: {output}')


if __name__ == '__main__':
    main()
