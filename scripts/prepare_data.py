import argparse
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIND_URLS = {
    'wt_walks_v1.zip': 'https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/wt_walks_v1.zip',
    'adjacency_wt_standard.csv': ('https://raw.githubusercontent.com/juangamella/'
                                   'causal-chamber-package/9fb5d82e391bb91a89a64f2e1c9b6ae8e7aed6d6/causalchamber/ground_truth/'
                                   'adjacencies/adjacency_wt_standard.csv'),
}
WIND_SHA256 = {
    'wt_walks_v1.zip': '1ef92d39fff7645a9e6e8cd41311e708362606491590a1534c6a5135d207736e',
    'adjacency_wt_standard.csv': '05088000010c094b9f6a29bcb62d67b154507395e54441ee7748eaf5ecd5024d',
}


def file_digest(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.part')
        print(f'Downloading {path.name}', flush=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'ISTC-data-preparation/0.1'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as target:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                target.write(block)
        temporary.replace(path)
    return path


def prepare_wind(data_dir):
    for filename, url in WIND_URLS.items():
        path = download(url, data_dir / 'CausalChambers' / filename)
        if file_digest(path) != WIND_SHA256[filename]:
            raise ValueError(f'The wind-tunnel input does not match this release: {filename}')
        print(f'Wind tunnel: {path}')


def main():
    parser = argparse.ArgumentParser(description='Download the official wind-tunnel inputs. Hong Kong and traffic CSVs are included in data/.')
    parser.add_argument('dataset', choices=['wind-tunnel'])
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    prepare_wind(args.data_dir)


if __name__ == '__main__':
    main()
