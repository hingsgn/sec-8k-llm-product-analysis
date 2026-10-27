"""Download the pinned public GGUF model without redistributing its weights."""
from pathlib import Path
import hashlib
import urllib.request
from sec_product_extractor import MODEL_FILE, MODEL_URL


def download_model(destination: Path = Path('models')) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / MODEL_FILE
    if target.is_file() and target.stat().st_size == 2104932768:
        return target
    temporary = target.with_suffix('.part')
    print('Downloading', MODEL_FILE, '(about 2.1 GB)')
    with urllib.request.urlopen(MODEL_URL, timeout=120) as source, temporary.open('wb') as output:
        while chunk := source.read(8 * 1024 * 1024):
            output.write(chunk)
    if temporary.stat().st_size != 2104932768:
        raise RuntimeError('Incomplete model download. Run this command again.')
    temporary.replace(target)
    return target


if __name__ == '__main__':
    print(download_model().resolve())
