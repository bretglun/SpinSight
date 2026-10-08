from pathlib import Path

PRESETS_PATH = Path(__file__).parent.resolve() / 'presets'
CUSTOM_PRESET = '(custom)'

def get_presets():
    presets = [CUSTOM_PRESET]
    for file in PRESETS_PATH.iterdir():
        if file.is_file() and file.suffix == '.toml':
            presets.append(file.stem)
    return presets


def get_file(stem):
    return PRESETS_PATH / f'{stem}.toml'