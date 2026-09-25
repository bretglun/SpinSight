from datetime import datetime
from pathlib import Path

import toml

from spinsight.gui import GUI


def get_version():
    try:
        return f', v {toml.load(Path(__file__).parent.parent / "pyproject.toml")["project"]["version"]}'
    except (FileNotFoundError, KeyError):
        return ''


def get_app(dark_mode=True, settings_filestem='', start_time=datetime.now(), lazy_sliders=True, sliders_with_text=False):

    version = get_version()
    settings_file = Path(settings_filestem).with_suffix('.toml') if bool(settings_filestem) else Path('')

    gui = GUI(settings_file, start_time, version, lazy_sliders, sliders_with_text, dark_mode)

    return gui.view()