from spinsight.gui import GUI
from datetime import datetime
import toml
from pathlib import Path


def get_version():
    try:
        return f', v {toml.load(Path(__file__).parent.parent / "pyproject.toml")["project"]["version"]}'
    except (FileNotFoundError, KeyError):
        return ''


def get_app(dark_mode=True, settings_filestem='', start_time=datetime.now(), lazy_sliders=True):

    version = get_version()
    settings_file = Path(settings_filestem).with_suffix('.toml') if bool(settings_filestem) else Path('')

    gui = GUI(settings_file, start_time, version, lazy_sliders, dark_mode)

    return gui.view()