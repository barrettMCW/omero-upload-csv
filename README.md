compiling:
* create and source venv
* install zeroc-ice wheel from [glencoe](https://github.com/glencoesoftware/zeroc-ice-py-macos-universal2/releases)
* install omero-py and nuitka
* run `python3 -m nuitka --onefile $(for f in .venv/lib/python3.12/site-packages/omero_*_ice.py; do echo --include-module=$(basename "$f" .py); done) ./omero-upload-kv.py`
