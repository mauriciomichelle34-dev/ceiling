"""Prepare a local environment if needed, then open the app."""
import argparse
import importlib
import importlib.metadata
import os
import re
from pathlib import Path
import runpy
import subprocess
import sys
import venv

HERE = Path(__file__).resolve().parent
VENV = HERE / '.venv'
VENV_PYTHON = VENV / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
DEPENDENCIES = [('torch', 'torch', (2, 6), (3,)), ('numpy', 'numpy', (2,), (3,)),
                ('opencv-python', 'cv2', (4, 10), (6,)), ('Pillow', 'PIL', (10,), (13,))]


def dependencies_ready():
    try:
        for package, module, minimum, maximum in DEPENDENCIES:
            importlib.import_module(module)
            version = tuple(int(v) for v in re.match(r'\d+(?:\.\d+)*', importlib.metadata.version(package)).group(0).split('.'))
            if not minimum <= version < maximum:
                return False
        return True
    except (ImportError, OSError, ValueError, importlib.metadata.PackageNotFoundError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check without installing anything.')
    parser.add_argument('--setup-only', action='store_true', help='Prepare dependencies without starting the website.')
    args, app_args = parser.parse_known_args()
    if sys.version_info < (3, 10):
        parser.error('Python 3.10 or newer is required; Python 3.11 is recommended.')
    if not (HERE/'mycelium_v2.pt').is_file() and '--model' not in app_args:
        parser.error('mycelium_v2.pt is missing. Extract the complete project.')
    ready = dependencies_ready()
    if args.check:
        print('Model found. Dependencies ready.' if ready else 'Dependencies missing. Run python start.py to set them up.')
        return 0 if ready else 1
    if not ready:
        if Path(sys.prefix).resolve() == VENV.resolve():
            target_python = Path(sys.executable)
        else:
            if not VENV_PYTHON.is_file():
                print('Creating local .venv...', flush=True)
                venv.EnvBuilder(with_pip=True).create(VENV)
            target_python = VENV_PYTHON
        check = subprocess.run([str(target_python), str(HERE/'start.py'), '--check'], cwd=HERE)
        if check.returncode:
            print('First run: downloading dependencies. This can take several minutes.', flush=True)
            subprocess.run([str(target_python), '-m', 'pip', 'install', 'torch>=2.6,<3', '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True, cwd=HERE)
            subprocess.run([str(target_python), '-m', 'pip', 'install', '-r', str(HERE/'requirements.txt')], check=True, cwd=HERE)
        if args.setup_only:
            print('Setup complete.', flush=True)
            return 0
        return subprocess.run([str(target_python), str(HERE/'start.py'), *app_args], cwd=HERE).returncode
    if args.setup_only:
        print('Dependencies ready; no installation needed.', flush=True)
        return 0
    sys.argv = [str(HERE/'local_app.py'), '--open', *app_args]
    runpy.run_path(str(HERE/'local_app.py'), run_name='__main__')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print('Setup failed. Check your internet connection and retry python start.py.', file=sys.stderr)
        raise SystemExit(exc.returncode)
    except KeyboardInterrupt:
        raise SystemExit(130)
