import sys
from pathlib import Path
import runpy
'''
all this does is make it so you can actually use the command I provided XD
allows use of python batch_runner.py/py batch_runner.py etc etc.
'''
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    runpy.run_module("batch_runner", run_name="__main__")