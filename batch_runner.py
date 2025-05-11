import sys
from pathlib import Path
'''
all this does is make it so you can actually use the command I provided XD
allows use of python batch_runner.py/py batch_runner.py etc etc.
'''
sys.path.insert(0, str(Path(__file__).parent / "src"))

from batch_runner import main

if __name__ == "__main__":
    main()