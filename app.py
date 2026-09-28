"""`streamlit run app.py` - the HALLUDETECT v2 demo UI.

Kept at the repo root so the usual command opens the current app. The page
itself is src/halludetect/ui/app.py; the original v1 app is in legacy/.
"""
import os
import sys

# Lets the command work from a fresh clone before `pip install -e .`.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from halludetect.ui.app import main  # noqa: E402

main()
