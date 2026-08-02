"""
Run just the Python function-signature demo on its own:
    python examples/demo_python.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.model_runner import ConstrainedModel
from src.grammars import PYTHON_GRAMMAR, PYTHON_TERMINALS, PYTHON_PROMPT

if __name__ == "__main__":
    llm = ConstrainedModel()
    output = llm.generate(
        PYTHON_PROMPT, PYTHON_GRAMMAR, PYTHON_TERMINALS,
        temperature=1.2, trace=True,
    )
    print("\nFinal output:", output)
