"""
Run just the JSON demo on its own:
    python examples/demo_json.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.model_runner import ConstrainedModel
from src.grammars import JSON_GRAMMAR, JSON_TERMINALS, JSON_PROMPT

if __name__ == "__main__":
    llm = ConstrainedModel()
    output = llm.generate(
        JSON_PROMPT, JSON_GRAMMAR, JSON_TERMINALS,
        temperature=1.2, trace=True,
    )
    print("\nFinal output:", output)
