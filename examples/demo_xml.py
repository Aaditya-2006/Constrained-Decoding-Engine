"""
Run just the XML tag demo on its own:
    python examples/demo_xml.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.model_runner import ConstrainedModel
from src.grammars import XML_GRAMMAR, XML_TERMINALS, XML_PROMPT

if __name__ == "__main__":
    llm = ConstrainedModel()
    output = llm.generate(
        XML_PROMPT, XML_GRAMMAR, XML_TERMINALS,
        temperature=1.2, trace=True,
    )
    print("\nFinal output:", output)
