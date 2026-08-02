from .json_grammar import GRAMMAR as JSON_GRAMMAR, TERMINALS as JSON_TERMINALS, DEMO_PROMPT as JSON_PROMPT
from .python_grammar import GRAMMAR as PYTHON_GRAMMAR, TERMINALS as PYTHON_TERMINALS, DEMO_PROMPT as PYTHON_PROMPT
from .xml_grammar import GRAMMAR as XML_GRAMMAR, TERMINALS as XML_TERMINALS, DEMO_PROMPT as XML_PROMPT

# Registry so `main.py` / examples can loop over "all demos" without
# hardcoding each import site -- add a new grammar file, add one line here.
REGISTRY = {
    "json": (JSON_GRAMMAR, JSON_TERMINALS, JSON_PROMPT),
    "python": (PYTHON_GRAMMAR, PYTHON_TERMINALS, PYTHON_PROMPT),
    "xml": (XML_GRAMMAR, XML_TERMINALS, XML_PROMPT),
}
