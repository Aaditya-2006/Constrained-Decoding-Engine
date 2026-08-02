"""Constrains output to a Python function signature: def name(arg, arg, ...):"""

GRAMMAR = r"""
    start: DEF SPACE NAME LPAREN arguments? RPAREN COLON
    arguments: NAME (COMMA SPACE NAME)*

    DEF: "def"
    LPAREN: "("
    RPAREN: ")"
    COLON: ":"
    COMMA: ","
    SPACE: " "
    NAME: /[a-zA-Z_][a-zA-Z0-9_]{0,15}/
"""

TERMINALS = {
    "DEF": r"def",
    "LPAREN": r"\(",
    "RPAREN": r"\)",
    "COLON": r":",
    "COMMA": r",",
    "SPACE": r" ",
    "NAME": r"[a-zA-Z_][a-zA-Z0-9_]{0,15}",
}

DEMO_PROMPT = "Write a Python function signature for calculating the trajectory of a rocket:\n"
