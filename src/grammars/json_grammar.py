"""Constrains output to: {"name": "<letters>", "age": <1-3 digit number>}"""

GRAMMAR = r"""
    start: LBRACE SPACE? NAME_KEY COLON SPACE? STRING COMMA SPACE? AGE_KEY COLON SPACE? NUMBER RBRACE
    LBRACE: "{"
    RBRACE: "}"
    NAME_KEY: "\"name\""
    AGE_KEY: "\"age\""
    COLON: ":"
    COMMA: ","
    SPACE: " "
    STRING: /"[a-zA-Z ]+"/
    NUMBER: /[0-9]{1,3}/
"""

TERMINALS = {
    "LBRACE": r"\{",
    "RBRACE": r"\}",
    "NAME_KEY": r'"name"',
    "AGE_KEY": r'"age"',
    "COLON": r":",
    "COMMA": r",",
    "SPACE": r" ",
    "STRING": r'"[a-zA-Z ]+"',
    "NUMBER": r"[0-9]{1,3}",
}

DEMO_PROMPT = "Generate a JSON profile for a random person:\n"
