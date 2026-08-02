"""Constrains output to: <user name="<letters>" age=<1-3 digit number>/>"""

GRAMMAR = r"""
    start: TAG_OPEN SPACE NAME_ATTR EQUALS STRING SPACE AGE_ATTR EQUALS NUMBER TAG_CLOSE

    TAG_OPEN: "<user"
    NAME_ATTR: "name"
    AGE_ATTR: "age"
    EQUALS: "="
    TAG_CLOSE: "/>"
    SPACE: " "
    STRING: /"[a-zA-Z ]+"/
    NUMBER: /[0-9]{1,3}/
"""

TERMINALS = {
    "TAG_OPEN": r"<user",
    "NAME_ATTR": r"name",
    "AGE_ATTR": r"age",
    "EQUALS": r"=",
    "TAG_CLOSE": r"/>",
    "SPACE": r" ",
    "STRING": r'"[a-zA-Z ]+"',
    "NUMBER": r"[0-9]{1,3}",
}

DEMO_PROMPT = "Write an XML tag for a user profile:\n"
