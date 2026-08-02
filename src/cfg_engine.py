"""
UniversalCFGEngine
------------------
Bridges two libraries that solve different halves of the problem:

- `lark` (LALR interactive parser) tracks *which terminals are legal next*
  given everything parsed so far -- this is the CFG / recursive-structure layer.
- `interegular` compiles each terminal's regex into an FSM -- this is the
  character-level layer that tells you whether a partial string (e.g. a
  partially-typed number) is still a valid prefix of that terminal.

The engine walks a candidate token character-by-character: while a terminal
is "active" (partway through being built), each new character either extends
it (interegular FSM step) or closes it off and hands control back to the
parser to pick the next legal terminal. This is what lets `is_token_valid`
answer "is this whole multi-character BPE token legal right now?" in one call.
"""

import interegular
from lark import Lark, Token


class UniversalCFGEngine:
    def __init__(self, grammar_str: str, terminal_patterns: dict):
        self.parser = Lark(grammar_str, parser="lalr")
        self.terminal_fsms = {
            name: interegular.parse_pattern(pattern).to_fsm()
            for name, pattern in terminal_patterns.items()
        }
        self.interactive = self.parser.parse_interactive("")
        self.active_terminal = None
        self.buffer = ""

    def clone(self):
        """Cheap copy so `is_token_valid` can speculatively try a whole
        token without mutating the engine's real state."""
        new_engine = UniversalCFGEngine.__new__(UniversalCFGEngine)
        new_engine.parser = self.parser
        new_engine.terminal_fsms = self.terminal_fsms
        new_engine.interactive = self.interactive.copy()
        new_engine.active_terminal = self.active_terminal
        new_engine.buffer = self.buffer
        return new_engine

    # ---------------------------------------------------------------
    # Character-level FSM helpers
    # ---------------------------------------------------------------
    def can_extend(self, term_name: str, s: str) -> bool:
        """Is `s` a valid *prefix* of terminal `term_name`? (may not be complete yet)"""
        fsm = self.terminal_fsms[term_name]
        state = fsm.initial
        for ch in s:
            try:
                symbol = fsm.alphabet[ch]
                if state in fsm.map and symbol in fsm.map[state]:
                    state = fsm.map[state][symbol]
                else:
                    return False
            except KeyError:
                return False
        return True

    def is_match(self, term_name: str, s: str) -> bool:
        """Is `s` a *complete*, acceptable instance of terminal `term_name`?"""
        fsm = self.terminal_fsms[term_name]
        state = fsm.initial
        for ch in s:
            try:
                symbol = fsm.alphabet[ch]
                if state in fsm.map and symbol in fsm.map[state]:
                    state = fsm.map[state][symbol]
                else:
                    return False
            except KeyError:
                return False
        return state in fsm.finals

    # ---------------------------------------------------------------
    # State advancement (mutates self)
    # ---------------------------------------------------------------
    def advance_char(self, ch: str) -> bool:
        accepts = self.interactive.accepts()
        if self.active_terminal:
            if self.can_extend(self.active_terminal, self.buffer + ch):
                self.buffer += ch
                return True
            else:
                if self.is_match(self.active_terminal, self.buffer):
                    try:
                        self.interactive.feed_token(Token(self.active_terminal, self.buffer))
                    except Exception:
                        return False
                    self.active_terminal = None
                    self.buffer = ""
                    return self.advance_char(ch)
                else:
                    return False
        else:
            for term in accepts:
                if term in self.terminal_fsms and self.can_extend(term, ch):
                    self.active_terminal = term
                    self.buffer = ch
                    return True
            return False

    # ---------------------------------------------------------------
    # Read-only validity check (does NOT mutate self)
    # ---------------------------------------------------------------
    def is_token_valid(self, token_str: str, accepts=None) -> bool:
        """Would feeding `token_str` (a whole candidate vocab token) in,
        one character at a time, stay legal all the way through?

        `accepts` is `self.interactive.accepts()`, computed by the caller.
        It's the same value for every candidate token checked at a given
        generation step (it only depends on parser state, not on the
        candidate string), so recomputing it per-candidate -- which this
        method used to do -- means recomputing it up to ~150,000 times per
        step for nothing. Pass it in once per step; this still recomputes
        it itself as a fallback if called standalone (e.g. from a test)."""
        active, buf = self.active_terminal, self.buffer
        if accepts is None:
            accepts = self.interactive.accepts()
        for ch in token_str:
            if active:
                if self.can_extend(active, buf + ch):
                    buf += ch
                else:
                    if self.is_match(active, buf):
                        engine_copy = self.clone()
                        for c in token_str:
                            if not engine_copy.advance_char(c):
                                return False
                        return True
                    else:
                        return False
            else:
                found = False
                for term in accepts:
                    if term in self.terminal_fsms and self.can_extend(term, ch):
                        active, buf, found = term, ch, True
                        break
                if not found:
                    return False
        return True
