"""
ConstrainedModel
----------------
Thin wrapper around a HF causal LM that plugs a UniversalCFGEngine into the
per-token sampling loop: at every step, mask out every vocab token the
engine considers illegal given the grammar's current state, renormalize,
sample, then advance the engine's state by the characters actually chosen.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console

from src.cfg_engine import UniversalCFGEngine
from src import trace as tr

console = Console()


class ConstrainedModel:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        tr.print_loading(console)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.model.eval()

        # Precompute once: token_id -> exact decoded string. This is the
        # vocab side of the "vocab <-> FSM" bridge described in the build guide.
        self.id_to_str = {
            i: self.tokenizer.decode([i], clean_up_tokenization_spaces=False)
            for i in range(len(self.tokenizer))
        }

    def _valid_token_ids(self, engine: UniversalCFGEngine) -> list:
        # Compute once per step, not once per candidate token -- see the
        # docstring on UniversalCFGEngine.is_token_valid for why this matters.
        accepts = engine.interactive.accepts()
        return [
            token_id
            for token_id, token_str in self.id_to_str.items()
            if token_str and engine.is_token_valid(token_str, accepts=accepts)
        ]

    def generate(
        self,
        prompt: str,
        grammar: str,
        terminals: dict,
        max_tokens: int = 40,
        temperature: float = 1.0,
        trace: bool = False,
    ) -> str:
        engine = UniversalCFGEngine(grammar, terminals)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        generated_text = ""

        if trace:
            tr.print_trace_start(console)

        for step in range(max_tokens):
            with torch.no_grad():
                raw_logits = self.model(input_ids).logits[0, -1]

            valid_ids = self._valid_token_ids(engine)
            if not valid_ids:
                if trace:
                    tr.print_no_valid_tokens(console)
                break

            mask = torch.full_like(raw_logits, float("-inf"))
            mask[valid_ids] = 0
            constrained_logits = (raw_logits / temperature) + mask
            probs = torch.softmax(constrained_logits, dim=-1)

            next_id = torch.multinomial(probs, num_samples=1).item()
            token_str = self.id_to_str[next_id]

            if trace:
                tr.print_step(
                    console, step, generated_text, raw_logits, probs,
                    valid_ids, self.id_to_str, next_id, token_str,
                )

            for ch in token_str:
                engine.advance_char(ch)

            generated_text += token_str
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=-1)

            if "$END" in engine.interactive.accepts():
                if trace:
                    tr.print_clean_end(console)
                break

        return generated_text.strip()
