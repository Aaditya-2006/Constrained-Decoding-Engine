"""
Phase 6 comparison: your engine vs. two production libraries, same grammar,
same prompt.

This is a single generation per backend (not a sweep like validity_rate.py
or latency.py), so it should run in well under a minute once models are
loaded -- most of the wall-clock time here is model *loading*, not generation.

Requires (only needed for this script, not the rest of the project):
    pip install outlines llama-cpp-python

llama.cpp also needs a local GGUF file. Get one with:
    huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
        qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir .

Run:
    python benchmarks/compare_backends.py
"""

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table

from src.model_runner import ConstrainedModel
from src.grammars import JSON_GRAMMAR, JSON_TERMINALS, JSON_PROMPT

console = Console()

GGUF_PATH = "./qwen2.5-0.5b-instruct-q4_k_m.gguf"


def run_custom(prompt: str):
    llm = ConstrainedModel("Qwen/Qwen2.5-0.5B-Instruct")
    start = time.perf_counter()
    output = llm.generate(prompt, JSON_GRAMMAR, JSON_TERMINALS, temperature=1.0, trace=False)
    elapsed = time.perf_counter() - start
    n_tokens = len(llm.tokenizer(output, add_special_tokens=False).input_ids)
    return llm, output, elapsed, n_tokens


def run_outlines(prompt: str, hf_model, hf_tokenizer):
    import outlines
    from pydantic import BaseModel

    class UserProfile(BaseModel):
        name: str
        age: int

    # Reuse the already-loaded HF model/tokenizer instead of loading a
    # second copy of the same weights from disk.
    outlines_model = outlines.from_transformers(hf_model, hf_tokenizer)

    start = time.perf_counter()
    result = outlines_model(prompt, UserProfile)
    elapsed = time.perf_counter() - start
    n_tokens = len(hf_tokenizer(str(result), add_special_tokens=False).input_ids)
    return str(result), elapsed, n_tokens


def run_llamacpp(prompt: str, gguf_path: str):
    from llama_cpp import Llama, LlamaGrammar

    if not os.path.exists(gguf_path):
        raise FileNotFoundError(
            f"GGUF file not found at {gguf_path}. Download one first -- see "
            "the module docstring at the top of this file."
        )

    llama_model = Llama(model_path=gguf_path, verbose=False)
    gbnf_grammar = LlamaGrammar.from_string(r"""
        root ::= "{" space "\"name\"" space ":" space string "," space "\"age\"" space ":" space number "}"
        space ::= " "?
        string ::= "\"" [a-zA-Z ]+ "\""
        number ::= [0-9] | [1-9] [0-9] | [1-9] [0-9] [0-9]
    """)

    start = time.perf_counter()
    raw = llama_model(prompt, max_tokens=40, temperature=1.0, grammar=gbnf_grammar)
    elapsed = time.perf_counter() - start
    output = raw["choices"][0]["text"].strip()
    n_tokens = raw["usage"]["completion_tokens"]
    return output, elapsed, n_tokens


def main():
    table = Table(title="Constrained Decoding: Backend Comparison")
    table.add_column("Implementation", style="cyan", no_wrap=True)
    table.add_column("Tokens/sec", justify="right", style="green")
    table.add_column("Total Time", justify="right", style="green")
    table.add_column("Output")

    console.print("[bold magenta]Loading custom engine + Qwen2.5-0.5B-Instruct...[/bold magenta]")
    llm, custom_output, custom_time, custom_tokens = run_custom(JSON_PROMPT)
    table.add_row(
        "Custom Engine (Python FSM)",
        f"{custom_tokens / custom_time:.2f}" if custom_time > 0 else "-",
        f"{custom_time:.2f}s",
        custom_output,
    )

    try:
        console.print("[bold magenta]Running Outlines (reusing loaded model)...[/bold magenta]")
        outlines_output, outlines_time, outlines_tokens = run_outlines(JSON_PROMPT, llm.model, llm.tokenizer)
        table.add_row(
            "Outlines",
            f"{outlines_tokens / outlines_time:.2f}" if outlines_time > 0 else "-",
            f"{outlines_time:.2f}s",
            outlines_output,
        )
    except ImportError:
        console.print("[dim]Skipping Outlines -- not installed (pip install outlines)[/dim]")
    except Exception as e:
        console.print(f"[bold red]Outlines failed:[/bold red] {e}")

    try:
        console.print("[bold magenta]Running llama.cpp...[/bold magenta]")
        llama_output, llama_time, llama_tokens = run_llamacpp(JSON_PROMPT, GGUF_PATH)
        table.add_row(
            "llama-cpp-python (C++ GGML)",
            f"{llama_tokens / llama_time:.2f}" if llama_time > 0 else "-",
            f"{llama_time:.2f}s",
            llama_output,
        )
    except ImportError:
        console.print("[dim]Skipping llama.cpp -- not installed (pip install llama-cpp-python)[/dim]")
    except FileNotFoundError as e:
        console.print(f"[dim]Skipping llama.cpp -- {e}[/dim]")
    except Exception as e:
        console.print(f"[bold red]llama.cpp failed:[/bold red] {e}")

    console.print("\n")
    console.print(table)


if __name__ == "__main__":
    main()
