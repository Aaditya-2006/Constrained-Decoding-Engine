"""
Console trace output for the generation loop.
Kept separate from model_runner.py so the sampling loop isn't cluttered
with print formatting -- toggle `trace=True` in ConstrainedModel.generate()
to turn this on.
"""

import torch
from rich.console import Console


def print_trace_start(console: Console):
    console.print(
        "\n[bold magenta]================ STARTING STEP-BY-STEP TRACE "
        "================[/bold magenta]\n"
    )


def print_step(
    console: Console,
    step: int,
    generated_text: str,
    raw_logits: torch.Tensor,
    probs: torch.Tensor,
    valid_ids: list,
    id_to_str: dict,
    next_id: int,
    token_str: str,
):
    raw_probs = torch.softmax(raw_logits, dim=-1)
    raw_top_p, raw_top_i = torch.topk(raw_probs, 3)
    raw_cand = [
        f"{repr(id_to_str[idx.item()])} ({p.item():.1%})"
        for p, idx in zip(raw_top_p, raw_top_i)
    ]

    m_top_p, m_top_i = torch.topk(probs, min(3, len(valid_ids)))
    m_cand = [
        f"{repr(id_to_str[idx.item()])} ({p.item():.1%})"
        for p, idx in zip(m_top_p, m_top_i)
    ]

    console.print(
        f"[bold cyan]Step {step + 1}[/bold cyan] | "
        f"Generated so far: [yellow]{repr(generated_text)}[/yellow]"
    )
    console.print(f"  ├─ [dim]1. Raw LLM Choices:[/dim]      {', '.join(raw_cand)}")
    console.print(
        f"  ├─ [dim]2. Logit Mask Action:[/dim]    "
        f"Allowed [bold green]{len(valid_ids)}[/bold green] / {len(id_to_str):,} tokens"
    )
    console.print(f"  ├─ [dim]3. Re-normalized Choices:[/dim] {', '.join(m_cand)}")
    console.print(f"  └─ [bold green]4. Sampled Token:[/bold green]        {repr(token_str)}\n")


def print_no_valid_tokens(console: Console):
    console.print("[bold red]Grammar naturally completed / No valid tokens.[/bold red]")


def print_clean_end(console: Console):
    console.print("[bold green]Reached $END state cleanly.[/bold green]\n")


def print_loading(console: Console):
    console.print("[bold green]Loading Universal Engine & Model...[/bold green]")
