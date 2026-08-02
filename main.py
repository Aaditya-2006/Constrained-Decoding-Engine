"""
Run all grammar demos back to back:
    python main.py
"""

from rich.console import Console

from src.model_runner import ConstrainedModel
from src.grammars import REGISTRY

console = Console()


def run_demo(llm: ConstrainedModel, name: str, grammar: str, terminals: dict, prompt: str, trace: bool = True):
    console.print(f"\n[bold yellow]--- TEST: Generating {name.upper()} ---[/bold yellow]")
    output = llm.generate(prompt, grammar, terminals, temperature=1.2, trace=trace)
    console.print(f"[bold cyan]Prompt:[/bold cyan] {prompt.strip()}")
    console.print(f"[bold green]Final Output:[/bold green] {output}\n")


def main():
    llm = ConstrainedModel()
    for name, (grammar, terminals, prompt) in REGISTRY.items():
        run_demo(llm, name, grammar, terminals, prompt)


if __name__ == "__main__":
    main()
