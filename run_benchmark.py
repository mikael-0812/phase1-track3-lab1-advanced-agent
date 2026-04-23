from __future__ import annotations
import json
from pathlib import Path
import typer
from rich import print
from src.reflexion_lab import runtime
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import dataset_label, load_dataset, save_jsonl
app = typer.Typer(add_completion=False)


def implemented_extensions(mode: str, reflexion: ReflexionAgent) -> list[str]:
    extensions = [
        "benchmark_report_json",
        "structured_evaluator",
        *reflexion.implemented_extensions(),
    ]
    if mode == "mock":
        extensions.append("mock_mode_for_autograding")
    return sorted(set(extensions))


@app.command()
def main(dataset: str = "data/hotpot_mini.json", out_dir: str = "outputs/sample_run", reflexion_attempts: int = 3) -> None:
    examples = load_dataset(dataset)
    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)
    mode = "mock" if runtime.use_mock() else "real"
    extensions = implemented_extensions(mode, reflexion)
    dataset_name = dataset_label(dataset)
    print(f"Loaded {len(examples)} examples from {dataset_name}")
    if mode == "real" and len(examples) < 100:
        print(f"[yellow]Warning[/yellow] Real benchmark dataset has only {len(examples)} examples; rubric expects at least 100.")
    react_records = [react.run(example) for example in examples]
    reflexion_records = [reflexion.run(example) for example in examples]
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(
        all_records,
        dataset_name=dataset_name,
        mode=mode,
        extensions=extensions,
    )
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(f"Mode: {mode}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
