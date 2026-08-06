import ast
from pathlib import Path


def test_lower_layers_do_not_import_pipelines():
    source_root = Path(__file__).resolve().parents[1] / "src" / "trading_system"
    lower_layers = (
        "data",
        "features",
        "labels",
        "models",
        "training",
        "evaluation",
        "backtest",
        "reporting",
        "experiments",
        "analysis",
    )
    offenders = []
    for layer in lower_layers:
        for path in (source_root / layer).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "trading_system.pipelines"
                ):
                    offenders.append(str(path.relative_to(source_root)))
                if isinstance(node, ast.Import):
                    if any(
                        alias.name.startswith("trading_system.pipelines")
                        for alias in node.names
                    ):
                        offenders.append(str(path.relative_to(source_root)))
    assert not offenders


def test_manual_ann_has_single_training_implementation():
    source_root = Path(__file__).resolve().parents[1] / "src" / "trading_system"
    definitions = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "forward_pass"
            ):
                definitions.append(path.relative_to(source_root).as_posix())
    assert definitions == ["models/manual_ann/manual_nn.py"]
