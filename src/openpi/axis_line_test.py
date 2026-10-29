"""This line stands alone: no import of the axis_qvla package, and no environment reads in its AXIS
modules or its client package."""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CLIENT = ROOT / "packages/openpi-client/src"
SOURCES = sorted(p for d in ("src/openpi", "scripts", "packages/openpi-client/src") for p in (ROOT / d).rglob("*.py"))
# The modules this line adds or rewrites for AXIS, and the client package it ships to benchmark
# clients: none of them may change behaviour through the environment.
AXIS_MODULES = sorted(
    [
        *(ROOT / "src/openpi/training").glob("axis_*.py"),
        ROOT / "src/openpi/training/weighted_loss.py",
        ROOT / "src/openpi/policies/axis_franka_policy.py",
        ROOT / "src/openpi/policies/policy_config.py",
        *CLIENT.rglob("*.py"),
    ]
)
ENV = ("environ", "getenv", "environb", "getenvb")


def _imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
            yield from (f"{node.module}.{a.name}" for a in node.names)


def _rel(p: pathlib.Path) -> str:
    return str(p.relative_to(ROOT))


def test_sources_were_found():
    assert len(SOURCES) > 50
    assert len([p for p in SOURCES if p.is_relative_to(CLIENT)]) >= 10
    for rel in (
        "src/openpi/training/axis_config.py",
        "src/openpi/training/axis_roots.py",
        "src/openpi/training/axis_bundle.py",
        "src/openpi/policies/axis_franka_policy.py",
        "src/openpi/policies/policy_config.py",
        "packages/openpi-client/src/openpi_client/websocket_client_policy.py",
    ):
        assert ROOT / rel in AXIS_MODULES, rel


@pytest.mark.parametrize("path", SOURCES, ids=_rel)
def test_no_axis_qvla_import(path):
    names = list(_imports(path))
    assert not [n for n in names if n.split(".")[0] in ("axis", "axis_qvla")]


@pytest.mark.parametrize("path", AXIS_MODULES, ids=_rel)
def test_axis_modules_read_no_environment(path):
    tree = ast.parse(path.read_text())
    bad = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in ENV]
    bad += [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module == "os" and any(a.name in ENV for a in n.names)
    ]
    assert not bad
