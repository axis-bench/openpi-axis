"""The serve command line: top-level flags before `policy:checkpoint`, a generated `.json` config,
and `--cfg-conditioning` refused before any model is loaded."""

import importlib.util
import pathlib
import sys

import pytest
import tyro

ROOT = pathlib.Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("axis_serve_script", ROOT / "scripts" / "serve_policy.py")
serve = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = serve  # tyro reads the dataclasses' source through their module
_spec.loader.exec_module(serve)


def test_the_documented_command_line_parses():
    args = tyro.cli(
        serve.Args,
        args=[
            "--port=8000",
            "--quality-tag=5",
            "--guidance-scale=0.3",
            "policy:checkpoint",
            "--policy.config=/r/runs/a/openpi_config.json",
            "--policy.dir=/r/ckpt/13579",
        ],
    )
    assert (args.port, args.quality_tag, args.guidance_scale, args.cfg_conditioning) == (8000, 5, 0.3, False)
    assert (args.policy.config, args.policy.dir) == ("/r/runs/a/openpi_config.json", "/r/ckpt/13579")


def test_a_top_level_flag_after_the_subcommand_is_refused():
    with pytest.raises(SystemExit):
        tyro.cli(serve.Args, args=["policy:checkpoint", "--policy.config=a.json", "--policy.dir=d", "--quality-tag=5"])


def test_cfg_conditioning_is_refused_before_loading():
    args = serve.Args(
        policy=serve.Checkpoint(config="/nonexistent/openpi_config.json", dir="/nonexistent"), cfg_conditioning=True
    )
    with pytest.raises(ValueError, match="--cfg-conditioning is not served"):
        serve.create_policy(args)
