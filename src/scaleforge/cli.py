from __future__ import annotations

from pathlib import Path

import typer

from scaleforge.protocol import sha256_file

app = typer.Typer(no_args_is_help=True)


@app.command()
def hash_file(path: Path) -> None:
    """Print a SHA-256 digest for an evidence artifact."""
    typer.echo(sha256_file(path))
