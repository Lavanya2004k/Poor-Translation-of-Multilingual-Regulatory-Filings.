"""Compatibility entry point for the filing translator CLI."""

from src.filing_translator.complete import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("src.filing_translator.complete", run_name="__main__")

