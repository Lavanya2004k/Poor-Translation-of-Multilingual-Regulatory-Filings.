"""Compatibility entry point for fine-tuned translation workflows."""

from src.filing_translator.finetuned import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("src.filing_translator.finetuned", run_name="__main__")

