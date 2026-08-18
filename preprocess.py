"""Compatibility entry point for PDF preprocessing."""

from src.filing_translator.preprocess import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("src.filing_translator.preprocess", run_name="__main__")

