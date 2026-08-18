"""Compatibility entry point for the translator agent."""

from src.filing_translator.translatoragent import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("src.filing_translator.translatoragent", run_name="__main__")

