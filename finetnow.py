"""Compatibility entry point for memory-optimized fine-tuning."""

from src.filing_translator.finetnow import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("src.filing_translator.finetnow", run_name="__main__")

