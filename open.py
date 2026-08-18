"""Compatibility entry point for OpenAI-assisted translation."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("src.filing_translator.open", run_name="__main__")

