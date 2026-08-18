"""Compatibility entry point for the Streamlit app."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("src.filing_translator.frontend", run_name="__main__")

