"""Streamlit dashboard (Phase 7).

`ui.dashboard_data` is the read-only query layer (no Streamlit dependency,
fully unit-testable); `ui.app` is the actual Streamlit page and requires
`streamlit` to be installed to import. This package stays a leaf import
(nothing here is re-exported at the top level) so importing
`research_assistant.ui.dashboard_data` alone never requires Streamlit.
"""
