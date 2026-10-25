"""Shared foundational models and enums.

`Article` and `Evidence` (Phase 2) are the canonical, source-grounded
representations later phases build on — import them from their own
submodules (`research_assistant.models.article`,
`research_assistant.models.evidence`) rather than this package's top level.
That keeps this `__init__.py` a leaf import (just `enums`), which config
and retrieval both depend on; eagerly re-exporting `Article`/`Evidence`
here would pull in `retrieval` (for `ExtractionMethod`/`ExtractionStatus`),
which itself depends on `config.schema`, which depends on `models.enums` —
a real import cycle. `Event`, `ResearchItem`, and `Digest` are introduced
in later phases (5+).
"""
