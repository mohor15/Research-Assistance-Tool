"""Research agents.

`PeerAgent`, `AIProductAgent`, `StartupAgent`, `AnalystAgent` (Phase 3/4),
and `ReportAgent` (Phase 6) are all implemented. Import concrete agents
from their own submodules (`research_assistant.agents.peer`, `.ai_product`,
`.startup`, `.analyst`, `.report`), not from this package's top level — see
`research_assistant.models`'s docstring for why this package stays a leaf
import (agents depend on `config`/`retrieval` transitively; re-exporting
here would risk the same import-cycle class of bug fixed in Phase 2).
"""
