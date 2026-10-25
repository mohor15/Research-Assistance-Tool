"""Email delivery: a separate capability layered around already-persisted
digests/newsletters -- never a redesign of the research pipeline itself.

    PipelineRunner._run_locked           ui/app.py (manual send controls)      scheduling/daemon.py (startup)
              |                                      |                                   |
    delivery.service.deliver_digest_if_eligible   delivery.service.send_manual   delivery.service.discover_and_deliver_pending
              \\______________________________________|___________________________________/
                                           |
                             delivery.renderer (HTML/text) -> delivery.mailer (SMTP) -> storage.repositories.email_delivery

Every entry point here is read-persisted-data-and-send-mail only: nothing
in this package ever triggers retrieval, an LLM call, or a new pipeline run.
"""
