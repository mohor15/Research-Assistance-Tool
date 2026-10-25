"""Nomura AI & Fintech Research Assistant."""

__version__ = "0.1.0"

try:
    # Makes `ssl`/`requests` validate certificates against the OS trust
    # store (Windows Certificate Store / macOS Keychain / Linux CA certs)
    # instead of only the bundled `certifi` CA list. Needed because local
    # TLS-intercepting software (antivirus HTTPS scanning, corporate
    # proxies) re-signs connections with a root CA the OS trusts but
    # `certifi` does not -- without this, every such connection fails with
    # SSLCertVerificationError even though a real browser on the same
    # machine connects fine. Optional: if `truststore` isn't installed,
    # retrieval still works, just without this extra trust source.
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover - exercised only when truststore is absent
    pass
