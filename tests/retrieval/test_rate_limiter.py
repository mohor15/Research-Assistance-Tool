"""Per-domain rate limiter: spacing enforced, independent per domain, disableable."""

from __future__ import annotations

from research_assistant.retrieval.rate_limiter import DomainRateLimiter


def test_first_request_to_a_domain_never_waits():
    clock = [0.0]
    limiter = DomainRateLimiter(min_delay_seconds=2.0, sleep_fn=lambda s: None, time_fn=lambda: clock[0])

    slept = limiter.wait("example.com")
    assert slept == 0.0


def test_second_request_too_soon_waits_the_remaining_delay():
    clock = [0.0]
    sleeps = []
    limiter = DomainRateLimiter(
        min_delay_seconds=2.0, sleep_fn=lambda s: sleeps.append(s), time_fn=lambda: clock[0]
    )

    limiter.wait("example.com")
    clock[0] = 0.5  # only 0.5s elapsed, need 2.0s spacing
    limiter.wait("example.com")

    assert sleeps == [1.5]


def test_request_after_delay_has_elapsed_does_not_wait():
    clock = [0.0]
    sleeps = []
    limiter = DomainRateLimiter(
        min_delay_seconds=2.0, sleep_fn=lambda s: sleeps.append(s), time_fn=lambda: clock[0]
    )

    limiter.wait("example.com")
    clock[0] = 5.0
    limiter.wait("example.com")

    assert sleeps == []


def test_domains_are_rate_limited_independently():
    clock = [0.0]
    sleeps = []
    limiter = DomainRateLimiter(
        min_delay_seconds=2.0, sleep_fn=lambda s: sleeps.append(s), time_fn=lambda: clock[0]
    )

    limiter.wait("a.com")
    limiter.wait("b.com")  # different domain, should not wait even though a.com was just hit

    assert sleeps == []


def test_zero_delay_disables_limiting():
    sleeps = []
    limiter = DomainRateLimiter(min_delay_seconds=0.0, sleep_fn=lambda s: sleeps.append(s))

    limiter.wait("example.com")
    limiter.wait("example.com")

    assert sleeps == []
