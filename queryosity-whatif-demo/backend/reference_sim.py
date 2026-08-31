"""
Reference clock-sweep buffer-pool simulator.

IMPORTANT SCOPE NOTE
--------------------
This module is used ONLY by the local MOCK backend (see ``simulator.py``),
so that the application can be exercised end-to-end without the real project
on the machine. It is a faithful *reference* re-implementation of the
page-level clock-sweep policy described in the Queryosity report (integer
usage counts capped at ``BM_MAX_USAGE_COUNT`` = 5), but it is NOT the
project's source-of-truth simulator.

On the VM the application calls the real
``simulate_schedule_page_level(page_sets, order, cap)`` instead
(see ``RealSimulatorBackend`` in ``simulator.py``). Numbers produced here are
labelled MOCK throughout the UI and API and must never be presented as real
benchmark results.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

BM_MAX_USAGE_COUNT = 5


def simulate_clock_sweep(
    sequence_page_sets: Sequence[Iterable[int]],
    cap: int,
) -> Tuple[int, int]:
    """Replay a schedule through a clock-sweep buffer pool.

    Args:
        sequence_page_sets: page sets already arranged in *schedule order*
            (i.e. the i-th element is the page set of the i-th executed query).
            Each page set is an iterable of page identifiers; each page is
            requested exactly once when its query executes.
        cap: buffer capacity in pages (frames). Must be a positive integer.

    Returns:
        ``(total_hits, total_requests)``. ``total_misses`` is
        ``total_requests - total_hits`` and the hit ratio is
        ``total_hits / total_requests`` (0 when there are no requests).
    """
    if not isinstance(cap, int) or cap <= 0:
        raise ValueError(f"cap must be a positive integer, got {cap!r}")

    frames: list[int] = []          # page id resident in each frame
    usage: dict[int, int] = {}      # page id -> usage count
    hand = 0
    total_hits = 0
    total_requests = 0

    for page_set in sequence_page_sets:
        for page in page_set:
            total_requests += 1
            u = usage.get(page)
            if u is not None:
                # Hit: bump usage count toward the cap.
                total_hits += 1
                if u < BM_MAX_USAGE_COUNT:
                    usage[page] = u + 1
                continue

            # Miss: load the page, evicting via the clock sweep if full.
            if len(frames) < cap:
                usage[page] = 1
                frames.append(page)
                continue

            # Advance the clock hand, decrementing usage counts, until a
            # frame with usage 0 is found; that frame is the victim.
            while True:
                victim = frames[hand]
                vu = usage[victim]
                if vu <= 0:
                    break
                usage[victim] = vu - 1
                hand = (hand + 1) % cap

            del usage[victim]
            frames[hand] = page
            usage[page] = 1
            hand = (hand + 1) % cap

    return total_hits, total_requests
