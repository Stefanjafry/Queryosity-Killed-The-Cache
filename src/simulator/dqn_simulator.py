"""
DQN-based fitness evaluation for query schedule optimization.

This module provides inference utilities for evaluating query schedules
using a pre-trained Deep Q-Network (DQN) exported in ONNX format. The
DQN serves as a surrogate fitness function for the genetic algorithm,
replacing the expensive LRU cache simulation during evolution.

The network accepts a state vector encoding the current buffer contents
and a candidate query's page access pattern, and outputs a Q-value
estimating the cumulative future cache hit rate achievable from that
state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import onnxruntime as ort
import numpy as np

from src.simulator.cache_simulator import ClockSweepCache
from src.simulator.access_profile import AccessProfile


logger = logging.getLogger(__name__)


def build_state(
    cache: ClockSweepCache,
    query_profile: AccessProfile,
    all_tables: list[str],
    max_pages: dict[str, int],
) -> list[float]:
    """
    Build a normalized state vector for DQN inference.

    Concatenates a buffer vector and a query vector, each of length
    ``len(all_tables)``, into a single flat vector of length
    ``2 * len(all_tables)``. Each entry is normalized to [0, 1] by
    dividing by the maximum observed page count for that table.

    The buffer vector encodes how many pages of each table are currently
    cached. The query vector encodes how many pages of each table the
    given query will access.

    Parameters
    ----------
    cache : ClockSweepCache
        Current cache state.  Per-table page counts are read via
        ``cache.pages_held(table)``.
    query_profile : AccessProfile
        Access profile for the candidate query, mapping table names to
        page counts.
    all_tables : List[str]
        Ordered list of all table names in the workload. Determines the
        dimensionality and ordering of the state vector.
    max_pages : Dict[str, int]
        Maximum page count observed per table across all query profiles.
        Used for normalization.

    Returns
    -------
    List[float]
        Flat normalized state vector of length ``2 * len(all_tables)``.
    """

    buffer_vec = [
        cache.pages_held(t) / max_pages[t]
        for t in sorted(all_tables)
    ]
    query_vec = [
        query_profile.table_pages.get(t, 0) / max_pages[t]
        for t in sorted(all_tables)
    ]
    return buffer_vec + query_vec


def dqn_fitness(
    schedule: list[int],
    profiles: list[AccessProfile],
    cache_capacity_pages: int,
    session: ort.InferenceSession,
    all_tables: list[str],
    max_pages: dict[str, int],
) -> float:
    """
    Evaluate a query schedule using the DQN surrogate fitness function.

    Replays the schedule through a fresh LRU cache, querying the DQN at
    each step to obtain a Q-value for the current (buffer state, query)
    pair. The fitness is the sum of Q-values across all scheduling
    decisions, where higher values indicate better expected cache
    utilization.

    This function is intended as a drop-in replacement for
    ``simulate_schedule`` during GA evolution, providing orders of
    magnitude faster evaluation by avoiding full cache simulation.

    Parameters
    ----------
    schedule : List[int]
        Permutation of query indices specifying the execution order.
    profiles : List[AccessProfile]
        Access profiles for all queries, indexed by query index.
    cache_capacity_pages : int
        LRU cache capacity in 8 KB pages.
    session : ort.InferenceSession
        ONNX runtime inference session loaded from the exported DQN
        model.
    all_tables : List[str]
        Ordered list of all table names in the workload. Must match the
        ordering used during model training.
    max_pages : Dict[str, int]
        Maximum page count observed per table across all query profiles.
        Must match the values used during model training.

    Returns
    -------
    float
        Sum of Q-values across all scheduling steps. Higher is better.
    """
    cache = ClockSweepCache(cache_capacity_pages)
    total_q = 0.0

    for idx in schedule:
        state = np.array(
            build_state(cache, profiles[idx], all_tables, max_pages),
            dtype=np.float32
        ).reshape(1, -1)

        q_value = cast(np.ndarray, session.run(['q_value'], {'state': state})[0])
        total_q += q_value.item()

        for table, pages in profiles[idx].table_pages.items():
            cache.access(table, pages)

    return total_q


class DQN:
    """
    DQN-based surrogate fitness evaluator for query schedule optimization.

    Wraps an ONNX runtime inference session and provides a fitness
    evaluation interface compatible with the ``FitnessFunction`` protocol.
    Intended as a drop-in replacement for the simulation-based fitness
    function during GA evolution, providing faster evaluation by
    substituting full LRU cache simulation with learned Q-value inference.

    Attributes
    ----------
    session : ort.InferenceSession
        ONNX runtime inference session loaded from the exported DQN model.
    all_tables : list[str]
        Ordered list of all table names in the workload. Must match the
        ordering used during model training.
    max_pages : dict[str, int]
        Maximum page count observed per table across all query profiles.
        Must match the values used during model training.
    """

    session: ort.InferenceSession
    all_tables: list[str]
    max_pages: dict[str, int]

    def __init__(
        self,
        onnx_path: Path,
        all_tables: list[str],
        max_pages: dict[str, int],
    ):
        self.session = ort.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']
        )
        self.all_tables = all_tables
        self.max_pages = max_pages

    def infer(
        self,
        individual: list[int],
        profiles: list[AccessProfile],
        cache_capacity_pages: int,
        page_sets: list[frozenset[int]] | None = None
    ) -> float:
        """
        Evaluate the fitness of a query schedule using DQN inference.

        Replays the schedule through a fresh LRU cache, querying the DQN
        at each step to obtain a Q-value for the current (buffer state,
        query) pair. Returns the sum of Q-values across all scheduling
        decisions as the fitness score.

        Parameters
        ----------
        individual : list[int]
            Permutation of query indices representing the schedule to
            evaluate.
        profiles : list[AccessProfile]
            Access profiles for all queries, indexed by query index.
        cache_capacity_pages : int
            LRU cache capacity in 8 KB pages.
        page_sets : list[frozenset[int]] or None, optional
            Unused. Accepted for compatibility with the
            ``FitnessFunction`` protocol. A warning is emitted if a
            non-None value is provided.

        Returns
        -------
        float
            Sum of Q-values across all scheduling steps. Higher is
            better.
        """
        if page_sets is not None:
            logger.warning('page_sets was provided to DQN.infer, which is ignored')
        return dqn_fitness(
            schedule=individual,
            profiles=profiles,
            cache_capacity_pages=cache_capacity_pages,
            session=self.session,
            all_tables=self.all_tables,
            max_pages=self.max_pages,
        )
