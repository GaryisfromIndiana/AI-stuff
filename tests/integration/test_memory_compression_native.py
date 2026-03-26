"""Native integration tests for memory compression module wiring."""

from __future__ import annotations

import pytest

from core.memory.compression import CompressionResult, MemoryCluster, MemoryCompressor

pytestmark = pytest.mark.integration


def test_memory_compressor_initializes() -> None:
    compressor = MemoryCompressor("empire-alpha")
    assert compressor.empire_id == "empire-alpha"
    assert compressor.min_cluster_size == 3


def test_compression_dataclasses_have_stable_defaults() -> None:
    result = CompressionResult()
    cluster = MemoryCluster(cluster_key="topic:test")
    assert result.clusters_found == 0
    assert result.compression_ratio == 0.0
    assert cluster.cluster_key == "topic:test"
    assert cluster.memories == []
