#!/usr/bin/env python3
"""
Optimized Data Loader for Training

This module provides an optimized data loader with prefetching, caching,
and efficient batch processing to improve training performance.

Author: Louis Chua Bean Chong
License: GPLv3
"""

import os
import queue
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler


class OptimizedDataset(Dataset):
    """
    Optimized dataset with caching and memory management.

    This dataset provides efficient data loading with optional caching
    and memory management to improve training performance.
    """

    def __init__(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        cache_size: Optional[int] = None,
        pin_memory: bool = True,
    ):
        """
        Initialize optimized dataset.

        Args:
            data: Input data tensor
            targets: Target tensor
            cache_size: Number of samples to cache in memory
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.data = data
        self.targets = targets
        self.cache_size = cache_size
        self.pin_memory = pin_memory

        # Initialize cache
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        if cache_size and cache_size > 0:
            print(f"Initializing cache with {cache_size} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Check cache first
        if self.cache_size and idx in self.cache:
            self.cache_hits += 1
            return self.cache[idx]

        self.cache_misses += 1

        # Get data
        sample_data = self.data[idx]
        sample_target = self.targets[idx]

        # Pin memory if requested
        if self.pin_memory and torch.cuda.is_available():
            sample_data = sample_data.pin_memory()
            sample_target = sample_target.pin_memory()

        # Cache if enabled
        if self.cache_size and len(self.cache) < self.cache_size:
            self.cache[idx] = (sample_data, sample_target)

        return sample_data, sample_target

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
        }


class PrefetchDataLoader:
    """
    Data loader with prefetching for improved performance.

    This data loader uses background threads to prefetch data,
    reducing the time spent waiting for data during training.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Initialize prefetch data loader.

        Args:
            dataset: Dataset to load
            batch_size: Batch size
            num_workers: Number of worker processes
            prefetch_factor: Number of batches to prefetch
            pin_memory: Whether to pin memory
            shuffle: Whether to shuffle data
            drop_last: Whether to drop incomplete batches
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Initialize data loader
        self.data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Prefetch queue
        self.prefetch_queue = queue.Queue(maxsize=prefetch_factor)
        self.prefetch_thread = None
        self.stop_prefetch = False

        # Start prefetching
        self._start_prefetch()

        print(f"PrefetchDataLoader initialized with {num_workers} workers")

    def _start_prefetch(self):
        """Start prefetching thread."""
        if self.prefetch_factor > 0:
            self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
            self.prefetch_thread.daemon = True
            self.prefetch_thread.start()

    def _prefetch_worker(self):
        """Worker thread for prefetching data."""
        try:
            for batch in self.data_loader:
                if self.stop_prefetch:
                    break

                # Put batch in queue (block if full)
                self.prefetch_queue.put(batch, block=True)
        except Exception as e:
            print(f"Prefetch worker error: {e}")

    def __iter__(self):
        """Iterate over prefetched batches."""
        return self

    def __next__(self):
        """Get next batch from prefetch queue."""
        if self.stop_prefetch:
            raise StopIteration

        try:
            # Get batch from prefetch queue
            batch = self.prefetch_queue.get(timeout=1.0)
            return batch
        except queue.Empty:
            # If queue is empty, get directly from data loader
            return next(self.data_loader.__iter__())

    def __len__(self):
        return len(self.data_loader)

    def stop(self):
        """Stop prefetching."""
        self.stop_prefetch = True
        if self.prefetch_thread:
            self.prefetch_thread.join()


class DynamicBatchSampler(Sampler):
    """
    Dynamic batch sampler that adjusts batch size based on memory availability.

    This sampler monitors system memory and adjusts batch sizes dynamically
    to optimize memory usage and training performance.
    """

    def __init__(
        self,
        dataset_size: int,
        base_batch_size: int = 32,
        max_batch_size: int = 128,
        memory_threshold: float = 0.8,
        adjustment_factor: float = 1.2,
    ):
        """
        Initialize dynamic batch sampler.

        Args:
            dataset_size: Size of the dataset
            base_batch_size: Base batch size
            max_batch_size: Maximum batch size
            memory_threshold: Memory usage threshold for adjustment
            adjustment_factor: Factor for batch size adjustment
        """
        self.dataset_size = dataset_size
        self.base_batch_size = base_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.adjustment_factor = adjustment_factor

        self.current_batch_size = base_batch_size
        self.batch_history = deque(maxlen=10)

        print(f"DynamicBatchSampler initialized with base batch size: {base_batch_size}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage as a fraction."""
        memory = psutil.virtual_memory()
        return memory.percent / 100.0

    def _adjust_batch_size(self):
        """Adjust batch size based on memory usage."""
        memory_usage = self._get_memory_usage()

        if memory_usage > self.memory_threshold:
            # Reduce batch size if memory usage is high
            self.current_batch_size = max(
                self.base_batch_size, int(self.current_batch_size / self.adjustment_factor)
            )
        else:
            # Increase batch size if memory usage is low
            self.current_batch_size = min(
                self.max_batch_size, int(self.current_batch_size * self.adjustment_factor)
            )

        self.batch_history.append(self.current_batch_size)

    def __iter__(self):
        """Generate batch indices."""
        indices = list(range(self.dataset_size))

        # Shuffle indices
        np.random.shuffle(indices)

        # Generate batches
        for i in range(0, len(indices), self.current_batch_size):
            batch_indices = indices[i : i + self.current_batch_size]

            # Adjust batch size for next iteration
            self._adjust_batch_size()

            yield batch_indices

    def __len__(self):
        return (self.dataset_size + self.current_batch_size - 1) // self.current_batch_size

    def get_stats(self) -> Dict[str, Any]:
        """Get sampler statistics."""
        return {
            "current_batch_size": self.current_batch_size,
            "base_batch_size": self.base_batch_size,
            "max_batch_size": self.max_batch_size,
            "memory_usage": self._get_memory_usage(),
            "batch_history": list(self.batch_history),
        }


class OptimizedDataLoader:
    """
    High-performance data loader with multiple optimizations.

    This data loader combines multiple optimization techniques:
    - Prefetching with background threads
    - Dynamic batch sizing
    - Memory pinning
    - Caching
    - Efficient memory management
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        shuffle: bool = True,
        drop_last: bool = False,
        use_dynamic_batching: bool = True,
        cache_size: Optional[int] = None,
    ):
        """
        Initialize optimized data loader.

        Args:
            dataset: Dataset to load
            batch_size: Base batch size
            num_workers: Number of worker processes
            prefetch_factor: Number of batches to prefetch
            pin_memory: Whether to pin memory
            shuffle: Whether to shuffle data
            drop_last: Whether to drop incomplete batches
            use_dynamic_batching: Whether to use dynamic batch sizing
            cache_size: Number of samples to cache
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.use_dynamic_batching = use_dynamic_batching
        self.cache_size = cache_size

        # Create optimized dataset if caching is enabled
        if cache_size and cache_size > 0:
            self.dataset = OptimizedDataset(
                dataset.data if hasattr(dataset, "data") else dataset,
                dataset.targets if hasattr(dataset, "targets") else None,
                cache_size=cache_size,
                pin_memory=pin_memory,
            )

        # Create sampler
        if use_dynamic_batching:
            self.sampler = DynamicBatchSampler(
                dataset_size=len(self.dataset),
                base_batch_size=batch_size,
                max_batch_size=batch_size * 4,
            )
        else:
            self.sampler = None

        # Create data loader
        self.data_loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            shuffle=shuffle if not use_dynamic_batching else False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Create prefetch loader
        self.prefetch_loader = PrefetchDataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        print(f"OptimizedDataLoader initialized with {num_workers} workers")

    def __iter__(self):
        """Iterate over batches."""
        return iter(self.prefetch_loader)

    def __len__(self):
        return len(self.data_loader)

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        stats = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "cache_enabled": self.cache_size is not None,
            "dynamic_batching": self.use_dynamic_batching,
        }

        if hasattr(self.dataset, "get_cache_stats"):
            stats.update(self.dataset.get_cache_stats())

        if self.sampler:
            stats.update(self.sampler.get_stats())

        return stats

    def stop(self):
        """Stop the data loader."""
        self.prefetch_loader.stop()


def create_optimized_loader(
    dataset: Dataset, batch_size: int = 32, num_workers: Optional[int] = None, **kwargs
) -> OptimizedDataLoader:
    """
    Create an optimized data loader with automatic configuration.

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        num_workers: Number of workers (auto-detect if None)
        **kwargs: Additional arguments

    Returns:
        OptimizedDataLoader: Configured data loader
    """
    if num_workers is None:
        # Auto-detect optimal number of workers
        num_workers = min(4, os.cpu_count() or 1)

    return OptimizedDataLoader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers, **kwargs
    )
