# SPDX-License-Identifier: Apache-2.0
"""GPU lane dispatcher for blocking inference calls."""

import asyncio
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


class LaneQueueFull(RuntimeError):
    """Raised when the lane queue is full."""


@dataclass
class _Lane:
    device_id: Optional[int]
    queue: "queue.Queue[tuple[asyncio.AbstractEventLoop, asyncio.Future, Callable, tuple, dict]]"


class GpuLaneDispatcher:
    """Dispatch blocking inference calls to per-GPU lanes."""

    def __init__(
        self,
        device_ids: Iterable[Optional[int]],
        max_queue_size: int = 8,
    ) -> None:
        self._lanes: list[_Lane] = []
        self._rr_lock = threading.Lock()
        self._rr_index = 0

        for device_id in device_ids:
            maxsize = max_queue_size if max_queue_size > 0 else 0
            lane_queue: "queue.Queue[tuple[asyncio.AbstractEventLoop, asyncio.Future, Callable, tuple, dict]]" = queue.Queue(
                maxsize=maxsize
            )
            lane = _Lane(device_id=device_id, queue=lane_queue)
            self._lanes.append(lane)
            thread = threading.Thread(
                target=self._worker_loop,
                args=(lane,),
                name=f"gpu-lane-{device_id if device_id is not None else 'cpu'}",
                daemon=True,
            )
            thread.start()

        logger.info(
            "gpu_lane_dispatcher ready lanes=%s queue_limit=%s",
            len(self._lanes),
            max_queue_size,
        )

    def _worker_loop(self, lane: _Lane) -> None:
        if lane.device_id is not None:
            try:
                import torch

                torch.cuda.set_device(lane.device_id)
            except Exception as exc:
                logger.warning("Failed to set CUDA device %s: %s", lane.device_id, exc)

        while True:
            loop, future, func, args, kwargs = lane.queue.get()
            try:
                result = self._run_task(func, args, kwargs)
                loop.call_soon_threadsafe(future.set_result, result)
            except Exception as exc:
                loop.call_soon_threadsafe(future.set_exception, exc)
            finally:
                lane.queue.task_done()

    @staticmethod
    def _run_task(func: Callable, args: tuple, kwargs: dict) -> Any:
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return asyncio.run(result)
        return result

    def _select_lane(self) -> _Lane:
        with self._rr_lock:
            sizes = [lane.queue.qsize() for lane in self._lanes]
            min_size = min(sizes)
            candidates = [idx for idx, size in enumerate(sizes) if size == min_size]
            idx = candidates[self._rr_index % len(candidates)]
            self._rr_index += 1
        return self._lanes[idx]

    async def submit(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        lane = self._select_lane()
        try:
            lane.queue.put_nowait((loop, future, func, args, kwargs))
        except queue.Full as exc:
            raise LaneQueueFull("Inference queue is full") from exc
        return await future
