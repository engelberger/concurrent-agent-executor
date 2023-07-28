import queue
import threading
from typing import Optional


class PriorityQueueMultiGet(queue.PriorityQueue):
    lock: threading.Lock

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()

    def empty(self):
        with self.lock:
            return super().empty()

    def size(self):
        with self.lock:
            return super().qsize()

    def put(self, *args, **kwargs):
        with self.lock:
            super().put(*args, **kwargs)

    def get(self, *args, **kwargs):
        with self.lock:
            return super().get(*args, **kwargs)

    def get_multiple(self, count: Optional[int] = None):
        if count is None:
            count = self.size()

        with self.lock:
            items = []
            while not super().empty() and len(items) < count:
                item = super().get()
                items.append(item)
            return items
