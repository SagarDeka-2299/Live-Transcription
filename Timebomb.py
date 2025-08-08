import asyncio
from typing import Callable, Awaitable

class ResettableTimer:
    def __init__(self, timeout_seconds, callback):
        self.timeout_seconds = timeout_seconds
        self.callback = callback
        self._task = None
    
    def reset(self):
        """Reset the timer - call when speech detected"""
        if self._task:
            self._task.cancel()
        self._task = asyncio.create_task(self._run())
    
    def cancel(self):
        """Stop the timer completely"""
        if self._task:
            self._task.cancel()
    
    async def _run(self):
        try:
            await asyncio.sleep(self.timeout_seconds)
            await self.callback()
        except asyncio.CancelledError:
            pass
