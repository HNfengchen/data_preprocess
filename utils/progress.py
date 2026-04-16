from tqdm import tqdm


class ProgressTracker:
    """tqdm progress bar wrapper with disable mode for tests/CI"""

    def __init__(self, total: int, desc: str = "处理中", enabled: bool = True):
        self._enabled = enabled
        self.pbar = tqdm(total=total, desc=desc, unit="张", disable=not enabled)

    def update(self, n: int = 1, msg: str = "") -> None:
        self.pbar.update(n)
        if msg:
            self.pbar.set_postfix_str(msg)

    def close(self) -> None:
        self.pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
