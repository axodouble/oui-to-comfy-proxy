import re
import time

MAX_N = 8
MIN_DIM = 32
MAX_DIM = 4096
SIZE_RE = re.compile(r"^(\d+)x(\d+)$")


class RequestError(Exception):
    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


def _round_dim(value: int) -> int:
    return max(MIN_DIM, min(MAX_DIM, int(round(value / 32)) * 32))


def parse_size(
    size: str | None, default: tuple[int, int] = (1024, 1024)
) -> tuple[int, int]:
    if not size:
        return default
    size = size.strip()
    if size.lower() == "auto":
        return default
    m = SIZE_RE.match(size)
    if not m:
        raise RequestError(f"invalid size {size!r}, expected 'WxH' or 'auto'")
    w, h = int(m.group(1)), int(m.group(2))
    if w <= 0 or h <= 0:
        raise RequestError("size dimensions must be positive")
    return _round_dim(w), _round_dim(h)


def validate_n(n: int | None) -> int:
    if n is None:
        return 1
    if n < 1:
        raise RequestError("n must be >= 1")
    if n > MAX_N:
        raise RequestError(f"n must be <= {MAX_N}")
    return n


def image_response(b64_list: list[str]) -> dict:
    return {"created": int(time.time()), "data": [{"b64_json": b} for b in b64_list]}
