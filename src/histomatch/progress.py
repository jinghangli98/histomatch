"""Progress bars for the long steps. One bar is active at a time; inner loops call tick()."""
import sys

from tqdm import tqdm

ENABLED = None          # None: only when stderr is a terminal; True / False: forced
_bar = None
_step = [0, 0]          # current step, number of steps


def plan(n_steps):
    _step[0], _step[1] = 0, n_steps


def start(total, desc):
    """Begin the next step with a bar of `total` units."""
    global _bar
    done()
    _step[0] += 1
    label = f"[{_step[0]}/{_step[1]}] {desc}" if _step[1] else desc
    _bar = tqdm(total=total, desc=f"{label:<34}", disable=None if ENABLED is None else not ENABLED,
                file=sys.stderr, dynamic_ncols=True, leave=True,
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    return _bar


def tick(n=1):
    if _bar is not None:
        _bar.update(n)


def extend(n):
    """The step turned out to have `n` more units of work."""
    if _bar is not None and _bar.total is not None:
        _bar.total += n
        _bar.refresh()


def done():
    global _bar
    if _bar is not None:
        if _bar.total is not None and _bar.n < _bar.total:
            _bar.total = _bar.n                      # finished early: show a full bar
            _bar.refresh()
        _bar.close()
        _bar = None


def write(msg=""):
    """Print a line without breaking an active bar."""
    if _bar is not None and not _bar.disable:
        tqdm.write(msg, file=sys.stdout)
    else:
        print(msg, flush=True)
