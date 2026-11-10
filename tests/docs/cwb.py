"""How cWB reads data quality files and turns them into analysis jobs, for the tests.

cWB is not published on PyPI, so the tests cannot run it. This is a line-for-line reimplementation of
the three routines of its toolbox (``CWB::Toolbox``) that decide which data an analysis reads, from
cWB's O4 release:

- ``readSegList(dqfile)``: reads one data quality file, ``start stop`` per line (format B) when the
  entry's ``c4`` field is false, ``index start stop duration`` (format A) when it is true; skips lines
  starting with ``#``; stops the analysis on a file with no segment or a segment that does not end
  after it starts; adds the entry's ``shift``.
- ``readSegList(nDQF, DQF, dqcat)``: intersects the lists of every file of category at most
  ``dqcat`` (``mergeSegLists``), across all interferometers.
- ``getJobList(ilist, segLen, segMLS, segEdge)``: rounds each segment inward to whole seconds, takes
  ``segEdge`` off both ends, and cuts the rest into jobs of at most ``segLen`` seconds, dropping a
  piece shorter than ``segMLS``. A job reads its data with ``segEdge`` on both sides, so it reads only
  inside the segment it came from.

Inverted files (vetoes) and categories above ``CWB_CAT0`` are not reimplemented: gwmock writes only
the ``CWB_CAT0`` lists of time to analyse.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

#: cWB's defaults: segment length, minimum segment length after CAT1, and the wavelet edge, seconds.
SEG_LEN = 600.0
SEG_MLS = 300.0
SEG_EDGE = 8.0

CWB_CAT0 = 0


class CWBExitError(RuntimeError):
    """Where cWB calls ``gSystem->Exit(1)``."""


@dataclass(frozen=True)
class DQFile:
    """One ``dqfile`` entry of ``user_parameters.C``."""

    ifo: str
    file: str
    cat: int
    shift: float
    invert: bool
    c4: bool


_DQFILE = re.compile(
    r'\{\s*"(?P<ifo>[^"]+)"\s*,\s*"(?P<file>[^"]+)"\s*,\s*CWB_CAT(?P<cat>\d)\s*,\s*(?P<shift>[-+0-9.eE]+)\s*,'
    r"\s*(?P<invert>true|false)\s*,\s*(?P<c4>true|false)\s*\}"
)


def parse_dqfiles(source: str) -> list[DQFile]:
    """The ``dqfile`` entries of a ``user_parameters.C`` fragment."""
    return [
        DQFile(
            ifo=match["ifo"],
            file=match["file"],
            cat=int(match["cat"]),
            shift=float(match["shift"]),
            invert=match["invert"] == "true",
            c4=match["c4"] == "true",
        )
        for match in _DQFILE.finditer(source)
    ]


def read_seg_list(dqf: DQFile, root: Path) -> list[tuple[float, float]]:
    """``readSegList(dqfile)``: one file's segments, sorted, refusing what cWB refuses."""
    if dqf.invert:
        raise NotImplementedError("inverted (veto) files are not reimplemented")
    lines = [line for line in (root / dqf.file).read_text().splitlines() if not line.startswith("#")]
    tokens = " ".join(lines).split()
    if not tokens:
        raise CWBExitError(f"CWB::Toolbox::readSegList - Error : File {dqf.file} is empty")
    width = 4 if dqf.c4 else 2
    if len(tokens) % width:
        raise CWBExitError(f"CWB::Toolbox::readSegList - malformed file {dqf.file}")
    segments = []
    for i in range(0, len(tokens), width):
        row = [float(token) for token in tokens[i : i + width]]
        start, stop = (row[1], row[2]) if dqf.c4 else (row[0], row[1])
        start, stop = start + dqf.shift, stop + dqf.shift
        if stop <= start:
            raise CWBExitError(f"CWB::Toolbox::readSegList - Error Ranges : {start} {stop}")
        segments.append((start, stop))
    segments.sort()
    for (_, previous_stop), (start, _) in pairwise(segments):
        if start <= 0 or start < previous_stop:
            raise CWBExitError(f"CWB::Toolbox::readSegList - Error in segment list file (duplicated veto) : {dqf.file}")
    return segments


def merge_seg_lists(first: list[tuple[float, float]], second: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """``mergeSegLists``: the intersection of two sorted lists."""
    merged, i, j = [], 0, 0
    while i < len(first) and j < len(second):
        start, stop = max(first[i][0], second[j][0]), min(first[i][1], second[j][1])
        if stop > start:
            merged.append((start, stop))
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return merged


def read_seg_lists(dqfs: list[DQFile], root: Path, dqcat: int = CWB_CAT0) -> list[tuple[float, float]]:
    """``readSegList(nDQF, DQF, dqcat)``: every file of category at most ``dqcat``, intersected."""
    selected = [dqf for dqf in dqfs if dqf.cat <= dqcat]
    if not selected:
        raise CWBExitError(f"CWB::Toolbox::readSegList - no CWB_CAT={dqcat} files in the list")
    merged = read_seg_list(selected[0], root)
    for dqf in selected[1:]:
        merged = merge_seg_lists(read_seg_list(dqf, root), merged)
    return merged


def get_job_list(
    segments: list[tuple[float, float]], seg_len: float = SEG_LEN, seg_mls: float = SEG_MLS, seg_edge: float = SEG_EDGE
) -> list[tuple[int, int]]:
    """``getJobList``: the analysis jobs, ``(start, stop)`` without their edges."""
    jobs: list[tuple[int, int]] = []
    for segment_start, segment_stop in segments:
        start = int(segment_start + 1) if math.fmod(segment_start, 1) != 0 else int(segment_start)
        stop = int(segment_stop)  # cWB truncates: 1.x -> 1.0
        start += int(seg_edge)
        stop -= int(seg_edge)
        length = stop - start
        if length <= 0:
            continue
        n = int(length / seg_len)
        if n == 0:
            if length >= seg_mls:
                jobs.append((start, stop))
            continue
        if n == 1:
            if length > seg_len:
                half = int(length / 2)
                if half >= seg_mls:
                    jobs += [(start, start + half), (start + half, stop)]
                else:
                    jobs.append((start, int(start + seg_len)))
            else:
                jobs.append((start, stop))
            continue
        job_stop = start
        for j in range(n - 1):
            job_start = int(seg_len * j + start)
            job_stop = int(job_start + seg_len)
            jobs.append((job_start, job_stop))
        half = int((stop - job_stop) / 2)
        if half >= seg_mls:
            jobs += [(job_stop, job_stop + half), (job_stop + half, stop)]
        else:
            jobs.append((job_stop, int(job_stop + seg_len)))
    return jobs


def read_windows(jobs: list[tuple[int, int]], seg_edge: float = SEG_EDGE) -> list[tuple[float, float]]:
    """The data each job reads: the job with ``segEdge`` on both sides."""
    return [(start - seg_edge, stop + seg_edge) for start, stop in jobs]
