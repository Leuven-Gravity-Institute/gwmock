# `gwmock.data.time_series.time_series`

`gwmock.data.TimeSeries` is `gwmock_data.MultichannelTimeSeries`, shared by the gwmock packages
and re-exported here under the name gwmock has always used; it is documented from
`gwmock_data.multichannel_series`.

::: gwmock_data.multichannel_series.MultichannelTimeSeries
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_object_full_path: true
      members_order: source
      filters:
        - '!^_'

## Injecting a chunk that is off the segment grid

`TimeSeries.inject` adds a chunk to a segment by GPS time. When the chunk's samples fall a
fraction of a sample off the segment's grid, the chunk is resampled before it is added.

A simulated signal usually does not take this path. When the signal backend's `simulate` takes an
`output_grid` parameter by keyword, as gwmock-signal's does from the release that added it, each per-event
chunk is projected onto the segment's grid and declared there, so it is added by whole samples; the
batched path does the same. That one-step projection is the more accurate: resampling a finished,
cut buffer cannot reproduce what the polarizations' band-limited interpolant puts past its ends, which
costs up to 1.5e-8 of the signal's peak at the buffer's first samples at 8192 Hz, decaying within a
few hundred samples, almost all of it near Nyquist. It grows as the signal's content approaches
Nyquist: the same binary black hole at 1024 Hz differs by up to about 4e-6 of its peak. The path below is what a chunk from a backend without that
option falls back to when it is off the grid, and what any chunk built off the grid takes.

- **Kernel.** The band-limited windowed-sinc kernel `gwmock_data.resampling.resample_uniform_sinc`,
  shared by the gwmock packages: gwmock-signal's projection and its `inject_strain` resample with
  it too. Earlier releases interpolated linearly, which at 0.8 x Nyquist was off by ~0.63
  of the signal's peak; the sinc kernel's error there is ~1e-12. Off-grid injections therefore
  produce different samples from earlier releases, most visibly at high frequency.
- **Whole chunk, once.** The whole chunk is resampled onto the segment's grid, extended past
  either end of the segment, rather than only the part inside it. The kernel needs about half its
  taps of context on each side, and resampling segment by segment would cut that context at every
  segment boundary. Only grid points within the chunk's own sample span are produced.
- **What comes back.** The part of the chunk past the segment's end is returned for the next
  segment. For an off-grid chunk it is the resampled series, already on the segment grid, so a
  contiguous next segment adds it without resampling it again; it keeps the chunk's metadata,
  channel names and units. A chunk ending inside the segment returns `None`, as does a single
  off-grid sample with no grid point inside its span (nothing is injected and a warning is
  logged). A chunk lying entirely before or after the segment is returned unchanged.

### Agreement with `inject_strain`

`gwmock_signal.injection.core.inject_strain` resamples off-grid injections with the same kernel
(from gwmock-signal 0.17.4; this package requires 0.22.0), so off-grid injection through gwmock
and through `inject_strain` produces the same samples.

### Test coverage

The end-to-end reference matrix does not exercise this path: its fixture event coalesces on a
whole second and its runs start on a 16-second boundary, so every chunk reaches injection on the
segment grid. The off-grid path is covered by unit tests against the analytic signal. An
integration test generates a real waveform with an off-grid coalescence time across a segment
boundary and checks it against the on-grid run shifted by the same sub-sample offset; with a
gwmock-signal that takes `output_grid` the chunk is projected onto the grid instead, and the test
checks that no chunk is resampled at assembly.
