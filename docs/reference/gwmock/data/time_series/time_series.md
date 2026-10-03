# `gwmock.data.time_series.time_series`

::: gwmock.data.time_series.time_series
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
fraction of a sample off the segment's grid, the chunk is resampled before it is added:

- **Kernel.** The band-limited windowed-sinc kernel
  `gwmock_signal.projection.resampling.resample_uniform_sinc`, the one gwmock-signal's projection
  resamples with. Earlier releases interpolated linearly, which at 0.8 x Nyquist was off by ~0.63
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

`gwmock_signal.injection.core.inject_strain` in gwmock-signal 0.17.3, the minimum version this
package requires, still interpolates cubically, so off-grid injection through gwmock and through
`inject_strain` does not yet produce identical samples. This is deliberate and temporary. The
kernel module is byte-identical between gwmock-signal 0.17.3 and the pending gwmock-signal change
that moves `inject_strain` onto the same kernel, so the two paths become equal once that change
is released and required here, with no further change on the gwmock side.

### Test coverage

The end-to-end reference matrix does not exercise this path: its fixture event coalesces on a
whole second and its runs start on a 16-second boundary, so every chunk reaches injection on the
segment grid. The off-grid path is covered by unit tests against the analytic signal and by an
integration test that generates a real waveform with an off-grid coalescence time across a
segment boundary and checks it against the on-grid run shifted by the same sub-sample offset.
