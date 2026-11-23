# Delta Operations: Deployment Notes

## Purpose
These notes record operational constraints observed when running local
inference servers in a shared university laboratory.

## Observed constraints
Delta Operations records that a second inference process on the same GPU
reduced available KV cache and caused the primary server to fail allocation at
startup. The recommended practice is to confirm the GPU is idle before
starting a benchmark, and to verify no background model remains resident.

## Measurement hygiene
Delta Operations recommends draining all in-flight requests to zero between
experimental conditions, since a batching scheduler will mix requests from
adjacent runs and contaminate timing. The notes also recommend interleaving
conditions rather than running one condition to completion before the next, to
avoid thermal and cache ordering effects.

## Reported token accounting
Delta Operations notes that token counts reported by the inference server
differ from client-side character-based estimates by between 8 and 14 percent,
and recommends using server-reported counts wherever available.

## Agreement with other reports
Delta Operations agrees with Alpha Lab that four simultaneous requests is a
reasonable working ceiling for a 46 GB card, and agrees with Beta Group that
concurrency does not improve per-agent latency.
