# Beta Group: Multi-Agent Latency Study (2025)

## Setup
Beta Group studied orchestration strategies for multiple language-model agents
answering a shared user request. Hardware was a single NVIDIA A5000 with 24 GB
of VRAM serving an 8-billion-parameter model. KV cache was limited to 5.4 GB,
which the authors state permitted roughly 1.2 concurrent long requests.

## Sequential versus concurrent dispatch
Running three agents one after another produced an end-to-end latency of 43.9
seconds. Running the same three agents concurrently produced 19.8 seconds, a
speedup of 2.22x. Beta Group emphasises that the concurrent arm did not reduce
per-agent latency; individual agents were marginally slower. The gain came
entirely from overlapping execution.

## Caveat on caching
Beta Group observed that concurrent agents sharing an identical prompt prefix
benefited from prefix caching, measuring a 48 percent prefix cache hit rate in
the concurrent arm versus 11 percent in the sequential arm. The authors warn
that this inflates the apparent benefit of concurrency and recommend repeating
the experiment with prefix caching disabled.

## Contradiction with Alpha Lab
Beta Group reports a practical concurrency ceiling of approximately two
simultaneous requests, which is lower than the figure of four reported by
Alpha Lab. Beta Group attributes the difference to the smaller KV cache on the
A5000 rather than to any scheduling difference.
