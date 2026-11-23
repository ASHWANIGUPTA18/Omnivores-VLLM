# Alpha Lab: On-Premise Inference Report (2025)

## Hardware and configuration
Alpha Lab evaluated a single NVIDIA L40S with 46 GB of VRAM. The served model
was an 8-billion-parameter dense transformer in bfloat16, occupying 15.3 GB of
weights. KV cache was allocated 11.2 GB. Maximum sequence length was 32768
tokens and the scheduler admitted at most 64 concurrent sequences.

## Measured throughput
Single-stream decode reached 41.2 tokens per second. With four concurrent
streams aggregate throughput reached 88.6 tokens per second, an effective
speedup of 2.15x over single-stream. Time to first token at one concurrent
request was 189 milliseconds; at four concurrent requests it rose to 642
milliseconds.

## Scheduling behaviour
Alpha Lab reports that continuous batching delivered most of its benefit
between one and four concurrent requests. Beyond six concurrent requests the
KV cache saturated and requests began queueing, with p95 latency rising
sharply. Alpha Lab concludes that for this hardware class the practical
concurrency ceiling is approximately four simultaneous long-context requests.

## Limitations
The evaluation used a single prompt template and did not vary output length.
No quality evaluation was performed. Alpha Lab notes that all figures come
from one machine and were not replicated.
