# Gamma Institute: Quality Under Concurrency (2026)

## Objective
Gamma Institute asked whether reducing latency through concurrent execution
degrades answer quality. Prior work, including Alpha Lab and Beta Group,
measured only latency and throughput.

## Method
Gamma Institute served an 8-billion-parameter model and evaluated 240
document-grounded questions under sequential and concurrent dispatch. Model
parameters, prompts, retrieved documents and conversation history were held
constant across both conditions. Answers were scored for citation correctness
and factual accuracy by two independent human raters.

## Findings
Citation correctness was 94.1 percent under sequential dispatch and 93.7
percent under concurrent dispatch, a difference the authors describe as within
measurement noise. Factual accuracy was 88.2 percent sequential and 88.0
percent concurrent. Gamma Institute concludes that concurrency did not
measurably degrade answer quality on this workload.

## Disagreement on the cause of speedup
Unlike Beta Group, Gamma Institute attributes the latency reduction primarily
to GPU scheduler batching rather than to prefix caching. Gamma Institute
repeated its measurements with prefix caching disabled and still observed a
1.87x speedup, which it argues isolates the scheduling effect.

## Limitations
Only one model family was tested. The authors did not vary the number of
agents beyond three, and did not test more than four simultaneous users.
