# Benchmarks

Three scripts, three questions.

| Script | Answers |
|---|---|
| `benchmark_inference.py` | How fast is the model server? |
| `benchmark_agents.py` | How fast is a whole agent turn? |
| `benchmark_rag.py` | How fast is indexing and retrieval? |

All of them run against whatever `backend/.env` configures, so the same command
measures the mock backend on a laptop and vLLM on a GPU box.

## Typical use

```bash
# Does concurrency help? Where does it stop helping?
python benchmarks/benchmark_inference.py --concurrency 1 4 8 16 32 --requests 20

# What does the application add on top of the model?
python benchmarks/benchmark_agents.py --requests 10

# Did that chunking change slow retrieval down?
python benchmarks/benchmark_rag.py --documents 50 --queries 100
```

## Comparing two configurations

```bash
python benchmarks/benchmark_inference.py --output results/qwen3-8b-fp16.json
# change MODEL_NAME or QUANTIZATION, restart vLLM
python benchmarks/benchmark_inference.py --output results/qwen3-8b-fp8.json
```

Both files carry the environment they ran under, so a comparison cannot
silently be between two different setups.

## Reading the numbers

* **TTFT** is prefill-dominated and grows with prompt length. It is what makes
  the UI feel responsive.
* **Decode speed** (tok/s per request) is what makes a long answer bearable. It
  falls as concurrency rises - that is expected.
* **Aggregate output tok/s** should keep rising with concurrency well past the
  point where per-request speed drops. That gap is continuous batching doing
  its job, and it is the argument for vLLM over one-request-at-a-time serving.
* **p95, not mean.** The tail is what users describe as "it's slow sometimes".

Every run warms up first. The first request against a cold vLLM server pays for
CUDA graph capture and an empty KV cache, and including it makes comparisons
meaningless.

## A note on the mock backend

With `LLM_PROVIDER=mock` these scripts measure the *application* - context
assembly, tool dispatch, event fan-out - and not a model. That is genuinely
useful for catching a regression in the agent loop, and the scripts say so in
their banner so the numbers are never mistaken for inference performance.
