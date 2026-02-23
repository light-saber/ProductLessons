# AI for PMs — Day 3: Model Quantization (Deep Dive)

## The Problem PMs Face

Your engineering team says: *"We need a bigger GPU cluster."*  
Your finance team says: *"Cut cloud spend by 30%."*

Somewhere in this tension, quantization lives. Most PMs skip straight to "evaluation frameworks" without understanding the fundamental tradeoff: **precision vs. cost**. This lesson fixes that.

---

## What Is Quantization? (The 60-Second Version)

AI models store numbers as "weights." A 70B parameter model has 70 billion of them. 

| Format | Bits per Weight | Model Size (70B) | VRAM Needed |
|--------|----------------|------------------|-------------|
| FP32 (Full) | 32 | 280 GB | 4× A100 GPUs |
| FP16 (Half) | 16 | 140 GB | 2× A100 GPUs |
| INT8 | 8 | 70 GB | 1× A100 GPU |
| INT4 | 4 | 35 GB | 1× RTX 4090 |
| Q4_K_M (GGUF) | ~4.5 | ~40 GB | Runs on a laptop |

**Quantization = Fewer bits per weight = Smaller model = Lower cost.**

---

## The Business Math PMs Care About

### Scenario: Walmart Catalog Team

**Task:** Classify 10M product images/month using a vision model.

| Setup | GPU Cost/Month | Accuracy | Latency |
|-------|---------------|----------|---------|
| GPT-4V API | $50,000 | 94% | 200ms |
| Llava-1.6 34B FP16 | $12,000 | 91% | 150ms |
| Llava-1.6 34B Q4 | $800 | 89% | 80ms |
| Llava-1.6 7B Q4 | $200 | 85% | 40ms |

**PM Decision Framework:**
- If 89% accuracy meets your SLA → **Q4 saves $11,200/month**
- If you need 91%+ → **FP16 is worth it**
- If 85% is enough → **Small Q4 model is a no-brainer**

This is the conversation you should have *before* building evals.

---

## Types of Quantization (What to Ask Your Engineers)

### 1. Post-Training Quantization (PTQ)
**What it is:** Take a trained model, shrink it.
**Pros:** Fast, no retraining needed.
**Cons:** Some accuracy loss.
**Use when:** You're deploying an open-source model quickly.

### 2. Quantization-Aware Training (QAT)
**What it is:** Train the model *knowing* it'll be quantized.
**Pros:** Better accuracy at low precision.
**Cons:** Requires training from scratch.
**Use when:** You're building a custom model for production.

### 3. GGUF/GGML (Consumer Hardware)
**What it is:** Format optimized for CPUs and consumer GPUs.
**Pros:** Run 70B models on a MacBook.
**Cons:** Slower than GPU-native formats.
**Use when:** Prototyping, edge deployment, cost-sensitive batch jobs.

---

## Quality Degradation: When Does It Matter?

### The "Quantization Cliff"

Not all models degrade equally. Some tasks are robust; others break suddenly.

| Task Type | 8-bit Loss | 4-bit Loss | Verdict |
|-----------|-----------|-----------|---------|
| Classification | ~0.5% | ~2% | Safe to quantize |
| Summarization | ~1% | ~3% | Safe for most use cases |
| Reasoning (math) | ~2% | ~8% | Test carefully |
| Code generation | ~3% | ~15% | Avoid 4-bit for production code |
| Multi-step agents | ~5% | ~20% | Often breaks at 4-bit |

**PM Insight:** The more "reasoning steps" a task requires, the more quantization hurts. A simple classifier is fine at 4-bit. An agent that chains 5 LLM calls together? That's where you'll see the cliff.

---

## Evaluating Quantized Models (The PM Playbook)

Don't let engineers tell you "it works." Define your thresholds *before* quantization.

### Step 1: Baseline Your Current Model
```
Full precision (FP16) accuracy: _____%
Full precision latency: _____ms
Full precision cost/query: $_____
```

### Step 2: Define Acceptable Tradeoffs
```
Minimum acceptable accuracy: _____% (usually 98-99% of baseline)
Maximum acceptable latency: _____ms
Maximum acceptable cost/query: $_____
```

### Step 3: Test Quantized Candidates

| Model | Bits | Accuracy | Latency | Cost/Month | Pass? |
|-------|------|----------|---------|------------|-------|
| Llama-3-8B | FP16 | 82% | 120ms | $4,800 | Baseline |
| Llama-3-8B | INT8 | 81.5% | 95ms | $2,400 | ✓ |
| Llama-3-8B | Q4_K_M | 79% | 60ms | $600 | ✗ (<80% threshold) |
| Llama-3-70B | Q4_K_M | 84% | 150ms | $2,000 | ✓ |

**Key insight:** Sometimes a *bigger quantized model* beats a *smaller full-precision one* on both accuracy AND cost.

---

## Real-World Deployment Patterns

### Pattern 1: The "Cascade"
Use cheap quantized model first. Escalate to expensive one only when uncertain.

```
User Query → Q4 Model (90% of traffic)
              ↓
       Confidence < 0.8?
              ↓
       Route to GPT-4 (10% of traffic)
```
**Result:** 70% cost reduction, 2% accuracy improvement.

### Pattern 2: The "Hybrid"
Different tasks → Different precisions.

| Task | Model | Precision | Why |
|------|-------|-----------|-----|
| Intent classification | DistilBERT | INT8 | Simple, robust |
| Entity extraction | Llama-3-8B | Q4 | Fast, good enough |
| Complex reasoning | GPT-4 | FP16 (API) | Fallback for edge cases |

### Pattern 3: The "Edge"
Run Q4 models on-device for latency/privacy.

- **iOS/Android:** Core ML / TensorFlow Lite with 8-bit
- **Browser:** Transformers.js with quantized ONNX
- **IoT:** GGML on Raspberry Pi

---

## Questions to Ask Your Engineering Team

1. **"What's our accuracy cliff?"** — At what precision does our specific task break?
2. **"Have we tested Q4 vs FP16 on our actual data?"** — Benchmarks lie. Your data matters.
3. **"What's the latency vs. cost curve?"** — Sometimes 8-bit is the sweet spot, not 4-bit.
4. **"Do we need dynamic quantization?"** — Different inputs → different precision needs.
5. **"What's our rollback plan?"** — If quantized model degrades, how fast can you switch?

---

## Common PM Mistakes

### ❌ "Let's just use the smallest model"
A 7B Q4 model might be worse than a 70B Q4 model at the same cost. Test the tradeoff.

### ❌ "Evals first, optimization later"
You can't evaluate properly without knowing your precision target. Quantization affects your baseline.

### ❌ "4-bit is always fine"
Code generation, math reasoning, and multi-hop queries often fail at 4-bit. Measure don't assume.

### ❌ "We'll quantize after launch"
Retraining evals for quantized models is expensive. Test during development.

---

## Key Takeaways

1. **Quantization is a PM decision disguised as engineering.** You own the accuracy/cost tradeoff.
2. **Start with your SLA.** Define minimum accuracy before testing models.
3. **Bigger quantized > Smaller full-precision.** A 70B Q4 often beats an 8B FP16.
4. **Task matters more than model.** Classification survives 4-bit. Reasoning may not.
5. **Cascade when possible.** Cheap first, expensive fallback.

---

## Further Reading

- **Thellama.cpp quantization wiki:** github.com/ggerganov/llama.cpp/blob/master/docs/quantization.md
- **Hugging Face Optimum:** Optimize Transformers with quantization
- **AWQ/GPTQ papers:** Advanced quantization methods with minimal loss

---

*Tomorrow: Day 4 — Evals That Actually Matter (and why most are waste)*
