# From Blind Spots to Gains: Diagnostic-Driven Iterative Training for Large Multimodal Models

## 🔔 Updates
**2026-2-26:** We released our paper and Models! 🎉 

📄 [Read the Paper](https://arxiv.org/abs/2602.22859) &nbsp;|&nbsp; 🌐 [Visit the Model Zoo](#-model-zoo)

---

**DPE (Diagnostic-driven Progressive Evolution)** is a self-evolving training framework for Large Multimodal Models (LMMs). Inspired by the "diagnose-and-correct" mechanism in educational psychology, DPE moves beyond indiscriminate data expansion. It prioritizes the **diagnosis of capability gaps** to steer targeted data generation and mixture optimization, effectively breaking the multimodal long-tail bottleneck.

## 🌟 Key Features

- **Adaptive Diagnosis Mechanism**: Before each evolution cycle, a diagnostic agent analyzes the model's failure patterns to identify specific weaknesses and capability blind spots. This insight dynamically optimizes the training data mixture.
- **Tool-Use Data Evolution**: Instead of relying on static datasets or simple text rewriting, DPE employs a multi-agent system equipped with image search and editing tools to source and annotate diverse visual content from external pools.
- **High Efficiency**: Broad improvements in multimodal reasoning can be achieved with only ~1,000 targeted training examples.
- **Enhanced Stability**: The closed-loop of diagnosis, generation, and reinforcement significantly improves training stability and mitigates capability regression on long-tail tasks like Mathematics and OCR.

## 🚀 Getting Started

### Installation

We recommend using Conda to manage the environment:

```bash
# Create and activate environment
conda create -n DPE python=3.11
conda activate DPE

# Run setup script
bash setup.sh
```

### Training Pipeline

DPE follows an iterative evolution workflow managed by `run_iterative_pipeline.sh`. The pipeline automates the following steps for each iteration:

1.  **Weakness Analysis**: Identifies current model gaps.
2.  **Question Generation**: Uses tool-calling agents to generate targeted training data based on the diagnosis.
3.  **Score & Filter**: Evaluates and selects high-quality samples.
4.  **Solver Training**: Conducts Reinforcement Learning (RL) or Supervised Fine-Tuning (SFT) to enhance capabilities.

To start the pipeline:

```bash
bash run_iterative_pipeline.sh
```

### Evaluation

We use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for comprehensive evaluation across 11 challenging benchmarks, including:
- **MMMU**: Multimodal University-level Understanding
- **MathVision**: Multimodal Mathematical Reasoning
- **CharXiv**: Realistic Charting Gaps
- **OCR & Localization**: Tasks involving fine-grained visual perception.

## 📦 Model Zoo

We provide multiple versions of DPE-enhanced models on Hugging Face:

| Base Model | Version | Hugging Face Link |
| :--- | :--- | :--- |
| **Qwen3-VL-8B-Instruct** | v3 | [hongruijia/Qwen3_VL_8B_Instruct_DPE_v3](https://huggingface.co/hongruijia/Qwen3_VL_8B_Instruct_DPE_v3) |
| | v2 | [hongruijia/Qwen3_VL_8B_Instruct_DPE_v2](https://huggingface.co/hongruijia/Qwen3_VL_8B_Instruct_DPE_v2) |
| | v1 | [hongruijia/Qwen3_VL_8B_Instruct_DPE_v1](https://huggingface.co/hongruijia/Qwen3_VL_8B_Instruct_DPE_v1) |
| **Qwen2.5-VL-7B-Instruct** | v3 | [hongruijia/Qwen2.5-VL-7B-Instruct_DPE_v3](https://huggingface.co/hongruijia/Qwen2.5-VL-7B-Instruct_DPE_v3) |
| | v2 | [hongruijia/Qwen2.5-VL-7B-Instruct_DPE_v2](https://huggingface.co/hongruijia/Qwen2.5-VL-7B-Instruct_DPE_v2) |
| | v1 | [hongruijia/Qwen2.5-VL-7B-Instruct_DPE_v1](https://huggingface.co/hongruijia/Qwen2.5-VL-7B-Instruct_DPE_v1) |

## 🏗️ Acknowledgements

The codebase is built upon and inspired by the following projects:
[VisPlay](https://github.com/bruno686/VisPlay), 
[Vision-SR1](https://github.com/zli12321/Vision-SR1), 
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

## 📑 Citation

If you find this work helpful, please consider citing our paper:

```latex
@misc{jia2026blindspotsgainsdiagnosticdriven,
      title={From Blind Spots to Gains: Diagnostic-Driven Iterative Training for Large Multimodal Models}, 
      author={Hongrui Jia and Chaoya Jiang and Shikun Zhang and Wei Ye},
      year={2026},
      eprint={2602.22859},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.22859}, 
}
```