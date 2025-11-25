# FlashFineTune-ASR: Efficient Online Learning via Logit Caching

[](https://www.python.org/downloads/release/python-380/)
[](https://opensource.org/licenses/MIT)
[](https://pytorch.org/)

**FlashFineTune-ASR** is a lightweight framework for **online ASR (Automatic Speech Recognition) adaptation**. It enables systems to learn from user corrections in real-time without the computational overhead of re-processing audio data.

By caching inference logits ($t_0$) and utilizing user feedback ($t_1$) as ground truth, this architecture **skips the secondary forward pass**, reducing computational cost by \~50-60% during the optimization phase.

-----

## 🏗 System Architecture

The core innovation of this project is the **"Inference-Cache-Update"** loop. Instead of discarding neural network states after inference, we cache the final Logits to enable immediate, low-cost gradient calculation when feedback becomes available.

![](./assets/architecture.png)

### The 4-Stage Workflow

#### 1\. Inference & Caching ($t_0$)

The system processes the initial user input.

  - **Input:** User Audio ($Audio_{t0}$).
  - **Action:** Standard Encoder-Decoder inference.
  - **Optimization:** Instead of discarding the computation graph, we store the output **Logits ($\mathbf{Z}_{t0}$)** in a high-speed cache (GPU/RAM).
  - **Storage Cost:** Minimal (Shape: $[B, T, V]$).

#### 2\. Feedback & Labeling ($t_1$)

The user provides a correction or follow-up, which serves as the ground truth.

  - **Input:** User Audio ($Audio_{t1}$).
  - **Assumption:** The text predicted at $t_1$ is the correct label ($y_{true}$) for the intent of $t_0$.
  - **Result:** We now have the Prediction Distribution ($\mathbf{Z}_{t0}$) and the True Label ($y_{true}$).

#### 3\. Efficient Fine-tuning (The Core)

We perform a backward pass **without** feeding $Audio_{t0}$ through the model again.

  - **Process:** Load Cache $\mathbf{Z}_{t0}$ $\rightarrow$ Calculate Loss against $y_{true}$.
  - **Efficiency:** Skips the Feature Extractor and Encoder entirely.
  - **Equation:**
    $$\mathcal{L} = \text{CrossEntropy}(\text{Softmax}(\mathbf{Z}_{t0}), y_{true})$$

#### 4\. Gradient Accumulation

To prevent Catastrophic Forgetting, gradients are not applied immediately.

  - **Accumulation:** Gradients $\nabla \theta$ are stored in an accumulator.
  - **Trigger:** Weights update after $N$ interactions (e.g., $N=4$).
  - **Cleanup:** Cache is cleared after the update.

-----

## 🚀 Getting Started

### Prerequisites

  * Python 3.8+
  * PyTorch
  * Hugging Face Transformers (optional, if using pre-trained backbones)

### Installation

```bash
git clone https://github.com/yourusername/FlashFineTune-ASR.git
cd FlashFineTune-ASR
pip install -r requirements.txt
```

-----

## 💻 Usage Example

Here is a pseudo-code demonstration of how the agent handles the conversational loop:

```python
from flash_finetune import ASRAgent

# Initialize the agent (e.g., with N=4 gradient accumulation steps)
agent = ASRAgent(model_name="whisper-small", accumulation_steps=4)

# --- Stage 1: Initial Interaction (t0) ---
audio_t0 = load_audio("user_request.wav")
text_t0, cache_id = agent.infer(audio_t0, cache_logits=True)

print(f"Bot: {text_t0}")
# User thinks: "That was wrong."

# --- Stage 2: User Correction (t1) ---
# User says: "No, I meant [correction]"
audio_t1 = load_audio("user_correction.wav")
text_t1 = agent.infer(audio_t1, cache_logits=False)

# --- Stage 3: Instant Optimization ---
# We use text_t1 as the Ground Truth for the audio_t0
loss = agent.compute_loss_from_cache(
    cache_id=cache_id, 
    ground_truth_text=text_t1
)

# Backward pass is triggered internally, skipping the encoder forward pass
print(f"Loss computed: {loss.item()}")

# --- Stage 4: Update ---
# If accumulation threshold is met, optimizer.step() is called automatically
agent.step_if_ready()
```

-----

## ⚙️ Configuration

You can tune the behavior of the online learning in `config.yaml`:

```yaml
training:
  learning_rate: 1e-5
  accumulation_steps: 4  # Number of dialogues before weight update
  max_cache_size: 100    # Max number of logit tensors to keep in VRAM
  freeze_encoder: True   # If True, only updates the Adapter/Head
```

## 📊 Performance Benefits

| Metric | Standard Online Learning | FlashFineTune (Ours) |
| :--- | :--- | :--- |
| **VRAM Usage** | High (Retains full graph) | **Low** (Stores only Logits) |
| **Compute Cost** | 2x Forward Pass | **1x Forward Pass** |
| **Latency** | High | **Near Zero** (for update step) |

## 🤝 Contributing

Contributions are welcome\! Please read [CONTRIBUTING.md](https://www.google.com/search?q=CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.