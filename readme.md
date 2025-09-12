# Creating Custom Language Models (Mini GPT)

Build and experiment with a **small GPT‑style, character‑level Transformer language model** in a single Jupyter Notebook. This repo shows the full workflow end‑to‑end: load raw text, tokenize (char level), train a mini‑Transformer in PyTorch, and **generate text** from a prompt.

> If you’re new to language models, this notebook is a compact, hands‑on way to learn how GPT‑like models work under the hood.

---

## 🎬 Video Walkthrough
[![Watch the video](https://img.youtube.com/vi/1zl3zkwRL80/hqdefault.jpg)](https://www.youtube.com/watch?v=1zl3zkwRL80)

*Click the thumbnail to watch the YouTube walkthrough.*

---

## 📦 Repository Contents
```
.
├── Creating_Custom_Language_Models.ipynb
├── data/                # put your training text files here (e.g., input.txt)
├── models/              # optional: saved checkpoints will go here
└── README.md
```
- **Creating_Custom_Language_Models.ipynb** — the main notebook (mini GPT at char level).
- **data/** — place your text dataset(s) here (e.g., from Project Gutenberg or your own corpus).
- **models/** — optional folder for saving trained weights/checkpoints.

---

## ⚙️ Requirements
- Python 3.9+
- [PyTorch](https://pytorch.org/) (GPU highly recommended for speed, but CPU also works)
- Jupyter Notebook/Lab
- NumPy

Install basics with:
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install --upgrade pip
# Install PyTorch matching your system (CUDA/CPU). See https://pytorch.org/get-started/locally/
pip install torch
pip install jupyter numpy
```

---

## 🚀 Quick Start (Local)
1. **Clone or download** this repository.
2. (Optional) **Create folders**:
   ```bash
   mkdir -p data models
   ```
3. **Add a text file** for training, e.g. `data/input.txt`.  
   - You can use any plain‑text corpus (ensure you have the rights to use it).  
   - If using public domain books (e.g., Project Gutenberg), check their license terms.
4. **Launch Jupyter** and open the notebook:
   ```bash
   jupyter notebook
   ```
5. **Edit the data path** at the top of the notebook to point to your file (e.g., `./data/input.txt`).  
6. **Run the cells** top‑to‑bottom to:
   - Build the character vocabulary
   - Prepare train/validation splits and mini‑batches (context windows)
   - Define the mini‑GPT model (embeddings, multi‑head attention, MLP, layer norms, dropout)
   - Train with cross‑entropy (e.g., using AdamW)
   - **Generate text** autoregressively from a seed prompt

> **Tip:** If you hit out‑of‑memory errors on GPU/CPU, reduce `batch_size`, `block_size`, number of layers/heads, or embedding size in the hyperparameters section of the notebook.

---

## ☁️ Run on Google Colab
- Upload `Creating_Custom_Language_Models.ipynb` to Colab.
- Upload or mount your dataset (e.g., upload `input.txt` into the Colab session or mount Google Drive).
- Make sure to set the **Runtime → Change runtime type → GPU** if available.
- Update the **file path cell** to your uploaded dataset (e.g., `/content/input.txt`), then run all cells.

---

## 🧪 Text Generation
After training, use the **generation cell** in the notebook to sample text. Common knobs you may see:
- `max_new_tokens` — how many new characters to generate
- `temperature` — higher values → more randomness; lower → more conservative
- `top_k` / `top_p` — sample from a smaller set of most likely tokens

> Try different prompts and sampling settings to see how the model’s style changes.

---

## 💾 Saving & Loading
The notebook includes simple snippets to **save** and **load** model weights (either uncomment or run the cells):
```python
# Save
import os, torch
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/mini_gpt_charlevel.pt")

# Load
model.load_state_dict(torch.load("models/mini_gpt_charlevel.pt", map_location=device))
model.eval()
```
Adjust filenames/paths as you prefer.

---

## 🧰 Tips & Troubleshooting
- **Slow training on CPU**: It’ll work, but a GPU is recommended. On CPU, reduce model size and training iterations.
- **CUDA OOM**: Lower `batch_size`, `block_size`, number of layers (`n_layer`), heads (`n_head`), or embedding size (`n_embd`).
- **Unicode errors**: Ensure your dataset is saved as UTF‑8 text.
- **Weird output**: Train longer, try a larger corpus, or tweak sampling (temperature/top‑k/p).

---

## ❓ FAQ
**Q: Can I use words or subwords instead of characters?**  
A: This notebook is char‑level for learning clarity. To switch, integrate a tokenizer (e.g., BPE/WordPiece) and adapt embeddings and data loaders.

**Q: Can I fine‑tune an existing model?**  
A: This notebook shows a from‑scratch mini‑model. You can adapt the data pipeline and architecture to fine‑tune pre‑trained models using libraries like Hugging Face Transformers.

**Q: Where do I change hyperparameters?**  
A: Near the top of the notebook you’ll find settings like `batch_size`, `block_size`, number of layers/heads, embedding size, dropout, learning rate, and training iterations.

---

## 📜 License
Add your license of choice (e.g., MIT). If you use external datasets (e.g., Project Gutenberg), follow their license/usage terms.

---

## 🙏 Acknowledgements
Thanks to the open‑source ML community and many educational resources that inspire learning‑oriented GPT implementations.
