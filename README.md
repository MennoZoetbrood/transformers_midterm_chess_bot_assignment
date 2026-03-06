# ♟️ MSweetbreadChess

A chess-playing agent powered by a QLoRA fine-tuned Qwen2.5-1.5B model, built for the `INFOMTALC` (2025–2026) course's midterm chess tournament.
## How It Works

The bot combines a language model backbone with deterministic chess heuristics to select moves:

1. **Move scoring** — For a given board position (FEN), every legal move is scored by computing its log-probability under the fine-tuned model. Moves are batched in chunks to stay within VRAM limits.
2. **Heuristic overlay** — Model scores are adjusted with classical chess logic:
   - Forced-mate search (depth 3)
   - Draw and stalemate avoidance
   - Material-aware capture bonuses
   - Pawn promotion incentives
   - Endgame pawn advancement

The highest-scoring legal move is returned in UCI format.

## Model

| | |
|---|---|
| **Base model** | [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) |
| **Fine-tuning** | QLoRA (4-bit quantisation, fp16 compute) |
| **Adapter** | [MSweetbread/qwen2.5-1.5b-chess-qlora](https://huggingface.co/MSweetbread/qwen2.5-1.5b-chess-qlora) |
| **Training data** | Lichess position evaluations |
| **Input format** | `FEN: <fen>\nBest move:` |
| **Output format** | UCI move string (e.g. `e2e4`) |

## Setup

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- CUDA-capable GPU (tested on Google Colab T4)
- ~4 GB VRAM (4-bit quantised model)

## Usage

```python
from player import TransformerPlayer

player = TransformerPlayer("MSweetbreadChess")
move = player.get_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
print(move)  # e.g. "e2e4"
```


## Approach & Design Decisions

- **Why Qwen2.5-1.5B?** — Fits comfortably on a free-tier Colab T4 GPU when quantised to 4-bit, while still being expressive enough to learn positional patterns from Lichess data.
- **Why QLoRA over full fine-tuning?** — Training a 1.5B-parameter model in full precision exceeds Colab's ~15 GB VRAM. QLoRA reduces memory by quantising the frozen base weights to 4-bit and only training low-rank adapters.
- **Why heuristics on top of the model?** — The language model can hallucinate illegal moves or miss forced mates. A thin heuristic layer enforces legality and exploits positions the model might undervalue.
