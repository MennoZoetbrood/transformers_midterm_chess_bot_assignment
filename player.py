from typing import Optional
from abc import ABC, abstractmethod

import torch
import chess
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------------------------------------------------------
# Player base class — the instructor package exposes the same interface,
# but we redefine it here so the file is self-contained during local testing.
# When the championship runner imports TransformerPlayer it will override
# the Player binding via its own package, which is fine because
# TransformerPlayer.__bases__ is re-resolved at import time.
# ---------------------------------------------------------------------------
try:
    from chess_tournament import Player          # instructor package (Colab)
except ImportError:
    class Player(ABC):
        def __init__(self, name: str):
            self.name = name

        @abstractmethod
        def get_move(self, fen: str) -> Optional[str]:
            pass


# ---------------------------------------------------------------------------
# Helper — must match the format used during fine-tuning exactly
# ---------------------------------------------------------------------------
def format_chess_prompt(fen: str, move: Optional[str] = None) -> str:
    """Format a FEN position into the prompt used during training."""
    prompt = f"FEN: {fen}\nBest move:"
    if move is not None:
        return f"{prompt} {move}"
    return prompt


# ---------------------------------------------------------------------------
# TransformerPlayer
# ---------------------------------------------------------------------------
class TransformerPlayer(Player):
    """Chess player using a QLoRA fine-tuned Qwen2.5-1.5B model.

    Scores every legal move by computing its log-probability under the
    fine-tuned model, then applies deterministic chess heuristics on top
    (forced-mate search, draw avoidance, material-aware captures, pawn
    promotion, endgame pawn push).
    """

    CHUNK_SIZE = 32  # moves scored per forward pass

    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    def __init__(
        self,
        name: str = "MSweetbreadChess",
        model_path: str = "MSweetbread/qwen2.5-1.5b-chess-qlora",
        base_model_id: str = "Qwen/Qwen2.5-1.5B",
    ):
        super().__init__(name)

        # ---- tokenizer ----
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # ---- 4-bit base model ----
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map="auto",
            trust_remote_code=True,
        )

        # ---- attach LoRA adapter ----
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    # ------------------------------------------------------------------
    # Model scoring
    # ------------------------------------------------------------------

    def score_all_moves(self, fen: str, legal_moves: list) -> list:
        """Return a log-probability score for each move in *legal_moves*."""
        prompt_text = format_chess_prompt(fen) + " "
        full_texts = [format_chess_prompt(fen, m.uci()) for m in legal_moves]

        prompt_len = self.tokenizer.encode(prompt_text, return_tensors="pt").shape[1]

        batch = self.tokenizer(
            full_texts,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**batch).logits

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        scores = []
        for i in range(len(legal_moves)):
            total_log_prob = 0.0
            seq_ids = batch["input_ids"][i]
            attn_mask = batch["attention_mask"][i]
            for pos in range(prompt_len - 1, seq_ids.shape[0] - 1):
                if attn_mask[pos + 1] == 0:
                    break
                next_token_id = seq_ids[pos + 1].item()
                total_log_prob += log_probs[i, pos, next_token_id].item()
            scores.append(total_log_prob)

        return scores

    # ------------------------------------------------------------------
    # Chess helpers
    # ------------------------------------------------------------------

    def count_material(self, board: chess.Board, color: bool) -> int:
        total = 0
        for piece_type, value in self.PIECE_VALUES.items():
            total += len(board.pieces(piece_type, color)) * value
        return total

    def is_endgame(self, board: chess.Board) -> bool:
        return len(board.piece_map()) <= 12

    def has_mate_in_n(self, board: chess.Board, max_depth: int = 3) -> Optional[int]:
        """Return the index of a forced-mate move, or None."""
        legal_moves = list(board.legal_moves)

        # Depth 1 — immediate checkmate
        for i, move in enumerate(legal_moves):
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return i
            board.pop()

        if max_depth < 3:
            return None

        # Depth 3 — every opponent reply still leads to checkmate
        for i, move in enumerate(legal_moves):
            board.push(move)
            if not board.is_game_over():
                opponent_moves = list(board.legal_moves)
                all_lead_to_mate = bool(opponent_moves)
                for opp_move in opponent_moves:
                    board.push(opp_move)
                    found = False
                    for my_move in list(board.legal_moves):
                        board.push(my_move)
                        if board.is_checkmate():
                            found = True
                        board.pop()
                        if found:
                            break
                    board.pop()
                    if not found:
                        all_lead_to_mate = False
                        break
                if all_lead_to_mate:
                    board.pop()
                    return i
            board.pop()

        return None

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------

    def apply_heuristics(self, board: chess.Board, legal_moves: list, scores: list):
        """Overlay chess heuristics on model scores.

        Returns either:
        - an int  → index of a forced-mate move (play it immediately), or
        - a list  → adjusted float scores for every move.
        """
        # Forced mate — skip everything else
        mate_idx = self.has_mate_in_n(board, max_depth=3)
        if mate_idx is not None:
            return mate_idx

        my_color = board.turn
        opp_color = not my_color
        material_advantage = (
            self.count_material(board, my_color)
            - self.count_material(board, opp_color)
        )
        endgame = self.is_endgame(board)

        # Normalise model scores to [0, 1]
        lo, hi = min(scores), max(scores)
        rng = hi - lo if hi != lo else 1.0
        adjusted = [(s - lo) / rng for s in scores]

        for i, move in enumerate(legal_moves):
            board.push(move)

            # Draw avoidance
            if board.is_repetition(2):
                adjusted[i] -= 5.0
            if board.is_stalemate():
                adjusted[i] -= 8.0
            if board.halfmove_clock >= 40:
                adjusted[i] -= 2.0

            # Check bonus
            if board.is_check():
                adjusted[i] += 0.4

            board.pop()

            # Capture bonuses (material-aware)
            if board.is_capture(move):
                if material_advantage >= 3:
                    adjusted[i] += 0.8
                elif material_advantage >= 1:
                    adjusted[i] += 0.4
                else:
                    adjusted[i] += 0.2

                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    adjusted[i] += self.PIECE_VALUES.get(captured_piece.piece_type, 0) * 0.1

            # Promotion
            if move.promotion:
                adjusted[i] += 3.0

            # Endgame: advance passed pawns
            if endgame:
                piece = board.piece_at(move.from_square)
                if piece and piece.piece_type == chess.PAWN:
                    rank = (
                        chess.square_rank(move.to_square)
                        if my_color == chess.WHITE
                        else 7 - chess.square_rank(move.to_square)
                    )
                    adjusted[i] += rank * 0.15

        return adjusted

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_move(self, fen: str) -> Optional[str]:
        """Return the best legal UCI move for the given FEN, or None."""
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        # Score moves in chunks to stay within VRAM
        all_scores: list = []
        for start in range(0, len(legal_moves), self.CHUNK_SIZE):
            chunk = legal_moves[start : start + self.CHUNK_SIZE]
            all_scores.extend(self.score_all_moves(fen, chunk))

        result = self.apply_heuristics(board, legal_moves, all_scores)

        if isinstance(result, int):
            return legal_moves[result].uci()

        return legal_moves[result.index(max(result))].uci()
