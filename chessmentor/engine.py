from contextlib import contextmanager
from pathlib import Path

import chess
import chess.engine


@contextmanager
def open_engine(config):
    engine_path = config["stockfish_path"]
    thread_count = config.get("thread_count", 1)
    hash_size = config.get("hash_size", 128)
    depth = config.get("depth", 15)
    movetime = config.get("movetime", 1000)

    if not Path(engine_path).exists():
        raise FileNotFoundError(
            f"Stockfish nicht gefunden: {Path(engine_path).resolve()}"
        )
    else:
        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
            engine.configure({"Threads": thread_count, "Hash": hash_size})
            yield engine, depth, movetime


def get_best_move(board, config):
    if isinstance(board, str):
        board = chess.Board(board)

    with open_engine(config) as (engine, depth, movetime):
        result = engine.play(
            board,
            chess.engine.Limit(depth=depth, time=movetime / 1000),
            info=chess.engine.INFO_ALL,
        )

    return result.move, result.info.get("score").relative.score(mate_score=100000)
