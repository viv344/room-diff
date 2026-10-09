"""python -m roomdiff BEFORE_VIDEO AFTER_VIDEO [-o OUT]"""
import argparse
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def main():
    ap = argparse.ArgumentParser(prog="roomdiff", description="What changed in the room? 3D change detection from two phone videos.")
    ap.add_argument("before", help="video of the room before the changes")
    ap.add_argument("after", help="video of the room after the changes")
    ap.add_argument("-o", "--out", default="out", help="output directory (default: out)")
    ap.add_argument("--frames", type=int, default=40, help="keyframes per video (default 40)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--claude", dest="claude", action="store_true", default=None,
                   help="use Claude for labels/descriptions (default: automatic if ANTHROPIC_API_KEY is set)")
    g.add_argument("--no-claude", dest="claude", action="store_false", help="CLIP labels only, fully offline")
    ap.add_argument("--fresh", action="store_true", help="ignore cached reconstruction in OUT/work")
    a = ap.parse_args()
    cache = os.path.join(a.out, "work", "scene.pkl")
    if a.fresh and os.path.exists(cache):
        os.remove(cache)
    from .pipeline import run
    run(a.before, a.after, a.out, n_frames=a.frames, use_claude=a.claude)


if __name__ == "__main__":
    main()
