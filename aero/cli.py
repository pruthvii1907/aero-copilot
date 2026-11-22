"""Interactive CLI:  python -m aero.cli   |   python -m aero.cli --once "question" -v"""
from __future__ import annotations

import argparse
import asyncio

from .agents_setup import Answer, copilot


def show(ans: Answer) -> None:
    if ans.blocked:
        print(f"\n[blocked by {ans.block_layer} guardrail] {ans.block_reason}\n")
        return
    print(f"\n[{ans.agent}]\n{ans.text}\n")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", help="ask one question and exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="print agent steps to stderr")
    args = ap.parse_args()

    async with copilot() as cp:
        if args.once:
            show(await cp.ask(args.once, verbose=args.verbose))
            return
        print("Aero Copilot. Ask an engineering question ('exit' to quit).")
        history: list = []
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"exit", "quit"}:
                break
            if not q:
                continue
            ans = await cp.ask(q, history, verbose=args.verbose)
            show(ans)
            if not ans.blocked:
                history = ans.history


if __name__ == "__main__":
    asyncio.run(main())
