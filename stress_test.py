"""
多进程压力测试：模拟真实上线后多个用户同时使用的场景。

每个"用户"是一个独立的 main.py 子进程，通过 stdin 喂入问题，
读取 stdout 的回复，测量端到端延迟。

用法：
    python stress_test.py                      # 默认 5 个并发用户
    python stress_test.py --workers 10         # 10 个并发用户
    python stress_test.py --load-index my.npz  # 指定索引文件
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

QUESTIONS = [
    "我的成绩什么时候出来？",
    "如何修改注册信息？",
    "比赛时间是什么时候？",
    "怎么阅卷，给考生评分？",
    "小程序在哪里下载？",
    "时间节点过了还能上传试卷吗？",
    "账号审核需要多久？",
    "往年题目在哪里？",
    "排名如何查询？",
    "如何参加线下赛季？",
]


def ask_one(question: str, user_id: str, load_index: str) -> dict:
    """Spawn a main.py process, send one question, capture the reply."""
    cmd = [
        sys.executable, "main.py",
        "--load-index", load_index,
        "--user", user_id,
        "--no-dual-path",    # halves API calls per request
        "--max-retries", "1",  # fail fast under load; don't queue retries
    ]
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # suppress pipeline logs from each process
            text=True,
        )
        # Send the question then "exit" to cleanly terminate the REPL
        stdout, _ = proc.communicate(input=f"{question}\nexit\n", timeout=300)
        latency = time.monotonic() - t0

        # Extract the "AI: ..." line from the output
        reply = ""
        for line in stdout.splitlines():
            if line.startswith("AI:"):
                reply = line[len("AI:"):].strip()
                break

        return {"ok": True, "user": user_id, "question": question,
                "reply": reply, "latency": latency}

    except subprocess.TimeoutExpired:
        proc.kill()
        return {"ok": False, "user": user_id, "question": question,
                "error": "TIMEOUT", "latency": time.monotonic() - t0}
    except Exception as exc:
        return {"ok": False, "user": user_id, "question": question,
                "error": str(exc), "latency": time.monotonic() - t0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-process stress test for main.py")
    parser.add_argument("--workers",    type=int, default=5,           help="Number of concurrent user processes")
    parser.add_argument("--load-index", type=str, default="cphos.npz", help="Path to .npz index file")
    args = parser.parse_args()

    questions = (QUESTIONS * ((args.workers // len(QUESTIONS)) + 1))[:args.workers]
    users     = [f"user_{i:02d}" for i in range(args.workers)]

    print(f"Spawning {args.workers} concurrent main.py processes…\n")
    wall_t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(ask_one, q, u, args.load_index)
            for q, u in zip(questions, users)
        ]
        results = [f.result() for f in as_completed(futures)]

    wall_elapsed = time.monotonic() - wall_t0

    # ── Print results ──────────────────────────────────────────────────────────
    ok      = [r for r in results if r["ok"]]
    failed  = [r for r in results if not r["ok"]]
    latencies = [r["latency"] for r in ok]

    print("─" * 72)
    for r in sorted(results, key=lambda x: x["latency"]):
        status = "✓" if r["ok"] else "✗"
        label  = f"{r['latency']:.2f}s"
        detail = (r["reply"][:60].replace("\n", " ") + "…") if r["ok"] else r["error"]
        print(f"  {status} [{label}]  [{r['user']}]  Q: {r['question']}")
        print(f"             A: {detail}")
    print("─" * 72)

    if latencies:
        print(f"\n  Processes     : {args.workers}")
        print(f"  Success / Fail: {len(ok)} / {len(failed)}")
        print(f"  Wall-clock    : {wall_elapsed:.2f}s")
        print(f"  Latency  avg  : {sum(latencies)/len(latencies):.2f}s")
        print(f"  Latency  min  : {min(latencies):.2f}s")
        print(f"  Latency  max  : {max(latencies):.2f}s")
        print(f"  Throughput    : {len(ok)/wall_elapsed:.2f} req/s\n")

    if failed:
        print("Failed:")
        for r in failed:
            print(f"  [{r['user']}] Q: {r['question']}  →  {r['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()

