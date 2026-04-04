"""
CLI entry point for the CPHOS AI Customer Service system.

Usage:
    # Interactive session with no knowledge base
    python main.py

    # Load one or more knowledge-base documents before starting
    python main.py --docs path/to/guide.txt path/to/faq.pdf

    # Load all YAML files from a references directory (with section tagging)
    python main.py --refs-dir references/

    # Start with pre-built index (skips re-embedding)
    python main.py --load-index index.npz

    # Index documents and save the index for later reuse
    python main.py --docs guide.txt --save-index index.npz
    python main.py --refs-dir references/ --save-index index.npz
"""
from __future__ import annotations

import argparse
import sys
import readline  # noqa: F401  — enables arrow-key history in input()
from pathlib import Path

import config
from agents.classifier import ClassifierAgent
from agents.critic     import CriticAgent
from agents.executor   import ExecutorAgent
from agents.verifier   import VerifierAgent
from rag.document      import load_documents
from rag.retriever     import Retriever
from pipeline          import Pipeline
from utils.logger      import ConversationLogger, get_logger

logger = get_logger(__name__)


def build_pipeline(
    doc_paths:        list[str] | None = None,
    refs_dir:         str | None = None,
    load_index:       str | None = None,
    save_index:       str | None = None,
    conv_log_path:    str = "logs",
    verbose:          bool = False,
    check_refs:       bool = False,
    top_k:            int | None = None,
    max_retries:      int | None = None,
    enable_dual_path: bool | None = None,
) -> Pipeline:
    """Instantiate all components and return a ready Pipeline."""
    if not config.OPENAI_API_KEY:
        print(
            "[ERROR] OPENAI_API_KEY is not set.\n"
            "        Create a .env file (see .env.example) or export the variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    retriever = Retriever(
        api_key        = config.OPENAI_API_KEY,
        embedding_model= config.EMBEDDING_MODEL,
        base_url       = config.LLM_BASE_URL,
    )

    _index_path = Path(load_index) if load_index else None
    _ref_files: list[Path] = []
    if refs_dir:
        _ref_files = sorted(Path(refs_dir).glob("*.yml")) + sorted(Path(refs_dir).glob("*.yaml"))
    if doc_paths:
        _ref_files += [Path(p) for p in doc_paths]

    def _index_stale() -> bool:
        """True when any source file is newer than the saved index."""
        if not _index_path or not _index_path.exists():
            return False
        index_mtime = _index_path.stat().st_mtime
        return any(f.stat().st_mtime > index_mtime for f in _ref_files if f.exists())

    if _index_path and _index_path.exists() and not (check_refs and _index_stale()):
        logger.info("Loading index from %r…", load_index)
        retriever.load(load_index)
        logger.info("Loaded %d chunks from index.", len(retriever._chunks))

    elif refs_dir or doc_paths:
        if _index_path:
            if not _index_path.exists():
                print(
                    f"[INFO] Index file {load_index!r} not found — building from {refs_dir!r}…",
                    file=sys.stderr,
                )
            else:
                stale = [f.name for f in _ref_files if f.exists() and f.stat().st_mtime > _index_path.stat().st_mtime]
                print(
                    f"[INFO] References changed ({', '.join(stale)}) — rebuilding index…",
                    file=sys.stderr,
                )
            # auto-save back to the same path so next run loads fast
            save_index = save_index or load_index

        # ── Load YAML reference files (with per-file section tags) ────────────
        if refs_dir:
            ref_path = Path(refs_dir)
            yaml_files = sorted(ref_path.glob("*.yml")) + sorted(ref_path.glob("*.yaml"))
            if not yaml_files:
                logger.warning("No YAML files found in %r.", refs_dir)
            else:
                logger.info("Loading %d YAML file(s) from %r…", len(yaml_files), refs_dir)
                for yml in yaml_files:
                    section = yml.stem  # filename without extension → section name
                    chunks  = load_documents([str(yml)])
                    logger.info(
                        "  %s → %d chunk(s) (section=%r)", yml.name, len(chunks), section
                    )
                    retriever.add_documents(chunks, section=section)

        # ── Load arbitrary document files ─────────────────────────────────────
        if doc_paths:
            logger.info("Loading %d additional document(s)…", len(doc_paths))
            for path in doc_paths:
                section = Path(path).stem
                chunks  = load_documents([path], word_length=config.CHUNK_WORD_LENGTH)
                logger.info(
                    "  %s → %d chunk(s) (section=%r)", Path(path).name, len(chunks), section
                )
                retriever.add_documents(chunks, section=section)

        logger.info(
            "Indexing complete. Total chunks: %d. "
            "(This called the Embeddings API.)",
            len(retriever._chunks),
        )
        if save_index:
            retriever.save(save_index)
            logger.info("Index saved to %r.", save_index)

    else:
        logger.warning(
            "No documents loaded. The assistant will have no knowledge base.\n"
            "          Use --docs or --refs-dir to load knowledge-base files."
        )

    return Pipeline(
        classifier   = ClassifierAgent(config.CLASSIFIER_MODEL, config.OPENAI_API_KEY, config.LLM_BASE_URL),
        executor     = ExecutorAgent  (config.EXECUTOR_MODEL,   config.OPENAI_API_KEY, config.LLM_BASE_URL),
        verifier     = VerifierAgent  (config.VERIFIER_MODEL,   config.OPENAI_API_KEY, config.LLM_BASE_URL),
        critic       = CriticAgent    (config.CRITIC_MODEL,     config.OPENAI_API_KEY, config.LLM_BASE_URL),
        retriever    = retriever,
        top_k        = top_k        if top_k        is not None else config.TOP_K_CHUNKS,
        max_retries  = max_retries  if max_retries  is not None else config.MAX_RETRIES,
        enable_dual_path = enable_dual_path if enable_dual_path is not None else config.ENABLE_DUAL_PATH,
        conv_logger  = ConversationLogger(conv_log_path, verbose=verbose),
    )


def run_interactive(
    pipeline: Pipeline,
    user_id:  str  = "anonymous",
    source:   str  = "cli",
) -> None:
    """Simple read-eval-print loop."""
    mode_hint = "  \033[33m[debug mode ON]\033[0m" if config.DEBUG_MODE else ""
    user_hint = f"  \033[36m[user: {user_id}]\033[0m" if user_id != "anonymous" else ""
    print(f"\nCPHOS AI Customer Service  (type 'exit' or Ctrl-C to quit){mode_hint}{user_hint}\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        reply = pipeline.answer(question, user_id=user_id, source=source)
        print(f"\nAI: {reply}\n")


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="CPHOS AI Customer Service — interactive Q&A",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", "--docs",
        nargs="*",
        metavar="FILE",
        default=None,
        help="Path(s) to .txt / .md / .pdf / .yml knowledge-base files to index.",
    )
    parser.add_argument(
        "-r", "--refs-dir",
        type=str,
        metavar="DIR",
        default="references/",
        help=(
            "Directory containing YAML knowledge-base files. "
            "Each file is loaded with its stem as the section tag, enabling "
            "topic-aware retrieval boosting."
        ),
    )
    parser.add_argument(
        "-l", "--load-index",
        type=str,
        metavar="FILE",
        default="cphos.npz",
        help=(
            "Load a pre-built embedding index from a .npz file. "
            "Used when the file exists; otherwise the index is built from --refs-dir / --docs."
        ),
    )
    parser.add_argument(
        "-s", "--save-index",
        type=str,
        metavar="FILE",
        default=None,
        help="After indexing --docs / --refs-dir, save the resulting index to this .npz file.",
    )
    parser.add_argument(
        "-u", "--user",
        type=str,
        metavar="ID",
        default="anonymous",
        help=(
            "User identifier recorded in the conversation log. "
            "For WeChat / WeCom integrations pass the nickname or WeCom userid."
        ),
    )
    parser.add_argument(
        "-c", "--conv-log",
        type=str,
        metavar="DIR",
        default="logs",
        help="Directory where per-session .jsonl and .log files are written.",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=config.TOP_K_CHUNKS,
        help="Number of knowledge-base chunks to retrieve per query.",
    )
    parser.add_argument(
        "-n", "--max-retries",
        type=int,
        default=config.MAX_RETRIES,
        help="Maximum Executor → Verifier retry cycles before escalating to human support.",
    )
    parser.add_argument(
        "--no-dual-path",
        action="store_true",
        default=False,
        help="Disable dual-path parallel execution and Critic selection; use single-path mode.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print each agent's full input / output to stderr for every turn.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Print INFO-level pipeline logs to stderr (classified category, latency, etc.).",
    )
    parser.add_argument(
        "--check-refs",
        action="store_true",
        default=False,
        help="Check if any reference file is newer than the index and rebuild automatically if so.",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()

    if args.debug:
        config.DEBUG_MODE = True

    pipeline = build_pipeline(
        doc_paths     = args.docs,
        refs_dir      = args.refs_dir,
        load_index    = args.load_index,
        save_index    = args.save_index,
        conv_log_path = args.conv_log,
        verbose       = args.verbose,
        check_refs    = args.check_refs,
        top_k         = args.top_k,
        max_retries   = args.max_retries,
        enable_dual_path = not args.no_dual_path,
    )
    run_interactive(pipeline, user_id=args.user, source="cli")


if __name__ == "__main__":
    main()
