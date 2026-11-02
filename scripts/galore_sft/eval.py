"""Independent full-state reload and token-weighted validation/test scoring."""

from __future__ import annotations

from common import encode, evaluate, metadata, parser, setup, trainer, write


def main():
    p = parser(__doc__)
    p.add_argument("--test", action="store_true")
    args = p.parse_args()
    log = setup(args, "eval")
    write(args.output_dir / "metadata.json", metadata(args))
    engine = trainer(args, resume=True)
    try:
        engine.init()
        result = {"status": "complete", "restored_sdk_step": engine._ctx.global_step}
        for split in ["validation", "test"] if args.test else ["validation"]:
            examples, stats = encode(engine.get_tokenizer(), args, split)
            result[split] = evaluate(engine, examples, args.micro_batch)
            result[split]["encoding"] = stats
        write(args.output_dir / "summary.json", result)
        log.info("%s", result)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
