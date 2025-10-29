import argparse

from train.arguments import Arguments
from train.main import benchmark

__doc__ = """
Starting point for fine-tuning language models on HuLU benchmarks.
"""


def cli() -> None:
    parser = argparse.ArgumentParser(description="HuLU evaluate CLI tool")

    parser.add_argument("--config-file", type=str, help="Path to config JSON file")
    parser.add_argument(
        "--output-dir", type=str, default="finetune_results", help="Output directory"
    )
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--tokenizer-name", type=str, help="Tokenizer name (defaults to model_name)"
    )
    parser.add_argument(
        "--train-epochs", type=int, default=6, help="Number of training epochs"
    )
    parser.add_argument("--train-batch", type=int, default=8, help="Batch size")
    parser.add_argument("--train-lr", type=float, default=2e-05, help="Learning rate")
    parser.add_argument("--train-warmup", type=int, default=0, help="Warmup steps")
    parser.add_argument(
        "--train-maxlen", type=int, default=256, help="Max sequence length"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--precision", type=str, default="fp32", help="Precision (e.g., fp16 or fp32)"
    )
    parser.add_argument(
        "--use-lora",
        type=lambda x: x.lower() == "true",
        help="Use LoRA for training",
        default=False,
    )
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA r parameter")
    parser.add_argument(
        "--lora-alpha", type=int, default=16, help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.1, help="LoRA dropout rate"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["cola", "rte", "wnli", "cb", "sst", "copa"],
        help="List of tasks to train on",
    )
    parser.add_argument(
        "--use-fsdp",
        action="store_true",
        help="Use FSDP for training",
        default=False,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )

    args = parser.parse_args()

    if args.config_file:
        arguments = Arguments.from_json(args.config_file)
    else:
        arguments = Arguments(
            output_dir=args.output_dir,
            model_name=args.model_name,
            tokenizer_name=args.tokenizer_name,
            train_epochs=args.train_epochs,
            train_batch=args.train_batch,
            train_lr=args.train_lr,
            train_warmup=args.train_warmup,
            train_maxlen=args.train_maxlen,
            train_seed=args.seed,
            precision=args.precision,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_fsdp=args.use_fsdp,
            tasks=args.tasks,
        )

    benchmark(arguments)


if __name__ == "__main__":
    cli()
