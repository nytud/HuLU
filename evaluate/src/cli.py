import os
import logging
import argparse

import eval as evaluate


log_level = int(os.getenv('LOG_LEVEL', logging.INFO))
logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def cli() -> None:

    parser = argparse.ArgumentParser(description='hugme cli tool')

    parser.add_argument('--model-name', type=str, metavar='S', required=True, help='model name or path')
    parser.add_argument('--tokenizer-name', type=str, metavar='S', help='tokenizer name or path')
    parser.add_argument(
        '--tasks', type=str, nargs="+",
        choices=["hucola", "hurte", "huwnli", "hucommitmentbank", "husst", "hucopa"],
        help='task name(s)'
    )
    parser.add_argument("--parameters-path", type=str, default=None, required=True, help="JSON config path for model params")
    parser.add_argument("--eval-test", type=lambda x: x.lower()=='true', default=False, help='evaluate on test set')

    parser.add_argument("--report-to", type=str, default=None, help='reporting dest e.g. wandb, mlflow')
    parser.add_argument("--report-uri", type=str, default=None, help='reporting URI for wandb, mlflow')
    parser.add_argument("--experiment-name", type=str, default="hulu-finetune", help='experiment/project name for reporting')
    parser.add_argument("--run-name", type=str, default=None, help='run name for reporting')

    parser.add_argument("--save-results-path", type=str, default="./results/", help='save results')

    args = parser.parse_args()

    evaluate.evaluate(args)


if __name__ == '__main__':
    cli()