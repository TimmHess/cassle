from argparse import ArgumentParser


def continual_args(parser: ArgumentParser):
    """Adds continual learning arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add dataset args to.
    """
    # base continual learning args
    parser.add_argument("--num_tasks", type=int, default=2)
    parser.add_argument("--task_idx", type=int, required=True)
    parser.add_argument("--iters_per_task", type=int, default=None)

    SPLIT_STRATEGIES = ["class", "data", "domain", "joint", "incremental_joint"]
    parser.add_argument("--split_strategy", choices=SPLIT_STRATEGIES, type=str, required=True)
    parser.add_argument("--use_max_num_workers", action="store_true")

    # distillation args
    parser.add_argument("--distiller", type=str, default=None)
