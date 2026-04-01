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

    SPLIT_STRATEGIES = ["class", "data", "domain", "joint", "incremental_joint", "incremental_joint_class"]
    parser.add_argument("--split_strategy", choices=SPLIT_STRATEGIES, type=str, required=True)
    parser.add_argument("--use_max_num_workers", action="store_true")

    # distillation args
    parser.add_argument("--distiller", type=str, default=None)

    # replay / rehearsal args
    parser.add_argument(
        "--replay",
        action="store_true",
        default=False,
        help="Enable experience replay: store samples from each task and "
             "interleave them when training subsequent tasks.",
    )
    parser.add_argument(
        "--replay_samples_per_task",
        type=int,
        default=200,
        help="Number of samples to store per task in the replay memory buffer. "
             "Total memory grows as replay_samples_per_task * num_tasks_seen. "
             "Ignored when --replay_memory_budget is set.",
    )
    parser.add_argument(
        "--replay_memory_budget",
        type=int,
        default=None,
        help="Fixed total number of samples across all tasks in the replay buffer. "
             "Slots are distributed evenly (budget // num_tasks_seen per task) and "
             "existing task files are trimmed when a new task is added. "
             "Mutually exclusive with --replay_samples_per_task.",
    )
    parser.add_argument(
        "--replay_ratio",
        type=float,
        default=None,
        help="Fraction of each batch drawn from the replay memory. "
             "If None (default), memory is merged with the task dataset and "
             "sampled proportionally (natural ratio = n_mem / (n_task + n_mem)). "
             "If set to a float in (0, 1), WeightedRandomSampler ensures each "
             "batch contains approximately replay_ratio * batch_size memory samples "
             "while keeping the total batch size unchanged.",
    )
    parser.add_argument(
        "--replay_double_batch",
        action="store_true",
        default=False,
        help="When enabled, each training step uses a batch of 2 * batch_size: "
             "exactly batch_size samples drawn exclusively from the current task "
             "and exactly batch_size samples drawn exclusively from the memory "
             "buffer.  Takes priority over --replay_ratio.",
    )
    parser.add_argument(
        "--replay_dir",
        type=str,
        default=None,
        help="Directory where replay memory buffer files are stored. "
             "Defaults to {checkpoint_dir}/replay/ if not specified.",
    )
