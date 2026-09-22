"""Opt-in purged CV controls shared by classifier and objective comparisons."""


def add_cv_arguments(parser):
    parser.add_argument("--cv-folds", type=int, default=None, help="Enable nested purged expanding CV with at least two outer folds.")
    parser.add_argument("--cv-initial-train-fraction", type=float, default=.5)
    parser.add_argument("--cv-inner-val-fraction", type=float, default=.2)
    parser.add_argument("--cv-gap-bars", type=int, default=0, help="Exclude these dates before inner validation and outer evaluation from fitting targets.")
    parser.add_argument("--cv-embargo-bars", type=int, default=0, help="Post-validation embargo; no additional exclusions for past-only expanding training.")
    parser.add_argument("--cv-score", default=None, help="Outer-fold ranking metric (classifier default: macro_f1).")
    parser.add_argument("--cv-final-test", action="store_true", help="Evaluate only the CV-selected configuration on the reserved final holdout.")


def validate_cv_arguments(args):
    if args.cv_folds is None and (args.cv_final_test or args.cv_gap_bars or args.cv_embargo_bars
                                 or args.cv_score is not None or args.cv_initial_train_fraction != .5
                                 or args.cv_inner_val_fraction != .2):
        raise ValueError("CV parameters require --cv-folds.")


def execute_cv(frame, config, parameter_sets, args, output_dir, *, loss_configs=None,
               gate_search=None):
    from trading_system.experiments.purged_search import run_purged_cv
    metric = args.cv_score or (args.selection_metric if loss_configs is not None else "macro_f1")
    return run_purged_cv(
        frame, config, parameter_sets, args.seeds, output_dir,
        n_splits=args.cv_folds, initial_train_fraction=args.cv_initial_train_fraction,
        inner_val_fraction=args.cv_inner_val_fraction, gap_bars=args.cv_gap_bars,
        embargo_bars=args.cv_embargo_bars, loss_configs=loss_configs,
        selection_metric=metric,
        final_test=args.cv_final_test or getattr(args, "final_test", False),
        save_artifacts=not args.no_run_artifacts, fail_fast=args.fail_fast,
        dataset_path=args.data, gate_search=gate_search,
    )
