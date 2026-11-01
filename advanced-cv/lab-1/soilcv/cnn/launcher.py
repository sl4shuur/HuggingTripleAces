"""Run independent CNN folds with the shared process queue."""

from soilcv.jobs import install_signal_handler, run_jobs

from .experiments import RECIPES, ROOT


def launch():
    jobs = [
        {"experiment": recipe["id"], "fold": fold}
        for recipe in RECIPES
        for fold in range(4)
    ]
    return run_jobs(jobs, ROOT, "soilcv.cnn.experiments")


if __name__ == "__main__":
    install_signal_handler()
    launch()
