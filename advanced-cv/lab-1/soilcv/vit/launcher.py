"""Run the selected Transformer study on two GPUs."""

from soilcv.jobs import install_signal_handler, run_jobs

from .experiments import ROOT, jobs


def launch():
    queue = [
        {"experiment": recipe, "fold": fold, "train_camera": camera}
        for recipe, fold, camera in jobs()
    ]
    return run_jobs(queue, ROOT, "soilcv.vit.experiments")


if __name__ == "__main__":
    install_signal_handler()
    launch()
