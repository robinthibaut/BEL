#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os

from experiment.design.forecast_error import by_mode
from ..config import Setup

if __name__ == "__main__":
    # Amount of information
    forecast_dir = Setup.Directories.forecasts_dir
    listit = os.listdir(forecast_dir)
    listit.remove("base")
    duq = list(
        filter(lambda f: os.path.isdir(os.path.join(forecast_dir, f)),
               listit))  # Folders of combinations

    by_mode(duq)
    by_mode(["818bf1676c424f76b83bd777ae588a1d"])
