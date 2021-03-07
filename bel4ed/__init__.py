#  Copyright (c) 2021. Robin Thibaut, Ghent University
import os
from datetime import date
from os.path import join as jp

from loguru import logger

from .hydro import forward_modelling
from .learning import bel_pipeline

__version__ = '1.0.dev0'

# Set up logger
logger.add(jp(os.getcwd(), "logs", f"{__name__}_{date.today()}.log"), backtrace=True, diagnose=True, enqueue=True)
logger.debug(f"Beginning logging session for {__name__}!")

__all__ = ['config', 'exceptions', 'utils', 'goggles',
           'algorithms', 'design', 'hydro', 'learning', 'processing', 'spatial']
