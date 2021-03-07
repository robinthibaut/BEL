#  Copyright (c) 2021. Robin Thibaut, Ghent University
import os
from datetime import date
from os.path import join as jp

from loguru import logger

from .hydro import forward_modelling
from .learning import bel_pipeline

__version__ = '1.0.dev0'

source = __name__.split('.')[-1]
# Set up logger
logger.add(jp(os.getcwd(), "logs", f"{source}_{date.today()}.log"), backtrace=True, diagnose=True, enqueue=True)
logger.debug(f"Beginning logging session for {source}!")

__all__ = ['config', 'exceptions', 'utils', 'goggles',
           'algorithms', 'design', 'hydro', 'learning', 'processing', 'spatial']
