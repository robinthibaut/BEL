#  Copyright (c) 2021. Robin Thibaut, Ghent University
import os
from os.path import join as jp
from datetime import date
from .learning import bel_pipeline
from .hydro import forward_modelling
from loguru import logger

__version__ = '1.0.dev0'

logger.add(jp(os.getcwd(), "logs", f"{date.today()}.log"), backtrace=True, diagnose=True, enqueue=True)
logger.debug("Beginning logging session!")

__all__ = ['config', 'exceptions', 'utils', 'goggles',
           'algorithms', 'design', 'hydro', 'learning', 'processing', 'spatial']

