import os
from os.path import join as jp
import numpy as np
from pysgems.algo.sgalgo import XML
from pysgems.dis.sgdis import Discretize
from pysgems.io.sgio import PointSet
from pysgems.sgems import sg

algo_dir = "C:/Users/Robin/PycharmProjects/BEL4ED/bel4ed/algorithms"
al = XML(algo_dir=algo_dir)
al.xml_reader("bel_sgsim")

# Modify xml below:
al.xml_update("Seed", "value", str(np.random.randint(1e9)), show=1)

al.xml_update("Seed", "value", str(np.random.randint(1e9)), show=1)

al.xml_update("Seed", "value", str(np.random.randint(1e9)), show=1)
