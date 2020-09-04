#  Copyright (c) 2020. Robin Thibaut, Ghent University

from experiment.base.inventory import MySetup
from os.path import join, dirname
import sys

libpath = dirname(dirname(MySetup.Directories.main_dir))

flopath = join(libpath, 'flopy')
sgpath = join(libpath, 'pysgems')

print(f'flopy located in {flopath}')
print(f'pysgems located in {sgpath}')

sys.path.append(flopath)
sys.path.append(sgpath)
