#  Copyright (c) 2020. Robin Thibaut, Ghent University
from os.path import join as jp
import joblib

from experiment.base.inventory import MySetup

md = MySetup.Directories()

root = '6a4d614c838442629d7a826cc1f498a8'
sources = '123456'
sdir = jp(md.forecasts_dir, root, sources)
post_obj = joblib.load(jp(sdir, 'obj', 'post.pkl'))
h_samples = post_obj.random_sample()
