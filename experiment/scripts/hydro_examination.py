#  Copyright (c) 2020. Robin Thibaut, Ghent University
import experiment.toolbox.filesio as fops

model_nam = '/Users/robin/PycharmProjects/BEL/experiment/storage/forwards/6a4d614c838442629d7a826cc1f498a8/whpa.nam'

flow = fops.load_flow_model(model_nam)

flow.d