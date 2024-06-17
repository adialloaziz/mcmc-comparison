
#_________________________________PREREQUIES_____________________________________


# This is required for pymc parallel evaluation in notebooks
import multiprocessing as mp
import platform

if platform.system() != "Windows":
    
    mp.set_start_method('forkserver')

    

import Calibrate as cal #Runing the calibration process and gathering results
from calibs_utilities import get_all_priors, get_targets, load_data
from models.models import model1 #All the models we design for the test
from Calibrate import plot_comparison_Bars

# Combining tagets and prior with our summer2 model in a BayesianCompartmentalModel (BCM)
from estival.model import BayesianCompartmentalModel


import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from typing import List

import pymc as pm

# We use estivals parallel tools to run the model evaluations
from estival.utils.parallel import map_parallel

# import numpyro
# from numpyro import distributions as dist
from numpyro import infer
import arviz as az

from pathlib import Path


BASE_PATH = Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH / "data"

targets_yml = Path(DATA_PATH/"target_yml.yml")


#Names of parameters and their ranges
params = {
    "contact_rate": (0.0,1.0),
    "recovery_rate": (0.0,1.0)

}
targets = get_targets(targets_yml) 
priors = get_all_priors(params)

#_______Model definition : A bayesian compartmental model with estival/SUMMER_____
model_1 = model1()


#________________Default parameters____________
parameters = {
    "contact_rate": 0.2,
    "recovery_rate": 0.1,
    #"active_cases_dispersion": 0.5,
}
bcm_model_1 = BayesianCompartmentalModel(model_1, parameters, priors, targets)
# print(bcm_model_1)

#_____________SAMPLING PROCESS________________________________________________________

D = 2 # Dimension of the parameter's space
samplers = [pm.Metropolis] #[infer.NUTS] # [pm.DEMetropolisZ]*2 + [pm.DEMetropolis]*2 + [pm.Metropolis]*4
Draws = [2000] #[4000]*6+ [8000]*2
Tunes = [1000] #[100] + [1000] + [100] + [1000] + [100] + [1000] + [100] + [1000]
chains = 2*D
results = []



#     #print(sampler)
idata, Time = cal.Sampling_calib(
    bcm_model = bcm_model_1,
    mcmc_algo = sampler,
    initial_params = parameters,
    draws = draws,
    tune = tune,
    cores = 4,
    chains = chains,
        )

results.append(cal.Compute_metrics(
        mcmc_algo = sampler,
        idata = idata,
        Time = Time,
        draws = draws, 
        chains = chains,
        tune = tune,
            )
        )


results_df = pd.concat(results)
results_df["Run"] = results_df.Sampler + "\nDraws=" + results_df.Chains.astype(str) + "\nTune=" + results_df.Tune.astype(str)

results_df = results_df.reset_index(drop=True)
# results_df.style.set_caption("MCMC COMPARISON")
