
#_________________________________PREREQUIES_____________________________________


# This is required for pymc parallel evaluation in notebooks
import multiprocessing as mp
import platform

# if platform.system() != "Windows":

#     mp.set_start_method('spawn')

    # mp.set_start_method('forkserver')
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
import pickle
import pymc as pm

# We use estivals parallel tools to run the model evaluations
from estival.utils.parallel import map_parallel

# import numpyro
# from numpyro import distributions as dist
from numpyro import infer
import arviz as az

from pathlib import Path

if __name__ == "__main__":
    if platform.system() != "Windows":
    
        mp.set_start_method('spawn')

    
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

        
    #_____________SAMPLING PROCESS________________________________________________________

    #__________________Simple run for algorithms with different draws and or tuning________
    
    D = 2 # Dimension of the parameter's space
    samplers = [infer.NUTS]*2 + [pm.DEMetropolisZ]*2 + [pm.DEMetropolis]*2 + [pm.Metropolis]*4
    Draws = [2000]*2 + [4000]*6 + [8000]*2
    Tunes = [100, 1000]*5
    chains = 2*D
    ##____Uniform Initialisation for each chain_________
    init_vals = []
    for c in range(chains):
        init_vals.append({param: np.random.uniform(0.0,1.0) for param in parameters.keys()})

    results_df = pd.DataFrame()

    for sampler, draws, tune in zip (samplers, Draws, Tunes):
        
        #calling the function multirun with only one iteration
        results = cal.multirun(sampler = sampler, 
            draws = draws,
            tune = tune,
            bcm_model = bcm_model_1,
            n_iterations = 1,
            n_jobs = 1,
            initial_params = init_vals
            )
            
        results_df = pd.concat([results_df,results])
    results_df = results_df.reset_index(drop=True)

    #Storing our results in to pickle file

    with open('./Results/Model_1/Simple_run_results.pkl', 'wb') as fp:
        pickle.dump(results_df, fp)
    #__________________________________________________________________


    #____________Multiple run for Staical Analysis___________________

    # all_results = dict()
    # sampler = infer.NUTS
    # all_results[sampler.__name__] = cal.multirun(sampler, draws = 2000,tune = 1000, bcm_model = bcm_model_1,n_iterations = 100,n_jobs = 4, initial_params = init_vals)

    # sampler = pm.DEMetropolis
    # all_results[sampler.__name__] = cal.multirun(sampler, draws = 4000,tune = 1000,bcm_model = bcm_model_1,n_iterations = 100,n_jobs = 4,initial_params = init_vals
    # )

    # sampler = pm.DEMetropolisZ
    # all_results[sampler.__name__] = cal.multirun(sampler, draws = 4000,tune = 1000,bcm_model = bcm_model_1,n_iterations = 100,n_jobs = 4,initial_params = init_vals
    # )

    # sampler = pm.Metropolis
    # all_results[sampler.__name__] = cal.multirun(sampler, draws = 8000,tune = 1000,bcm_model = bcm_model_1,n_iterations = 100,n_jobs = 4,initial_params = init_vals
    # )
    # # Storing the results in a pickle file
    # with open('./Results/Model_1/Multiple_run_results.pkl', 'wb') as fp:
    #     pickle.dump(all_results, fp)




