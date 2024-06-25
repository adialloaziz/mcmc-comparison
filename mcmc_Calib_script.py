
#_________________________________PREREQUIES_____________________________________

import logging

# Disable warning messages from pytensor.tensor.blas
logging.getLogger('pytensor.tensor.blas').setLevel(logging.ERROR)


import multiprocessing as mp # This is required for pymc parallel evaluation in (notebooks)
import platform

# if platform.system() != "Windows":

#     mp.set_start_method('spawn')

    # mp.set_start_method('forkserver')
import Calibrate as cal #Runing the calibration process and gathering results
from calibs_utilities import get_all_priors, get_targets
from models.models import model1 #All the models we design for the test

# Combining tagets and prior with our summer2 model in a BayesianCompartmentalModel (BCM)
from estival.model import BayesianCompartmentalModel

import pandas as pd
from jax import numpy as jnp
import numpy as np
import pickle
import pymc as pm
from time import time

# We use estivals parallel tools to run the model evaluations
from estival.utils.parallel import map_parallel

# import numpyro
# from numpyro import distributions as dist
from numpyro import infer

from pathlib import Path
import argparse

if __name__ == "__main__":

    #_____Handling command line arguments_____

    parser = argparse.ArgumentParser(
    prog='mcmc_calibration',
    description="""This script is designed to run full analysis for MCMC sampler comparison for 3 different applications:
                Application 1: A very simple SIR model with 2 parameters to calibrate
                Application 2: A SEIR stratified by age and infectious compartment
                Application 3: A simulation test
                Included MCMC algorithms are: NUTS (from numpyro), RWM, DEMetropolis, DEMetropolisZ (from pymc)"""
                )
    parser.add_argument(
                    "-n_jobs","--n_jobs", type=int, default = 2,
                    help="""This is the number of jobs used while doing multiple-run for the sampler in purpose of statistical analysis (default is 4)
                      The multiple-run uses a parallel loop(joblib mapping) to accelerate the process"""
                      )
    parser.add_argument(
                    "-apl","--application", type=int, choices=range(1,4), default = 1,
                    help="""Specify wich application you want to run (1,2 or 3). Default is 1 the SIR"""
                      )

    args = parser.parse_args()

    # print("Number of jobs for multirun = ", args.n_jobs)
    if platform.system() != "Windows":
    
        mp.set_start_method('forkserver')

    
    if args.application == 1 : #Runing analysis for application 1
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
        init_vals_pymc = []
        for c in range(chains):
            init_vals_pymc.append({param: np.random.uniform(0.0,1.0) for param in parameters.keys()})

        init_vals_nuts = {param: jnp.array(np.random.uniform(0.0,1.0, 4)) for param in parameters.keys()}

        results_df = pd.DataFrame()
        start = time()
        print("Simple run involved sampler with varying draws/tune...\n")
        for sampler, draws, tune in zip (samplers, Draws, Tunes):
            if sampler.__name__ == "NUTS": #Because the format for sampler in pymc and numpyro are different 
                init_vals = init_vals_nuts
            else :
                init_vals = init_vals_pymc
            #calling the function multirun with only one iteration
            results = cal.multirun(sampler = sampler, 
                draws = draws,
                tune = tune,
                bcm_model = bcm_model_1,
                n_iterations = 1,
                n_jobs = 1, #args.n_jobs,
                initial_params = init_vals
                )
                
            results_df = pd.concat([results_df,results])
        results_df = results_df.reset_index(drop=True)
        end = time()

        print("Simple run walltime:\n", end - start)

        #Storing our results in to pickle file

        with open('./Results/Model_1/Simple_run_results.pkl', 'wb') as fp:
            pickle.dump(results_df, fp)
        #__________________________________________________________________


        #____________Multiple run for Staical Analysis___________________
        print("Multiple runs...")
        n_jobs = args.n_jobs
        all_results = dict()

        sampler = infer.NUTS
        start = time()
        all_results[sampler.__name__] = cal.multirun(sampler, draws = 2000,tune = 1000, bcm_model = bcm_model_1,n_iterations = 100,n_jobs = n_jobs  , initial_params = init_vals_nuts)
        end = time()
        print("NUTS walltime: ", end - start)


        sampler = pm.DEMetropolis
        start = time()
        all_results[sampler.__name__] = cal.multirun(sampler, draws = 4000,tune = 1000,bcm_model = bcm_model_1,n_iterations = 100, n_jobs= n_jobs,initial_params = init_vals
        )
        end = time()
        print("DEMetropolis walltime:\n", end - start)

        sampler = pm.DEMetropolisZ
        start = time()
        all_results[sampler.__name__] = cal.multirun(sampler, draws = 4000,tune = 1000,bcm_model = bcm_model_1,n_iterations = 100,n_jobs = n_jobs,initial_params = init_vals
        )
        end = time()
        print("DEMetropolisZ walltime:\n", end - start)

        sampler = pm.Metropolis
        start = time()
        all_results[sampler.__name__] = cal.multirun(sampler, draws = 8000,tune = 1000,bcm_model = bcm_model_1,n_iterations = 100,n_jobs= n_jobs,initial_params = init_vals
        )
        end = time()
        print("Metropolis walltime:\n", end - start)

        # Storing the results in a pickle file
        with open('./Results/Model_1/Multiple_run_results.pkl', 'wb') as fp:
            pickle.dump(all_results, fp)




