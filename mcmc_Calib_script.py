
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
from models.models import model1, model2, bcm_seir_age_strat, bcm_sir #All the models we design for the test

# Combining tagets and prior with our summer2 model in a BayesianCompartmentalModel (BCM)
from estival.model import BayesianCompartmentalModel
import estival.targets as est
import estival.priors as esp

import pandas as pd
from jax import numpy as jnp
import numpy as np
import pickle
import pymc as pm
from time import time
from datetime import datetime

# We use estivals parallel tools to run the model evaluations
from estival.utils.parallel import map_parallel

# import numpyro
# from numpyro import distributions as dist
from numpyro import infer
import numpyro
from numpyro import distributions as dist

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
        #Defining  a Bayesian Compartmental Model
        #Targets and priors are already defined
        #See models.py for the costumization
        bcm_model_1 = bcm_sir()

        #We need to define a numpyro compatible model.
        def nmodel1():
            sampled = {k:numpyro.sample(k, dist.Uniform(0.0,1.0)) for k in bcm_model_1.parameters}
            ll = numpyro.factor("ll", bcm_model_1.loglikelihood(**sampled))
                
        #_____________SAMPLING PROCESS________________________________________________________
        def init_uniform(num_chains, parameters):
            init_vals = []
            for c in range(num_chains):
                init_vals.append({param: np.random.uniform(0.0,1.0) for param in parameters.keys()})
    
            return init_vals
        
        
        ##____Uniform Initialisation for each chain_________
        init_vals_4 = init_uniform(4,bcm_model_1.parameters)
        init_vals_6 = init_uniform(6,bcm_model_1.parameters)
        init_vals_nuts = {param: jnp.array(np.random.uniform(0.0,1.0, 4)) for param in bcm_model_1.parameters.keys()}
        #__________________Simple run for algorithms with different draws and or tuning________

        samplers =  [infer.NUTS,pm.sample_smc, pm.Metropolis,pm.DEMetropolisZ]+ [pm.DEMetropolis]*2
        Draws = [2000]*2 + [10000] + [8000]*3
        # Tunes = [0] + [100, 1000]*5
        Init = [init_vals_nuts] + [init_vals_4]*4 + [init_vals_6]
        Chains = [4]*5 + [6]
        results_df = pd.DataFrame()
        start = time()
        print("Single analyse runing...\n")
        for sampler, draws, chains, init in zip (samplers, Draws, Chains, Init):
            results = cal.multirun(sampler = sampler, 
            draws = draws,
            chains=chains,
            cores = chains,
            tune = 1000,
            bcm_model = bcm_model_1,
            nmodel= nmodel1,
            n_iterations = 1,
            n_jobs = 1,
            initial_params = init)
            
            results_df = pd.concat([results_df,results])

        results_df = results_df.reset_index(drop=True)
        end = time()

        print("Simple run walltime:\n", end - start)

        #Storing our results in to pickle file

        with open('./Results/Model_1/Simple_run_results_3_script.pkl', 'wb') as fp:
            pickle.dump(results_df, fp)
        #__________________________________________________________________


        # #____________Multiple run for Staical Analysis___________________
        # print("Multiple runs...")
        # n_jobs = args.n_jobs
        # all_results = dict()

        # sampler = infer.NUTS
        # start = time()
        # all_results[sampler.__name__] = cal.multirun(sampler, draws = 2000,tune = 1000, bcm_model = bcm_model_1,n_iterations = 100,n_jobs = n_jobs  , initial_params = init_vals_nuts)
        # end = time()
        # print("NUTS walltime: ", end - start)


        # sampler = pm.DEMetropolis
        # start = time()
        # all_results[sampler.__name__] = cal.multirun(sampler, draws = 4000,tune = 1000,bcm_model = bcm_model_1,n_iterations = 100, n_jobs= n_jobs,initial_params = init_vals
        # )
        # end = time()
        # print("DEMetropolis walltime:\n", end - start)

        # sampler = pm.DEMetropolisZ
        # start = time()
        # all_results[sampler.__name__] = cal.multirun(sampler, draws = 4000,tune = 1000,bcm_model = bcm_model_1,n_iterations = 100,n_jobs = n_jobs,initial_params = init_vals
        # )
        # end = time()
        # print("DEMetropolisZ walltime:\n", end - start)

        # sampler = pm.Metropolis
        # start = time()
        # all_results[sampler.__name__] = cal.multirun(sampler, draws = 8000,tune = 1000,bcm_model = bcm_model_1,n_iterations = 100,n_jobs= n_jobs,initial_params = init_vals
        # )
        # end = time()
        # print("Metropolis walltime:\n", end - start)

        # # Storing the results in a pickle file
        # with open('./Results/Model_1/Multiple_run_results.pkl', 'wb') as fp:
        #     pickle.dump(all_results, fp)

    if args.application == 2 : #Runing analysis for the SEIR age-stratified model
        bcm_model_2 = bcm_seir_age_strat()

        ##____Uniform Initialisation for each chain_________
        chains = 4
        init_vals = []
        for c in range(chains):
            temp = {param: np.random.uniform(0.0,1.0) for param in list(bcm_seir_age_strat.parameters.keys())[:-2]}
            temp["incubation_period"] = np.random.uniform(1.,15.) 
            temp["infectious_period"] = np.random.uniform(1.,15.)
            init_vals.append(temp)



