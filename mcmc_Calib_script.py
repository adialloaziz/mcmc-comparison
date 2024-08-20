
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
                    "-apl","--application", type=int, choices=range(1,4), default = 2,
                    help="""Specify wich application you want to run (1,2 or 3). Default is 2 the SEIR"""
                      )

    args = parser.parse_args()

    # print("Number of jobs for multirun = ", args.n_jobs)
    if platform.system() != "Windows":
    
        mp.set_start_method('forkserver')

    BASE_PATH = Path(__file__).parent.resolve()
    today_analysis = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

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
        model_config = {"compartments": ("S", "E","I","R"), # "Ip","Ic", "Is", "R"),
        "population": 56490045, #England population size 2021
        # "seed": 100.0,
        "start_time": datetime(2020, 6, 1),
        "end_time": datetime(2020, 11, 30)
            }
        bcm_model_2 = bcm_seir_age_strat(model_config)
        model_2 = model2(model_config)

        ##____Uniform Initialisation for each chain_________
        chains = 4
        init_vals = []
        # for c in range(chains):
        #     temp = {param: np.random.uniform(0.0,1.0) for param in list(bcm_seir_age_strat.parameters.keys())[:-2]}
        #     temp["incubation_period"] = np.random.uniform(1.,15.) 
        #     temp["infectious_period"] = np.random.uniform(1.,15.)
        #     init_vals.append(temp)

        def nmodel_2():
            unif_priors = list(bcm_model_2.parameters)[:-3]
            sampled = {k:numpyro.sample(k, dist.Uniform(0.0,1.0)) for k in unif_priors}
            sampled["seed"] = numpyro.sample("seed", dist.Uniform(1.0,1000))
            #Adding the normal priors for the incubation and infectious periods
            # sampled["incubation_period"] = numpyro.sample("incubation_period", dist.TruncatedNormal(7.3, 2.0, low=1., high=14.))
            # sampled["infectious_period"] = numpyro.sample("infectious_period", dist.TruncatedNormal(5.4, 3.0, low=1., high=14.))
            # Log-likelihood
            disp_params = {k:v.ppf(0.5) for k,v in bcm_model_2.priors.items() if "_disp" in k}

            log_likelihood = bcm_model_2.loglikelihood(**sampled | disp_params)

            numpyro.factor("ll",log_likelihood)


        #_______________Sampling____________________________
        #_______Multiple Run______________________
        #_______________Sampling____________________________

        all_results = dict()
        start = time()
        sampler = pm.sample_smc
        all_results[sampler.__name__] = cal.multirun(
        sampler, 
        draws = 10000,
        tune = 0,
        chains=4,
        cores=4, 
        bcm_model = bcm_model_2,
        n_iterations = 100,
        n_jobs = 4,
        )
        end = time()
        print("SMC Multi run elapsed time", end-start)


        #Storing our results in to pickle file
        output_root_dir = Path.home() / "sh30/users/rragonnet/outputs/"
        # output_root_dir = BASE_PATH / "Results/" #If I'm on my own computer 

        Dir_path = Path(output_root_dir/f"{today_analysis}_Results/Model_2/")
        Dir_path.mkdir(parents=True, exist_ok=True)
        # file_path = Dir_path/"Multi_run_SMC.pkl"
        # with open(file_path, 'wb') as fp:
        #     pickle.dump(all_results, fp)

        sampler = pm.DEMetropolisZ
        start = time()
        all_results[sampler.__name__] = cal.multirun(
        sampler, 
        draws = 10000,
        tune = 1000,
        chains=4,
        cores=4, 
        bcm_model = bcm_model_2,
        n_iterations = 100,
        n_jobs = 4,
        )
        end = time()
        print("DEMZ Multi run elapsed time", end-start)

        with open(Dir_path/"Multi_run_DEMZ.pkl", 'wb') as fp:
            pickle.dump(all_results, fp)
        
        
        sampler = pm.Metropolis
        start = time()

        all_results[sampler.__name__] = cal.multirun(
        sampler, 
        draws = 10000,
        tune = 1000,
        chains=4,
        cores=4, 
        bcm_model = bcm_model_2,
        n_iterations = 100,
        n_jobs = 4,
        )
        end = time()
        print("MH Multi run elapsed time", end-start)
        with open(Dir_path/"Multi_run_SMC_DEMZ_MH.pkl", 'wb') as fp:
            pickle.dump(all_results, fp)

        
        sampler = infer.NUTS

        start = time()
        all_results[sampler.__name__] = cal.multirun(
        sampler, 
        draws = 10000,
        tune = 1000,
        chains=4,
        cores=4, 
        bcm_model = bcm_model_2,
        n_iterations = 100,
        nmodel=nmodel_2,
        n_jobs = 4,
        )
        end = time()
        print("NUTS Multi run elapsed time", end-start)
        with open(Dir_path/"Multi_run_SMC_DEMZ_MH_NUTS.pkl", 'wb') as fp:
            pickle.dump(all_results, fp)
        
        sampler = pm.DEMetropolis
        start = time()
        all_results[sampler.__name__] = cal.multirun(
        sampler, 
        draws = 10000,
        tune = 1000,
        chains=4,
        cores=4, 
        bcm_model = bcm_model_2,
        n_iterations = 100,
        n_jobs = 4,
        )
        end = time()
        print("DEM Multi run elapsed time", end-start) 

        with open(Dir_path/"Multi_run_all.pkl", 'wb') as fp:
            pickle.dump(all_results, fp)


