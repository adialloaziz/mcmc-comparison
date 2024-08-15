import logging

# Disable warning messages from pytensor.tensor.blas
logging.getLogger('pytensor.tensor.blas').setLevel(logging.ERROR)

from summer2 import CompartmentalModel
from summer2.parameters import Parameter
import arviz as az
import Calibrate as cal
import seaborn as sns
from jax.scipy.stats import gaussian_kde
from jax import lax

import jax.numpy as jnp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

import pymc as pm
from estival.wrappers import pymc as epm
from estival.model import BayesianCompartmentalModel
import estival.priors as esp
import estival.targets as est
from estival.sampling import tools as esamp
import numpyro
from numpyro import infer
from numpyro import distributions as dist
from jax import random
import pickle

import multiprocessing as mp
import platform
from time import time 



if __name__ == "__main__":
    if platform.system() != "Windows":
         mp.set_start_method('forkserver')

    BASE_PATH = Path(__file__).parent.resolve()

     #------------------Build a simple SIR MODEL------------
    def build_model():
        sir_model = CompartmentalModel([0.0,100.0],["S","I","R"],["I"])
        sir_model.set_initial_population({"S": 999.0, "I": 1.0})
        sir_model.add_infection_frequency_flow("infection",Parameter("contact_rate"),"S","I")
        sir_model.add_transition_flow("recovery",Parameter("recovery_rate"),"I","R")

        sir_model.request_output_for_flow("incidence", "infection")
        
        return sir_model

    sir_model = build_model()
    parameters = {
    "contact_rate": 0.3,
    "recovery_rate": 0.1
    }
 #_____________SAMPLING FROM A KNOWN DISTRIBUTION_________________________________
    with open(BASE_PATH/"true_sample.pkl", 'rb') as fp:
        samples = pickle.load(fp)

    #_____________RUNNING THE MODEL FORWARD______________

    priors = [
    esp.UniformPrior("contact_rate", [0, 1]),
    ]
    targets = []
    bcm = BayesianCompartmentalModel(model=sir_model,priors=priors, targets=targets,parameters=parameters)
    samples_for_estival = [{"contact_rate": samples.iloc[i]} for i in range(len(samples))]

    model_runs = esamp.model_results_for_samples(samples_for_estival, bcm)

    #__________COLLECTING SYNTHETIC DATA FROM THE PREVIOUS MODEL RUN_______

    data_times = list(range(10, 91, 10))
    len(data_times)
    likelihood_comps = {t: gaussian_kde(jnp.array(model_runs.results['incidence'].loc[t]), bw_method=0.01) for t in data_times}

    #_____________REFIT THE MODEL TO THE SYNTHETIC DATA___________

    # Flat prior
    priors = [
        esp.UniformPrior("contact_rate", [0.1, 0.5]),
    ]
    n_data_points = len(data_times)
    # Define a custom target using the likelihood components
    def make_eval_func(t):
        def eval_func(modelled, obs, parameters, time_weights):
            likelihood_comp = likelihood_comps[t](modelled) 
            likelihood_comp = jnp.max(jnp.array([likelihood_comp, jnp.array([1.e-300])]))  # to avoid zero values.
            return jnp.log(likelihood_comp) / n_data_points

        return eval_func

    targets = [est.CustomTarget(f"likelihood_comp_{t}", pd.Series([0.], index=[t]), make_eval_func(t), 
                                model_key='incidence') for t in data_times]

    refit_bcm = BayesianCompartmentalModel(model=sir_model,priors=priors, targets=targets,parameters=parameters)

    #_______________Sampling____________________________

    all_results = dict()
    start = time()
    sampler = pm.sample_smc
    all_results[sampler.__name__] = cal.multirun(
    sampler, 
    draws = 500,
    tune = 0,
    chains=4,
    cores=4, 
    bcm_model = refit_bcm,
    n_iterations = 4,
    n_jobs = 2,
    )
    end = time()
    print("SMC Multi run elapsed time", end-start)


    #Storing our results in to pickle file
    Dir_path = Path(BASE_PATH/"Results/Reverse_Ingineering/").mkdir(parents=True, exist_ok=True)
    file_path = Dir_path/"Multi_run_SMC.pkl"
    with open(file_path, 'wb') as fp:
        pickle.dump(all_results, fp)

    # sampler = pm.DEMetropolisZ
    # start = time()
    # all_results[sampler.__name__] = cal.multirun(
    # sampler, 
    # draws = 100,
    # tune = 1000,
    # chains=4,
    # cores=4, 
    # bcm_model = refit_bcm,
    # n_iterations = 100,
    # n_jobs = 2,
    # )
    # end = time()
    # print("DEMZ Multi run elapsed time", end-start)

    # with open(Path(BASE_PATH/"Results/Reverse_Ingineering/Multi_run_DEMZ.pkl"), 'wb') as fp:
    #     pickle.dump(all_results, fp)
    
    
    # sampler = pm.Metropolis
    # start = time()

    # all_results[sampler.__name__] = cal.multirun(
    # sampler, 
    # draws = 10000,
    # tune = 1000,
    # chains=4,
    # cores=4, 
    # bcm_model = refit_bcm,
    # n_iterations = 100,
    # n_jobs = 2,
    # )
    # end = time()
    # print("MH Multi run elapsed time", end-start)
    # with open(Path(BASE_PATH/"Results/Reverse_Ingineering/Multi_run_MH.pkl"), 'wb') as fp:
    #     pickle.dump(all_results, fp)

    # def nmodel():
    #     sampled = {"contact_rate":numpyro.sample("contact_rate", dist.Uniform(0.0,1.0))}
    #     ll = numpyro.factor("ll", refit_bcm.loglikelihood(**sampled))
    
    # sampler = infer.NUTS

    # start = time()
    # all_results[sampler.__name__] = cal.multirun(
    # sampler, 
    # draws = 5000,
    # tune = 1000,
    # chains=4,
    # cores=4, 
    # bcm_model = refit_bcm,
    # n_iterations = 100,
    # nmodel=nmodel,
    # n_jobs = 2,
    # )
    # end = time()
    # print("NUTS Multi run elapsed time", end-start)
    # with open(Path(BASE_PATH/"Results/Reverse_Ingineering/Multi_run_NUTS.pkl"), 'wb') as fp:
    #     pickle.dump(all_results, fp)
    
    # sampler = pm.DEMetropolis
    # start = time()
    # all_results[sampler.__name__] = cal.multirun(
    # sampler, 
    # draws = 10000,
    # tune = 1000,
    # chains=4,
    # cores=4, 
    # bcm_model = refit_bcm,
    # n_iterations = 100,
    # n_jobs = 2,
    # )
    # end = time()
    # print("DEM Multi run elapsed time", end-start) 

    # with open(Path(BASE_PATH/"Results/Reverse_Ingineering/Multi_run_DEM.pkl"), 'wb') as fp:
    #     pickle.dump(all_results, fp)

    