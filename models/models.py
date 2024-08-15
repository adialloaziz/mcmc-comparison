# In this file we build all the models we want to calibrate
from summer2 import CompartmentalModel, Stratification, AgeStratification
from summer2.parameters import Parameter
from datetime import datetime
from pathlib import Path
from estival.model import BayesianCompartmentalModel
from calibs_utilities import get_all_priors, get_targets
import pandas as pd
import numpy as np

import estival.targets as est
import estival.priors as esp

# from typing import List, Dict


def get_sir_model(
    config: dict,
) -> CompartmentalModel:
    """
    Args: config : A dictionary containing the model configuration/Initial configuration; Items should be: compartments a alist of the model's compartments,
                  'population' with the total size, 'seed' the number of initial infectious and 'end_time' the final duration of the spread.

        Example: config = {
        "compartments" : ("S", "I", "R")
        "population": 1e6,
        "seed": 100.0,
        "end_time": 365.0,
        }
    Return:  A simple SIR model which could be complexified with stratification (age groupes or compartments splitting)
    """
    model = CompartmentalModel(
        times=(0.0, config["end_time"]),
        compartments=config["compartments"],
        infectious_compartments=("I",),
    )
    model.set_initial_population(
        distribution=
        {
            "S": config["population"] - config["seed"], 
            "I": config["seed"],
        },
    )

    #Adding the recovery transition flow from Infectious compartment
    model.add_transition_flow(
        name="recovery",
        fractional_rate=Parameter("recovery_rate"),
        source="I",
        dest="R",
    )
    model.request_output_for_compartments(name="active_cases", compartments=["I"])

    return model



def model1():
    #------Defining here the model's congiguration/ We can also if necessar import it from another file
    model_config = {
    "compartments": ("S", "I", "R"),
    "population": 1e6,
    "seed": 100.0,
    "end_time": 365.0,
    }
    m = get_sir_model(model_config)
    m.set_initial_population(
        distribution=
        {
            "S": model_config["population"] - model_config["seed"], 
            "I": model_config["seed"],
        },
    )

    m.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("contact_rate"), 
        source="S", 
        dest="I",
    )
    return m

def bcm_sir():
    BASE_PATH = Path(__file__).parent.parent.resolve()
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
    model_1 = model1()
    bcm_model_1 = BayesianCompartmentalModel(model_1, parameters, priors, targets)  

    return bcm_model_1

def model2(
        model_config: dict = 
        {"compartments": ("S", "E","I","R"), # "Ip","Ic", "Is", "R"),
        "population": 56490045, #England population size 2021
        # "seed": Parameter("seed"),
        "start_time": datetime(2020, 6, 1),
        "end_time": datetime(2020, 11, 30),
    },
           ):
    #----We've added the compartiment Exposed to the SIR model
    m = CompartmentalModel(
        times=(model_config["start_time"], model_config["end_time"]),
        compartments=model_config["compartments"],
        infectious_compartments=("I"), #"Ip","Ic", "Is",),
        ref_date=datetime(2020, 6, 1)
    )
    m.set_initial_population(
        distribution=
        {
            "S": model_config["population"] - Parameter("seed"), 
            "I": Parameter("seed"),
        },
    )
    #m.add_transition_flow()
    # fixed parameters
    incubation_period = 5.4
    infectious_period = 7.3
    m.add_infection_frequency_flow(name="infection", contact_rate = 1., source = "S", dest ="E")
    m.add_transition_flow('progression', 1.0 / incubation_period, source = 'E', dest='I')
    m.add_transition_flow('recovery', 1.0 / infectious_period, source = 'I', dest = 'R')
    strata = [i for i in range(0, 65, 5)] 
    #All the compartments are age-strafied
    age_strat = Stratification(name='age', strata=strata,compartments=model_config["compartments"])
    age_proportion = [0.055, 0.06, 0.06, 0.057, 0.06, 0.066, 0.07, 0.06, 0.063, 0.064, 0.069, 0.067, 0.242]
    age_strat.set_population_split({f"{i}": val for i, val in zip(range(0,65,5),age_proportion)}
            )
    # Stratify the model first
    #We suppose that only the susceptibility varies by age
    age_suscept = {str(catgr): Parameter(f"age_transmission_rate_{str(catgr)}") for catgr in strata } 
    age_strat.set_flow_adjustments('infection', adjustments=age_suscept)



    m.stratify_with(age_strat)
    for strat in strata:

        m.request_output_for_flow(f"IXage_{str(strat)}", "infection", source_strata={"age": str(strat)}, save_results=True)
    
    m.request_output_for_compartments(name="incidence", compartments=["I"])


    return m

def bcm_seir_age_strat(model_config: dict = 
        {"compartments": ("S", "E","I","R"), # "Ip","Ic", "Is", "R"),
        "population": 56490045, #England population size 2021
        # "seed": Parameter("seed"),
        "start": datetime(2020, 8, 1),
        "end_time": datetime(2020, 11, 30)}):
    #------DATA-MANAGEMENT------------------------
    BASE_PATH = Path(__file__).parent.parent.resolve()
    DATA_PATH = BASE_PATH / "data"
    df = pd.read_csv(DATA_PATH/"new_cases_England_2020.csv")
    df["date"] = pd.to_datetime(df.date)
    df.set_index(["age","date"], inplace=True)

    # ages_labels = [f"{i:02}_{i+4:02}" for i in range(0,60, 5)] + ["60+"]
    # targets_data = dict()
    # for age in ages_labels:
    #     targets_data[age] = df.loc[age]

    ages_labels = [f"{i:02}_{i+4:02}" for i in range(0,60, 5)] + ["60+"]
    age_strat = [f"{i}" for i in range(0,65,5)]

    parameters = {
    'age_transmission_rate_'+ str(age) : 0.25 for age in age_strat
        }
    #Default parameters
    parameters["seed"] = 100.

    parameters['incubation_period']= 5.4
    parameters['infectious_period'] = 7.3

    

    
    # A uniform prior is defined for all the transmission rate
    params = {param: (0.0,1.0) for param in parameters.keys() if param not in ("seed","incubation_period","infectious_period")}
    priors_seed = [esp.UniformPrior("seed",(1,1000))]
    # # A normal prior for the incubation and infectious periods
    # normal_priors = [ 
    #     esp.TruncNormalPrior("incubation_period",5.4, 3.0, (1,15)),
    #     esp.TruncNormalPrior("infectious_period",7.3, 2.0, (1,15)),
    #     ]
    uniform_priors = get_all_priors(params)
    # uniform_priors.append(esp.UniformPrior("inc_disp",(0.1, 2000)))

    priors = priors_seed + uniform_priors #[:-2]

    #The fitted period is Aug 2020 to Nov 2020
    # We define a normal target with fixed std for each age catergory
    # We rolle the data for 14 day to discard fluctuations
    # Inc_disp = esp.UniformPrior("inc_disp",(0.1, 2000))
    targets = [
                est.TruncatedNormalTarget("IXage_"+str(age),
                                        df.loc[age_cat]["Aug 2020":"Nov 2020"].rolling(14).mean().iloc[14:]["cases"],
                                          (0.0,np.inf),
                                1200) for age_cat, age in zip(ages_labels, age_strat)
            ]

    model_2 = model2(model_config)
    #Defining  a Bayesian Compartmental Model

    bcm_model_2 = BayesianCompartmentalModel(model_2, parameters, priors,targets)
    return bcm_model_2