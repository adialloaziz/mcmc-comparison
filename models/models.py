# In this file we build all the models we want to calibrate
from summer2 import CompartmentalModel, Stratification, AgeStratification
from summer2.parameters import Parameter
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


def model2():
    from datetime import datetime
    model_config = {
    "compartments": ("S", "E","I","R"), # "Ip","Ic", "Is", "R"),
    "population": 10500,
    "seed": 100.0,
    "end_time": datetime(2020, 12, 8),
    }
    #----We've added the compartiment Exposed to the SIR model
    m = CompartmentalModel(
        times=(datetime(2019, 12, 8), model_config["end_time"]),
        compartments=model_config["compartments"],
        infectious_compartments=("I"), #"Ip","Ic", "Is",),
        ref_date=datetime(2019, 12, 8)
    )
    m.set_initial_population(
        distribution=
        {
            "S": model_config["population"] - model_config["seed"], 
            "E": model_config["seed"],
        },
    )
    #m.add_transition_flow()
    m.add_infection_frequency_flow(name="infection", contact_rate = 1., source = "S", dest ="E")
    m.add_transition_flow('progression', 1.0 / Parameter('incubation_period'), source = 'E', dest='I')
    m.add_transition_flow('recovery', 1.0 / Parameter('infectious_period'), source = 'I', dest = 'R')
    strata = [i for i in range(0, 65, 5)] 
    #All the compartments are age-strafied
    age_strat = Stratification(name='age', strata=strata,compartments=model_config["compartments"])
    #age_strat.set_population_split({"0": .7, "15": 0.3})
    # Stratify the model first
    #We suppose that only the susceptibility varies by age
    age_suscept = {str(catgr): Parameter(f"age_transmission_rate_{str(catgr)}") for catgr in strata } 
    age_strat.set_flow_adjustments('infection', adjustments=age_suscept)



    m.stratify_with(age_strat)
    for strat in strata:

        m.request_output_for_flow(f"incX{str(strat)}", "infection", source_strata={"age": str(strat)}, save_results=True)
    
    m.request_output_for_compartments(name="incidence", compartments=["I"])


    return m


    




#

# def Build_model(
#     config: dict,
#     compartments: List[str],
#     #latent_compartments: List[str],
#     infectious_compartments: List[str],
#     #age_strata: List[int],
#     #fixed_params: Dict[str, any],
    
# ) -> CompartmentalModel:
#     """
#     Builds and returns a compartmental model for epidemiological studies, incorporating
#     various flows and stratifications based on age, organ status, and treatment outcomes.

#     Args:
#         compartments: List of compartment names in the model.
#         latent_compartments: List of latent compartment names.
#         infectious_compartments: List of infectious compartment names.
#         age_strata: List of age groups for stratification.
#         time_start: Start time for the model simulation.
#         time_end: End time for the model simulation.
#         time_step: Time step for the model simulation.
#         fixed_params: Dictionary of parameters with fixed values.

#     Returns:
#         A configured CompartmentalModel object.
#     """

#     model = CompartmentalModel(
#         times=(0.0, config["end_time"]),
#         compartments = compartments,
#         infectious_compartments = infectious_compartments,
#         timestep = config["timestep"]
#     )

#     model.set_initial_population(
#         distribution=
#         {
#             "susceptible": config["population"] - config["seed"], 
#             "infectious": config["seed"],
#         },
#     )
    
#     model.add_infection_frequency_flow(
#         name="infection", 
#         contact_rate=Parameter("contact_rate"), 
#         source="susceptible", 
#         dest="infectious",
#     )
#     model.add_transition_flow(
#         name="recovery",
#         fractional_rate=Parameter("recovery"),
#         source="infectious",
#         dest="recovered",
#     )
    
#     return model