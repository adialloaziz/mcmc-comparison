# In this file we build all the models we want to calibrate
from summer2 import CompartmentalModel
from summer2.parameters import Parameter
from typing import List, Dict


def get_sir_model(
    config: dict,
) -> CompartmentalModel:

    model = CompartmentalModel(
        times=(0.0, config["end_time"]),
        compartments=(
            "susceptible", 
            "infectious", 
            "recovered",
        ),
        infectious_compartments=("infectious",),
    )
    model.set_initial_population(
        distribution=
        {
            "susceptible": config["population"] - config["seed"], 
            "infectious": config["seed"],
        },
    )
    
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("contact_rate"), 
        source="susceptible", 
        dest="infectious",
    )
    model.add_transition_flow(
        name="recovery",
        fractional_rate=Parameter("recovery_rate"),
        source="infectious",
        dest="recovered",
    )
    model.request_output_for_compartments(name="active_cases", compartments=["infectious"])

    return model


#------Defining here the model's congiguration/ We can also if necessar import it from another file
model_config = {
    "population": 1e6,
    "seed": 100.0,
    "end_time": 365.0,
}
def model1():
    return  get_sir_model(model_config)


    




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