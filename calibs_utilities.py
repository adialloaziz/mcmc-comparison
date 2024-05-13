### Here we define function that build targets and prior distribution for observed data of an model
import estival.priors as esp
import estival.targets as est
import numpy as np
import pandas as pd
import yaml
# ___________________________________________________________________________________________________________

def get_all_priors(params) -> list:
    """
    Get all priors used in any of the analysis types.
    Args:
    - params (dict): List of all the parameters and their corresponding range 
    Returns:
        All the priors used under any analyses. For all parameters we set a Uniform prior.
    """
    return [ 
        esp.UniformPrior(parameter, value_range) for parameter, value_range in params.items()
        #esp.UniformPrior("contact_rate", (0.0,0.5)),
        #esp.GammaPrior("recovery_rate", 2.0, 0.1),
    ]


#_________________________________________________________________________________________________

def load_data(
        data_yml
        )-> dict:
    """
    Args:
        - data_yml : A YAML file containing the data your want to load. 
    Returns:
    - Convert the loaded YAML data to a dictionary which each compenent is a Penda Series data 
        (assuming the data structure allows for it)
        This example assumes the YAML file contains a dictionary at its root
    """
    with open(data_yml, 'r') as file:
        data = yaml.safe_load(file)

    
    return {key: pd.Series(value) for key, value in data.items()}
    

#_____________________________________________________________________________________________________________

def get_targets(targets_yml) -> list:
    """
    Loads target data for a model and constructs a list of target distributions instances.

    This function is designed to load external target data, presumably for the purpose of
    model calibration or validation. It then constructs and returns a list of target distributions
    instances, each representing a specific target metric with associated observed values
    and standard deviations. These targets are essential for fitting the model to observed
    data, allowing for the estimation of model parameters that best align with real-world
    observations.
    Has to be modified with respect to the model's parameters (names, range,...).
    Args: 
    - target_yml (yml): A yml file containing the observed data which will be used to 
     construct the target distributions.

     -"Soon we can add a list containing the names of target with its correspondind stdev " So we can make a loop for the Normaltarget

    Returns:
    - list: A list of Target instances.
    """
    target_data = load_data(data_yml=targets_yml) #Converting the yml file into a pd.Series
    return [
        est.TruncatedNormalTarget("active_cases", target_data["active_cases"], (0.0,np.inf), stdev=
        #esp.UniformPrior("active_cases_dispersion",(0.1, targets_data.max()*0.1))) #Calibration de l'ecart type
       0.5) #Utiliser un nombre fixe pour ne pas l'inclure dans la calibration
    ]

#_____________________________________________________________________________________
