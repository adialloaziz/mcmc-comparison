### In this script we develop a procedure that return a list or a file containing
## all the metrics and information to assess a MCMC algorithm given a compartmental model 

from estival.model import BayesianCompartmentalModel
import pymc as pm
import pandas as pd
from estival.wrappers import pymc as epm
import arviz as az #For convergence stats and diagnostics
from arviz import InferenceData
import time
import matplotlib .pyplot as plt
from typing import Callable, Optional

from numpyro import infer
from jax import random

def compute_max_Rhat(
        idata: InferenceData
        
) -> float:
    Rhat = az.rhat(idata).variables
    # Extract data from the Frozen object
    data = {key: value.values.item() for key, value in Rhat.items()}

    # Create DataFrame from the extracted data
    max_rhat_df = pd.DataFrame(data, index=[0])
    return max_rhat_df.values.max()
#_________________________________________________________

def compute_mean_Rhat(
        idata: InferenceData
        
) -> float:
    Rhat = az.rhat(idata).variables
    # Extract data from the Frozen object
    data = {key: value.values.item() for key, value in Rhat.items()}

    # Create DataFrame from the extracted data
    mean_rhat_df = pd.DataFrame(data, index=[0])
    return mean_rhat_df.values.mean()

#__________________________________________________________
def compute_mean_Ess(
        idata: InferenceData
        
) -> float:
    Ess = az.ess(idata).variables
    # Extract data from the Frozen object
    data = {key: value.values.item() for key, value in Ess.items()}

    # Create DataFrame from the extracted data
    mean_ess_df = pd.DataFrame(data, index=[0])
    return mean_ess_df.values.mean()

#_____________________________________________________________
def compute_min_Ess(
        idata: InferenceData
        
) -> float:
    Ess = az.ess(idata).variables
    # Extract data from the Frozen object
    data = {key: value.values.item() for key, value in Ess.items()}

    # Create DataFrame from the extracted data
    min_ess_df = pd.DataFrame(data, index=[0])
    return min_ess_df.values.min()
    
    

def Sampling_calib(
        bcm_model : BayesianCompartmentalModel,
        mcmc_algo: str,
        draws : int,
        tune: int,
        chains : int | None,
        cores : int | None,
        nmodel: Optional[Callable] = None,
        initial_params: Optional[dict | list] = None,

        # hmc_warmup = 1000,
                ) -> list[InferenceData, float] :

    """
    Args:
    - bcm_model : A BayesianCompartmentalModel, within wich the targets and priors distributions will be extracted.
    -nmodel : The numpyro NUTS require a defined model to perform sampling.
            Example of definition:  
            def nmodel():
                sampled = {k:numpyro.sample(k, dist.Uniform(0.0,1.0)) for k in bcm_model.parameters}
                ll = numpyro.factor("ll", bcm_model.loglikelihood(**sampled))

    - mcmc algo (str) : A Markov Chain Monte Carlo algorithm can be from the Pymc/Numpyro list of MCMC algorithms or a disigned sampler algorithm.
                  please refer do the Pymc documentation for more details.
    - initial_params (dict) : The MCMC algorithm starting points. 
    - draws(int) : The size of the chains (each chain if running multiple chains). Default 1000 samples.
    - tune (int, default 100): Number of tune for Pymc MCMC algo, number of warmup for HMC family 
    - chains(int, default 1): Number of chains for multiple chains sampling
    -cores(int, default 1 ): Number of cpu cores to use for mutlichain sampling.

    Returns:
     - An arviz InferenceData which provides the stats and diagnostics of the chains convergence.
     We may add soon the ESS/second, the minimum number of iterations to obtain the given Ess etc..
    """

    sampler_name = mcmc_algo.__name__ #Name of the sampler
    if sampler_name == "NUTS": # We need to set some specifications for this sampler. 
                                # Use of numpyro and/or BlackJax 
        # pass
        # def nmodel():
        #     sampled = {k:numpyro.sample(k, dist.Uniform(0.0,1.0)) for k in bcm_model.parameters}
        #     ll = numpyro.factor("ll", bcm_model.loglikelihood(**sampled))

        kernel = infer.NUTS(nmodel)
        mcmc = infer.MCMC(kernel, num_warmup=tune, num_chains=chains, num_samples=draws, progress_bar=False)

        start = time.time()
        mcmc.run(random.PRNGKey(0), init_params=initial_params)
        end = time.time()

        Time = end - start
        idata = az.from_numpyro(mcmc)

    else :
        with pm.Model() as model:
                
            variables = epm.use_model(bcm_model)

            step_kwargs = {} #dict(variables,proposal_dist = pm.NormalProposal)
            # step = mcmc_algo(**step_kwargs)
            start = time.time()

            if sampler_name == "sample_smc": ## SEQUENTIAL MONTE CARLO SAMPLING

                idata = pm.sample_smc(kernel=pm.smc.IMH, 
                                        start = None, 
                                        draws=draws,
                                        chains=chains,
                                        threshold = 0.1,
                                        correlation_threshold=0.5,
                                        progressbar=False,
                                        cores=cores,
                                        )
            else :  
                idata = pm.sample(step = mcmc_algo(**step_kwargs),
                                    initvals = initial_params,
                                    draws=draws,
                                    tune = tune ,
                                    cores=cores,
                                    chains=chains,
                                    progressbar=False
                                    )   
            end = time.time()
            Time = end - start

    return [idata, Time] # Will use arviz to examine outputs (trace)

def Compute_metrics(
        mcmc_algo: str,
        idata: InferenceData,
        Time: float,
        draws: int, 
        chains: int,
        tune: int,
                    ) :
    
    ess_mean = compute_mean_Ess(idata)
    ess_min = compute_min_Ess(idata)

    rhat_mean = compute_mean_Rhat(idata)
    rhat_max = compute_max_Rhat(idata)

    results = dict(
        Sampler = mcmc_algo.__name__,
        Draws = draws,
        Chains = chains,
        Tune = tune,
        Time = Time,
        Mean_ESS = ess_mean,
        Min_Ess = ess_min,
        Ess_per_sec = ess_mean/Time, 
        Mean_Rhat = rhat_mean,
        Rhat_max = rhat_max,

        Trace = [idata],
    )
    
    return pd.DataFrame(results)


def Single_analysis(
            sampler : str,
            draws : int,
            chains : int | None,
            cores : int | None,
            tune : int,
            bcm_model: BayesianCompartmentalModel,
            initial_params: Optional[dict | list] = None,
            nmodel: Optional[Callable] = None,
            )->pd.DataFrame:
    
    df = pd.DataFrame()

    [idata, Time] = Sampling_calib(bcm_model=bcm_model,
                    mcmc_algo = sampler,
                    initial_params = initial_params,
                    draws = draws,
                    tune = tune,
                    chains = chains,
                    cores = cores,
                    nmodel = nmodel,
                    )
    results = Compute_metrics(
                    mcmc_algo = sampler,
                    idata = idata,
                    Time = Time,
                    draws = draws, 
                    chains = chains,
                    tune = tune,
                        )
                
    df = pd.concat([df, results])
    df["Run"] = df.Sampler + "\nDraws=" + df.Draws.astype(str) + "\nTune=" + df.Tune.astype(str) +"\nChains=" + df.Chains.astype(str)

    df = df.reset_index(drop=True)
    return df


from joblib import Parallel, delayed


#Leveraging multiprocess or multithreading while using a loop to perform many runs
def multirun(sampler : str,
            draws : int,
            chains : int | None,
            cores : int | None,
            tune : int,
            bcm_model: BayesianCompartmentalModel,
            n_iterations : int,
            n_jobs: int,
            initial_params: Optional[dict] = None,
            nmodel: Optional[Callable] = None,
            )->pd.DataFrame:

    iterations = range(n_iterations)
    def run_analysis():
        df = Single_analysis(sampler = sampler,
                            draws = draws,
                            tune = tune,
                            chains= chains,
                            cores = cores,
                            bcm_model = bcm_model,
                            nmodel = nmodel,
                            initial_params = initial_params
                            )
        return df
        #Parallel loop
    if sampler.__name__ in ("NUTS", "DEMetropolis"):
        backend = 'processes'
    else :
        backend = 'threads'
    results = Parallel(n_jobs=n_jobs, prefer=backend,timeout=1000)(delayed(run_analysis)() for i in iterations)

    return pd.concat(results, ignore_index=True)



def plot_comparison_bars(results_df):
    pd.options.plotting.backend = "matplotlib"
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    ax = axes[0]
    results_df.plot.bar(y="Ess_per_sec", x="Run", ax=ax, legend=False)
    ax.set_title("Ess_per_sec")
    ax.set_xlabel("")
    labels = ax.get_xticklabels()

    ax = axes[1]
    results_df.plot.bar(y="Mean_Rhat", x="Run", ax=ax, legend=False)
    ax.set_title(r"$\hat{R}$")
    ax.set_xlabel("")
    ax.set_ylim(1)
    labels = ax.get_xticklabels()
    plt.suptitle(f"Comparison MCMC runs", fontsize=12)
    plt.tight_layout()


def group_summary(results_df):
    s = []
    for row in results_df.index:
        trace = results_df.Trace.loc[row].posterior
        summary = az.summary(trace)
        summary["Sampler"] = results_df.at[row, "Sampler"]
        s.append(summary)
    summaries = pd.concat(s)

    summaries["Variables"] = summaries.index
    summaries.set_index("Sampler", inplace=True)

    sum_dict = dict()
    summary_means = pd.DataFrame()

    for sampler in ["sample_smc","NUTS", "DEMetropolis", "DEMetropolisZ", "Metropolis"]:
        sum_dict[sampler] = summaries.loc[sampler].groupby("Variables").mean().reset_index(drop=False)

    summary_means = pd.concat(sum_dict)
    #computing the pecentage of Mean rhat <= 1.1
    prcnt_success= pd.DataFrame()
    prcnt_success["Mean rhat <= 1.1(%)"] = results_df.groupby('Sampler')['Mean_Rhat'].apply(lambda x: (x <= 1.1).mean() * 100)
    prcnt_success["Mean time(s)"] = results_df.groupby('Sampler')['Time'].mean()
    return summary_means, prcnt_success

# def compute_MSE(
#         sampler_results: pd.DataFrame,
#         true_posterior: pd.DataFrame,
#         variable: str,
#         )->pd.DataFrame:
#     df = sampler_results.copy()
#     #create a new column named "MSE" containing the mean squared error for each sampler
#     df["MSE"] = 10
#     for row in sampler_results.index:
#         idata = sampler_results.Trace.loc[row]
#         pred_posterior = pd.DataFrame(idata.posterior.to_dataframe()[variable].to_list())
#         pred_posterior = pred_posterior.rename(columns={0:variable})

#         # Ensure the DataFrames are aligned
#         # true_sample, posterior_sample = true_sample.align(posterior_sample, join='inner', axis=1)
#         squared_diff = (true_posterior[variable] - pred_posterior) ** 2
#         df.MSE.loc[row] = squared_diff.mean().values

#         # print("Mean Squared Error:", mse)

    
#     return df

def fitting_test(idata, bcm, model):
    from estival.sampling.tools import likelihood_extras_for_samples
    likelihood_df = likelihood_extras_for_samples(idata.posterior, bcm)
    # likelihood_df = likelihood_extras_for_idata(idata, bcm)
    ldf_sorted = likelihood_df.sort_values(by="logposterior",ascending=False)

    # Extract the parameters from the calibration samples
    map_params = idata.posterior.to_dataframe().loc[ldf_sorted.index[0]].to_dict()
    bcm.loglikelihood(**map_params), ldf_sorted.iloc[0]["loglikelihood"]
    # Run the model with these parameters
    model.run(map_params)
    # ...and plot some results
    return model.get_outputs_df()



# def plot_comparison_Bars(results_df: pd.DataFrame):
#     fig, axes = plt.subplots(1, 2, figsize=(10, 8))
#     ax = axes[0]
#     ax.bar(x=results_df["Run"], height=results_df["Ess_per_sec"],width= 0.2)#, legend=False)
#     ax.set_title("ESS per Second")
#     # ax.set_xlabel('Run',  rotation='vertical',fontsize=28)
#     ax.set_xlabel("")
#     labels = ax.get_xticklabels()
#     """
#     ax = axes[1]
#     results_df.plot.bar(y="ESS_pct", x="Run", ax=ax, legend=False)
#     ax.set_title("ESS Percentage")
#     ax.set_xlabel("")
#     labels = ax.get_xticklabels()
#     """
#     ax = axes[1]
#     ax.bar(x=results_df["Run"], height=results_df["Mean_Rhat"],width= 0.2)#, legend=False)
#     ax.set_title(r"$\hat{R}$")
#     ax.set_xlabel("")
#     ax.set_ylim(1)
#     labels = ax.get_xticklabels()
#     plt.suptitle(f"Comparison of MCMC runs using the simple SIR model", fontsize=12)

#     plt.tight_layout()
#     plt.show()



