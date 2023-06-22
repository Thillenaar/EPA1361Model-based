import os
from ema_workbench import (
    ema_logging,
    Model,
    MultiprocessingEvaluator,
    Scenario,
    SequentialEvaluator
)

ema_logging.log_to_stderr(ema_logging.INFO)

from ema_workbench.em_framework.optimization import (ArchiveLogger, EpsilonProgress)
from ema_workbench.util import ema_logging
import pandas as pd
from our_problem_formulation import get_model_for_problem_formulation


# Function to create a list of scenarios from a dataframe
def create_scenarios(df_scenario_discovery):
    # create scenarios list
    scenarios = []

    # loop through dataframe
    for index, row in df_scenario_discovery.iterrows():
        # create values dictionary
        reference_values = {}
        # save dataframe row values to the dictionary
        for i in range(1, 6, 1):
            reference_values[f"A.{i}_Bmax"] = df_scenario_discovery[f"A.{i}_Bmax"][index]
            reference_values[f"A.{i}_Brate"] = df_scenario_discovery[f"A.{i}_Brate"][index]
            reference_values[f"A.{i}_pfail"] = df_scenario_discovery[f"A.{i}_pfail"][index]

        reference_values["discount rate 0"] = df_scenario_discovery["discount rate 0"][index]
        reference_values["discount rate 1"] = df_scenario_discovery["discount rate 1"][index]
        reference_values["discount rate 2"] = df_scenario_discovery["discount rate 2"][index]
        reference_values["A.0_ID flood wave shape"] = df_scenario_discovery["A.0_ID flood wave shape"][index]

        # create a dictionary with all stored uncertainty values
        scen1 = {}
        for key in model.uncertainties:
            scen1.update({key.name: reference_values[key.name]})

        # create scenario object and save in scenario list
        ref_scenario = Scenario(index, **scen1)
        scenarios.append(ref_scenario)

    return scenarios


# Function to run the optimizer to find policy levers for minimized model outcomes
def optimize_scenarios(scenario, nfe, model, epsilons, number_of_seeds):

    # save number of seeds per scenario
    seeds_dict = {"number of seeds": number_of_seeds}
    df_seeds = pd.DataFrame(seeds_dict, index=[0])
    seeds_file_path = os.path.join("data", "optimize_results", "number_of_seeds.csv",)
    df_seeds.to_csv(seeds_file_path)

    # start optimization process
    archives_folder_path = os.path.join("data", "archives")
    with MultiprocessingEvaluator(model) as evaluator:
        for i in range(number_of_seeds):
            convergence_metrics = [
                ArchiveLogger(
                    archives_folder_path,
                    # filter model levers and outcomes names on invalid python identifiers
                    [l.name for l in model.levers],
                    [o.name for o in model.outcomes],
                    base_filename=f"multi_MORDM_{scenario.name}_seed_{i}.tar.gz",
                ),
                EpsilonProgress(),
            ]

            # run optimizer
            # also, specify the seed because of reproducibility
            result, convergence = evaluator.optimize(nfe=nfe, searchover='levers',
                                                        convergence=convergence_metrics,
                                                        epsilons=epsilons,
                                                        reference=scenario,
                                                        seed=i)

            # save results and convergence in folder (optimize_results)
            result_file_path = os.path.join("data", "optimize_results", f"results_scenario_{scenario.name}_seed_{i}.csv")
            result.to_csv(result_file_path)
            convergence_file_path = os.path.join("data", "optimize_results", f"convergence_scenario_{scenario.name}_seed_{i}.csv")
            convergence.to_csv(convergence_file_path)


### Run Script ###
if __name__ == "__main__":

    print("\nMulti-MORDM outcome optimization script is running...\n")

    ema_logging.log_to_stderr(ema_logging.INFO)

    # get model
    model, steps = get_model_for_problem_formulation()
    print("Model is loaded.")

    # get scenarios from scenario discovery
    scenarios_file_path = os.path.join("data", "scenario_discovery", "reference_scenarios.csv")
    df_scenario_discovery = pd.read_csv(scenarios_file_path)

    # create list of scenario from scenario discovery
    scenarios = create_scenarios(df_scenario_discovery)
    print("Scenarios are loaded.")

    # specify epsilons
    # these values should correspond to the scale of the specific model outcome and
    # ensure convergence in a reasonable time span
    epsilons = [50000000,  # A1_Expected_Annual_Damage
                50000000,  # A1_Dike_Investment_Costs
                1,  # A1_Expected_Number_of_Deaths
                50000000,  # A2_Expected_Annual_Damage
                50000000,  # A2_Dike_Investment_Costs
                1,  # A2_Expected_Number_of_Deaths
                50000000,  # A3_Expected_Annual_Damage
                50000000,  # A3_Dike_Investment_Costs
                1,  # A3_Expected_Number_of_Deaths
                50000000,  # A4_Expected_Annual_Damage
                50000000,  # A4_Dike_Investment_Costs
                1,  # A4_Expected_Number_of_Deaths
                50000000,  # A5_Expected_Annual_Damage
                50000000,  # A5_Dike_Investment_Costs
                1,  # A5_Expected_Number_of_Deaths
                500000000,  # RfR_Total_Costs
                500000000,  # Expected_Evacuation_Costs
                ]

    # save these epsilons for further multi-scenario MORDM steps
    eps_dict = {"epsilons": epsilons}
    df_eps = pd.DataFrame(eps_dict, index=[*range(17)])
    eps_file_path = os.path.join("data", "optimize_results", "epsilons.csv")
    df_eps.to_csv(eps_file_path)
    print("Epsilons are determined and saved.")

    # set number of function evaluations
    # this value needs to be large enough for converged solutions, but also limit computing time
    nfe = 100000

    # search for optimized results per scenario
    # set number of seeds to increase the variance in solution spaces
    number_of_seeds = 3
    print(f"Optimization will run for {len(scenarios)} scenarios, {number_of_seeds} seeds and {nfe} NFEs:\n")
    for scenario in scenarios:
        optimize_scenarios(scenario, nfe, model, epsilons, number_of_seeds)

    # end of script
    print("\nMulti-MORDM optimization script is finished.")
    results_folder_path = os.path.join("data", "optimize_results")
    print(f"Results are exported to: {os.path.abspath(results_folder_path)}")


