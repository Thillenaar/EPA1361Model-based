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


#CRITERIA SELECTION BASED ON SUBSPACE PARTITIONING


def create_scenarios(df_scenario_discovery):
    scenarios = []
    for index, row in df_scenario_discovery.iterrows():

        reference_values = {}

        for i in range(1, 6, 1):
            reference_values[f"A.{i}_Bmax"] = df_scenario_discovery[f"A.{i}_Bmax"][index]
            reference_values[f"A.{i}_Brate"] = df_scenario_discovery[f"A.{i}_Brate"][index]
            reference_values[f"A.{i}_pfail"] = df_scenario_discovery[f"A.{i}_pfail"][index]

        reference_values["discount rate 0"] = df_scenario_discovery["discount rate 0"][index]
        reference_values["discount rate 1"] = df_scenario_discovery["discount rate 1"][index]
        reference_values["discount rate 2"] = df_scenario_discovery["discount rate 2"][index]
        reference_values["A.0_ID flood wave shape"] = df_scenario_discovery["A.0_ID flood wave shape"][index]

        scen1 = {}

        for key in model.uncertainties:
            scen1.update({key.name: reference_values[key.name]})

        ref_scenario = Scenario(index, **scen1)
        scenarios.append(ref_scenario)

    return scenarios


def optimize_scenarios(scenario, nfe, model, epsilons, number_of_seeds):

    # save number of seeds per scenario
    seeds_dict = {"number of seeds": number_of_seeds}
    df_seeds = pd.DataFrame(seeds_dict, index=[0])
    df_seeds.to_csv("data/optimize_results/number_of_seeds.csv")

    with MultiprocessingEvaluator(model) as evaluator:
        for i in range(number_of_seeds):
            convergence_metrics = [
                ArchiveLogger(
                    "data/archives",
                    # filter model levers and outcomes names on invalid python identifiers
                    [l.name for l in model.levers],
                    [o.name for o in model.outcomes],
                    base_filename=f"multi_MORDM_{scenario.name}_seed_{i}.tar.gz",
                ),
                EpsilonProgress(),
            ]

            result, convergence = evaluator.optimize(nfe=nfe, searchover='levers',
                                                        convergence=convergence_metrics,
                                                        epsilons=epsilons,
                                                        reference=scenario,
                                                        seed=i)

            # save results and convergence in folder (optimize_results)
            result.to_csv(f"data/optimize_results/results_scenario_{scenario.name}_seed_{i}.csv")
            convergence.to_csv(f"data/optimize_results/convergence_scenario_{scenario.name}_seed_{i}.csv")


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    # get model
    model, steps = get_model_for_problem_formulation()

    # get scenarios from scenario discovery
    df_scenario_discovery = pd.read_csv(r'data/scenario_discovery/reference_scenarios.csv')

    # create list of scenario from scenario discovery
    scenarios = create_scenarios(df_scenario_discovery)

    # specify epsilons
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
    # save these epsilons
    eps_dict = {"epsilons": epsilons}
    df_eps = pd.DataFrame(eps_dict, index=[*range(17)])
    df_eps.to_csv("data/optimize_results/epsilons.csv")

    # set number of functional evaluations
    # note that 100000 nfe is again rather low to ensure proper convergence
    nfe = 100000

    # search for optimized results per scenario
    number_of_seeds = 3
    for scenario in scenarios:
        optimize_scenarios(scenario, nfe, model, epsilons, number_of_seeds)

    # end of script
    print("\nMulti-MORDM optimization script is finished.")
    print("Results are exported to: final assignment/data/optimize_results")



    # outcomes = results.loc[:,
    #            ['A.1 Expected Annual Damage', 'A.1_Expected Number of Deaths', 'A.2 Expected Annual Damage',
    #             'A.2_Expected Number of Deaths', 'RfR Total Costs', 'Expected Evacuation Costs']]
    # 'A.3 Expected Annual Damage', 'A.3_Expected Number of Deaths', 'A.4 Expected Annual Damage', 'A.4_Expected Number of Deaths', 'A.5 Expected Annual Damage', 'A.5_Expected Number of Deaths', 'A.1 Dike Investment Costs', 'A.2 Dike Investment Costs', 'A.3 Dike Investment Costs', 'A.4 Dike Investment Costs', 'A.5 Dike Investment Costs']]
