from ema_workbench import MultiprocessingEvaluator, ema_logging
from ema_workbench.em_framework.evaluators import BaseEvaluator

from ema_workbench.em_framework.optimization import (ArchiveLogger,
                                                         EpsilonProgress,
                                                         to_problem, epsilon_nondominated)

ema_logging.log_to_stderr(ema_logging.INFO)

from ema_workbench import (
    Model,
    MultiprocessingEvaluator,
    ScalarOutcome,
    IntegerParameter,
    optimize,
    Scenario,
    SequentialEvaluator
)
from ema_workbench import Constraint
from ema_workbench.em_framework.optimization import EpsilonProgress
from ema_workbench.util import ema_logging
import pandas as pd
from problem_formulation import get_model_for_problem_formulation
import matplotlib.pyplot as plt
import seaborn as sns


#CRITERIA SELECTION BASED ON SUBSPACE PARTITIONING


def optimize_scenarios(scenario, nfe, model, epsilons):
    results = []
    convergences = []

    with MultiprocessingEvaluator(model) as evaluator:
        for i in range(5):
            convergence_metrics = [
                ArchiveLogger(
                    "data/archives",
                    # filter model levers and outcomes names on invalid python identifiers
                    [filter_invalid_identifiers(l.name) for l in model.levers],
                    [filter_invalid_identifiers(o.name) for o in model.outcomes],
                    base_filename=f"multi_MORDM_{scenario.name}_seed_{i}.tar.gz",
                ),
                EpsilonProgress(),
            ]

            result, convergence = evaluator.optimize(nfe=nfe, searchover='levers',
                                                        convergence=convergence_metrics,
                                                        epsilons=epsilons,
                                                        reference=scenario,
                                                        seed=i)

            # rename column names for optimization.py (ema_workbench library code)
            for column in result.columns:
                new_name = column.replace(" ", "")
                new_name = new_name.replace(".", "")
                result = result.rename(columns={column: new_name})
                if column[0].isnumeric():
                    result = result.rename(columns={new_name: "d" + new_name})

            # save results and convergence in folder (optimize_results)
            result.to_excel(f"data/optimize_results/results_scenario_{scenario.name}_seed_{i}.xlsx")
            convergence.to_excel(f"data/optimize_results/convergence_scenario_{scenario.name}_seed_{i}.xlsx")

            results.append(result)
            convergences.append(convergence)

    # merge the results using a non-dominated sort
    # reference_set = epsilon_nondominated(results, epsilons, problem)
    #print(reference_set)
    # return reference_set, convergences

# Function to filter string on invalid Python identifiers
def filter_invalid_identifiers(given_string):
    new_string = given_string.replace(" ", "")
    new_string = new_string.replace(".", "")

    if given_string[0].isnumeric():
        new_string = "d" + new_string

    return new_string


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    model, steps = get_model_for_problem_formulation(3)

    df_scenario_discover = pd.read_excel(r'data/scenario_discovery/scenario.xlsx')
    #print(df_scenario_discover)
    scenarios = []
    for index, row in df_scenario_discover.iterrows():


        reference_values = {
            "Bmax": df_scenario_discover["Bmax"][index],
            "Brate": df_scenario_discover["Brate"][index],
            "pfail": df_scenario_discover["pfail"][index],
            "discount rate 0": df_scenario_discover["discount rate 0"][index],
            "discount rate 1": df_scenario_discover["discount rate 1"][index],
            "discount rate 2": df_scenario_discover["discount rate 2"][index],
            "ID flood wave shape": df_scenario_discover["ID flood wave shape"][index],
        }

        scen1 = {}

        for key in model.uncertainties:
            name_split = key.name.split("_")

            if len(name_split) == 1:
                scen1.update({key.name: reference_values[key.name]})

            else:
                scen1.update({key.name: reference_values[name_split[1]]})

        ref_scenario = Scenario(index, **scen1)
        scenarios.append(ref_scenario)

    #print(scenarios)
    convergence_metrics = [EpsilonProgress()]
    # print(len(model.outcomes))
    # for i in model.outcomes:
    #     print(i)
    epsilons = [10000000] * len(model.outcomes)
    # espilons = [1e3] * len(model.outcomes) #originele setting

    nfe = 10  # 200 #proof of principle only, way to low for actual use

    results = []

    for scenario in scenarios:  #
        # print(scenario)
        # print(scenario.name)

        optimize_scenarios(scenario, 1e2, model, epsilons)

        # note that 100000 nfe is again rather low to ensure proper convergence
        # results.append(optimize_scenarios(scenario, 1e2, model, epsilons))  # 1e5

    # outcomes = results.loc[:,
    #            ['A.1 Expected Annual Damage', 'A.1_Expected Number of Deaths', 'A.2 Expected Annual Damage',
    #             'A.2_Expected Number of Deaths', 'RfR Total Costs', 'Expected Evacuation Costs']]
    # 'A.3 Expected Annual Damage', 'A.3_Expected Number of Deaths', 'A.4 Expected Annual Damage', 'A.4_Expected Number of Deaths', 'A.5 Expected Annual Damage', 'A.5_Expected Number of Deaths', 'A.1 Dike Investment Costs', 'A.2 Dike Investment Costs', 'A.3 Dike Investment Costs', 'A.4 Dike Investment Costs', 'A.5 Dike Investment Costs']]
