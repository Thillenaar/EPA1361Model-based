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


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    model, steps = get_model_for_problem_formulation(3)

    reference_values = {
        "Bmax": 175,
        "Brate": 1.5,
        "pfail": 0.5,
        "discount rate 0": 3.5,
        "discount rate 1": 3.5,
        "discount rate 2": 3.5,
        "ID flood wave shape": 4,
    }
    scen1 = {}

    for key in model.uncertainties:
        name_split = key.name.split("_")

        if len(name_split) == 1:
            scen1.update({key.name: reference_values[key.name]})

        else:
            scen1.update({key.name: reference_values[name_split[1]]})

    ref_scenario = Scenario("reference", **scen1)

    convergence_metrics = [EpsilonProgress()]
    #print(len(model.outcomes))
    # for i in model.outcomes:
    #     print(i)
    espilon = [100000] * len(model.outcomes)
    #espilon = [1e3] * len(model.outcomes) #originele setting

    nfe = 10  # 200 #proof of principle only, way to low for actual use

    with MultiprocessingEvaluator(model) as evaluator:
        results, convergence = evaluator.optimize(
            nfe=nfe,
            searchover="levers",
            epsilons=espilon,
            convergence=convergence_metrics,
            reference=ref_scenario,
            constraints=[Constraint("A.1 Expected Annual Damage", outcome_names="A.1 Expected Annual Damage", function=lambda x:max(0, x - 500000)),
                         Constraint("A.2 Expected Annual Damage", outcome_names="A.2 Expected Annual Damage", function=lambda x:max(0, x - 500000))],
        )

    pd.set_option("display.max_columns", None)
    print(results)
    #Zelf toegevoegd op basis van code uitleg MOEA.
    from ema_workbench.analysis import parcoords
    outcomes = results.loc[:, ['A.1 Expected Annual Damage', 'A.1_Expected Number of Deaths', 'A.2 Expected Annual Damage', 'A.2_Expected Number of Deaths', 'RfR Total Costs', 'Expected Evacuation Costs']]
                               #'A.3 Expected Annual Damage', 'A.3_Expected Number of Deaths', 'A.4 Expected Annual Damage', 'A.4_Expected Number of Deaths', 'A.5 Expected Annual Damage', 'A.5_Expected Number of Deaths', 'A.1 Dike Investment Costs', 'A.2 Dike Investment Costs', 'A.3 Dike Investment Costs', 'A.4 Dike Investment Costs', 'A.5 Dike Investment Costs']]

    # plot lines and determine legend
    import matplotlib.patches as mpatches
    legend_handles = []
    import numpy as np
    hsv = plt.get_cmap('hsv')
    colors = hsv(np.linspace(0, 1.0, len(outcomes)))

    limits = parcoords.get_limits(outcomes)
    axes = parcoords.ParallelAxes(limits)

    for i, (index, row) in enumerate(outcomes.iterrows()):
        label = f"Policy {str(index)}"
        clr = colors[i]

        axes.plot(row.to_frame().T, color=clr, label=label)

        patch = mpatches.Patch(color=clr, label=label)
        legend_handles.append(patch)

    plt.legend(handles=legend_handles)
    plt.show()

    # export csv with results (levers and outcomes)
    results.to_csv("OptimizationResults.csv")

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True)
    fig, ax1 = plt.subplots(ncols=1)
    ax1.plot(convergence.epsilon_progress)
    ax1.set_xlabel("nr. of generations")
    #hoe kan dit niet in duizendtallen zijn?
    ax1.set_ylabel(r"$\epsilon$ progress")
    sns.despine()
    #op plot 2 kan hypervolume nog bij als assessing convergence tool
    #dan moeten ook bij de outcomes expected ranges worden aangegeven.
    plt.show()
