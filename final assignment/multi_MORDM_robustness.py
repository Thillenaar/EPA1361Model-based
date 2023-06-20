import matplotlib.pyplot as plt
import pandas as pd
from ema_workbench.analysis import parcoords
import matplotlib.patches as mpatches
import numpy as np


def calculate_regret(x):
    # policy is non numeric, so min is not defined for this
    # all the outcomes need to be minimized.
    best = x.min(numeric_only=True)

    regret = x.loc[:, best.index] - best

    # we add policy back into our regret dataframe
    # so we know the regret for each policy
    regret['policy'] = x.policy
    return regret


### Run script ###
if __name__ == "__main__":

    # get computed experiment data
    df_experiments = pd.read_csv("data/robustness_experiments/experiments.csv")
    df_outcomes = pd.read_csv("data/robustness_experiments/outcomes.csv")
    df_number_of_experiments = pd.read_csv("data/robustness_experiments/number_of_experiments.csv")

    # convert outcomes dataframe to dictionary
    outcomes = {}
    for column in df_outcomes:
        outcomes[column] = np.array(df_outcomes[column].tolist())

    # get number of experiments
    number_of_experiments = df_number_of_experiments["number of experiments"][0]


    ### Domain criterion ###
    # set thresholds for outcome preferences
    thresholds = {'A1_Expected_Annual_Damage': 1000000, 'A1_Dike_Investment_Costs': 5000000,
                  'A1_Expected_Number_of_Deaths': 1, 'A2_Expected_Annual_Damage': 1000000,
                  'A2_Dike_Investment_Costs': 5000000, 'A2_Expected_Number_of_Deaths': 1,
                  # 'A3_Expected_Annual_Damage': 1000000,
                  # 'A3_Dike_Investment_Costs': 5000000, 'A3_Expected_Number_of_Deaths': 1,
                  # 'A4_Expected_Annual_Damage': 1000000,
                  # 'A4_Dike_Investment_Costs': 5000000, 'A4_Expected_Number_of_Deaths': 1,
                  # 'A5_Expected_Annual_Damage': 1000000,
                  # 'A5_Dike_Investment_Costs': 5000000, 'A5_Expected_Number_of_Deaths': 1,
                  'RfR_Total_Costs': 200000000, 'Expected_Evacuation_Costs': 1000000}

    # compare experiment results to thresholds
    experiments = df_experiments
    overall_scores = {}
    for policy in experiments.policy.unique():
        logical = experiments.policy == policy
        scores = {}
        for k, v in outcomes.items():
            try:
                n = np.sum(v[logical] >= thresholds[k])
            except KeyError:
                continue
            scores[k] = n / number_of_experiments
        overall_scores[policy] = scores

    # store and show the scores
    overall_scores = pd.DataFrame(overall_scores).T
    overall_scores.to_excel("data/robustness_results/overall_scores.xlsx")

    # plot threshold compliance
    limits = parcoords.get_limits(overall_scores)
    axes = parcoords.ParallelAxes(limits)
    # setup legend and colors
    legend_handles = []
    hsv = plt.get_cmap('hsv')
    colors = hsv(np.linspace(0, 1.0, len(overall_scores)))
    # process data and set legend info
    for i, (index, row) in enumerate(overall_scores.iterrows()):
        label = f"Policy {str(index)}"
        clr = colors[i]
        axes.plot(row.to_frame().T, color=clr, label=label)
        patch = mpatches.Patch(color=clr, label=label)
        legend_handles.append(patch)

    # save figure
    plt.savefig("data/robustness_results/threshold_compliance.png")
    # plot figure
    plt.show()

    # save figure of legend
    plt.legend(handles=legend_handles, loc='center')
    plt.savefig("data/robustness_results/domain_legend.png")


    ### Regret criterion ###
    # setup a dataframe for the outcomes
    # we add scenario and policy as additional columns
    # we need scenario because regret is calculated on a scenario by scenario basis
    # we add policy because we need to get the maximum regret for each policy.

    outcomes = pd.DataFrame(outcomes)
    outcomes['scenario'] = experiments.scenario
    outcomes['policy'] = experiments.policy


    # we want to calculate regret on a scenario by scenario basis
    regret = outcomes.groupby('scenario', group_keys=False).apply(calculate_regret)

    # as last step, we calculate the maximum regret for each policy
    max_regret = regret.groupby('policy').max()

    # I reorder the columns
    max_regret = max_regret[['A1_Expected_Annual_Damage', 'A1_Dike_Investment_Costs',
                             'A1_Expected_Number_of_Deaths', 'A2_Expected_Annual_Damage',
                             'A2_Dike_Investment_Costs', 'A2_Expected_Number_of_Deaths',
                             'RfR_Total_Costs', 'Expected_Evacuation_Costs']]

    limits = parcoords.get_limits(max_regret)
    paraxes = parcoords.ParallelAxes(max_regret)
    paraxes.plot(max_regret, lw=1, alpha=0.75)
    # setup legend and colors
    legend_handles = []
    color = plt.get_cmap('hsv')
    colors = color(np.linspace(0, 1.0, len(max_regret)))
    # process data and set legend info
    for i, (index, row) in enumerate(max_regret.iterrows()):
        label = f"Policy {str(index)}"
        clr = colors[i]
        paraxes.plot(row.to_frame().T, color=clr, label=label)
        patch = mpatches.Patch(color=clr, label=label)
        legend_handles.append(patch)

    # save figure
    plt.savefig("data/robustness_results/min_max_regret.png")
    # plot figure
    plt.show()

    # save figure of legend
    plt.legend(handles=legend_handles, loc='center')
    plt.savefig("data/robustness_results/regret_legend.png")


