import os
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
    regret["policy"] = x.policy
    return regret


### Run script ###
if __name__ == "__main__":

    print("\nMulti-MORDM robustness script is running...\n")

    # get computed experiment data
    experiments_file_path = os.path.join("data", "robustness_experiments", "experiments.csv")
    df_experiments = pd.read_csv(experiments_file_path)
    outcomes_file_path = os.path.join("data", "robustness_experiments", "outcomes.csv")
    df_outcomes = pd.read_csv(outcomes_file_path)
    number_file_path = os.path.join("data", "robustness_experiments", "number_of_experiments.csv")
    df_number_of_experiments = pd.read_csv(number_file_path)

    # convert outcomes dataframe to dictionary
    outcomes = {}
    for column in df_outcomes:
        outcomes[column] = np.array(df_outcomes[column].tolist())

    # get number of experiments
    number_of_experiments = df_number_of_experiments["number of experiments"][0]

    print("Previously created experiments and outcomes are loaded.")


    ### Domain criterion ###
    print("\nThe domain criterion is checked:")
    # set thresholds for outcome preferences
    # threshold values should be set regarding the scale of the outcome
    thresholds = {'A1_Expected_Annual_Damage': 10000000,
                  'A1_Dike_Investment_Costs': 20000000, 'A1_Expected_Number_of_Deaths': 1,
                  'A2_Expected_Annual_Damage': 10000000,
                  'A2_Dike_Investment_Costs': 20000000, 'A2_Expected_Number_of_Deaths': 1,
                   'A3_Expected_Annual_Damage': 10000000,
                   'A3_Dike_Investment_Costs': 20000000, 'A3_Expected_Number_of_Deaths': 1,
                   'A4_Expected_Annual_Damage': 10000000,
                   'A4_Dike_Investment_Costs': 20000000, 'A4_Expected_Number_of_Deaths': 1,
                   'A5_Expected_Annual_Damage': 10000000,
                   'A5_Dike_Investment_Costs': 20000000, 'A5_Expected_Number_of_Deaths': 1,
                   'RfR_Total_Costs': 100000000, 'Expected_Evacuation_Costs': 1000000}

    print("Domain criterion thresholds are set.")

    # compare experiment results to thresholds
    experiments = df_experiments
    overall_scores = {}
    for policy in experiments.policy.unique():
        logical = experiments.policy == policy
        scores = {}
        for k, v in outcomes.items():
            try:
                # because we want to minimize the outcomes, <= is used to assess whether the value is
                # under the given threshold
                n = np.sum(v[logical] <= thresholds[k])
            except KeyError:
                continue
            scores[k] = n / number_of_experiments
        overall_scores[policy] = scores

    # the calculated scores indicate in how many percent of the new scenarios, the outcomes stay under their threshold
    # store the scores
    overall_scores = pd.DataFrame(overall_scores).T
    scores_file_path = os.path.join("data", "robustness_results", "overall_scores.xlsx")
    overall_scores.to_excel(scores_file_path)

    print("Domain criterion scores are calculated.")
    print("Domain criterion plot will be created and shown.")

    # plot threshold compliance
    limits = parcoords.get_limits(overall_scores)
    paraxes = parcoords.ParallelAxes(limits)
    # setup legend and colors
    legend_handles = []
    hsv = plt.get_cmap("hsv")
    colors = hsv(np.linspace(0, 1.0, len(overall_scores)+1))
    # process data and set legend info
    for i, (index, row) in enumerate(overall_scores.iterrows()):
        label = f"Policy {str(index)}"
        clr = colors[i]
        paraxes.plot(row.to_frame().T, color=clr, label=label)
        patch = mpatches.Patch(color=clr, label=label)
        legend_handles.append(patch)

    # format figure
    fig = plt.gcf()
    fig.set_size_inches(20, 8)
    fig.subplots_adjust(bottom=0.5, left=0.03, right=0.75, top=0.95)
    fig.suptitle("Domain Criterion", fontsize=16, fontweight=800, y=0.98)
    fig.legend(handles=legend_handles, loc="upper right", fontsize=18)

    # save figure
    domain_plot_path = os.path.join("data", "robustness_results", "threshold_compliance.png")
    plt.savefig(domain_plot_path)
    # show figure
    plt.show()


    ### Regret criterion ###
    print("\nThe regret criterion is checked:")
    # setup a dataframe for the outcomes
    # we add scenario and policy as additional columns
    # we need scenario because regret is calculated on a scenario by scenario basis
    # we add policy because we need to get the maximum regret for each policy.

    # add scenario and policy info to the experiment outcomes
    outcomes = pd.DataFrame(outcomes)
    outcomes["scenario"] = experiments.scenario
    outcomes["policy"] = experiments.policy

    # calculate regret scenario by scenario
    regret = outcomes.groupby("scenario", group_keys=False).apply(calculate_regret)

    # for all policies, calculate max regret
    max_regret = regret.groupby("policy").max()

    # reorder columns of max regret
    max_regret = max_regret[['A1_Expected_Annual_Damage', 'A1_Dike_Investment_Costs',
                             'A1_Expected_Number_of_Deaths', 'A2_Expected_Annual_Damage',
                             'A2_Dike_Investment_Costs', 'A2_Expected_Number_of_Deaths',
                             'A3_Expected_Annual_Damage', 'A3_Dike_Investment_Costs',
                             'A3_Expected_Number_of_Deaths', 'A4_Expected_Annual_Damage',
                             'A4_Dike_Investment_Costs', 'A4_Expected_Number_of_Deaths',
                             'A5_Expected_Annual_Damage', 'A5_Dike_Investment_Costs',
                             'A5_Expected_Number_of_Deaths', 'RfR_Total_Costs',
                             'Expected_Evacuation_Costs']]

    # save max regret file
    regret_file_path = os.path.join("data", "robustness_results", "regret_values.xlsx")
    max_regret.to_excel(regret_file_path)

    print("Outcomes of interest for the regret criterion are set.")
    print("Regret criterion plot will be created and shown.")

    # plot regret criterion
    limits = parcoords.get_limits(max_regret)
    paraxes = parcoords.ParallelAxes(max_regret)
    paraxes.plot(max_regret, lw=1, alpha=0.75)
    # setup legend and colors
    legend_handles = []
    color = plt.get_cmap("hsv")
    colors = color(np.linspace(0, 1.0, len(max_regret)+1))
    # process data and set legend info
    for i, (index, row) in enumerate(max_regret.iterrows()):
        label = f"Policy {str(index)}"
        clr = colors[i]
        paraxes.plot(row.to_frame().T, color=clr, label=label)
        patch = mpatches.Patch(color=clr, label=label)
        legend_handles.append(patch)

    # format figure
    fig = plt.gcf()
    fig.set_size_inches(26, 21)
    fig.subplots_adjust(bottom=0.5, left=0.05, right=0.95, top=0.95)
    fig.suptitle("Regret Criterion", fontsize=16, fontweight=800, y=0.98)
    fig.legend(handles=legend_handles, loc="lower center", fontsize=18)

    # save figure
    regret_plot_path = os.path.join("data", "robustness_results", "min_max_regret.png")
    plt.savefig(regret_plot_path)
    # show figure
    plt.show()

    # end of script
    print("\nMulti-MORDM robustness script is finished.")
    results_folder_path = os.path.join("data", "robustness_results")
    print(f"The robustness criterion plots are exported to: {os.path.abspath(results_folder_path)}")

