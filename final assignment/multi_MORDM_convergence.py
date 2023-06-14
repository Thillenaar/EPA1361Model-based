import os
import pandas as pd
import numpy as np
from ema_workbench import Policy
from problem_formulation import get_model_for_problem_formulation
from ema_workbench.em_framework.optimization import (ArchiveLogger, EpsilonProgress,
                                                        to_problem, epsilon_nondominated,
                                                        rebuild_platypus_population)
from ema_workbench import (Model, MultiprocessingEvaluator, ScalarOutcome, IntegerParameter,
                            optimize, Scenario, SequentialEvaluator)
from ema_workbench.analysis import parcoords
from platypus import Hypervolume
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches


# Function to read all available files in a list
def get_files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


# Function to get results and convergence data from optimizer script
def retrieve_results(folder_path):
    results = []
    convergences = []
    scenarios = []
    number_of_seeds = 0

    # loop through all files in optimize_results folder
    for file in get_files(folder_path):
        # process results
        if "results" in file:
            df_result = pd.read_excel(folder_path + "/" + file)
            results.append(df_result)

        # process convergence
        elif "convergence" in file:
            df_convergence = pd.read_excel(folder_path + "/" + file)
            convergences.append(df_convergence)

        # get processed scenario name
        strings = file.split("_")
        scenario_name = strings[2]
        # save scenario name
        scenarios.append(scenario_name)

        # get seed (and remove excel extension)
        this_seed = strings[4].split(".")[0]


    # calculate number of scenarios
    number_of_scenarios = len(set(scenarios))

    # split results list per scenario
    results_per_scenario = []
    convergences_per_scenario = []
    # first index of list
    start = 0
    # last index of list
    end = len(results)
    # number of items per split (number of seeds per scenario)
    step = 5
    # perform list splits
    for i in range(start, end, step):
        x = i
        results_per_scenario.append(results[x:x + step])
        convergences_per_scenario.append(convergences[x:x + step])

    # return variables
    return results_per_scenario, convergences_per_scenario, number_of_scenarios


# Function to merge results using the non dominance sort
def sort_non_dominance(results, epsilons, problem):

    # retrieve reference set from the non dominance sort function
    reference_set = epsilon_nondominated(results, epsilons, problem)

    return reference_set

# Function to calculate convergence_metrics
def calculate_convergence_metrics(problem, archives_file):
    hv = Hypervolume(minimum=[0, ] * len(model.outcomes), maximum=[12, ] * len(model.outcomes))
    archives = ArchiveLogger.load_archives(archives_file)
    metrics = []
    for nfe, archive in archives.items():
        population = rebuild_platypus_population(archive, problem)
        metrics.append(dict(hypervolume=hv.calculate(population), nfe=nfe))

    metrics = pd.DataFrame.from_dict(metrics)
    metrics.sort_values(by="nfe", inplace=True, ignore_index=True)
    return metrics


### Run script ###
if __name__ == "__main__":

    # get model reference
    model, steps = get_model_for_problem_formulation(3)
    # determine scenarios of model
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

    # create problem object
    problem = to_problem(model, searchover="levers")
    # create epsilons
    # check consistency of epsilons in all multi_MORDM_.py scripts!!!
    epsilons = [10000 ] * len(model.outcomes)

    # retrieve results from multi_MORDM_optimize.py (in folder: optimize_results)
    results_per_scenario, convergences_per_scenario, number_of_scenarios = retrieve_results(r"data/optimize_results")
    #print(results_per_scenario)
   # print(number_of_scenarios)
    # search for nondominance in results per scenario
    dominant_results = []
    for i in range(number_of_scenarios):
        print('results per scenario i')
        print(results_per_scenario[i])
        reference_set = sort_non_dominance(results_per_scenario[i], epsilons, problem)
        print('reference set')
        print(reference_set)
        # save dominant results (with nondominant convergence???)
        dominant_results.append((reference_set, convergences_per_scenario[i]))
        print('dominant results')
        print(dominant_results)

    # calculate convergence
    convergence_calculations = []
    for (refset, eps_progress), scenario in zip(dominant_results, scenarios):
        for seed, seed_eps in zip(range(5), eps_progress):
            archive_file = f"data/archives/multi_MORDM_{scenario.name}_seed_{seed}.tar.gz"
            metrics = calculate_convergence_metrics(problem, archive_file)
            metrics["seed"] = seed
            metrics["scenario"] = scenario.name
            metrics["epsilon_progress"] = seed_eps.epsilon_progress

            convergence_calculations.append(metrics)
    convergence = pd.concat(convergence_calculations, ignore_index=True)

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    colors = sns.color_palette()

    legend_items = []
    for (scenario_name, scores), color in zip(convergence.groupby("scenario"), colors):
        # we use this for a custom legend
        legend_items.append((mpl.lines.Line2D([0, 0], [1, 1], c=color), scenario_name))
        for seed, score in scores.groupby("seed"):
            ax1.plot(score.nfe, score.hypervolume, c=color, lw=1)
            ax2.plot(score.nfe, score.epsilon_progress, c=color, lw=1)

    ax1.set_ylabel('hypervolume')
    ax1.set_xlabel('nfe')
    ax2.set_ylabel('$\epsilon$ progress')
    ax2.set_xlabel('nfe')

    # create our custom legend
    artists, labels = zip(*legend_items)
    fig.legend(artists, labels, bbox_to_anchor=(1, 0.9))

    plt.show()

    #print('domin')
    #print(dominant_results[0][0].columns)
    #print(dominant_results)

    policies = []
    # print('results')
    # print(results)
    for i, (result, _) in enumerate(dominant_results):
        result = result.iloc[:, 0:31]
        # print('result')
        # print(result)
        for j, row in result.iterrows():
            policy = Policy(f'scenario {i} option {j}', **row.to_dict())
            #print('policy')
            #print(policy)
            #print('j')
            #print(j)
            policies.append(policy)
    #print('policies')
    #print(policies)
    #print(len(policies))

    with MultiprocessingEvaluator(model) as evaluator:
        reeevaluation_results = evaluator.perform_experiments(10, policies=policies)

    experiments, outcomes = reeevaluation_results

    thresholds = {'A.1 Expected Annual Damage': 100, 'A.1 Dike Investment Costs': 500,
                  'A.1_Expected Number of Deaths': 10000000000, 'A.2 Expected Annual Damage': 100,
                  'A.2 Dike Investment Costs': 500, 'A.2_Expected Number of Deaths': 100000000,
                  'RfR Total Costs': 500000000000, 'Expected Evacuation Costs': 1000000000}

    overall_scores = {}
    for policy in experiments.policy.unique():
        logical = experiments.policy == policy
        scores = {}
        for k, v in outcomes.items():
            try:
                n = np.sum(v[logical] >= thresholds[k])
            except KeyError:
                continue
            scores[k] = n / 10 #wij hadden hier nog 1000
        overall_scores[policy] = scores

    overall_scores = pd.DataFrame(overall_scores).T
    print('overall_scores')
    print(overall_scores)

    # plot lines and determine legend


    # legend_handles = []


    # hsv = plt.get_cmap('hsv')
    # colors = hsv(np.linspace(0, 1.0, len(outcomes)))
    #
    # limits = parcoords.get_limits(outcomes)
    # axes = parcoords.ParallelAxes(limits)
    #
    # for i, (index, row) in enumerate(outcomes.iterrows()):
    #     label = f"Policy {str(index)}"
    #     clr = colors[i]
    #
    #     axes.plot(row.to_frame().T, color=clr, label=label)
    #
    #     patch = mpatches.Patch(color=clr, label=label)
    #     legend_handles.append(patch)
    #
    # plt.legend(handles=legend_handles)
    # plt.show()

    pd.set_option("display.max_columns", None)
    limits = parcoords.get_limits(overall_scores)
    print('limits')
    print(limits)
    paraxes = parcoords.ParallelAxes(limits)
    paraxes.plot(overall_scores)
    plt.show()

    #Kolomnamen die uit reference set komen hebben weer spatie trouwens --> veroorzaakt geen problemen.

    #Kijk naar het stukje code hierboven, vanaf lijn 225, waar we het domain criteria toepassen.
    #Let heel goed op. Uitkomsten, plots kloppen wel. Ligt aan de scores, die zijn genormaliseerd (tussen 0 en 1).
    #Je moet dan ook niet delen door 1000, maar door 10, als je maar 10 scenarios gebruikt.
    ##We kunnen die >= vervangen door <= als dat duidelijker is! Bij Kwakkel ging het om maximizen,
    #bij ons om minimizen.
    # Die code, en dus plot, geeft aan in hoeveel procent van de gevallen je threshold is overschreden
    # in dit geval door je policy over de verschillende scenarios. Wij kunnen daar ook percentage niet over
    # schreden van maken.