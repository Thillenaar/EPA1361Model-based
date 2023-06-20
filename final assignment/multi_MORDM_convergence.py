import os
import pandas as pd
from ema_workbench import Policy
from our_problem_formulation import get_model_for_problem_formulation
from ema_workbench.em_framework.optimization import (ArchiveLogger, EpsilonProgress,
                                                        to_problem, epsilon_nondominated,
                                                        rebuild_platypus_population)
from ema_workbench import (MultiprocessingEvaluator, Scenario, SequentialEvaluator)
from platypus import Hypervolume
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# Function to create a list of scenarios from the scenario discovery selection
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

    # get number of seeds per scenario
    df_seeds = pd.read_csv("data/optimize_results/number_of_seeds.csv")
    number_of_seeds = df_seeds["number of seeds"][0]

    # loop through all files in optimize_results folder
    for file in get_files(folder_path):
        # process results
        if "results" in file:
            df_result = pd.read_csv(folder_path + "/" + file)
            results.append(df_result)

        # process convergences
        elif "convergence" in file:
            df_convergence = pd.read_csv(folder_path + "/" + file)
            convergences.append(df_convergence)

        # process the file as long as it contains results/convergences
        if ("results" in file) or ("convergence" in file):
            # get processed scenario name
            strings = file.split("_")
            scenario_name = strings[2]
            # save scenario name
            scenarios.append(scenario_name)

            # get seed (and remove ".csv" extension with .split())
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
    step = number_of_seeds
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
    model, steps = get_model_for_problem_formulation()
    # get scenarios from scenario discovery
    df_scenario_discovery = pd.read_csv(r'data/scenario_discovery/reference_scenarios.csv')
    # create list of scenario objects
    scenarios = create_scenarios(df_scenario_discovery)
    # create problem object
    problem = to_problem(model, searchover="levers")
    # retrieve specified epsilons
    df_eps = pd.read_csv("data/optimize_results/epsilons.csv")
    eps_list = df_eps["epsilons"]
    epsilons = [float(x) for x in eps_list]

    # retrieve results from multi_MORDM_optimize.py (in folder: optimize_results)
    results_per_scenario, convergences_per_scenario, number_of_scenarios = retrieve_results(r"data/optimize_results")

    # search for non dominance in results per scenario
    dominant_results = []
    for i in range(number_of_scenarios):
        reference_set = sort_non_dominance(results_per_scenario[i], epsilons, problem)
        # save dominant results and store convergences of previous optimized results
        dominant_results.append((reference_set, convergences_per_scenario[i]))

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

    # fig, (ax1, ax2) = plt.subplots(ncols=2)
    fig, ax2 = plt.subplots(ncols=1)

    colors = sns.color_palette()

    legend_items = []
    for (scenario_name, scores), color in zip(convergence.groupby("scenario"), colors):
        # we use this for a custom legend
        legend_items.append((mpl.lines.Line2D([0, 0], [1, 1], c=color), scenario_name))
        for seed, score in scores.groupby("seed"):
            # ax1.plot(score.nfe, score.hypervolume, c=color, lw=1)
            ax2.plot(score.nfe, score.epsilon_progress, c=color, lw=1)

    # ax1.set_ylabel('hypervolume')
    # ax1.set_xlabel('nfe')
    ax2.set_ylabel('$\epsilon$ progress')
    ax2.set_xlabel('nfe')

    # create our custom legend
    artists, labels = zip(*legend_items)
    fig.legend(artists, labels, bbox_to_anchor=(1, 0.9))
    # save figure
    plt.savefig("data/robustness_experiments/convergence_figure.png")
    # show figure
    plt.show()


    policies = []
    for i, (result, _) in enumerate(dominant_results):
        result = result.iloc[:, 0:31]
        # print('result')
        # print(result)
        for j, row in result.iterrows():
            policy = Policy(f'scenario {i} option {j}', **row.to_dict())
            policies.append(policy)

    # test policies on new scenarios
    number_of_experiments = 1000
    with MultiprocessingEvaluator(model) as evaluator:
        reevaluation_results = evaluator.perform_experiments(number_of_scenarios, policies=policies)

    experiments, outcomes = reevaluation_results

    # save these experiments and outcomes
    experiments.to_csv("data/robustness_experiments/experiments.csv", index=False)
    df_outcomes = pd.DataFrame().from_dict(outcomes)
    df_outcomes.to_csv("data/robustness_experiments/outcomes.csv", index=False)

    # save number of experiments
    number_dict = {"number of experiments": number_of_experiments}
    df_number = pd.DataFrame(number_dict, index=[0])
    df_number.to_csv("data/robustness_experiments/number_of_experiments.csv")



### End notes ###

    #Kolomnamen die uit reference set komen hebben weer spatie trouwens --> veroorzaakt geen problemen.

    #Kijk naar het stukje code hierboven, vanaf lijn 225, waar we het domain criteria toepassen.
    #Let heel goed op. Uitkomsten, plots kloppen wel. Ligt aan de scores, die zijn genormaliseerd (tussen 0 en 1).
    #Je moet dan ook niet delen door 1000, maar door 10, als je maar 10 scenarios gebruikt.
    ##We kunnen die >= vervangen door <= als dat duidelijker is! Bij Kwakkel ging het om maximizen,
    #bij ons om minimizen.
    # Die code, en dus plot, geeft aan in hoeveel procent van de gevallen je threshold is overschreden
    # in dit geval door je policy over de verschillende scenarios. Wij kunnen daar ook percentage niet over
    # schreden van maken.
