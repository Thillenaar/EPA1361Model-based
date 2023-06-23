import pandas as pd
from collections import defaultdict
from ema_workbench import Model, MultiprocessingEvaluator, Policy, Scenario

from ema_workbench.em_framework.evaluators import perform_experiments
from ema_workbench.em_framework.samplers import sample_uncertainties
from ema_workbench.util import ema_logging
import time
from our_problem_formulation import get_model_for_problem_formulation

# pick config from ['ref', 'subspace']
CONFIG = 'subspace'
# outcomes will be saved with _{CONFIG} appended to file name

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    dike_model, planning_steps = get_model_for_problem_formulation()

    # Build a user-defined scenario and policy:
    reference_values = {
        "Bmax": 175,
        "Brate": 1.5,
        "pfail": 0.5,
        "ID flood wave shape": 4,
        "planning steps": 2,
    }
    reference_values.update({f"discount rate {n}": 3.5 for n in planning_steps})
    scen1 = {}

    for key in dike_model.uncertainties:
        name_split = key.name.split("_")

        if len(name_split) == 1:
            scen1.update({key.name: reference_values[key.name]})

        else:
            scen1.update({key.name: reference_values[name_split[1]]})

    ref_scenario = Scenario("reference", **scen1)
    print(scen1)

    # no dike increase, no warning, none of the rfr
    zero_policy = {"DaysToThreat": 0}
    zero_policy.update({f"DikeIncrease_t{n}": 0 for n in planning_steps})
    zero_policy.update({f"rfr_t{n}": 0 for n in planning_steps})
    pol0 = {}

    for key in dike_model.levers:
        name_split = key.name.split("_")
        if len(name_split) > 2:
            s1, s2, s3 = name_split
            if s1 == 'rfr':
                pol0.update({key.name: zero_policy[f'{s1}_{s3}']})
            else:
                pol0.update({key.name: zero_policy[f'{s2}_{s3}']})
        else:
            s1, s2 = name_split
            pol0.update({key.name: zero_policy[s2]})


    policy0 = Policy("Policy 0", **pol0)

    # Call random scenarios or policies:
    #    n_scenarios = 5
    #    scenarios = sample_uncertainties(dike_model, 50)
    #    n_policies = 10

    # single run
    #    start = time.time()
    #    dike_model.run_model(ref_scenario, policy0)
    #    end = time.time()
    #    print(end - start)
    #    results = dike_model.outcomes_output

    # series run
    config_options = {'ref': {'models': dike_model, 'scenarios': ref_scenario, 'policies': 5},
                      'subspace': {'models': dike_model, 'scenarios': 10000, 'policies': policy0}}
    experiments, outcomes = perform_experiments(**config_options[CONFIG])

    # export
    experiments.to_csv(f'data/experiments/experiments_{CONFIG}.csv')

    packed = defaultdict(list)
    for exp in range(len(experiments)):
        outcome = dict(map(lambda x: (x[0], x[1][exp]), outcomes.items()))
        packed['experiment'].append(exp)
        for key, val in outcome.items():
            packed[key].append(val)

    df = pd.DataFrame(packed)
    df.set_index('experiment', drop=True, inplace=True)
    df.to_csv(f'data/experiments/outcomes_{CONFIG}.csv')