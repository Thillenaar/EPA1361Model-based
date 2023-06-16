from multi_MORDM_convergence import reevaluation_results
import numpy as np
import pandas as pd

experiments, outcomes = reevaluation_results

thresholds = {'A.1 Expected Annual Damage':1000000000, 'A.1 Dike Investment Costs':500000000,
       'A.1_Expected Number of Deaths':10, 'A.2 Expected Annual Damage':1000000000,
       'A.2 Dike Investment Costs':500000000, 'A.2_Expected Number of Deaths':10,
       'RfR Total Costs':50000000, 'Expected Evacuation Costs': 10000000}

overall_scores = {}
for policy in experiments.policy.unique():
    logical = experiments.policy == policy
    scores = {}
    for k, v in outcomes.items():
        try:
            n = np.sum(v[logical] >= thresholds[k])
        except KeyError:
            continue
        scores[k] = n / 1000
    overall_scores[policy] = scores

overall_scores = pd.DataFrame(overall_scores).T