# Summary of model in- and outputs
Main files to inspect in order to understand model are [dike_model_function.py](dike_model_function.py), 
[problem_formulation.py](problem_formulation.py) and [dike_model_simulation.py](dike_model_simulation.py).

The files starting with 'funs_(...)' represent technical representations of aspects the system. 

The [our_problem_formulation.py](our_problem_formulation.py) file is created and used to scope and frame the problem
in stead of the already provided options in [problem_formulation.py](problem_formulation.py).

The outcome files of the model are represented by the python file names starting with:
multi_MORDM_(...)'. In these files, multi MORDM is used to find interesting policies to recommend
to Dike Ring 1 and 2.

## Uncertainties
The model includes the following uncertainties:
1. Breach max (Bmax): real number in range [30, 350], in meters
2. Probability of failure (pfail): real number in range [0, 1], dimensionless
3. Breach rate (Brate): categorical among (1.0, 1.5, 10) in meter/day
4. Discount rate (discount_rate): categorical among (1.5, 2.5, 3.5, 4.5), dimensionless
5. Wave shape: integer in range [0, 132]

Note that Bmax, pfail and Brate are disaggregated over five locations (=dike rings), 
and that discount rate is disaggregated over three planning phases.

This means we have 3 * 5 + 1 * 3 + 1 = 19 uncertainty parameters

## Levers
The model includes the following policy levers:
1. Dike Increase: real number in range [0, 10], in decimeters
2. Room for River project: boolean, indicating whether project is implemented
3. Early Warning System: integer in range [0, 4], in days

Note that all three parameters are disaggregated over both the three planning phases
as well as the five locations (dike rings).

This means we have 3 * 3 * 5 = 45 policy levers

Together, the uncertainties and policy levers span a total 54 variable parameters 
that together define an experiment

## Output
This section needs to be expanded, but first impression is that there are 6 ways of exporting results,
based on six different problem definitions:

 0. Total cost, and casualties
 1. Expected damages, costs, and casualties
 2. expected damages, dike investment costs, rfr costs, evacuation cost, and casualties
 3. costs and casualties disaggregated over dike rings, and room for the river and evacuation costs
 4. Expected damages, dike investment cost and casualties disaggregated over dike rings and room for the river and evacuation costs
 5. disaggregate over time and space

See [results.ipynb](notebooks/results.ipynb)  for what results look like, when problem definition is set to 5
