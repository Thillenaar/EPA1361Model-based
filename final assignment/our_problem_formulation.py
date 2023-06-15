from dike_model_function import DikeNetwork
from problem_formulation import sum_over
from ema_workbench import (
    Model,
    MultiprocessingEvaluator,
    ScalarOutcome,
    IntegerParameter,
    RealParameter,
    CategoricalParameter,
    Scenario,
)

# Load the model:
def get_model_for_problem_formulation():
    function = DikeNetwork()
    model = Model("dikesnet", function=function)

    model.uncertainties = [
        RealParameter("A.1_Bmax", 30, 350),
        RealParameter("A.1_pfail", 0, 1),
        CategoricalParameter("A.1_Brate", [1.0, 1.5, 10]),
        RealParameter("A.2_Bmax", 30, 350),
        RealParameter("A.2_pfail", 0, 1),
        CategoricalParameter("A.2_Brate", [1.0, 1.5, 10]),
        RealParameter("A.3_Bmax", 30, 350),
        RealParameter("A.3_pfail", 0, 1),
        CategoricalParameter("A.3_Brate", [1.0, 1.5, 10]),
        RealParameter("A.4_Bmax", 30, 350),
        RealParameter("A.4_pfail", 0, 1),
        CategoricalParameter("A.4_Brate", [1.0, 1.5, 10]),
        RealParameter("A.5_Bmax", 30, 350),
        RealParameter("A.5_pfail", 0, 1),
        CategoricalParameter("A.5_Brate", [1.0, 1.5, 10]),
        CategoricalParameter("discount rate 0", (1.5, 2.5, 3.5, 4.5)),
        CategoricalParameter("discount rate 1", (1.5, 2.5, 3.5, 4.5)),
        CategoricalParameter("discount rate 2", (1.5, 2.5, 3.5, 4.5)),
        IntegerParameter("A.0_ID flood wave shape", 0, 132),
    ]

    model.levers = [
        IntegerParameter("EWS_DaysToThreat", 0, 4),
        IntegerParameter("rfr_0_t0", 0, 1, variable_name="0_RfR 0"),
        IntegerParameter("rfr_0_t1", 0, 1, variable_name="0_RfR 1"),
        IntegerParameter("rfr_0_t2", 0, 1, variable_name="0_RfR 2"),
        IntegerParameter("rfr_1_t0", 0, 1, variable_name="1_RfR 0"),
        IntegerParameter("rfr_1_t1", 0, 1, variable_name="1_RfR 1"),
        IntegerParameter("rfr_1_t2", 0, 1, variable_name="1_RfR 2"),
        IntegerParameter("rfr_2_t0", 0, 1, variable_name="2_RfR 0"),
        IntegerParameter("rfr_2_t1", 0, 1, variable_name="2_RfR 1"),
        IntegerParameter("rfr_2_t2", 0, 1, variable_name="2_RfR 2"),
        IntegerParameter("rfr_3_t0", 0, 1, variable_name="3_RfR 0"),
        IntegerParameter("rfr_3_t1", 0, 1, variable_name="3_RfR 1"),
        IntegerParameter("rfr_3_t2", 0, 1, variable_name="3_RfR 2"),
        IntegerParameter("rfr_4_t0", 0, 1, variable_name="4_RfR 0"),
        IntegerParameter("rfr_4_t1", 0, 1, variable_name="4_RfR 1"),
        IntegerParameter("rfr_4_t2", 0, 1, variable_name="4_RfR 2"),
        IntegerParameter(
            "A1_DikeIncrease_t0", 0, 10, variable_name="A.1_DikeIncrease 0"
        ),
        IntegerParameter(
            "A1_DikeIncrease_t1", 0, 10, variable_name="A.1_DikeIncrease 1"
        ),
        IntegerParameter(
            "A1_DikeIncrease_t2", 0, 10, variable_name="A.1_DikeIncrease 2"
        ),
        IntegerParameter(
            "A2_DikeIncrease_t0", 0, 10, variable_name="A.2_DikeIncrease 0"
        ),
        IntegerParameter(
            "A2_DikeIncrease_t1", 0, 10, variable_name="A.2_DikeIncrease 1"
        ),
        IntegerParameter(
            "A2_DikeIncrease_t2", 0, 10, variable_name="A.2_DikeIncrease 2"
        ),
        IntegerParameter(
            "A3_DikeIncrease_t0", 0, 10, variable_name="A.3_DikeIncrease 0"
        ),
        IntegerParameter(
            "A3_DikeIncrease_t1", 0, 10, variable_name="A.3_DikeIncrease 1"
        ),
        IntegerParameter(
            "A3_DikeIncrease_t2", 0, 10, variable_name="A.3_DikeIncrease 2"
        ),
        IntegerParameter(
            "A4_DikeIncrease_t0", 0, 10, variable_name="A.4_DikeIncrease 0"
        ),
        IntegerParameter(
            "A4_DikeIncrease_t1", 0, 10, variable_name="A.4_DikeIncrease 1"
        ),
        IntegerParameter(
            "A4_DikeIncrease_t2", 0, 10, variable_name="A.4_DikeIncrease 2"
        ),
        IntegerParameter(
            "A5_DikeIncrease_t0", 0, 10, variable_name="A.5_DikeIncrease 0"
        ),
        IntegerParameter(
            "A5_DikeIncrease_t1", 0, 10, variable_name="A.5_DikeIncrease 1"
        ),
        IntegerParameter(
            "A5_DikeIncrease_t2", 0, 10, variable_name="A.5_DikeIncrease 2"
        ),
    ]

    direction = ScalarOutcome.MINIMIZE
    model.outcomes = [
        ScalarOutcome(
            "A1_Total_Costs",
            variable_name=("A.1_Expected Annual Damage", "A.1_Dike Investment Costs"),
            function=sum_over,
            kind=direction,
        ),
        ScalarOutcome(
            "A1_Expected_Number_of_Deaths",
            variable_name="A.1_Expected Number of Deaths",
            function=sum_over,
            kind=direction,
        ),
        ScalarOutcome(
            "A2_Total_Costs",
            variable_name=("A.2_Expected Annual Damage", "A.2_Dike Investment Costs"),
            function=sum_over,
            kind=direction,
        ),
        ScalarOutcome(
            "A2_Expected_Number_of_Deaths",
            variable_name="A.2_Expected Number of Deaths",
            function=sum_over,
            kind=direction,
        ),
        ScalarOutcome(
            "A3_Total_Costs",
            variable_name=("A.3_Expected Annual Damage", "A.3_Dike Investment Costs"),
            function=sum_over,
            kind=direction,
        ),
        ScalarOutcome(
            "A3_Expected_Number_of_Deaths",
            variable_name="A.3_Expected Number of Deaths",
            function=sum_over,
            kind=direction,
        ),
        ScalarOutcome(
            "A4_Total_Costs",
            variable_name=("A.4_Expected Annual Damage", "A.4_Dike Investment Costs"),
            function=sum_over,
            kind=direction,
        ),
        ScalarOutcome(
            "A4_Expected_Number_of_Deaths",
            variable_name="A.4_Expected Number of Deaths",
            function=sum_over,
            kind=direction,
        ),
        ScalarOutcome(
            "A5_Total_Costs",
            variable_name=("A.5_Expected Annual Damage", "A.5_Dike Investment Costs"),
            function=sum_over,
            kind=direction,
        ),
        ScalarOutcome(
            "A5_Expected_Number_of_Deaths",
            variable_name="A.5_Expected Number of Deaths",
            function=sum_over,
            kind=direction,
        ),
        ScalarOutcome(
            "RfR_Total_Costs",
            variable_name="RfR Total Costs",
            function=sum_over,
            kind=direction,
        ),
        ScalarOutcome(
            "Expected_Evacuation_Costs",
            variable_name="Expected Evacuation Costs",
            function=sum_over,
            kind=direction,
        ),
    ]

    return model, function.planning_steps
