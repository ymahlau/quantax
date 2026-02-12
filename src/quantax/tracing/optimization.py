from quantax.unitful.tracer import UnitfulTracer, OperatorNode
from quantax.functional.constraints import CONSTRAINTS_DICT
from quantax.core.glob import TraceData
from ortools.math_opt.python import mathopt


def solve_scale_assignment(
    args,
    kwargs,
    trace_args,
    trace_kwargs,
    trace_output,
    trace_data: TraceData,
) -> dict[int | tuple[str, int, int], int]:
    # TODO: handle zero inputs
    model = mathopt.Model(name="Scale_Optimization")
    var_ops_dict, var_tracer_dict = collect_variables(model, trace_data)
    
    add_operator_constraints(
        model=model,
        trace_data=trace_data,
        var_ops_dict=var_ops_dict,
        var_tracer_dict=var_tracer_dict,
    )
    add_objective(
        model=model,
        var_ops_dict=var_ops_dict,
    )
    
    result = mathopt.solve(model, mathopt.SolverType.GSCIP)
    assert result.termination.reason == mathopt.TerminationReason.OPTIMAL
    primal_solution = result.solutions[0].primal_solution
    assert primal_solution is not None
    variable_assignment = primal_solution.variable_values
    
    result: dict[int | tuple[str, int, int], int] = {}
    for k, v in var_ops_dict.items():
        result[k] = round(variable_assignment[v])
    for k, v in var_tracer_dict.items():
        result[k] = round(variable_assignment[v])
    return result


def collect_variables(
    model: mathopt.Model,
    trace_data: TraceData,
) -> tuple[
    dict[tuple[str, int, int], mathopt.Variable],
    dict[int, mathopt.Variable],
]:
    # collect tracer nodes
    var_tracer_dict: dict[int, mathopt.Variable] = {}
    for t in trace_data.tracer_nodes:
        cur_var = model.add_integer_variable(name=str(t.id))
        var_tracer_dict[t.id] = cur_var
    
    # collect op nodes
    var_ops_dict = {}
    for n in trace_data.operator_nodes:
        for k, a in n.args.items():
            if isinstance(a, UnitfulTracer):
                key = (k, a.id, n.id)
                cur_var = model.add_integer_variable(name=str(key))
                var_ops_dict[key] = cur_var
                
    return var_ops_dict, var_tracer_dict
    

def add_operator_constraints(
    model: mathopt.Model,
    trace_data: TraceData,
    var_ops_dict: dict[tuple[str, int, int], mathopt.Variable],
    var_tracer_dict: dict[int, mathopt.Variable],
):
    for n in trace_data.operator_nodes:
        c_fun = CONSTRAINTS_DICT[n.op_name]
        c_outs = [var_tracer_dict[t.id] for t in n.output_tracer]
        # TODO: handle non-traced inputs
        c_kwargs = {}
        for k, t in n.args.items():
            # for each operation, the input is the variable plus optional scale adjustment factor
            factor_var = var_ops_dict[(k, t.id, n.id)]
            input_var = var_tracer_dict[t.id]
            c_kwargs[k] = input_var + factor_var
        c_kwargs = {k: var_tracer_dict[t.id] for k, t in n.args.items()}
        c_kwargs['out'] = c_outs
        cur_constraints = c_fun(**c_kwargs)
        for c in cur_constraints:
            model.add_linear_constraint(c)

    
def add_objective(
    model: mathopt.Model,
    var_ops_dict: dict[tuple[str, int, int], mathopt.Variable],
):
    new_vars = []
    for k, v in var_ops_dict.items():
        # minimize absolute value of variables (technically non-linear, but can be achieved using slack variable)
        helper = model.add_variable(lb=0, name=f"helper_{k}")
        model.add_linear_constraint(helper >= v)
        model.add_linear_constraint(helper >= -v)
        new_vars.append(helper)
    # minimize sum of helper variables (equivalent to sum of absolute original values)
    model.minimize(sum(new_vars))
