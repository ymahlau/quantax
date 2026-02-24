from __future__ import annotations

from collections import defaultdict

import jax
import jax.numpy as jnp
from ortools.math_opt.python import mathopt

from quantax.core.fraction import IntFraction
from quantax.core.glob import GlobalTraceData, ScaleAssignment
from quantax.functional.collection import CONSTRAINTS_DICT
from quantax.unitful.tracer import UnitfulTracer
from quantax.unitful.unitful import Unitful

_ROUND_EPS = 1e-12

def solve_scale_assignment(
    trace_args,
    trace_kwargs,
    trace_output,
    trace_data: GlobalTraceData,
) -> ScaleAssignment:
    model = mathopt.Model(name="Scale_Optimization")
    var_ops_dict, var_tracer_dict = collect_variables(model, trace_data)

    add_input_constraints(
        model=model,
        trace_data=trace_data,
        var_tracer_dict=var_tracer_dict,
    )
    add_operator_constraints(
        model=model,
        trace_data=trace_data,
        var_ops_dict=var_ops_dict,
        var_tracer_dict=var_tracer_dict,
    )
    add_objective(
        model=model,
        var_ops_dict=var_ops_dict,
        var_tracer_dict=var_tracer_dict,
    )

    solver_result = mathopt.solve(model, mathopt.SolverType.GSCIP)
    assert solver_result.termination.reason == mathopt.TerminationReason.OPTIMAL
    primal_solution = solver_result.solutions[0].primal_solution
    assert primal_solution is not None
    variable_assignment = primal_solution.variable_values

    # output transformation
    tracer_scales, tracer_pre_transforms = {}, {}
    for t_id, (v1, v2) in var_tracer_dict.items():
        cur_v1 = variable_assignment[v1]
        cur_v2 = variable_assignment[v2]
        round_v2 = round(cur_v2)
        assert abs(round_v2 - cur_v2) < _ROUND_EPS
        tracer_scales[t_id] = cur_v1
        tracer_pre_transforms[t_id] = cur_v2

    node_input_transforms = {}
    for k, v in var_ops_dict.items():
        cur_v = variable_assignment[v]
        round_v = round(cur_v)
        assert abs(round_v - cur_v) < _ROUND_EPS
        node_input_transforms[k] = cur_v

    assignment = ScaleAssignment(
        tracer_scales=tracer_scales,
        tracer_pre_transforms=tracer_pre_transforms,
        node_input_transforms=node_input_transforms,
    )
    return assignment


def collect_variables(
    model: mathopt.Model,
    trace_data: GlobalTraceData,
) -> tuple[
    dict[tuple[str, int, int], mathopt.Variable],
    dict[int, tuple[mathopt.Variable, mathopt.Variable]],
]:
    # collect tracer nodes
    var_tracer_dict: dict[int, tuple[mathopt.Variable, mathopt.Variable]] = {}
    for t in trace_data.tracer_nodes.values():
        # variable representing scale of array
        cur_var = model.add_integer_variable(name=f"x{t.id}")
        # variable representing scale shift for variable
        cur_var2 = model.add_integer_variable(name=f"s{t.id}")
        var_tracer_dict[t.id] = (cur_var, cur_var2)

    # collect op nodes
    var_ops_dict = {}
    for n in trace_data.operator_nodes.values():
        for k, a in n.op_kwargs.items():
            # variable for input scale shift due to constraints of operations
            key = (k, a.id, n.id)
            cur_var = model.add_integer_variable(name=f"{k}.{a.id}.{n.id}")
            var_ops_dict[key] = cur_var

    return var_ops_dict, var_tracer_dict


def add_input_constraints(
    model: mathopt.Model,
    trace_data: GlobalTraceData,
    var_tracer_dict: dict[int, tuple[mathopt.Variable, mathopt.Variable]],
):
    for t in trace_data.tracer_nodes.values():
        if t.static_unitful is not None:
            cur_unitful = t.static_unitful
        elif t.value is not None and isinstance(t.value, Unitful):
            cur_unitful = t.value
        else:
            continue

        if jnp.all(cur_unitful.val == 0):  # TODO: change this to unitful equal comparison once implemented
            # if all input values are zero, we do not know the correct scale for this input value
            continue

        cur_scale = cur_unitful.scale
        if isinstance(cur_scale, IntFraction):
            cur_scale = cur_scale.value()
        cur_var, _ = var_tracer_dict[t.id]
        model.add_linear_constraint(cur_var == cur_scale)


def add_operator_constraints(
    model: mathopt.Model,
    trace_data: GlobalTraceData,
    var_ops_dict: dict[tuple[str, int, int], mathopt.Variable],
    var_tracer_dict: dict[int, tuple[mathopt.Variable, mathopt.Variable]],
):
    for n in trace_data.operator_nodes.values():
        # TODO: handle non-traced inputs
        # Operator input variables
        c_kwargs = {}
        for k, t in n.op_kwargs.items():
            # for each operation, the input is the variable plus optional scale adjustment factor
            factor_var = var_ops_dict[(k, t.id, n.id)]
            input_var, _ = var_tracer_dict[t.id]
            c_kwargs[k] = input_var + factor_var

        # generate tree of output variables
        leaves, treedef = jax.tree.flatten(n.output, is_leaf=lambda x: isinstance(x, UnitfulTracer))
        c_out_list = [(var_tracer_dict[t.id][0] + var_tracer_dict[t.id][1]) for t in leaves]
        c_out_tree = jax.tree.unflatten(treedef, c_out_list)

        # prepare kwargs for node constraint function calling
        c_kwargs["out"] = c_out_tree

        # constraints for the input/output variables given by the operation
        c_fun = CONSTRAINTS_DICT[n.op_name]
        cur_constraints = c_fun(**c_kwargs)
        for c in cur_constraints:
            model.add_linear_constraint(c)


def add_objective(
    model: mathopt.Model,
    var_ops_dict: dict[tuple[str, int, int], mathopt.Variable],
    var_tracer_dict: dict[int, tuple[mathopt.Variable, mathopt.Variable]],
):
    new_vars = []
    # minimize absolute value of variables (technically non-linear, but can be achieved using slack variable)
    for k, v in var_ops_dict.items():
        helper = model.add_variable(lb=0, name=f"h.{k[0]}.{k[1]}.{k[2]}")
        model.add_linear_constraint(helper >= v)
        model.add_linear_constraint(helper >= -v)
        new_vars.append(helper)

    for k, (_, v) in var_tracer_dict.items():
        helper = model.add_variable(lb=0, name=f"h.{k}")
        model.add_linear_constraint(helper >= v)
        model.add_linear_constraint(helper >= -v)
        new_vars.append(helper)

    # minimize sum of helper variables (equivalent to sum of absolute original values)
    model.minimize(sum(new_vars))


def export_model_to_lp(model: mathopt.Model) -> str:
    """
    Exports a mathopt.Model to a string in LP format.
    """
    # 1. Get the underlying Protocol Buffer representation
    proto = model.export_model()

    output = []

    # --- Helpers ---
    # Map variable IDs to names (or default names if missing)
    var_id_to_name = {}
    for i, var_id in enumerate(proto.variables.ids):
        name = proto.variables.names[i] if i < len(proto.variables.names) else ""
        if not name:
            name = f"x{var_id}"
        var_id_to_name[var_id] = name

    def format_term(coeff, var_name):
        """Formats a single term like '+ 3.5 x1'"""
        if coeff == 0:
            return ""
        sign = " + " if coeff >= 0 else " - "
        return f"{sign}{abs(coeff)} {var_name}"

    # --- Objective ---
    output.append("Maximize" if proto.objective.maximize else "Minimize")

    obj_terms = []
    # Objective offset
    if proto.objective.offset:
        obj_terms.append(f"{proto.objective.offset}")

    # Objective coefficients (SparseDoubleVectorProto uses .values)
    for i, var_id in enumerate(proto.objective.linear_coefficients.ids):
        coeff = proto.objective.linear_coefficients.values[i]
        name = var_id_to_name[var_id]
        obj_terms.append(format_term(coeff, name))

    obj_str = "".join(obj_terms).lstrip("+ ")
    output.append(" obj: " + (obj_str if obj_str else "0"))

    # --- Constraints ---
    output.append("\nSubject To")

    # Group matrix terms by constraint (row) ID
    # matrix structure: row_ids[k], column_ids[k], coefficients[k]
    constraints_terms = defaultdict(list)
    matrix = proto.linear_constraint_matrix

    # FIXED LINE BELOW: Changed .values to .coefficients
    for i, row_id in enumerate(matrix.row_ids):
        col_id = matrix.column_ids[i]
        coeff = matrix.coefficients[i]
        constraints_terms[row_id].append((coeff, var_id_to_name[col_id]))

    # Iterate through all constraints
    for i, c_id in enumerate(proto.linear_constraints.ids):
        c_name = proto.linear_constraints.names[i] if i < len(proto.linear_constraints.names) else ""
        if not c_name:
            c_name = f"c{c_id}"

        terms = constraints_terms.get(c_id, [])
        # Build expression string
        expr_parts = [format_term(c, v) for c, v in terms]
        expr_str = "".join(expr_parts).lstrip("+ ")
        if not expr_str:
            expr_str = "0"

        lb = proto.linear_constraints.lower_bounds[i]
        ub = proto.linear_constraints.upper_bounds[i]

        # Format: lhs <= expr <= rhs
        if lb > -float("inf") and ub < float("inf") and lb == ub:
            output.append(f" {c_name}: {expr_str} = {lb}")
        elif lb > -float("inf") and ub < float("inf"):
            output.append(f" {c_name}_lb: {expr_str} >= {lb}")
            output.append(f" {c_name}_ub: {expr_str} <= {ub}")
        elif lb > -float("inf"):
            output.append(f" {c_name}: {expr_str} >= {lb}")
        elif ub < float("inf"):
            output.append(f" {c_name}: {expr_str} <= {ub}")

    # --- Bounds ---
    output.append("\nBounds")
    for i, var_id in enumerate(proto.variables.ids):
        name = var_id_to_name[var_id]
        lb = proto.variables.lower_bounds[i]
        ub = proto.variables.upper_bounds[i]

        # Default in LP is 0 <= x <= +inf. Only write if different.
        if lb == 0 and ub == float("inf"):
            continue

        if ub == float("inf"):
            output.append(f" {lb} <= {name}")
        elif lb == -float("inf"):
            output.append(f" {name} <= {ub}")
        else:
            output.append(f" {lb} <= {name} <= {ub}")

    # --- Integers / Binaries ---
    integers = []
    binaries = []
    for i, is_int in enumerate(proto.variables.integers):
        if is_int:
            var_id = proto.variables.ids[i]
            lb = proto.variables.lower_bounds[i]
            ub = proto.variables.upper_bounds[i]
            name = var_id_to_name[var_id]
            if lb == 0 and ub == 1:
                binaries.append(name)
            else:
                integers.append(name)

    if binaries:
        output.append("\nBinaries")
        output.append(" " + " ".join(binaries))

    if integers:
        output.append("\nGenerals")
        output.append(" " + " ".join(integers))

    output.append("\nEnd")
    return "\n".join(output)
