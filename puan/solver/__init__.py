import math
import numpy
import puan.ndarray

def our_revised_simplex(polyhedron: puan.ndarray.ge_polyhedron, objective_function: numpy.ndarray, prio_list = []):
    """
    Solves the linear program given by the polyhedron and the objective function
    using a revised simplex algorithm.

    Parameters
    ----------
        polyhedron : puan.ndarray.ge_polyhedron
        objective_function : numpy.ndarray
        prio_list : list
            incase of multiple candidates for incoming variable, first occurence in the prio list is picked

    Returns
    -------
        out : tuple
            out[0] : objective value at the optimal solution
            out[1] : numpy.ndarray with the variable values at the optimal solution
            out[2] : numpy.ndarray with pivoted A matrix from polyhedron at the optimal solution
            out[3] : solution information as string, e.g. 'Solution is unique' or 'Solution does not exists'

    Notes
    -----
        Variables cannot take values below zero. Any suitable substitutions must be performed prio to calling the function.

    Examples
    --------
        >>> polyhedron = puan.ndarray.ge_polyhedron(numpy.array([[-100, -2, -1], [-80, -1, -1], [-40, -1, 0]]), variables=[puan.variable("b", int, True, -numpy.inf, numpy.inf), puan.variable("x1", int, False, 0, numpy.inf), puan.variable("x2", int, False, 0, numpy.inf )])
        >>> objective_function = numpy.array([30, 20])
        >>> our_revised_simplex(polyhedron, objective_function)
            (
                1800,
                numpy.array([20, 60]),
                numpy.array([
                    [ 0.,  1., -1.,  2.,  0.],
                    [ 0.,  0., -1.,  1.,  1.],
                    [ 1.,  0.,  1., -1.,  0.]
                ]),
                'Solution is unique'
            )
    """
    def _our_revised_simplex(A, b, C, B_vars, N_vars, prio_list):
        # Simplex start
        A = A.astype("float64")
        B = A[:,B_vars]
        N = A[:,N_vars]
        C_N = C[N_vars]
        C_B = C[B_vars]
        C_orig = C.copy()
        z_upb = 0 if (C<=0).all() else float("inf")
        substituted_vars = []
        B_inv = numpy.linalg.inv(B)
        B_inv_tilde = numpy.eye(B.shape[0])
        X_B = b
        dual_solution = numpy.dot(C_B, B_inv)
        C_tilde_N = C_N - numpy.dot(dual_solution, N)
        try:
            u_j = numpy.concatenate((numpy.array(list(map(lambda x: getattr(x, "ub"), A.variables))), numpy.repeat(numpy.inf, len(B_vars)+len(N_vars)-len(A.variables))), axis=0)
        except:
            u_j = numpy.repeat(numpy.inf, len(B_vars)+len(N_vars))
        while ((C_tilde_N > 0).any()):
            if numpy.dot(X_B, C_B) == z_upb:
                break
            incoming_candidates = N_vars[C_tilde_N == C_tilde_N.max()]
            if incoming_candidates.shape[0] == 1 or len(incoming_candidates[numpy.isin(incoming_candidates, prio_list)]) < 2:
                incoming_variable = incoming_candidates[0]
            else:
                incoming_candidates = incoming_candidates[numpy.isin(incoming_candidates, prio_list)]
                incoming_variable = incoming_candidates[numpy.array([numpy.where(x==prio_list)[0][0] for x in incoming_candidates]).argmin()]
            incoming_variable_index = numpy.where(N_vars == incoming_variable)[0][0]
            incoming_variable = N_vars[incoming_variable_index]
            B_inv_N_j = numpy.dot(B_inv, A[:, incoming_variable])
            t1 = numpy.where(B_inv_N_j > 0, numpy.divide(X_B, B_inv_N_j), numpy.inf).min()
            t2 = u_j[incoming_variable]
            t3 = numpy.where(B_inv_N_j < 0, numpy.divide(u_j[B_vars]-X_B, -B_inv_N_j), numpy.inf).min()
            if min(t1, t2, t3) == numpy.inf:
                # TODO: return proper XB
                return (numpy.inf, X_B, B_vars, B_inv_tilde, "Solution is unbounded")
            if (t3 < t1 and t3 < t2):
                outgoing_variable_index = numpy.argwhere(numpy.where(B_inv_N_j < 0, numpy.divide(u_j[B_vars]-X_B, -B_inv_N_j), numpy.inf)==t3).max()
                outgoing_variable = B_vars[outgoing_variable_index]
                b = b - u_j[outgoing_variable]*A[:, outgoing_variable]
                A[:, outgoing_variable] = -A[:, outgoing_variable]
                C[outgoing_variable] = -C[outgoing_variable]
                E = numpy.eye(B.shape[0])
                E[:, outgoing_variable_index] = -B_inv_N_j/B_inv_N_j[outgoing_variable_index]
                E[outgoing_variable_index, outgoing_variable_index] = 1/B_inv_N_j[outgoing_variable_index]
                B_inv_tilde = numpy.dot(E, B_inv)
                substituted_vars.append(outgoing_variable)
            elif (t2 < t1 and t2 <= t3):
                b = b - u_j[incoming_variable]*A[:, incoming_variable]
                A[:, incoming_variable] = -A[:, incoming_variable]
                C[incoming_variable] = -C[incoming_variable]
                outgoing_variable_index = 0
                substituted_vars.append(incoming_variable)
                incoming_variable = B_vars[outgoing_variable_index]
                outgoing_variable = N_vars[incoming_variable_index]
            else: # t1 is smallest
                outgoing_variable_index = numpy.argwhere(numpy.where(B_inv_N_j > 0, numpy.divide(X_B, B_inv_N_j), numpy.inf) == t1).max()
                outgoing_variable = B_vars[outgoing_variable_index]
                E = numpy.eye(B.shape[0])
                E[:, outgoing_variable_index] = -B_inv_N_j/B_inv_N_j[outgoing_variable_index]
                E[outgoing_variable_index, outgoing_variable_index] = 1/B_inv_N_j[outgoing_variable_index]
                B_inv_tilde = numpy.dot(E, B_inv)
            X_B = numpy.dot(B_inv_tilde, b)
            B_vars[outgoing_variable_index] = incoming_variable
            N_vars[incoming_variable_index] = outgoing_variable
            B_inv = B_inv_tilde
            C_N = C[N_vars]
            C_B = C[B_vars]
            dual_solution = numpy.dot(C_B, B_inv_tilde)
            C_tilde_N = C_N - numpy.dot(dual_solution, A[:, N_vars])

        if (C_tilde_N == 0).any():
            solution_information = "Solution is not unique"
        else:
            solution_information = "Solution is unique"

        X_B = numpy.where(numpy.in1d(B_vars, numpy.array(substituted_vars)), u_j[B_vars] - X_B, X_B)
        X_N = numpy.where(numpy.in1d(N_vars, numpy.array(substituted_vars)), u_j[N_vars], 0)
        X_B = numpy.concatenate((X_B, X_N[X_N>0]), axis=0)
        B_vars = numpy.concatenate((B_vars, N_vars[X_N>0]), axis=0)
        z = numpy.dot(C_orig[B_vars], X_B)
        return (z, X_B, B_vars, B_inv_tilde, solution_information)

    def _lb_constraint(x, ind, l):
        res = numpy.zeros(l)
        res[0] = x.lb
        res[ind] = 1
        return res

    lb_constraints = numpy.array([_lb_constraint(x, ind, polyhedron.shape[1]) for ind, x in enumerate(polyhedron.variables) if x.lb>0])
    if len(lb_constraints)>0:
        polyhedron = puan.ndarray.ge_polyhedron(numpy.append(polyhedron, lb_constraints, axis=0), variables=polyhedron.variables)
    A, b = polyhedron.to_linalg()
    n_orig_vars = A.shape[1]
    n_constraints = A.shape[0]
    # introducing slack variables
    slack_variables = -numpy.eye(n_constraints)
    n_slack_vars = slack_variables.shape[0]
    A_prim = puan.ndarray.integer_ndarray(numpy.append(A, slack_variables, axis=1), variables=A.variables)
    A_prim[b<0,:] = -1*A_prim[b<0,:]
    b = abs(b)
    B_vars = numpy.array(range(n_orig_vars, n_orig_vars+n_slack_vars))
    if (numpy.linalg.inv(A_prim[:, B_vars])*b >= 0).all():
        # origio is a BFS
        N_vars = numpy.array(range(n_orig_vars))
        C = numpy.append(objective_function, numpy.zeros(n_slack_vars))
        (z, X_B, B_vars, B_inv_tilde, solution_information) = _our_revised_simplex(A_prim, b, C, B_vars, N_vars, prio_list)
    else:
        #find BFS
        C = numpy.zeros(n_orig_vars+n_slack_vars)
        A_prim = puan.ndarray.integer_ndarray(numpy.append(A_prim, numpy.eye(n_constraints)[:,(numpy.linalg.inv(A_prim[:, B_vars])*b < 0).any(axis=0)], axis=1), variables = A.variables)
        n_artificial_vars = A_prim.shape[1] - (n_orig_vars+n_slack_vars)
        C = numpy.append(C, -numpy.ones(n_artificial_vars))
        N_vars = numpy.append(numpy.array(range(n_orig_vars)), numpy.array(range(n_orig_vars, n_orig_vars+n_slack_vars))[(numpy.linalg.inv(A_prim[:, B_vars])*b < 0).any(axis=0)])
        #B_vars = numpy.append(numpy.array(range(n_orig_vars, n_orig_vars+n_slack_vars))[(numpy.linalg.inv(A_prim[:, B_vars])*b >= 0).all(axis=0)], numpy.array(range(n_orig_vars+n_slack_vars, n_orig_vars+n_slack_vars+n_artificial_vars)))
        B_vars = ((A_prim!=0)*range(A_prim.shape[1])).argmax(axis=1)
        (z, X_B, B_vars, B_inv_tilde, solution_information) = _our_revised_simplex(A_prim, b, C, B_vars, N_vars, prio_list)
        if (z < 0):
            # No BFS exists
            return (None, None, None, "No feasible solution exists")
        A_prim = numpy.dot(B_inv_tilde, A_prim)
        A_prim = puan.ndarray.integer_ndarray(A_prim[:, :n_orig_vars+n_slack_vars], variables=A.variables)
        N_vars = numpy.array([i for i in N_vars if i not in range(n_orig_vars+n_slack_vars, n_orig_vars+n_slack_vars+n_artificial_vars)])
        for ind, val in enumerate(B_vars):
            if val >= n_orig_vars+n_slack_vars:
                B_vars[ind] = N_vars[0]
                N_vars = numpy.delete(N_vars, 0)
        C = numpy.append(objective_function, numpy.zeros(n_slack_vars))
        C[:B_inv_tilde.shape[0]] = numpy.dot(C[:B_inv_tilde.shape[0]], B_inv_tilde)
        (z, X_B, B_vars, B_inv_tilde, solution_information) = _our_revised_simplex(A_prim, X_B, C, B_vars, N_vars, prio_list)

    solution = numpy.zeros(A_prim.shape[1])
    solution[B_vars] = X_B
    A_prim = numpy.dot(B_inv_tilde, A_prim)
    return (z, solution[:n_orig_vars], A_prim, solution_information)

def our_land_doig_dankins(polyhedron: numpy.ndarray, objective_function: numpy.ndarray, prio_list = []):
    """
    Solves the linear program given by the polyhedron and the objective function
    using a revised simplex algorithm.

    Parameters
    ----------
        polyhedron : puan.ndarray.ge_polyhedron
        objective_function : numpy.ndarray
        prio_list : list
            incase of multiple candidates for incoming variable, first occurence in the prio list is picked

    Returns
    -------
        out : tuple
            out[0] : objective value at the optimal solution
            out[1] : numpy.ndarray with the variable values at the optimal solution
            out[2] : numpy.ndarray with pivoted A matrix from polyhedron at the optimal solution
            out[3] : solution information as string, e.g. 'Solution is unique' or 'Solution does not exists'

    Notes
    -----
        Variables cannot take values below zero. Any suitable substitutions must be performed prio to calling the function.

    Examples
    --------
        >>> polyhedron = puan.ndarray.ge_polyhedron(numpy.array([[-100, -2, -1], [-80, -1, -1], [-40, -1, 0]]), variables=[puan.variable("b", int, True, -numpy.inf, numpy.inf), puan.variable("x1", int, False, 0, numpy.inf), puan.variable("x2", int, False, 0, numpy.inf )])
        >>> objective_function = numpy.array([30, 20])
        >>> our_revised_simplex(polyhedron, objective_function)
            (
                1800,
                numpy.array([20, 60]),
                numpy.array([
                    [ 0.,  1., -1.,  2.,  0.],
                    [ 0.,  0., -1.,  1.,  1.],
                    [ 1.,  0.,  1., -1.,  0.]
                ]),
                'Solution is unique'
            )
    """
    z_ub = numpy.inf
    x = None
    A = None
    solution_information = "No feasible solution exists"
    nodes_to_explore = [(polyhedron, objective_function, prio_list)]
    n_vars = polyhedron.shape[1]
    while(nodes_to_explore):
        polyhedron, objective_function, prio_list = nodes_to_explore.pop()
        (z, X_B, A_prim, solution_information) = our_revised_simplex(polyhedron, objective_function, prio_list)
        if not z:
            # No feasible solution
            continue
        if z > z_ub:
            # Not possible to find better solution in this area
            continue
        first_non_integer = numpy.array([not i%1 for i in X_B]).argmin()
        if X_B[first_non_integer].is_integer():
            # We've found a best solution in this area
            if z < z_ub:
                z_ub = z
                x = X_B
                A = A_prim
            continue
        else:
            cond1 = numpy.zeros((1,n_vars))
            cond1[0,first_non_integer+1]=-1
            cond1[0][0] = -math.floor(X_B[first_non_integer])
            nodes_to_explore.append((puan.ndarray.ge_polyhedron(numpy.append(polyhedron.copy(), cond1, axis=0), variables=polyhedron.variables), objective_function, prio_list))
            cond2 = numpy.zeros((1,n_vars))
            cond2[0,first_non_integer+1]=1
            cond2[0][0] = math.ceil(X_B[first_non_integer])
            nodes_to_explore.append((puan.ndarray.ge_polyhedron(numpy.append(polyhedron.copy(), cond2, axis=0), variables=polyhedron.variables), objective_function, prio_list))
    return (z_ub, x, A, solution_information)