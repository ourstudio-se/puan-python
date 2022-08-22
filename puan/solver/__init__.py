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
    # Functions
    def _lb_constraint(x, ind, l):
        res = numpy.zeros(l)
        res[0] = x.lb
        res[ind] = 1
        return res
    
    def _revised_simplex(A, b, B_vars, N_vars, B_inv, C, u_j, substituted_vars):
        def _update_constraint_column(B_inv_N_j, index, B_inv):
            E = numpy.eye(B.shape[0])
            E[:, index] = -B_inv_N_j/B_inv_N_j[index]
            E[index, index] = 1/B_inv_N_j[index]
            return numpy.dot(E, B_inv)
        def _perform_ub_substitution(A, b, C, variable, limit, substituted_vars):
            b = b - limit*A[:, variable]
            A[:, variable] = -A[:, variable]
            C[variable] = -C[variable]
            substituted_vars.append(variable)
            return A, b, C, substituted_vars
        def _update_basis(B_vars, N_vars, incoming, outgoing):
            B_vars[outgoing] = incoming
            N_vars[incoming] = outgoing
            return B_vars, N_vars

        B = A[:, B_vars]
        N = A[:, N_vars]
        C_N = C[N_vars] # Objective function for non-basis variables
        C_B = C[B_vars] # Objective function for basis variables
        C_tilde_N = C_N - numpy.dot(numpy.dot(C_B, B_inv), N) # Reduced cost for non-basis variables
        # Begin simplex iterations
        X_B = numpy.dot(B_inv, b) # X values a current solution
        while((C_tilde_N>0).any()):
            incoming_candidates = N_vars[C_tilde_N == C_tilde_N.max()]
            if incoming_candidates.shape[0] == 1 or not numpy.isin(incoming_candidates, prio_list).any():
                incoming_variable = incoming_candidates[0]
            else:
                incoming_candidates = incoming_candidates[numpy.isin(incoming_candidates, prio_list)]
                incoming_variable = incoming_candidates[numpy.array([numpy.where(x==prio_list)[0][0] for x in incoming_candidates]).argmin()]
            incoming_variable_index = numpy.where(N_vars == incoming_variable)[0][0]
            incoming_variable = N_vars[incoming_variable_index]
            B_inv_N_j = numpy.dot(B_inv, A[:, incoming_variable])
            _t1 = numpy.where(B_inv_N_j > 0, numpy.divide(X_B, B_inv_N_j), numpy.inf)
            t1 = _t1.min()
            t2 = u_j[incoming_variable]
            _t3 = numpy.where(B_inv_N_j < 0, numpy.divide(u_j[B_vars]-X_B, -B_inv_N_j), numpy.inf)
            t3 = _t3.min()
            if min(t1, t2, t3) == numpy.inf:
                # TODO: return proper XB
                return (numpy.inf, X_B, B_vars, B_inv_tilde, "Solution is unbounded")
            if (t3 < t1 and t3 < t2):
                outgoing_variable_index = _t3.argmin()
                outgoing_variable = B_vars[outgoing_variable_index]
                A, b, C, substituted_vars = _perform_ub_substitution(A, b, C, outgoing_variable, u_j[outgoing_variable], substituted_vars)
                B_inv_tilde = _update_constraint_column(B_inv_N_j, outgoing_variable_index, B_inv)
                B_vars, N_vars = _update_basis(B_vars, N_vars, incoming_variable, outgoing_variable)
            elif (t2 < t1 and t2 <= t3):
                A, b, C, substituted_vars = _perform_ub_substitution(A, b, C, incoming_variable, u_j[incoming_variable], substituted_vars)
                B_inv_tilde = B_inv
            else: # t1 is smallest
                outgoing_variable_index = _t1.argmin()
                outgoing_variable = B_vars[outgoing_variable_index]
                B_inv_tilde = _update_constraint_column(B_inv_N_j, outgoing_variable_index, B_inv)
                B_vars, N_vars = _update_basis(B_vars, N_vars, incoming_variable, outgoing_variable)
            B_inv = B_inv_tilde
            C_N = C[N_vars]
            C_B = C[B_vars]
            C_tilde_N = C_N - numpy.dot(numpy.dot(C_B, B_inv_tilde), A[:, N_vars])
            X_B = numpy.dot(B_inv, b) # X values a current solution
        return A, b, B_vars, N_vars, B_inv, C, substituted_vars
    
    def _phaseI(A: puan.ndarray.integer_ndarray, b, substituted_vars: list):
        n_orig_and_slack_vars = A.shape[1]
        n_artificial_vars = 0
        B_vars = numpy.array(range(n_orig_and_slack_vars-A.shape[0], n_orig_and_slack_vars))
        B_inv = numpy.linalg.inv(A[:, B_vars])
        valid_constraints_at_origo = (numpy.dot(B_inv,b) >= 0)
        if valid_constraints_at_origo.all():
            # origio is a BFS
            N_vars = numpy.array(range(n_orig_and_slack_vars-A.shape[0]))
            B_inv = numpy.linalg.inv(A[B_vars])
            try:
                u_j = numpy.concatenate((numpy.array(list(map(lambda x: getattr(x, "ub"), A.variables))), numpy.repeat(numpy.inf, n_orig_and_slack_vars-len(A.variables))), axis=0)
            except:
                u_j = numpy.repeat(numpy.inf, n_orig_and_slack_vars)
        else:
            C_phaseI = numpy.zeros(n_orig_and_slack_vars)
            A = puan.ndarray.integer_ndarray(numpy.append(numpy.eye(A.shape[0])[:,~valid_constraints_at_origo], A, axis=1), variables = A.variables)
            n_artificial_vars = A.shape[1] - n_orig_and_slack_vars
            C_phaseI = numpy.append(-numpy.ones(n_artificial_vars), C_phaseI)
            N_vars = numpy.append(numpy.array(range(n_artificial_vars, n_orig_and_slack_vars-A.shape[0]+n_artificial_vars)), numpy.array(range(A.shape[1]-A.shape[0], A.shape[1]))[~valid_constraints_at_origo])
            B_vars = numpy.append(numpy.array(range(n_artificial_vars)), numpy.array(range(A.shape[1] - A.shape[0], A.shape[1]))[(valid_constraints_at_origo).any(axis=0)]) #((A!=0)*range(A.shape[1])).argmax(axis=1)
            try:
                u_j = numpy.concatenate((numpy.repeat(numpy.inf, n_artificial_vars), numpy.array(list(map(lambda x: getattr(x, "ub"), A.variables))), numpy.repeat(numpy.inf, n_orig_and_slack_vars - len(A.variables))), axis=0)
            except:
                u_j = numpy.repeat(numpy.inf, n_artificial_vars+n_orig_and_slack_vars)
            A, b, B_vars, N_vars, B_inv, C_phaseI, substituted_vars = _revised_simplex(A, b, B_vars, N_vars, numpy.eye(A.shape[0]), C_phaseI, u_j, substituted_vars)
            if (numpy.dot(C_phaseI[B_vars], numpy.dot(B_inv, b)))>0:
                # No feasible solution exists
                return False, None, None, None, None, None, None, None
            else:
                N_vars = N_vars[N_vars>=n_artificial_vars]-n_artificial_vars
                A = A[:, n_artificial_vars:]
                B_vars = B_vars - n_artificial_vars
                u_j = u_j[n_artificial_vars:]
                substituted_vars = numpy.array(substituted_vars)-n_artificial_vars
            return (True, A, b, B_vars, N_vars, B_inv, u_j, substituted_vars)
    
    def _phaseII(A, b, B_vars, N_vars, B_inv, C, u_j, substituted_vars):
        C_phaseII = numpy.append(C, numpy.zeros(A.shape[0]))
        if substituted_vars:
            C_phaseII[substituted_vars] = -C_phaseII[substituted_vars]
        return _revised_simplex(A, b, B_vars, N_vars, B_inv, C_phaseII, u_j, substituted_vars)

    # Pre-processing
    #    Constructing lower bound constraints
    lb_constraints = numpy.array([_lb_constraint(x, ind, polyhedron.shape[1]) for ind, x in enumerate(polyhedron.variables) if x.lb>0])
    if len(lb_constraints)>0:
        polyhedron = puan.ndarray.ge_polyhedron(numpy.append(polyhedron, lb_constraints, axis=0), variables=polyhedron.variables)
    A, b = polyhedron.to_linalg()
    A = A.astype("float64")
    substituted_vars = []
    #    Introducing slack variables
    slack_variables = -numpy.eye(A.shape[0]) # One slack variable for each constraint
    A = puan.ndarray.integer_ndarray(numpy.append(A, slack_variables, axis=1), variables=A.variables)
    #    Converting to standard form
    A[b<0,:] = -1*A[b<0,:]
    b = abs(b)
    # PhaseI
    bfs_exists, A, b, B_vars, N_vars, B_inv, u_j, substituted_vars = _phaseI(A, b, substituted_vars)
    if not bfs_exists:
        return
    # PhaseII
    A, b, B_vars, N_vars, B_inv, C, substituted_vars = _phaseII(A, b, B_vars, N_vars, B_inv, objective_function, u_j, substituted_vars)
    
    if (C[N_vars] == 0).any():
        solution_information = "Solution is not unique"
    else:
        solution_information = "Solution is unique"
    X_B = numpy.dot(B_inv, b)
    X_B = numpy.where(numpy.in1d(B_vars, numpy.array(substituted_vars)), u_j[B_vars] - X_B, X_B)
    X_N = numpy.where(numpy.in1d(N_vars, numpy.array(substituted_vars)), u_j[N_vars], 0)
    X_B = numpy.concatenate((X_B, X_N[X_N>0]), axis=0)
    B_vars = numpy.concatenate((B_vars, N_vars[X_N>0]), axis=0)
    z = numpy.dot(objective_function[B_vars[B_vars<(A.shape[1]-A.shape[0])]], X_B[B_vars<(A.shape[1]-A.shape[0])])
    return (z, X_B, solution_information)

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
    solution_information = "No feasible solution exists"
    nodes_to_explore = [(polyhedron, objective_function, prio_list)]
    n_vars = polyhedron.shape[1]
    while(nodes_to_explore):
        polyhedron, objective_function, prio_list = nodes_to_explore.pop()
        (z, X_B, solution_information) = our_revised_simplex(polyhedron, objective_function, prio_list)
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
    return (z_ub, x, solution_information)