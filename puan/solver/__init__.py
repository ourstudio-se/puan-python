from cmath import phase
import itertools
import math
from timeit import repeat
import numpy
import puan.ndarray
class Our_simplex_solver(object):
    """
    Solves the linear program given by the polyhedron and the objective function
    using a revised simplex algorithm.

    Parameters
    ----------
        polyhedron : puan.ndarray.ge_polyhedron
        bounds : list of tuples [(lower_bound, upper_bound)]
        objective_function : numpy.ndarray
        objective_functions : list
        integer_solution : bool
        prio_list : list
            incase of multiple candidates for incoming variable, first occurence in the prio list is picked

    Returns
    -------
        out : tuple
            out[0] : objective value at the optimal solution
            out[1] : numpy.ndarray with the variable values at the optimal solution
            out[2] : numpy.ndarray with pivoted A matrix from polyhedron at the optimal solution
            out[3] : solution information as string, e.g. 'Solution is unique' or 'Solution does not exists'

    Examples
    --------
        >>> polyhedron = puan.ndarray.ge_polyhedron(numpy.array([[-100, -2, -1], [-80, -1, -1], [-40, -1, 0]]))
        >>> bounds = [(0, numpy.inf), (0, numpy.inf)]
        >>> objective_function = numpy.array([30, 20])
        >>> our_revised_simplex(polyhedron, bounds, objective_function)
        (1800.0, array([20, 60]), array([[0, 1, -1, 2, 0],
               [0, 0, -1, 1, 1],
               [1, 0, 1, -1, 0]]), 'Solution is unique')
    """
    def __init__(self) -> None:
        pass
    def __call__(self, polyhedron, bounds, objective_function=None, objective_functions=[], integer_solution=False, prio_list=[]):
        self.prio_list = prio_list
        self.objectives, self.A, self.b, self.bounds_lower, self.bounds_upper, self.B_vars, self.B_inv, self.bfs, self.C_phaseI, self.N_vars, self.ORIG_VARS, self.solution_information, self.substituted_vars = itertools.repeat(None, 13)
        self._setup(polyhedron, bounds, objective_function, objective_functions)
        if integer_solution:
            solutions = [self.run_ldd(polyhedron, bounds, objective) for objective in self.objectives]
        else:
            solutions = [self.run(objective) for objective in self.objectives]
        if len(objective_functions)<1:
            return solutions[0]
        return solutions
    
    def _setup(self, polyhedron, bounds, objective_function, objective_functions):
        self.bounds_lower = numpy.array([x[0] for x in bounds])
        self.bounds_upper = numpy.array([x[1] for x in bounds])
        self.A, self.b = polyhedron.to_linalg()
        if self.A.shape[1] != len(bounds):
            raise ValueError(f"The shapes of polyhedron.A and bounds must match, {self.A.shape[1]}!={len(bounds)}")
        #if objective_function.any() and len(objective_functions)>0 or (not objective_function.any() and len(objective_functions<1)):
        #    raise ValueError("One and only one of objective_function and objective_functions must be given")
        self.objectives = objective_functions or [objective_function]
        # Pre-processing
        #    Constructing lower bound constraints
        self.b = numpy.append(self.b, numpy.array(self.bounds_lower[self.bounds_lower>0]), axis=0)
        _lb_constraint = numpy.zeros(self.A.shape[1])
        _lb_constraint[self.bounds_lower>0] = 1
        if ((_lb_constraint>0).any()):
            self.A = numpy.append(self.A, _lb_constraint[self.bounds_lower>0], axis=0)
        self.ORIG_VARS = self.A.shape[1]
        if ((self.bounds_lower<0).any()):
            self.A = numpy.append(self.A, -self.A[:, self.bounds_lower<0], axis=1)
            for i, of in enumerate(self.objectives):
                self.objectives[i] = numpy.append(of, -of[self.bounds_lower<0])
            self.bounds_lower = numpy.append(self.bounds_lower, numpy.repeat(0, sum(self.bounds_lower<0)))
            self.bounds_upper = numpy.append(self.bounds_upper, -self.bounds_lower[self.bounds_lower<0])
        self.ADDED_VARS = sum(self.bounds_lower<0)
        self.A = self.A.astype("float64")
        self.substituted_vars = []
        #    Introducing slack variables
        slack_variables = -numpy.eye(self.A.shape[0]) # One slack variable for each constraint
        self.A = numpy.append(self.A, slack_variables, axis=1)
        self.bounds_lower = numpy.append(self.bounds_lower, numpy.repeat(0, slack_variables.shape[0]))
        self.bounds_upper = numpy.append(self.bounds_upper, numpy.repeat(numpy.inf, slack_variables.shape[0]))
        #    Converting to standard form
        self.A[self.b<0,:] = -1*self.A[self.b<0,:]
        self.b = abs(self.b)
        self.bfs = False

    def _revised_simplex_method(self, phaseI: bool=False):
        def _update_constraint_column(B_inv_N_j, index, B_inv):
            E = numpy.eye(self.B_vars.shape[0])
            E[:, index] = numpy.divide(-B_inv_N_j, B_inv_N_j[index], out=numpy.repeat(numpy.inf, self.B_vars.shape[0]), where=B_inv_N_j[index]!=0)
            E[index, index] = numpy.divide(1, B_inv_N_j[index], out=numpy.repeat(numpy.inf, 1), where=B_inv_N_j[index]!=0)
            return numpy.dot(E, B_inv)
        def _perform_ub_substitution(A, b, C, variable, limit, substituted_vars):
            b = b - limit*A[:, variable]
            A[:, variable] = -A[:, variable]
            C[variable] = -C[variable]
            substituted_vars.append(variable)
            return A, b, C, substituted_vars
        def _update_basis(B_vars, N_vars, incoming, outgoing):
            _tmp = B_vars[outgoing]
            B_vars[outgoing] = N_vars[incoming]
            N_vars[incoming] = _tmp
            return B_vars, N_vars
        if phaseI:
            C = self.C_phaseI
        else:
            C = self.C

        N = self.A[:, self.N_vars]
        C_N = C[self.N_vars] # Objective function for non-basis variables
        C_B = C[self.B_vars] # Objective function for basis variables
        C_tilde_N = C_N - numpy.dot(numpy.dot(C_B, self.B_inv), N) # Reduced cost for non-basis variables
        # Begin simplex iterations
        X_B = numpy.dot(self.B_inv, self.b) # X values a current solution
        while((C_tilde_N>0).any()):
            incoming_candidates = self.N_vars[C_tilde_N == C_tilde_N.max()]
            if incoming_candidates.shape[0] == 1 or not numpy.isin(incoming_candidates, self.prio_list).any():
                incoming_variable = incoming_candidates[0]
            else:
                incoming_candidates = incoming_candidates[numpy.isin(incoming_candidates, self.prio_list)]
                incoming_variable = incoming_candidates[numpy.array([numpy.where(x==self.prio_list)[0][0] for x in incoming_candidates]).argmin()]
            incoming_variable_index = numpy.where(self.N_vars == incoming_variable)[0][0]
            incoming_variable = self.N_vars[incoming_variable_index]
            B_inv_N_j = numpy.dot(self.B_inv, self.A[:, incoming_variable])
            _t1 = numpy.divide(X_B, B_inv_N_j, out=numpy.repeat(numpy.inf, X_B.shape[0]), where=B_inv_N_j>0)
            t1 = _t1.min()
            t2 = self.bounds_upper[incoming_variable]
            _t3 = numpy.divide(self.bounds_upper[self.B_vars]-X_B, -B_inv_N_j, out=numpy.repeat(numpy.inf, X_B.shape[0]), where=B_inv_N_j<0)
            t3 = _t3.min()
            if min(t1, t2, t3) == numpy.inf:
                self.B_inv = _update_constraint_column(B_inv_N_j, 0, self.B_inv)
                self.B_vars, self.N_vars = _update_basis(self.B_vars, self.N_vars, incoming_variable_index, 0)
                return
            if (t3 < t1 and t3 < t2):
                outgoing_variable_index = _t3.argmin()
                outgoing_variable = self.B_vars[outgoing_variable_index]
                self.A, self.b, C, self.substituted_vars = _perform_ub_substitution(self.A, self.b, C, outgoing_variable, self.bounds_upper[outgoing_variable], self.substituted_vars)
                B_inv_tilde = _update_constraint_column(B_inv_N_j, outgoing_variable_index, self.B_inv)
                self.B_vars, self.N_vars = _update_basis(self.B_vars, self.N_vars, incoming_variable_index, outgoing_variable_index)
            elif (t2 < t1 and t2 <= t3):
                self.A, self.b, C, self.substituted_vars = _perform_ub_substitution(self.A, self.b, C, incoming_variable, self.bounds_upper[incoming_variable], self.substituted_vars)
                B_inv_tilde = self.B_inv
            else: # t1 is smallest
                outgoing_variable_index = _t1.argmin()
                outgoing_variable = self.B_vars[outgoing_variable_index]
                B_inv_tilde = _update_constraint_column(B_inv_N_j, outgoing_variable_index, self.B_inv)
                self.B_vars, self.N_vars = _update_basis(self.B_vars, self.N_vars, incoming_variable_index, outgoing_variable_index)
            self.B_inv = B_inv_tilde
            C_N = C[self.N_vars]
            C_B = C[self.B_vars]
            C_tilde_N = C_N - numpy.dot(numpy.dot(C_B, B_inv_tilde), self.A[:, self.N_vars])
            X_B = numpy.dot(self.B_inv, self.b) # X values a current solution
        if phaseI:
            self.C_phaseI = C
        else:
            self.C = C

    def _phaseI(self):
        n_orig_and_slack_vars = self.A.shape[1]
        n_artificial_vars = 0
        self.B_vars = numpy.array(range(n_orig_and_slack_vars-self.A.shape[0], n_orig_and_slack_vars))
        self.B_inv = numpy.linalg.inv(self.A[:, self.B_vars])
        valid_constraints_at_origo = (numpy.dot(self.B_inv, self.b) >= 0)
        if valid_constraints_at_origo.all():
            # origio is a BFS
            self.bfs = True
            self.N_vars = numpy.array(range(n_orig_and_slack_vars-self.A.shape[0]))
            self.B_inv = numpy.linalg.inv(self.A[:, self.B_vars])
        else:
            self.C_phaseI = numpy.zeros(n_orig_and_slack_vars)
            self.A = numpy.append(numpy.eye(self.A.shape[0])[:,~valid_constraints_at_origo], self.A, axis=1)
            n_artificial_vars = self.A.shape[1] - n_orig_and_slack_vars
            self.B_vars = self.B_vars + n_artificial_vars
            self.bounds_lower = numpy.append(numpy.repeat(0, n_artificial_vars), self.bounds_lower)
            self.bounds_upper = numpy.append(numpy.repeat(numpy.inf, n_artificial_vars), self.bounds_upper)
            self.C_phaseI = numpy.append(-numpy.ones(n_artificial_vars), self.C_phaseI)
            self.N_vars = numpy.append(numpy.array(range(n_artificial_vars, n_orig_and_slack_vars-self.A.shape[0]+n_artificial_vars)), numpy.array(range(self.A.shape[1]-self.A.shape[0], self.A.shape[1]))[~valid_constraints_at_origo])
            self.B_vars[~valid_constraints_at_origo] = numpy.array(range(n_artificial_vars))
            self.B_inv = numpy.linalg.inv(self.A[:, self.B_vars])
            self._revised_simplex_method(phaseI=True)
            if (numpy.dot(self.C_phaseI[self.B_vars], numpy.dot(self.B_inv, self.b)))<0:
                # No feasible solution exists
                self.bfs = False
                pass
            else:
                self.N_vars = self.N_vars[self.N_vars>=n_artificial_vars]-n_artificial_vars
                self.A = self.A[:, n_artificial_vars:]
                self.B_vars = self.B_vars - n_artificial_vars
                self.bounds_lower = self.bounds_lower[n_artificial_vars:]
                self.bounds_upper = self.bounds_upper[n_artificial_vars:]
                self.substituted_vars = numpy.array(self.substituted_vars)-n_artificial_vars
                self.bfs = True
    
    def _phaseII(self, objective_function):
        self.C = numpy.append(objective_function, numpy.zeros(self.A.shape[0]))
        if len(self.substituted_vars)>0:
                self.C[self.substituted_vars] = -self.C[self.substituted_vars]
        self._revised_simplex_method(phaseI=False)

    #def revised_simplex(A, b, c, bounds):
    def run(self, objective_function):
        if not self.bfs:
            # PhaseI
            self._phaseI()
            if not self.bfs:
                self.solution_information = "No feasible solution exists"
                return (None, None, None, self.solution_information)
        # PhaseII
        self._phaseII(objective_function)
        
        if (numpy.dot(self.B_inv, self.b) == numpy.inf).any():
            self.solution_information = "Solution is unbounded"
        elif (self.C[self.N_vars] - numpy.dot(numpy.dot(self.C[self.B_vars], self.B_inv), self.A[:, self.N_vars])==0).any():
            self.solution_information = "Solution is not unique"
        else:
            self.solution_information = "Solution is unique"
        X_B = numpy.dot(self.B_inv, self.b)
        X_B = numpy.where(numpy.in1d(self.B_vars, numpy.array(self.substituted_vars)), self.bounds_upper[self.B_vars] - X_B, X_B)
        X_N = numpy.where(numpy.in1d(self.N_vars, numpy.array(self.substituted_vars)), self.bounds_upper[self.N_vars], 0)
        X_B = numpy.concatenate((X_B, X_N[X_N>0]), axis=0)
        B_vars_sol = numpy.concatenate((self.B_vars, self.N_vars[X_N>0]), axis=0)
        X_B[B_vars_sol[(B_vars_sol >= self.ORIG_VARS) & (B_vars_sol < self.ORIG_VARS + self.ADDED_VARS)]] = -X_B[B_vars_sol[(B_vars_sol >= self.ORIG_VARS) & (B_vars_sol < self.ORIG_VARS + self.ADDED_VARS)]]
        B_vars_sol[(B_vars_sol >= self.ORIG_VARS) & (B_vars_sol < self.ORIG_VARS + self.ADDED_VARS)] = B_vars_sol[(B_vars_sol >= self.ORIG_VARS) & (B_vars_sol < self.ORIG_VARS + self.ADDED_VARS)] - self.ORIG_VARS
        sol = numpy.zeros(self.ORIG_VARS)
        sol[B_vars_sol[B_vars_sol<self.ORIG_VARS]] = X_B[B_vars_sol<self.ORIG_VARS]
        z = numpy.dot(objective_function[:self.ORIG_VARS], sol)
        return (z, sol, numpy.dot(self.B_inv, self.A), self.solution_information)
    
    def run_ldd(self, polyhedron, bounds, objective_function):
        z_ub = numpy.inf
        x = None
        solution_information = "No feasible solution exists"
        nodes_to_explore = [(polyhedron, bounds, objective_function)]
        explored_nodes = []
        n_vars = polyhedron.shape[1]
        while(nodes_to_explore):
            polyhedron, bounds, objective_function = nodes_to_explore.pop()
            self._setup(polyhedron, bounds, objective_function, [])
            (z, X_B, B_inv, solution_information) = self.run(objective_function)
            if not z:
                # No feasible solution
                continue
            if z > z_ub:
                # Not  possible to find better solution in this area
                continue
            first_non_integer = numpy.array([not i%1 for i in X_B]).argmin()
            if X_B[first_non_integer].is_integer():
                # We've found a best solution in this area
                if z < z_ub:
                    z_ub = z
                    x = X_B
                    B_inv = self.B_inv
                    A = self.A
                continue
            else:
                current_node = (first_non_integer, math.ceil(X_B[first_non_integer]))
                if current_node not in explored_nodes:
                    cond1 = numpy.zeros((1,n_vars))
                    cond1[0,first_non_integer+1]=-1
                    cond1[0][0] = -math.floor(X_B[first_non_integer])
                    nodes_to_explore.append((puan.ndarray.ge_polyhedron(numpy.append(polyhedron.copy(), cond1, axis=0)), bounds, objective_function))
                    cond2 = numpy.zeros((1,n_vars))
                    cond2[0,first_non_integer+1]=1
                    cond2[0][0] = math.ceil(X_B[first_non_integer])
                    nodes_to_explore.append((puan.ndarray.ge_polyhedron(numpy.append(polyhedron.copy(), cond2, axis=0)), bounds, objective_function))
                    explored_nodes.append(current_node)
        return (z_ub.astype("int"), x, numpy.dot(B_inv, A), solution_information)

our_revised_simplex = Our_simplex_solver()