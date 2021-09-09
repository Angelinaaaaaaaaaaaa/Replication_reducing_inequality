#!/usr/bin/env python
# coding: utf-8
# code adapted from https://github.com/mkusner/reducing_discriminatory_impact

import time
from itertools import product, combinations

import numpy as np
import pandas as pd
import gurobipy as gb
from sklearn.linear_model import LinearRegression


SOCIAL_CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

for TAU_VALUE in [0.566, None]:
    count_columns = [f'n_{category}' for category in SOCIAL_CATEGORIES]
    frac_columns = [f'frac_{category}' for category in SOCIAL_CATEGORIES]
    y_frac_columns = ['frac_sat_act']
    X_columns = ['frpl_rate', 'calculus', 'ap_ib', 'counselors']

    df = pd.read_csv('features.csv')

    X = df[X_columns]
    n = len(X)
    neighbor_distance_matrix = np.load('neighbor_distance_matrix.npy')

    A_frac = df[frac_columns]

    ap_ib = X['ap_ib'].values
    max_Sij_Pj = np.max(neighbor_distance_matrix * ap_ib.T,
                        axis=1).reshape(n, 1)
    a_max_Sij_Pj = A_frac * max_Sij_Pj

    calculus = X['calculus'].values
    max_Sij_Cj = np.max(neighbor_distance_matrix * calculus.T,
                        axis=1).reshape(n, 1)
    a_max_Sij_Cj = A_frac * max_Sij_Cj

    counselors = X['counselors'].values
    a_Fj = A_frac * counselors.reshape(n, 1)

    a = A_frac

    X_train = np.concatenate((
        a_max_Sij_Pj,
        a_max_Sij_Cj,
        a_Fj,
        a
    ), axis=1)
    y_train = df['frac_sat_act'].values

    linmod = LinearRegression(fit_intercept=False).fit(X_train, y_train)
    model_weights = linmod.coef_
    param_dims = len(SOCIAL_CATEGORIES)
    weight_dict = {
        'alpha': model_weights[param_dims:param_dims*2],
        'beta': model_weights[:param_dims],
        'gamma': model_weights[param_dims*2:param_dims*3],
        'theta': model_weights[-param_dims:],
    }
    params = pd.DataFrame(weight_dict)

    ALPHA, BETA, GAMMA, THETA = (params['alpha'].values, params['beta'].values, 
                                 params['gamma'].values, params['theta'].values)

    AP_IB = X['ap_ib'].values
    COUNSELORS = X['counselors'].values
    FRPL = np.ones_like(X['frpl_rate'].values)
    A_FRAC = df[frac_columns]
    A_MATRIX = A_FRAC.values
    NEIGHBOR_INDEX_MATRIX = np.load('neighbor_index_matrix.npy')
    NEIGHBOR_DISTANCE_MATRIX = np.load('neighbor_distance_matrix.npy')

    NUM_SCHOOLS = X.shape[0]
    weight_df = pd.read_csv('params_7_disagg.csv', index_col=0)
    WEIGHT_MATRIX = weight_df.values
    NUM_NEIGHBORS = NEIGHBOR_INDEX_MATRIX.shape[1]
    intervention_sample_spaces = [(0, 1)] * NUM_NEIGHBORS
    POSSIBLE_INTERVENTIONS_MATRIX = np.array(list(
        product(*intervention_sample_spaces)
    ))
    NUM_POSSIBLE_INTERVENTIONS = POSSIBLE_INTERVENTIONS_MATRIX.shape[0]

    BUDGET = 100

    NUM_CATEGORIES = WEIGHT_MATRIX.shape[0]
    CATEGORIES = list(range(NUM_CATEGORIES))
    CATEGORY_PAIRS = list(combinations(CATEGORIES, 2))

    DEMOGRAPHIC_COUNTERFACTUALS = [0, 1]
    NUM_COUNTERFACTUALS = len(DEMOGRAPHIC_COUNTERFACTUALS)

    TOTAL_STUDENTS = df['total_students'].values
    R_COUNTS = df[count_columns].values
    R_COUNTS_TOTAL = R_COUNTS.sum(axis=0)

    CALCULUS = X['calculus']
    A_DIMENSION = A_MATRIX.shape[1]
    
    WHETHER_OR_NOT_CALCULUS_GIVEN_INTERFERENCE = np.max(
        NEIGHBOR_DISTANCE_MATRIX * CALCULUS.values, axis=1)


    def expected_impact_i(index, intervention_array, a):
        nearest_neighbor_indices = NEIGHBOR_INDEX_MATRIX[index, :]
        neighbor_distances = NEIGHBOR_DISTANCE_MATRIX[index,
                                                      nearest_neighbor_indices]
        
        calculus_term = np.dot(a, ALPHA) * np.max(
            neighbor_distances * intervention_array
        )
    
        ap_ib_term = np.dot(a, BETA) * np.max(
            neighbor_distances * AP_IB[nearest_neighbor_indices]
        )
    
        counselors_term = np.dot(a, GAMMA) * COUNSELORS[index]
    
        race_term = np.dot(a, THETA)
    
        impact = calculus_term + ap_ib_term + counselors_term + race_term
    
        impact = max(min(impact, 1), 0)
    
        return impact
    
    
    def expected_impact(intervention_arrays):
        impact = 0
        for i in range(NUM_SCHOOLS):
            factual_A_dist_i = A_MATRIX[np.newaxis, i, :]
            impact += expected_impact_i(i, intervention_arrays[i], factual_A_dist_i)
        return impact[0]


    def all_possible_impacts(index, a):
        possible_impacts = np.empty(NUM_POSSIBLE_INTERVENTIONS)
        for k in range(NUM_POSSIBLE_INTERVENTIONS):
            intervention_array = POSSIBLE_INTERVENTIONS_MATRIX[k]
            possible_impacts[k] = expected_impact_i(index, intervention_array, a)

        return possible_impacts


    def optimize_interventions(tau_value):
        print(f'Running optimization for tau={tau_value}')
        model = gb.Model()

        interventions = model.addVars(
            list(range(NUM_SCHOOLS)),
            lb=0,
            ub=1,
            vtype=gb.GRB.BINARY
        )

        num_interventions = gb.LinExpr()
        num_interventions += sum(interventions.values())
        model.addConstr(num_interventions, gb.GRB.LESS_EQUAL, BUDGET, 'intervention_budget')

        def add_auxiliary_constraint(index, tau_value=None):
            factual_A_dist = A_MATRIX[np.newaxis, index, :]
            possible_factual_impacts = all_possible_impacts(index, a=factual_A_dist)

            auxiliary_variables = model.addVars(
                list(range(NUM_POSSIBLE_INTERVENTIONS)),
                lb=0,
                ub=1,
                obj=possible_factual_impacts,
                vtype=gb.GRB.CONTINUOUS
            )
            model.update()

            for j in range(NUM_POSSIBLE_INTERVENTIONS):
                for k in range(NUM_NEIGHBORS):
                    which_neighbor = NEIGHBOR_INDEX_MATRIX[index, k]
                    intervention_or_not = interventions[which_neighbor]
                    if POSSIBLE_INTERVENTIONS_MATRIX[j, k] == 1:
                        model.addConstr(
                            auxiliary_variables[j] <=
                            intervention_or_not
                        )
                    else:
                        model.addConstr(
                            auxiliary_variables[j] <=
                            1 - intervention_or_not
                        )

            model.addConstr(auxiliary_variables.sum() == 1)

            if tau_value is not None:
                counterfactual_impacts = np.empty((A_DIMENSION,
                                                   NUM_POSSIBLE_INTERVENTIONS))
                counterfactual_A_dists = np.eye(A_DIMENSION)
                for i, counterfactual_A_dist in enumerate(counterfactual_A_dists):
                    counterfactual_impacts[i] = all_possible_impacts(
                        index,
                        a=counterfactual_A_dist
                    )

                counterfactual_privilege = (-counterfactual_impacts +
                                            possible_factual_impacts)

                for a_dim in range(A_DIMENSION):
                    model.addConstr(sum(
                        auxiliary_variables[j] * counterfactual_privilege[a_dim, j]
                        for j in range(NUM_POSSIBLE_INTERVENTIONS)
                    ) <= tau_value)

            return auxiliary_variables

        all_auxiliary_variables = list(
            map(
                lambda index: add_auxiliary_constraint(index, tau_value=tau_value),
                range(NUM_SCHOOLS)
            )
        )

        model.setObjective(model.getObjective(), gb.GRB.MAXIMIZE)
        model.optimize()

        if model.status == gb.GRB.Status.OPTIMAL:
            print('Optimal solution found for:')
            optimal_interventions = np.array(
                [interventions[i].X for i in range(len(interventions))]
            ).round().astype(bool)
        else:
            raise InfeasibleModelError('optimization failed.')

        return optimal_interventions


    class InfeasibleModelError(Exception):
        pass


    def expected_disagg_impact_i(index, intervention_array, a):
        nearest_neighbor_indices = NEIGHBOR_INDEX_MATRIX[index, :]
        neighbor_distances = NEIGHBOR_DISTANCE_MATRIX[index,
                                                      nearest_neighbor_indices]

        calculus_value = a * max(
            WHETHER_OR_NOT_CALCULUS_GIVEN_INTERFERENCE[index], 
            np.max(neighbor_distances * intervention_array)
        )

        ap_ib_value = a * np.max(
            neighbor_distances * AP_IB[nearest_neighbor_indices]
        )

        counselors_value = a * COUNSELORS[index]
        frpl_value = a
        r_frac = A_FRAC.iloc[index].values
        calculus_value *= r_frac
        ap_ib_value *= r_frac
        counselors_value *= r_frac
        frpl_value *= r_frac

        vector = np.concatenate((
            calculus_value,
            ap_ib_value,
            counselors_value,
            frpl_value
        ), axis=0)

        impact = np.matmul(WEIGHT_MATRIX, vector)

        for i in range(len(impact)):
            impact[i] = max(min(impact[i], 1), 0)

        return impact


    def expected_disaggregated_impact(intervention_arrays):
        """Compute sum_i(E[Y^{(i)}(a^{(i)}, z)]))."""
        disaggregated_impact = np.zeros((NUM_CATEGORIES,))
        for i in range(NUM_SCHOOLS):
            factual_A_i = FRPL[i]
            disaggregated_impact += expected_disagg_impact_i(
                i, intervention_arrays[i], factual_A_i
            ) * R_COUNTS[i]
        return disaggregated_impact


    def absolute_expected_impact_gap(intervention_arrays):
        impact_gap = 0
        for r, r_prime in CATEGORY_PAIRS:
            total_r = 0
            total_r_prime = 0
            for i in range(NUM_SCHOOLS):
                factual_A_i = FRPL[i]
                impact_by_r = expected_disagg_impact_i(i, intervention_arrays[i], factual_A_i)
                total_r += impact_by_r[r] * R_COUNTS[i, r]
                total_r_prime += impact_by_r[r_prime] * R_COUNTS[i, r_prime]
            rate_r = total_r / R_COUNTS_TOTAL[r]
            rate_r_prime = total_r_prime / R_COUNTS_TOTAL[r_prime]
            absolute_gap = abs(rate_r - rate_r_prime)
            impact_gap += absolute_gap
        return impact_gap


    if TAU_VALUE is not None:
        optimal_interventions = None
        while optimal_interventions is None:
            try:
                optimal_interventions = optimize_interventions(TAU_VALUE)
            except InfeasibleModelError:
                print(f'Tau={TAU_VALUE} infeasible.')
                TAU_VALUE += 0.001
                print(f'Trying Tau={TAU_VALUE}.')
        EXPERIMENT_DESCRIPTION = f'7_original_tau_{TAU_VALUE * 10 ** 3:0.0f}'
    else:
        optimal_interventions = optimize_interventions(TAU_VALUE)
        EXPERIMENT_DESCRIPTION = '7_original_tau_none'

    all_neighbor_interventions = np.isin(
        NEIGHBOR_INDEX_MATRIX, np.where(optimal_interventions))
    impact_gap = absolute_expected_impact_gap(all_neighbor_interventions)
    print('='*80)
    print(f'Experiment: {EXPERIMENT_DESCRIPTION}')
    print(f'Tau: {TAU_VALUE}')
    print(f'Impact gap: {impact_gap}')

    print('Optimal interventions:')
    print(np.where(optimal_interventions))

    intervention_arrays = []
    null_arrays = []
    for i in range(NUM_SCHOOLS):
        nearest_neighbor_indices = NEIGHBOR_INDEX_MATRIX[i, :]
        intervention_arrays.append(optimal_interventions[nearest_neighbor_indices])
        null_arrays.append(np.zeros(6))

    null_intervention_counts = expected_disaggregated_impact(null_arrays)
    post_intervention_counts = expected_disaggregated_impact(intervention_arrays)

    print(f'R Counts: {R_COUNTS_TOTAL}')
    print(f'Null counts: {null_intervention_counts}')
    print(f'Post counts: {post_intervention_counts}')

    null_rates = null_intervention_counts / R_COUNTS_TOTAL
    post_rates = post_intervention_counts / R_COUNTS_TOTAL

    print(f'Percentage rate change: {(post_rates - null_rates) / null_rates * 100}')

    print(f'Null rates: {null_rates}')
    print(f'Post rates: {post_rates}')

    print(f'Null disparity gap: {absolute_expected_impact_gap(null_arrays)}')
    print(f'Post disparity gap: {absolute_expected_impact_gap(intervention_arrays)}')

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_path = f'{EXPERIMENT_DESCRIPTION}_{timestamp}.npy'
    np.save(file_path, optimal_interventions)
    print(f'Results output to file: {file_path}')
    print('=' * 80)
