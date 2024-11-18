#!/usr/bin/env python
# coding: utf-8
# code adapted from https://github.com/mkusner/reducing_discriminatory_impact

import time
from itertools import product, combinations

import os

os.environ['GRB_WLSACCESSID'] = 'ccc2c36a-db14-4956-b2e3-60adc45e9957'
os.environ['GRB_WLSSECRET'] = '1e0e3dbf-7933-44dc-8f81-e0482ded7ac8'
os.environ['GRB_LICENSEID'] = '2586688'

import numpy as np
import pandas as pd
import gurobipy as gb
from sklearn.linear_model import LinearRegression


for SOCIAL_CATEGORIES, NO_HARM, EXPERIMENT_DESCRIPTION in [
    (['A', 'B', 'C', 'D', 'E', 'F', 'G'], False, '7_disagg'),
    (['A', 'B', 'C', 'D', 'E', 'F', 'G'], True, '7_disagg_no_harm'),
    (['A_m', 'B_m', 'C_m', 'D_m', 'E_m', 'F_m', 'G_m',
      'A_f', 'B_f', 'C_f', 'D_f', 'E_f', 'F_f', 'G_f'], False, '14_disagg'),
    (['A_m', 'B_m', 'C_m', 'D_m', 'E_m', 'F_m', 'G_m',
      'A_f', 'B_f', 'C_f', 'D_f', 'E_f', 'F_f', 'G_f'], True, '14_disagg_no_harm')]:
    count_columns = [f'n_{category}' for category in SOCIAL_CATEGORIES]
    frac_columns = [f'frac_{category}' for category in SOCIAL_CATEGORIES]
    y_frac_columns = [f'frac_sat_act_{category}' for category in SOCIAL_CATEGORIES]
    y_count_columns = [f'n_sat_act_{category}' for category in SOCIAL_CATEGORIES]
    X_columns = ['frpl_rate', 'calculus', 'ap_ib', 'counselors']

    df = pd.read_csv('features.csv')

    X = df[X_columns]
    n = len(X)
    neighbor_distance_matrix = np.load('neighbor_distance_matrix.npy')
    Li = X['frpl_rate'].values.reshape(n, 1)
    Li = np.ones_like(Li)
    A_frac = df[frac_columns].values

    calculus = X['calculus'].values
    max_Sij_Cj = np.max(neighbor_distance_matrix * calculus.T,
                        axis=1).reshape(n, 1)
    a_Li_max_Sij_Cj = A_frac * Li * max_Sij_Cj

    ap_ib = X['ap_ib'].values
    max_Sij_Pj = np.max(neighbor_distance_matrix * ap_ib.T,
                        axis=1).reshape(n, 1)
    a_Li_max_Sij_Pj = A_frac * Li * max_Sij_Pj

    counselors = X['counselors'].values
    a_Li_Fi = A_frac * Li * counselors.reshape(n, 1)

    a_Li = A_frac * Li

    X_train = np.concatenate((
        a_Li_max_Sij_Cj,
        a_Li_max_Sij_Pj,
        a_Li_Fi,
        a_Li
    ), axis=1)

    y_train = df[y_frac_columns].values

    linmod = LinearRegression(fit_intercept=False).fit(X_train, y_train)
    model_weights = linmod.coef_

    column_names = [
        f'{param}_{category}'
        for param in ['alpha', 'beta', 'gamma', 'theta']
        for category in SOCIAL_CATEGORIES
    ]
    row_names = SOCIAL_CATEGORIES
    weight_df = pd.DataFrame(model_weights, columns=column_names)
    weight_df.index = row_names
    if EXPERIMENT_DESCRIPTION == '7_disagg':
        weight_df.to_csv(f'params_7_disagg.csv')
    print('Parameters fit.')

    AP_IB = X['ap_ib'].values
    COUNSELORS = X['counselors'].values
    FRPL = np.ones_like(X['frpl_rate'].values)
    R_FRAC = df[frac_columns]
    R_MATRIX = R_FRAC.values
    NEIGHBOR_INDEX_MATRIX = np.load('neighbor_index_matrix.npy')
    NEIGHBOR_DISTANCE_MATRIX = np.load('neighbor_distance_matrix.npy')
    NUM_SCHOOLS = X.shape[0]
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
    
    WHETHER_OR_NOT_CALCULUS_GIVEN_INTERFERENCE = np.max(
        NEIGHBOR_DISTANCE_MATRIX * CALCULUS.values, axis=1)


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
        r_frac = R_FRAC.iloc[index].values
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


    class InfeasibleModelError(Exception):
        pass


    def optimize_interventions():
        print('=' * 80)
        print('=' * 80)
        print(f'Running optimization for budget={BUDGET}')
        model = gb.Model()

        interventions = model.addVars(
            list(range(NUM_SCHOOLS)),
            lb=0,
            ub=1,
            vtype=gb.GRB.BINARY
        )

        num_interventions = gb.LinExpr()
        num_interventions += sum(interventions.values())
        # model.addConstr(num_interventions, gb.GRB.LESS_EQUAL, BUDGET, 'intervention_budget')
        model.addConstr(num_interventions <= BUDGET, name='intervention_budget')


        def add_auxiliary_constraint(index):
            auxiliary_variables = model.addVars(
                list(range(NUM_POSSIBLE_INTERVENTIONS)),
                lb=0,
                ub=1,
                obj=0.0,
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

            return auxiliary_variables

        print('Setting auxiliary variables...')
        all_auxiliary_variables = list(
            map(
                lambda index: add_auxiliary_constraint(index),
                range(NUM_SCHOOLS)
            )
        )

        if NO_HARM:
            print('Add no harm constraints...')
            null_interventions = np.zeros((NUM_SCHOOLS, NUM_NEIGHBORS))
            original_expected_rates_by_r = expected_disaggregated_impact(null_interventions) / R_COUNTS_TOTAL

            for r in range(NUM_CATEGORIES):
                total_students_r = 0
                for i in range(NUM_SCHOOLS):
                    factual_A = FRPL[i]
                    num_students_i_r = R_COUNTS[i, r]

                    rate_i_r = 0
                    for j in range(NUM_POSSIBLE_INTERVENTIONS):
                        intervention_array = POSSIBLE_INTERVENTIONS_MATRIX[j]

                        impacts_by_r = expected_disagg_impact_i(i, intervention_array, factual_A)

                        rate_i_r += all_auxiliary_variables[i][j] * impacts_by_r[r]

                    total_students_r += num_students_i_r * rate_i_r

                rate_r = total_students_r / R_COUNTS_TOTAL[r]

                model.addConstr(rate_r >= original_expected_rates_by_r[r])

        print('Building objective function...')
        objective_function_terms = []
        for r, r_prime in CATEGORY_PAIRS:
            total_students_r = 0
            total_students_r_prime = 0
            for i in range(NUM_SCHOOLS):
                factual_A = FRPL[i]
                num_students_i_r = R_COUNTS[i, r]
                num_students_i_r_prime = R_COUNTS[i, r_prime]

                rate_i_r = 0
                rate_i_r_prime = 0
                for j in range(NUM_POSSIBLE_INTERVENTIONS):
                    intervention_array = POSSIBLE_INTERVENTIONS_MATRIX[j]

                    impacts_by_r = expected_disagg_impact_i(i, intervention_array, factual_A)

                    rate_i_r += all_auxiliary_variables[i][j] * impacts_by_r[r]
                    rate_i_r_prime += all_auxiliary_variables[i][j] * impacts_by_r[r_prime]

                total_students_r += num_students_i_r * rate_i_r
                total_students_r_prime += num_students_i_r_prime * rate_i_r_prime

            rate_r = total_students_r / R_COUNTS_TOTAL[r]
            rate_r_prime = total_students_r_prime / R_COUNTS_TOTAL[r_prime]

            objective_function_terms.append(rate_r - rate_r_prime)

        print('Constraining objective function absolute value terms...')
        objective_variables = model.addVars(
            len(objective_function_terms),
            lb=-gb.GRB.INFINITY,
            ub=gb.GRB.INFINITY,
            obj=0.0,
            vtype=gb.GRB.CONTINUOUS
        )
        model.update()

        for i in range(len(objective_function_terms)):
            model.addConstr(objective_variables[i] == objective_function_terms[i])

        absolute_objective_variables = model.addVars(
            len(objective_function_terms),
            lb=0,
            ub=gb.GRB.INFINITY,
            obj=1.0,
            vtype=gb.GRB.CONTINUOUS
        )
        model.update()

        for i in range(len(objective_function_terms)):
            model.addGenConstrAbs(absolute_objective_variables[i], objective_variables[i])

        model.setObjective(model.getObjective(), gb.GRB.MINIMIZE)
        model.optimize()

        if model.status == gb.GRB.Status.OPTIMAL:
            print('Optimal solution found for:')
            optimal_interventions = np.array(
                [interventions[i].X for i in range(len(interventions))]
            ).round().astype(bool)
        else:
            raise InfeasibleModelError('optimization failed.')

        print('=' * 80)

        return optimal_interventions


    def expected_disaggregated_impact(intervention_arrays):
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


    optimal_interventions = optimize_interventions()

    all_neighbor_interventions = np.isin(
        NEIGHBOR_INDEX_MATRIX, np.where(optimal_interventions))
    impact_gap = absolute_expected_impact_gap(all_neighbor_interventions)
    print(f'Experiment: {EXPERIMENT_DESCRIPTION}')
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

    print(f'Percentage rate change: {(post_rates - null_rates)/null_rates * 100}')

    print(f'Null rates: {null_rates}')
    print(f'Post rates: {post_rates}')

    print(f'Null disparity gap: {absolute_expected_impact_gap(null_arrays)}')
    print(f'Post disparity gap: {absolute_expected_impact_gap(intervention_arrays)}')
    print('=' * 80)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_path = f'{EXPERIMENT_DESCRIPTION}_{timestamp}.npy'
    np.save(file_path, optimal_interventions)
    print(f'Results output to file: {file_path}')
