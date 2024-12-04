#!/usr/bin/env python
# coding: utf-8
import time
from itertools import product, combinations
import numpy as np
import pandas as pd
import gurobipy as gb
from sklearn.linear_model import LinearRegression

# WLS credentials
WLSACCESSID = 'ccc2c36a-db14-4956-b2e3-60adc45e9957'
WLSSECRET = '1e0e3dbf-7933-44dc-8f81-e0482ded7ac8'
LICENSEID = 2586688

# Create the Gurobi environment with parameters
try:
    env = gb.Env(empty=True)  # Start with an empty environment
    env.setParam('WLSACCESSID', WLSACCESSID)
    env.setParam('WLSSECRET', WLSSECRET)
    env.setParam('LICENSEID', LICENSEID)
    env.start()  # Start the environment
except gb.GurobiError as e:
    print(f"Error initializing Gurobi environment: {e}")
    exit(1)

print("Gurobi environment successfully initialized.")

# List of experiments
for SOCIAL_CATEGORIES, NO_HARM, EXPERIMENT_DESCRIPTION in [
    (['A', 'B', 'C', 'D', 'E', 'F', 'G'], False, '7_disagg'),
    (['A', 'B', 'C', 'D', 'E', 'F', 'G'], True, '7_disagg_no_harm'),
    (['A_m', 'B_m', 'C_m', 'D_m', 'E_m', 'F_m', 'G_m',
      'A_f', 'B_f', 'C_f', 'D_f', 'E_f', 'F_f', 'G_f'], False, '14_disagg'),
    (['A_m', 'B_m', 'C_m', 'D_m', 'E_m', 'F_m', 'G_m',
      'A_f', 'B_f', 'C_f', 'D_f', 'E_f', 'F_f', 'G_f'], True, '14_disagg_no_harm')]:
    
    print(f"Starting experiment: {EXPERIMENT_DESCRIPTION}")
    
    count_columns = [f'n_{category}' for category in SOCIAL_CATEGORIES]
    frac_columns = [f'frac_{category}' for category in SOCIAL_CATEGORIES]
    y_frac_columns = [f'frac_sat_act_{category}' for category in SOCIAL_CATEGORIES]
    y_count_columns = [f'n_sat_act_{category}' for category in SOCIAL_CATEGORIES]
    X_columns = ['frpl_rate', 'calculus', 'ap_ib', 'counselors']

    # Load data
    df = pd.read_csv('features.csv')

    X = df[X_columns]
    n = len(X)
    neighbor_distance_matrix = np.load('neighbor_distance_matrix.npy')
    Li = X['frpl_rate'].values.reshape(n, 1)
    Li = np.ones_like(Li)
    A_frac = df[frac_columns].values

    calculus = X['calculus'].values
    max_Sij_Cj = np.max(neighbor_distance_matrix * calculus.T, axis=1).reshape(n, 1)
    a_Li_max_Sij_Cj = A_frac * Li * max_Sij_Cj

    ap_ib = X['ap_ib'].values
    max_Sij_Pj = np.max(neighbor_distance_matrix * ap_ib.T, axis=1).reshape(n, 1)
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

    # Fit the linear model
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

    # Optimization function
    def optimize_interventions():
        print('=' * 80)
        print(f'Running optimization for budget={BUDGET}')
        try:
            model = gb.Model(env=env)  # Pass the environment to the model

            interventions = model.addVars(
                list(range(NUM_SCHOOLS)),
                lb=0,
                ub=1,
                vtype=gb.GRB.BINARY
            )

            # Add budget constraint
            num_interventions = gb.LinExpr()
            num_interventions += sum(interventions.values())
            model.addConstr(num_interventions <= BUDGET, name='intervention_budget')

            # Objective function placeholder (to be implemented)
            model.setObjective(num_interventions, gb.GRB.MINIMIZE)

            model.optimize()

            if model.status == gb.GRB.Status.OPTIMAL:
                print('Optimal solution found.')
                return np.array([interventions[i].X for i in range(NUM_SCHOOLS)])
            else:
                raise InfeasibleModelError('Optimization failed.')
        except gb.GurobiError as e:
            print(f"Gurobi error during optimization: {e}")
            exit(1)

    # Run optimization
    BUDGET = 100
    NUM_SCHOOLS = len(df)
    try:
        optimal_interventions = optimize_interventions()
        print(f"Optimal interventions: {optimal_interventions}")
    except InfeasibleModelError as e:
        print(f"Error: {e}")
