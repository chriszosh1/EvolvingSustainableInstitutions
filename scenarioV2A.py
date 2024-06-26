from hillclimbing import  Iterated_Local_Search
import time
from model import Model, run_model
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os

#Version summary
'''
   As before, Agents play a public goods game
   facing some policy. Agents must learn and decide
   how much they should harvest (from {0,1,...,5})
   to maximize their own benefit.

   According to theory:
    -Socially optimal choice: 2
    -Individually optimal choice: max = 5
   
   ---The new stuff---

   In this Scenario, we introduce a social planner
   who aims to pick policy to maximize social welfare.
   These fines take the form of a vector, where each
   entry is the amount of fine an agent would have to pay
   if the agent chose the action corresponding to the 
   position of the fine in the list.
   eg. suppose fine_vector = [0, 2, 4, 6, 8, 10],
       (Note the first position is for taking nothing.)
       then if I chose to harvest 1, I would get a fine
       of 2.
     
    According to theory:
    -Optimal policy (fine structure): Suppose associated with choosing
    the socially optimal level (2) is denoted f(c_so)
    Then an optimal fine structure is any vector of fines st..
    1) for all choices c != 2, EU(PO(c) - f(c)) <= EU(PO(c_so) - f(c_so))
'''

def model_helper_fnc(agent_count, game_variables, agent_variables, verbose_variables, fine_vector,
                     data_to_output, social_planner_vars, steps, return_indv_welfare, return_last_act):
    #Running the model:
    trial = Model(agent_count = agent_count, game_variables = game_variables, agent_variables = agent_variables,
                        verbose_variables = verbose_variables, fine_vector = fine_vector,
                        data_to_output = data_to_output, social_planner_vars = social_planner_vars)
    run_model(model = trial, steps = steps)
    #Creating desired output:
    output = {}
    if return_indv_welfare:
        output['indv_welfare'] = [0]*agent_count
        for a in range(agent_count):
            output['indv_welfare'][a] += trial.agents[a].lifetime_felicity
    output['social_welfare'] = trial.avg_social_welfare
    if return_last_act:
        output['last_act'] = {}
        for a_v in trial.agents.values():
            for var_name, var_val in a_v.period_choices.items():
                if var_name not in output['last_act']:
                    output['last_act'][var_name] = []
                output['last_act'][var_name].append(var_val)
    return output

def run_run_model(agent_count, game_variables, agent_variables, verbose_variables, 
                  fine_vector, data_to_output, steps, trials_per_policy, social_planner_vars = None,
                  return_last_act = False, return_indv_welfare = False, cores = 1):
    '''A function which finds the expected social welfare given a policy (fine vector) fv.'''
    parallel = (cores > 1)
    run_results = []
    if not parallel:
        for r in range(trials_per_policy):
            run_results.append(model_helper_fnc(agent_count = agent_count, game_variables = game_variables, agent_variables = agent_variables,
                        verbose_variables = verbose_variables, fine_vector = fine_vector,
                        data_to_output = data_to_output, social_planner_vars = social_planner_vars, steps = steps,
                        return_indv_welfare=return_indv_welfare, return_last_act=return_last_act))
    else:
        para_results = []
        with ProcessPoolExecutor(os.cpu_count()) as threadpool:
            for r in range(trials_per_policy):
                para_results.append(threadpool.submit(model_helper_fnc, agent_count = agent_count, game_variables = game_variables, agent_variables = agent_variables,
                        verbose_variables = verbose_variables, fine_vector = fine_vector,
                        data_to_output = data_to_output, social_planner_vars = social_planner_vars, steps = steps,
                        return_indv_welfare=return_indv_welfare, return_last_act=return_last_act))
            threadpool.shutdown(wait = True)
            for r in para_results:
                run_results.append(r.result())
    #print(f'Run Results:\n{run_results}\n\n')
    #COLLECTING RESULTS:
    rr_results = {}
    #Collecting avg social welfare:
    csw = 0
    for pos in range(len(run_results)):
        csw += run_results[pos]['social_welfare']
    csw = csw / trials_per_policy
    rr_results['csw'] = csw
    #Collecting avg iw for each agent:
    if return_indv_welfare:
        iw = [0]*agent_count
        for pos in range(len(run_results)):
            iw = [a + b for a, b in zip(iw, run_results[pos]['indv_welfare'])]
    rr_results['iw'] = iw
    #Collecting each final action taken by players: <---- Double check this grooves with it's existing use case...
    if return_last_act:
        final_choices = []
        for pos in range(len(run_results)):
            final_choices.append(run_results[pos]['last_act'])
        rr_results['last_act'] = final_choices
    return rr_results

#Game params:
gv = {'return_type': 'negative_externality', 'landscape_size':1, 'alpha':.8, 'beta':.05} 
#Shepherd params:
av = {'vision':1, 'action_set':{'sheep_count_if_seen': {'min': 0, 'max': 5, 'grain':1}}}
#Output params:
vv = []
dto = {'tag': 'S2_test', 'files':[]}

#Search params:
sv = {'distribution': 'normal', 'explore_range': .1, 'prob_mutate':.5, 'cores':3} #Note: Cores is for how many // processes allowed
ils_sv = {'distribution': 'normal', 'ils_depth': 20, 'explore_range': 2, 'downhill_coeff': None}

#Fitness function (run_run_model) params:
pv = {'fine_vector':{'type':'vector','min':0, 'max':10, 'size':6}}
ffa = {'agent_count': 4, 'game_variables': gv, 'agent_variables': av, 'verbose_variables': vv,
        'data_to_output': dto, 'steps':50_000, 'trials_per_policy':10, 'social_planner_vars':pv, 'return_last_act': True}

def main():
    starttime = time.time()
    result = Iterated_Local_Search(param_vars=pv, hc_search_vars = sv, pop_size = sv['cores']-1, hc_depth = 500,
                                    fitness_fnc = run_run_model, ils_depth = 5, ils_search_vars = ils_sv,
                                    fitness_fnc_args = ffa, verbose = False, output_to_file = True)
    print(f'result: {result}')
    print(f'Runtime: {time.time() - starttime}')

if __name__ == '__main__':
    main()