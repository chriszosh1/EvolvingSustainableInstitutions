from hillclimbing import  Iterated_Local_Search
import time
from model import Model, run_model
from scenarioV2A import run_run_model
from copy import deepcopy

def run_model_bundle(gv, av, vv, dto, sv, ils_sv, pv, ffa, tag = '', rrm_fnc = run_run_model, finetune=False):
    '''This function runs 3 version of the model together (listed below).'''
    #---Running V0 - Agents face no policy (altruistic)---:
    starttime = time.time()
    av_v0 = deepcopy(av)
    av_v0['altruism'] = 1
    dto_v0 = deepcopy(dto)
    dto_v0['tag'] = f'v0_{tag}'
    for r in range(ffa['trials_per_policy']):
        scenario1 = Model(agent_count = ffa['agent_count'], game_variables = gv, agent_variables = av_v0,
                        verbose_variables = vv, data_to_output = dto_v0, run = r)
        results = run_model(model = scenario1, steps = ffa['steps'])
    print(f'Altruist no policy done')
    print(f'Runtime: {time.time() - starttime}\n\n')

    #---Running V1 - Agents face no policy (selfish)---:
    starttime = time.time()
    av_v1 = deepcopy(av)
    av_v1['altruism'] = 0
    dto_v1 = deepcopy(dto)
    dto_v1['tag'] = f'v1_{tag}'
    for r in range(ffa['trials_per_policy']):
        scenario1 = Model(agent_count = ffa['agent_count'], game_variables = gv, agent_variables = av_v1,
                      verbose_variables = vv, data_to_output = dto_v1, run = r)
        results = run_model(model = scenario1, steps = ffa['steps'])
    print(f'Selfish no policy done')
    print(f'Runtime: {time.time() - starttime}\n\n')

    #Running V2A - Agents face policy from social planner:
    #Step 1. Run general search
    starttime = time.time()
    result = Iterated_Local_Search(param_vars=pv, hc_search_vars = sv, pop_size = sv['cores']-1, hc_depth = sv['hc_depth'],
                                    fitness_fnc = rrm_fnc, ils_depth = ils_sv['ils_depth'], ils_search_vars = ils_sv,
                                    fitness_fnc_args = ffa, verbose = False, output_to_file = True, tag = f'v2A_{tag}')
    print(f'Policy run done.\n{result}')
    print(f'Runtime: {time.time() - starttime}\n\n')

    #Step2. Run higher resolution close around results
    if finetune:
        starttime = time.time()
        sv_v2b = deepcopy(sv)
        sv_v2b['explore_range'] *= .1
        ffa_v2b = deepcopy(ffa)
        ffa_v2b['trials_per_policy'] *= 3
        result2 = Iterated_Local_Search(param_vars=pv, hc_search_vars = sv_v2b, pop_size = sv_v2b['cores']-1, hc_depth = sv['hc_depth'],
                                        fitness_fnc = rrm_fnc, ils_depth = 1, ils_search_vars = ils_sv,
                                        fitness_fnc_args = ffa_v2b, verbose = False, output_to_file = True, tag = f'ft_v2A_{tag}',
                                        starting_point = result)
        print(f'Policy fine-tune done.\n{result2}')
        print(f'Runtime: {time.time() - starttime}\n\n')