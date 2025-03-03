from random import uniform, choice
from numpy.random import normal
from copy import deepcopy
from math import exp
from concurrent.futures import ProcessPoolExecutor

#---For hillclimber variants---
def _roll_canidate(param_vars, search_vars, best_val = None, round_decimal_places = 2):
    '''Determines what new explored choice will be for a hillclimber.
       
       Vars to include:
       -parm_vars ~ {'var1': [min, max], 'var2': [min, max], ...}
       -search_vars ~ {'distribution': distribution, 'explore_range': explore_range}
            Distribution can be 'normal' or 'uniform'
       -best_val should be given if not turn 0 of search
    '''
    actions = {}
    if best_val:
        if 'prob_mutate' in search_vars:
            prob_mutate = search_vars['prob_mutate']
        else:
            prob_mutate = .5
        for name, ranges in param_vars.items():
            if ranges['type'] == 'vector':
                size = ranges['size']
                actions[name] = [ranges['max'] + 1 for i in range(size)]
                for pos in range(size):
                    roll = uniform(0,1)
                    if roll < prob_mutate:
                        #while actions[name][pos] < ranges['min'] or actions[name][pos] > ranges['max']:
                        #Try grabbing with specified distribution:
                        if search_vars['distribution'] == 'uniform':
                            sign = choice([-1,1])
                            epsilon = uniform(0, search_vars['explore_range']) * sign
                        elif search_vars['distribution'] == 'normal':
                            epsilon = normal(loc=0, scale=(ranges['max']-ranges['min'])*search_vars['explore_range'], size=None)
                        actions[name][pos] = round(best_val[name][pos] + epsilon, round_decimal_places)
                        #If out of range, just grab something in range:
                        if actions[name][pos] < ranges['min']:
                            actions[name][pos] = round(uniform(ranges['min'], best_val[name][pos]), round_decimal_places)
                        elif actions[name][pos] > ranges['max']:
                            actions[name][pos] = round(uniform(best_val[name][pos], ranges['max']), round_decimal_places)
                    else:
                        actions[name][pos] = best_val[name][pos]
            else:
                actions[name] = ranges['max'] + 1 #Starts out of range
                roll = uniform(0,1)
                if roll < prob_mutate:
                    #Try grabbing with specified distribution:
                    if search_vars['distribution'] == 'uniform':
                        sign = choice([-1,1])
                        epsilon = uniform(0, search_vars['explore_range']) * sign
                    elif search_vars['distribution'] == 'normal':
                        epsilon = normal(loc=0, scale=(ranges['max']-ranges['min'])*search_vars['explore_range'], size=None)
                    actions[name] = round(best_val[name] + epsilon, round_decimal_places)
                    #If out of range, just grab something in range:
                    if actions[name] < ranges['min']:
                            actions[name] = round(uniform(ranges['min'], best_val[name]), round_decimal_places)
                    elif actions[name] > ranges['max']:
                        actions[name] = round(uniform(best_val[name], ranges['max']), round_decimal_places)
                else:
                    actions[name] = best_val[name]
    else:
        for name, ranges in param_vars.items():
            if 'size' in ranges:
                size = ranges['size']
                actions[name] = []
                for pos in range(size):
                    actions[name].append(round(uniform(ranges['min'], ranges['max']), round_decimal_places))
            else:
                actions[name] = round(uniform(ranges['min'], ranges['max']), round_decimal_places)
    return actions

def _establish_memory():
    '''Creates memory object for a hillclimbing agent to use.'''
    memory_object = {'action': None, 'fitness': None, 'final_choices': 'NotStored'}
    return memory_object

def _update_memory(memory, action, feedback, verbose = False, final_choices = 'NotStored'):
    '''Updates the hillclimbers memory object with info from this period.'''
    #Case 1: Exploited old action, so update fitness with most recent:
    if memory['action'] == action:
        new_memory = {'action': memory['action'], 'fitness': feedback, 'final_choices':final_choices}
        if verbose:
            print(f'Same action {action}, updating payoff')
    #Case 2: Explored new action, so keep if did better than old one:
    else:
        if not memory['fitness']:
            if verbose:
                print(f'Tried initial action {action}')
            new_memory = {'action': action, 'fitness': feedback, 'final_choices':final_choices}
        elif memory['fitness'] < feedback:
            if verbose:
                print(f'New action {action} is better than old action {memory["action"]}')
            new_memory = {'action': action, 'fitness': feedback, 'final_choices':final_choices}
        else:
            if verbose:
                print(f'New action {action} is worse than old action {memory["action"]}')
            new_memory = deepcopy(memory)
    return new_memory

def simple_hillclimb(param_vars, search_vars, depth, fitness_fnc):
    '''
    Runs simple hillclimbing alg. given search depth.
    NOTE: This is a MAXIMIZATION function over fitness.
    '''
    #Initializing:
    bv = None
    mem = _establish_memory()
    if 'explore_range' not in search_vars:
        search_vars['explore_range'] = 1
    #Running rounds of search:
    for i in range(depth):
        a1 = _roll_canidate(param_vars, search_vars, bv)
        sc = fitness_fnc(a1)
        mem = _update_memory(mem, a1, sc)
        bv = mem['action']
    return mem

def _run_hillclimb_helper(param_vars, search_vars, bv, fitness_fnc_args, fitness_fnc, roll_new = True):
    '''A helper function for parallel processing in the SA hillclimb fnc.'''
    if roll_new:
        a = _roll_canidate(param_vars, search_vars, bv)
    else:
        a = bv
    if fitness_fnc_args:
        return [a, fitness_fnc(**a, **fitness_fnc_args)]
    else:
        return [a, fitness_fnc(**a)]

def SA_hillclimb(param_vars, search_vars, pop_size, depth, fitness_fnc, starting_point = None, fitness_fnc_args = None,
                 output_to_file = False, tag = ''):
    '''
    Runs a steepest ascent hillclimbing alg. for a given search depth.

    -fitness_fnc_args allows you to input other arguments for your fitness function that aren't your parameter values specifically.
    '''
    #Initializing:
    if starting_point:
        bv = starting_point['action']
        mem = starting_point
    else:
        bv = None
        mem = _establish_memory()
    if 'explore_range' not in search_vars:
        search_vars['explore_range'] = 1
    if 'cores' in search_vars:
        cores = search_vars['cores']
    else:
        cores = 1
    if 'return_last_act' in fitness_fnc_args:
        return_last_act = fitness_fnc_args['return_last_act']
    else:
        return_last_act = False
    #Setting up column titles and first data row for log file:
    if output_to_file:
        file_name = f'{tag}_hc_log.txt'
        lol = open(file_name,'w')
        if return_last_act:
            lol.write(f'best_policy, fitness, final_choices\n')
        else:
            lol.write(f'best_policy, fitness\n')
        lol.close()

    #Running rounds of search:
    for d in range(depth):
        print(f'  HC Depth {d}')
        #Step 1 - Create and evaluate pop_size variants:
        results = []
        performances = []
        with ProcessPoolExecutor(cores) as threadpool:
            #Resubmitting bv
            if bv:
                results.append(threadpool.submit(_run_hillclimb_helper, param_vars = param_vars, 
                                                    search_vars = search_vars, bv = bv, 
                                                    fitness_fnc_args = fitness_fnc_args, 
                                                    fitness_fnc = fitness_fnc, roll_new = False))
            for i in range(pop_size):
                results.append(threadpool.submit(_run_hillclimb_helper, param_vars = param_vars, 
                                                 search_vars = search_vars, bv = bv, 
                                                 fitness_fnc_args = fitness_fnc_args, 
                                                 fitness_fnc = fitness_fnc, roll_new = True))
            threadpool.shutdown(wait = True)
        for r in results:
            if not r.exception():
                performances.append(r.result())
            else:
                print(r.exception())
                raise Exception(f"Got exception for a result from parallel process")
        #Step 2 - Compare each to existing to see which to keep:
        for p in performances:
            #print(p)
            if return_last_act:
                mem = _update_memory(mem, p[0], p[1]['csw'], final_choices = p[1]['last_act'])
            else:
                mem = _update_memory(mem, p[0], p[1]['csw'])
            bv = mem['action']
            
        if output_to_file:
            lol = open(file_name,'a')
            if return_last_act:
                lol.write(f'{mem["action"]}, {mem["fitness"]}, {mem["final_choices"]},\n')
            else:
                lol.write(f'{mem["action"]}, {mem["fitness"]},\n')
            lol.close()
    return mem

def Iterated_Local_Search(param_vars, hc_search_vars, pop_size, hc_depth, fitness_fnc, ils_depth, 
                           ils_search_vars, fitness_fnc_args = None, verbose = False,
                           output_to_file = False, tag = '', starting_point = None):
    '''Hill climbs over local optima found with SA_hillclimb.

    -ils_search_vars ~ {'distribution': distribution, 'explore_range': explore_range, 'downhill_coeff': downhill_coeff}
        Distribution can be 'normal' or 'uniform'
        Downhill coeff partly determines how often we'll accept a new homebase which is a bit worse.
            smaller is better. Choose from (0,1]. .1 by default
    -fitness_fnc_args lets you give arguments to the fitness function besides the parameter values
    '''
    #Step 1: Finding a local optimum with SA_hillclimb
    print('\nILS Depth 0')
    local_optimum = SA_hillclimb(param_vars, hc_search_vars, pop_size, hc_depth, fitness_fnc,
                                 starting_point = starting_point, fitness_fnc_args = fitness_fnc_args,
                                 output_to_file=output_to_file, tag = f'{tag}_ILS0')
    if verbose:
        print(f'LO: {local_optimum}')
    if output_to_file:
        #Setting up column titles and first data row for log file:
        file_name = f'{tag}_local_optimum_log.txt'
        lol = open(file_name,'w')
        var_title = ''
        var_vals = ''
        for k, v in local_optimum['action'].items():
            if type(v) == list:
                for i in range(len(v)):
                    var_title += f', {k}_{i}'
                    var_vals += f', {v[i]}'
            else:
                var_title += f', {k}'
                var_vals += f', {v}'
        lol.write(f'num{var_title}, fitness\n0{var_vals}, {local_optimum["fitness"]}\n')
        lol.close()
    home_base = local_optimum #We'll use this to decide what are new starting points should be near.
    best_ever = local_optimum #We'll use this to hold onto the best solution we've ever found
    if 'explore_range' not in ils_search_vars:
        ils_search_vars['explore_range'] = 1
    if 'downhill_coeff' not in ils_search_vars:
        ils_search_vars['downhill_coeff'] = None
    for k in range(ils_depth-1):
        print(f'\nILS Depth {k+1}')
        #Step 2: Finding a starting point (which are variants of our home_base)
        variant_hb = _roll_canidate(param_vars, ils_search_vars, best_val = home_base['action'])
        new_sp = {'action': variant_hb, 'fitness': None}
        #Step 3: Find local optimum from that starting point
        local_optimum = SA_hillclimb(param_vars, hc_search_vars, pop_size, hc_depth, fitness_fnc, starting_point = new_sp,
                                     fitness_fnc_args = fitness_fnc_args, output_to_file=output_to_file, tag = f'{tag}_ILS{k+1}')
        if output_to_file:
            lol = open(file_name,'a')
            var_vals = ''
            for v in local_optimum['action'].values():
                if type(v) == list:
                    for i in range(len(v)):
                        var_vals += f', {v[i]}'
                else:
                    var_vals += f', {v}'
            lol.write(f'{k+1}{var_vals}, {local_optimum["fitness"]}\n')
            lol.close()
        #Step 4: Updating our home_base and storing the best solution so far.
        if best_ever['fitness'] < local_optimum['fitness']:
            best_ever = deepcopy(local_optimum)
        score = local_optimum['fitness'] - home_base['fitness']
        downhill_coeff = ils_search_vars['downhill_coeff']
        if ils_search_vars['downhill_coeff']:
            try:
                ans = exp((score)/downhill_coeff)
            except OverflowError:
                ans = 0
            prob_downhill = max(.5, ans)
            roll = uniform(0,1)
            if score > 0 or roll < prob_downhill:
                home_base = local_optimum
        #Step 5: Updating output
        if verbose:
            print(f'LO: {local_optimum}')
        if output_to_file:
            file_name2 = f'{tag}_best_peak.txt'
            bp = open(file_name2,'w')
            bvv = ''
            for bv in best_ever['action'].values():
                if type(bv) == list:
                    for i in range(len(bv)):
                        bvv += f', {bv[i]}'
                else:
                    bvv += f', {bv}'
            bp.write(f'depth{var_title}, fitness\n{k+1}{bvv}, {best_ever["fitness"]}\n')
            bp.close()
    return best_ever

def N_Party_Platforms():
    '''In a similar fashion to SA, policy agents will choose competing policies to get votes.'''