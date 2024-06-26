from hillclimbing import _roll_canidate
from scenarioV2A import run_run_model
import random
import collections
from copy import copy, deepcopy


#---N Party System---:
def initialize_platforms(N, policy_vars):
    '''Creates initial positions in policy space for each of the N parties.'''
    platforms = {}
    for i in range(N):
        platforms[i] = _roll_canidate(param_vars = policy_vars, search_vars = {'distribution':'uniform'})
    return platforms
    
def _eval_policy(lag_win_info, policy_id, policy, newp_runs, agent_count, game_variables, agent_variables, verbose_variables,
                 data_to_output, steps, spv = None, rla = False, verbose = False, cores = 1):
    '''Runs a policy runs times, then returns either social welfare, or a vector
     of each players utility (if indv is True).'''
    policy_eval = run_run_model(agent_count, game_variables, agent_variables, verbose_variables, fine_vector = policy['fine_vector'],
                                data_to_output=data_to_output, steps=steps, trials_per_policy = newp_runs, social_planner_vars = spv,
                                return_last_act = rla, return_indv_welfare = True, cores = cores)
    #print(f'Policy_eval: {policy_eval}')
    if verbose:
        print(f"Policy:\n{policy}\nPlayer round utility est.: {policy_eval['iw']}")
    if lag_win_info['id'] == policy_id:
        if verbose:
            print(f'...this policy has occured before, so will incorperate past outcomes in utility estimate..')
        s = lag_win_info['streak'] + 1
        policy_eval['csw'] = (lag_win_info['sw'] * (s-1)/s) + (policy_eval['csw']/s) #Calculating weighted avg of sw observed...
        new_avg_iw = []
        if verbose:
            print(f'lag_win_info: {lag_win_info}')
        for pos in range(len(lag_win_info['iw'])):
            a = (lag_win_info['iw'][pos]*(s-1)/s)
            b = (policy_eval['iw'][pos]/s)
            new_avg_iw.append(a+b)
        policy_eval['iw'] = deepcopy(new_avg_iw)
        if verbose:
            print(f"Player updated round utility est. (incorperating past performance): {policy_eval['iw']}")
    return [policy_eval['csw'], policy_eval['iw']]

def vote(lag_win_info, newp_runs, platforms, agent_count, gv, av, vv, dto, steps, spv, verbose=False, cores = 1):
    '''Creates a dict of evals for each player for each policy in platforms using
       _eval_policy. Returns vote (policy which yeilds best estimated EU) for each agent.'''
    #Step 1: Create dict to store agent's favorite policies
    votes = [-1 for a in range(agent_count)]
    util_under_voted_pol = [-1_000_000 for a in range(agent_count)]
    util_under_all_pol = []
    sw_under_each_pol = {}
    #Step 2: Evaluate each policy, and store it for each agent if better than best so far:
    for pid, plat in platforms.items():
        policy_utility_forcast = _eval_policy(lag_win_info, policy_id = pid, policy = plat, newp_runs = newp_runs,
                                               agent_count = agent_count, game_variables = gv,
                                               agent_variables = av, verbose_variables = vv, data_to_output = dto, steps = steps,
                                               spv = spv, verbose = verbose, cores = cores)
        sw_under_each_pol[pid] = policy_utility_forcast[0]
        util_under_all_pol.append(copy(policy_utility_forcast[1]))
        for aid in range(len(policy_utility_forcast[1])):
            if  policy_utility_forcast[1][aid]> util_under_voted_pol[aid]:     #If new policy is better...
                votes[aid] = pid #Store it as the best option considered so far
                util_under_voted_pol[aid] = policy_utility_forcast[1][aid]
    if verbose:
        print(f'Player Votes: {votes}')
        print(f'Player util under their fav policy: {util_under_voted_pol}')
    #print(f'Player utility under all policies: {util_under_all_pol}')
    return votes, sw_under_each_pol, util_under_all_pol

def policy_winner(N, votes, agg_type = 'majority', verbose = False):
    '''Takes list of agent policy votes and returns winning policy ID given agg_type 
    used to aggregate the votes.'''
    if agg_type == 'majority': #Returns most popular policy (with ties broken randomly)
        counts = collections.Counter(votes)
        max_freq = max(counts.values())
        vote_shares = []
        for pol_id in range(N):
            if pol_id in counts.keys():
                vote_shares.append(counts[pol_id] / len(votes))
            else:
                vote_shares.append(0)
        most_frequent_integers = [num for num, freq in counts.items() if freq == max_freq]
        winningPolicy = random.choice(most_frequent_integers)
        if verbose:
            print(f'Vote shares this round: {vote_shares}')
            print(f'Winning policy based on majority vote: {winningPolicy}')
        return winningPolicy, vote_shares
    else:
        print('ERROR: policy_winner was given bad agg_type argument.')
        return None

def update_platforms(winID, platforms, lag_platforms, vote_shares, lag_vote_shares, policy_vars, search_vars, verbose = False):
    '''Winning platform remains fixed, while others all choose a new platform near themselves. 
    Returns the resulting platform for next round.'''
    new_platforms = {}
    for pid, plat in platforms.items():
        if pid != winID:
            if vote_shares[pid] >= lag_vote_shares[pid]:
                new_platforms[pid] = _roll_canidate(param_vars = policy_vars, search_vars = search_vars, best_val=plat)
                if verbose:
                    print(f'Policy maker {pid} will try similar to their new platform.')
            else:
                new_platforms[pid] = _roll_canidate(param_vars = policy_vars, search_vars = search_vars, best_val=lag_platforms[pid])
                if verbose:
                    print(f'Policy maker {pid} will return to something similar to their old platform.')
        else:
            new_platforms[pid] = deepcopy(plat)
    return new_platforms

def run_N_party_democracy(rounds, N, newp_runs, policy_vars,
                          agent_count, gv, av, vv, dto, steps, sv, agg_type = 'majority',
                          verbose = False, cores = 1):
    '''Runs an N platform democracy for a number of rounds.'''
    #Initializing the model:
    lag_best_eval_info ={'id':-100, 'streak': -100, 'iw': None, 'sw': None}
    lag_vote_shares = [0 for i in range(N)]
    platforms = initialize_platforms(N = N, policy_vars = policy_vars) #<---- For later: Custom initial positions?
    lag_platforms = None
    #Prepping file writing:
    npfile = open(f'{N}Party_{dto["tag"]}_data.txt','w')
    var_titles = ''
    for vname, var in policy_vars.items():
        for i in range(var['size']):
            var_titles += f',{vname}_{i}'
    npfile.write(f'N,t,winning_id,vote_share,{var_titles},fitness\n') #Writing variable names for the file
    npfile.close()
    #Stepping..
    for t in range(rounds):
        if verbose:
            print(f'\nRound {t}')
            print(f'Policies:\n{platforms}')
        #Agents vote:
        agent_votes = vote(lag_win_info = lag_best_eval_info, newp_runs = newp_runs,
                           platforms = platforms, agent_count = agent_count, gv = gv, av = av, vv = vv, dto = dto,
                           steps = steps, spv = policy_vars, verbose = verbose, cores = cores)
        #Votes are collected and a winning policy is determined (and the implications of the policy are recorded):
        winning_id, vote_shares = policy_winner(N=N, votes = agent_votes[0], agg_type = agg_type, verbose = verbose)
        sw_under_p  = agent_votes[1][winning_id]
        if verbose:
            print(f'Avg social welfare under winning policy: {sw_under_p}')
        iw_under_p = agent_votes[2][winning_id]
        if verbose:
            print(f'Indiv. welfare under winning policy: {iw_under_p}')
        #Losing parties consider new policies for next election:
        platforms = update_platforms(winID = winning_id, platforms = platforms, lag_platforms=lag_platforms, vote_shares=vote_shares, 
                                     lag_vote_shares=lag_vote_shares, policy_vars = policy_vars, search_vars = sv, verbose=verbose)
        lag_platforms = deepcopy(platforms)
        if verbose:
            print(f'New policies for next round:\n{platforms}')
        #Writing results to file:
        if winning_id != lag_best_eval_info['id'] or t == rounds-1: #Write only if last round OR new policy won
            npfile = open(f'{N}Party_{dto["tag"]}_data.txt','a')
            tth_entry = ''
            for pvar in platforms[winning_id].values():
                for p in pvar:
                    tth_entry += f',{p}'
            npfile.write(f'{N},{t},{winning_id},{vote_shares[winning_id]}{tth_entry},{sw_under_p}\n')
            npfile.close()
        #Updating lag info for next round:
        if winning_id == lag_best_eval_info['id']:
            lag_best_eval_info['streak'] += 1
        else:
            lag_best_eval_info['id'] = winning_id
            lag_best_eval_info['streak'] = 1
        lag_best_eval_info['sw'] = sw_under_p
        lag_best_eval_info['iw'] = iw_under_p
        lag_vote_shares = copy(vote_shares)
    return winning_id, platforms[winning_id]
