from bundler_V0_V1_V2A import run_model_bundle

#Version summary
'''
   As before, Agents play a public goods game
   facing some policy. Agents must learn and decide
   how much they should harvest (from {0,1,...,5})
   to maximize their own benefit.

    According to theory:
    -Socially optimal choice: 2
    -Individually optimal choice: max = 5

   Also, as in V2A, we have a social planner
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

    ---The new stuff---
    
    This Scenario now considers an alteration to agent learning in the 
    form of similarity. Agents now consider actions informative about other
    actions close by (eg. choosing 2 might tell me something about what
    happens when I choose 3). Specifically, agents use the inverse exponential
    distance as their similarity weight, and they only consider their next
    closest neighboring actions as possibly similar. Let's see how this, if 
    anything, changes the 'optimal' policy chosen by the social planner.
'''

#Game params:
gv = {'return_type': 'negative_externality', 'landscape_size':1, 'alpha':.8, 'beta':.05, 'transfers': .5} #<-- Transfers arg
#Agent params:
av = {'vision':1, 'action_set':{'sheep_count_if_seen': {'min': 0, 'max': 5, 'grain':1}},
      'similarity':True} #<-- Sim arg
#Output params:
vv = []
dto = {'tag': '', 'files':['final']}
#Policy search params:
sv = {'distribution': 'normal', 'explore_range': .1, 'prob_mutate':.5, 'hc_depth': 1_000, 'cores':8} #Note: Cores is for how many // processes allowed
ils_sv = {'distribution': 'normal', 'ils_depth': 20, 'explore_range': 2, 'downhill_coeff': None}
#Fitness function (run_run_model) params:
pv = {'fine_vector':{'type':'vector','min':0, 'max':10, 'size':6}}
ffa = {'agent_count': 4, 'game_variables': gv, 'agent_variables': av, 'verbose_variables': vv,
        'data_to_output': dto, 'steps':50_000, 'trials_per_policy':5, 'social_planner_vars':pv, 'return_last_act': True}

def main():
    run_model_bundle(gv = gv, av=av, vv = vv, dto = dto, sv = sv, ils_sv = ils_sv,
                     pv = pv, ffa = ffa, tag = 'Bundle_w_half_transfers_and_similarity', finetune=False)

if __name__ == '__main__':
    main()