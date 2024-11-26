from bundler_V0_V1_V2A import run_model_bundle

#Version summary
'''
Runs 
'''

#Game params:
gv = {'return_type': 'negative_externality', 'landscape_size':1, 'alpha':.8, 'beta':.05} 
#Agent params:
av = {'vision':1, 'action_set':{'sheep_count_if_seen': {'min': 0, 'max': 5, 'grain':1}}}
#Output params:
vv = []
dto = {'tag': 'wide_no_simcase', 'files':['final']}
#Policy search params:
sv = {'distribution': 'normal', 'explore_range': .1, 'prob_mutate':.5, 'hc_depth': 500, 'cores':8} #Note: Cores is for how many // processes allowed
ils_sv = {'distribution': 'normal', 'ils_depth': 50, 'explore_range': 2, 'downhill_coeff': None}
#Fitness function (run_run_model) params:
pv = {'fine_vector':{'type':'vector','min':0, 'max':10, 'size':6}}
ffa = {'agent_count': 4, 'game_variables': gv, 'agent_variables': av, 'verbose_variables': vv,
        'data_to_output': dto, 'steps':50_000, 'trials_per_policy':5, 'social_planner_vars':pv, 'return_last_act': True}

def main():
    print(f'av:\n{av}')
    run_model_bundle(gv = gv, av=av, vv = vv, dto = dto, sv = sv, ils_sv = ils_sv,
                     pv = pv, ffa = ffa, tag = 'Bundle', finetune=False)

if __name__ == '__main__':
    main()