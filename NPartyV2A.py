from government import run_N_party_democracy
import time


gv = {'return_type': 'negative_externality', 'landscape_size':1, 'alpha':.8, 'beta':.05} 
av = {'vision':1, 'action_set':{'sheep_count_if_seen': {'min': 0, 'max': 5, 'grain':1}}}
vv = []
dto = {'tag': 'TEST', 'files':[]}
sv = {'distribution': 'normal', 'explore_range': .05, 'prob_mutate':.5, 'cores':3}
pv = {'fine_vector':{'type':'vector','min':0, 'max':10, 'size':6}}

def main():
    testrun = run_N_party_democracy(rounds=20_000, N=2, newp_runs=5, policy_vars = pv,
                                    agent_count=5, gv = gv, av = av, vv = vv, dto = dto,
                                    steps=50_000, sv = sv, agg_type = 'majority', verbose = False, cores = sv['cores'])
    print(f'Final results:\n{testrun}')

if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f'Finnished in :{time.time()-start_time}')