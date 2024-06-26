from model import Model, run_model


'''
Version summary
   In this scenario, selfish agents play a public goods game
   facing no policy. Agents must learn and decide
   how much they should harvest (from {0,1,...,5})
   to maximize their own benefit.

   According to theory:
    -Socially optimal choice: 2
    -Individually optimal choice: max = 5

'''

gv = {'return_type': 'negative_externality', 'landscape_size':1, 'alpha':.8, 'beta':.05}
av = {'vision':1,
      'action_set':{'sheep_count_if_seen': {'min': 0, 'max': 5, 'grain':1}}}
#vv = ['period','payoff', 'strategies','actions']
vv = []
#dto = {'tag': 'V1_fullrul', 'files':['per_period']}
dto = {'tag': 'S1_print_test', 'files':['final']}

#Run the model many times:
for r in range(1):
    scenario1 = Model(agent_count = 4, game_variables = gv, agent_variables = av, verbose_variables = vv, data_to_output = dto, run = r)
    results = run_model(model = scenario1, steps = 50_000)
    print('V1 done')