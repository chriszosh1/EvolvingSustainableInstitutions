from agent import Shepherd
from random import uniform, randint, shuffle
from copy import copy, deepcopy


class Model():
  """
  A repeated (spatial) commons game with learning agents facing (and developing) policy.

  VARIABLE DETAILS:
  - self.game_variables ~ a dictionary which holds game level parameters:
    - 'return_type' ~ determines how per sheep return is calculated
                      'root' by default ~ return is the square root of sum of sheep.
                      
  - self.agent_variables ~ a dictionary which holds agent characteristics:
    - 'init_pos' ~ determines the agent's initial position if spatial component.
                   can give a particular init_pos value, or can specify 'random' to draw from U[min,max]
    - 'action_set' ~ a dict of dictionaries w/ format var_name:{min, max, and discrete}.
  """

  def __init__(self, agent_count, game_variables, agent_variables, data_to_output = {},
                verbose_variables = [], fine_vector = None, run = 0, social_planner_vars = None,
                store_attraction_snapshots = []):
    self.game_variables = game_variables
    if 'return_type' not in game_variables:
      self.game_variables['return_type'] == 'negative_externality'
    if 'transfers' in self.game_variables:
      self.transfers = self.game_variables['transfers']
    else:
      self.transfers = False
    self.agent_variables = agent_variables
    self.social_planner_vars = social_planner_vars
    self.action_names = [i for i in self.agent_variables['action_set'].keys()]
    self.verbose_variables = verbose_variables
    self.fine_vector = fine_vector
    self.period = 0
    self.agent_count = agent_count
    self.data_to_output = data_to_output
    self.monitoring_rate = []
    self.avg_lifetime_harvest = 0
    self.avg_lifetime_fines = 0
    self.avg_lifetime_harvest_w_fines = 0
    self.avg_felicity = []
    self.avg_social_welfare = 0
    self.run = run
    self.converged = False
    self.store_attraction_snapshots = store_attraction_snapshots
    self.attraction_snaps = {}
    if 'tag' in self.data_to_output:
      self.tag = self.data_to_output['tag']
    else:
      self.tag = ''
    #print(f'  Trial {self.run}')
    if 'alpha' in self.game_variables: #coeff of the linear part of externality
      self.alpha = self.game_variables['alpha']
    if 'beta' in self.game_variables: #coeff on the squ part of externality
      self.beta = self.game_variables['beta']
    #Establishing bounds for payoffs:
    if 'sheep_count_if_seen' in self.agent_variables['action_set']:
      max_sheep = self.agent_variables['action_set']['sheep_count_if_seen']['max']
      min_sheep = self.agent_variables['action_set']['sheep_count_if_seen']['min']
    else:
      print('ERROR action not in actionset')
    if self.fine_vector:
      #Min personal harvest - externality when you're the only one not harvesting at max - max fine allowed
      self.min_payoff = min_sheep -self._calc_externality((self.agent_count-1)*max_sheep) - self.social_planner_vars['fine_vector']['max']
      #Max personal harvest - externality when you're the only one harvesting at max - min fine allowed:
      self.max_payoff = max_sheep - self._calc_externality(max_sheep) - self.social_planner_vars['fine_vector']['min']
    else:
      self.min_payoff = min_sheep - self._calc_externality((self.agent_count-1)*max_sheep)
      self.max_payoff = max_sheep - self._calc_externality(max_sheep)
    #print(f'Minimum payoff is: {self.min_payoff}')
    #print(f'Maximum payoff is: {self.max_payoff}')
    self.agents = {}
    if 'landscape_size' in self.game_variables:
      upper_range = self.game_variables['landscape_size']
      self.density =  self.game_variables['landscape_size']/self.agent_count
    else:
      upper_range = 0
    #Creating Agents:
    for id in range(self.agent_count):
      self.agents[id] = Shepherd(aid = id, agent_variables = agent_variables, init_pos = randint(0,upper_range+1),
                                 landscape_size = self.game_variables['landscape_size'], min_payoff = self.min_payoff,
                                 max_payoff = self.max_payoff, model = self)

  def is_agent_seen(self, agent):
    '''Determines if a particular agent is visible to other agents this period.'''
    agent_seen = False
    for other_a in self.agents.values(): 
      dist = abs(other_a.pos - agent.pos)
      if other_a.id != agent.id and dist <= other_a.agent_variables['vision']:
        agent_seen = True
    return agent_seen

  def _calc_externality(self, total_sheep):
    '''Returns the payoff per sheep as a function of total sheep.'''
    if self.game_variables['return_type'] == 'negative_externality':
      avg_sheep = total_sheep/self.agent_count
      return self.alpha * avg_sheep + self.beta * (avg_sheep ** 2)

  def calculate_harvests(self):
    '''Returns period payoff for each agent as a fnc of own and total sheep count.'''
    total_sheep = 0
    for agent in self.agents.values():
      total_sheep += agent.period_choices['sheep_count']
    externality = self._calc_externality(total_sheep)
    total_period_harvest = 0
    for agent in self.agents.values():
      agent.period_payoff = agent.period_choices['sheep_count'] - externality
      total_period_harvest += agent.period_payoff
    self.avg_lifetime_harvest += (total_period_harvest / self.agent_count)

  def apply_fines(self):
    '''Determines fines (if any) applied to each agent as a function of sheepcount choice.'''
    self.avg_period_fines = 0
    if self.fine_vector:
      #1. Applying fines
      for a in self.agents.values():
        if ('move_if_seen' in self.action_names and a.seen) or 'move_if_seen' not in self.action_names:
          fine = self.fine_vector[a.period_choices['sheep_count']]
          a.period_payoff -= fine
          self.avg_period_fines -= fine
      #2. Redistributing if transfers
      if self.transfers:
        for a in self.agents.values():
          a.period_payoff -= self.transfers*self.avg_period_fines #This is - because avg fines are negative.
    #3. Updating lifetime fines var
    self.avg_lifetime_fines += (self.avg_period_fines/self.agent_count)
    

  def calc_total_selfish_felicity(self):
    '''Takes each agent felicity from own consumption only and adds them up.'''
    total_selfish_felicity = 0
    for agent in self.agents.values():
      total_selfish_felicity += agent.period_selfish_felicity
    return total_selfish_felicity

  def calc_social_welfare(self):
    '''Takes each agent felicity (including altruism payoffs) and adds them up.'''
    sw = 0
    for agent in self.agents.values():
      sw += agent.period_felicity
      self.period_total_felicity = sw
    return sw

  def take_stock(self):
    '''Updates/calculates a few values for storage/outputting.'''
    #Calculating period monitoring rate:
    p_seen = 0
    for a in self.agents.values():
      if 'move_if_seen' in self.action_names:
        p_seen += int(a.seen == True)
    p_seen = p_seen / len(self.agents)
    #Storing the values:
    avg_felicity = self.period_total_felicity / self.agent_count
    self.avg_felicity.append(avg_felicity)
    self.avg_social_welfare += avg_felicity
    self.monitoring_rate.append(p_seen)

  def print_output(self):
    '''Allows for printing of output for debugging/analysis purposes.'''
    if self.verbose_variables:
      print()
    if 'period' in self.verbose_variables:
      print(f'---Period {self.period}---')
    if 'strategies' in self.verbose_variables:
      print(f'\nPeriod {self.period} strategies')
      for aid, a in self.agents.items():
        print(f'Agent {aid} had strategies...')
        for name, strat in a.period_strategy.items():
          print(f'    {name}: {strat}')
    if 'actions' in self.verbose_variables:
      print(f'\nPeriod {self.period} choices')
      for aid, a in self.agents.items():
        print(f'Agent {aid} choose...')
        for name, choice in a.period_choices.items():
          print(f'    {name}: {choice}')
    if 'payoff' in self.verbose_variables:
      print(f'\nPeriod {self.period} payoffs')
      for aid, a in self.agents.items():
        print(f'Agent {aid} got payoff {a.period_payoff}')
    if 'LearningAttractions' in self.verbose_variables:
      print(f'\nPeriod {self.period} strategies')
      for aid, a in self.agents.items():
        print(f'Agent {aid} had strategies...')
        for name, strat in a.period_strategy.items():
          print(f'    {name}: {strat}')

  def update_datafile(self, last = False):
    '''Creates and updates all datafiles requested.'''
    if 'per_period' or 'final' in self.data_to_output['files']:
      if self.period == 0 and self.run == 0:
        file = open(f'{self.tag}_data.txt','w')
        if 'move_if_seen' in self.action_names:
          file.write(f'run,period,agent_count,landscape_size,pop_density,agent_id,position,move_if_seen,best_move_if_seen,move_if_unseen,best_move_if_unseen,sheep_count_if_seen,best_sheep_count_if_seen,sheep_count_if_unseen,best_sheep_count_if_unseen,avg_social_welfare,payoff,avg_period_felicity,monitored,monitoring_rate\n')
        else:
          file.write(f'run,period,agent_count,landscape_size,agent_id,sheep_count_if_seen,best_sheep_count_if_seen,payoff,avg_period_felicity,avg_social_welfare\n')
      else:
        endpoint_update = last and 'final' in self.data_to_output['files']
        if 'per_period' in self.data_to_output['files'] or endpoint_update: 
          file = open(f'{self.tag}_data.txt','a')
          for a in self.agents.values():
            if 'move_if_seen' in self.action_names:
              file.write(f'{self.run},{self.period},{self.agent_count},{self.game_variables["landscape_size"]},{self.density},{a.id},{a.pos},{a.period_strategy["move_if_seen"]},{a.memory["move_if_seen"].index(max(a.memory["move_if_seen"]))},{a.period_strategy["move_if_unseen"]},{a.memory["move_if_unseen"].index(max(a.memory["move_if_unseen"]))},{a.period_strategy["sheep_count_if_seen"]},{a.memory["sheep_count_if_seen"].index(max(a.memory["sheep_count_if_seen"]))},{a.period_strategy["sheep_count_if_unseen"]},{a.memory["sheep_count_if_unseen"].index(max(a.memory["sheep_count_if_unseen"]))},{self.avg_social_welfare},{a.period_payoff},{self.avg_felicity[-1]},{a.seen},{self.monitoring_rate[-1]}\n')
            else:
              file.write(f'{self.run},{self.period},{self.agent_count},{self.game_variables["landscape_size"]},{a.id},{a.period_strategy["sheep_count_if_seen"]},{a.memory["sheep_count_if_seen"].index(max(a.memory["sheep_count_if_seen"]))},{a.period_payoff},{self.avg_felicity[-1]},{self.avg_social_welfare}\n')

  def collect_agent_attractions(self):
    '''Collect average agent attraction per snapshot.'''
    temp_list = []
    self.attraction_snaps[self.period] = {}
    for a in self.agents.values():
      temp_list.append(a.store_attraction_snapshot(self.period)['sheep_count_if_seen'])
    for d in temp_list:
      for k,v in d.items():
        if k not in self.attraction_snaps[self.period]:
          self.attraction_snaps[self.period][k] = v/len(temp_list)
        else:
          self.attraction_snaps[self.period][k] += v/len(temp_list)
          
  def step(self):
    #Step0: Store some top of the round info
    if self.period in self.store_attraction_snapshots:
      self.collect_agent_attractions()
    #Step1: Decide strategies for the period
    for agent in self.agents.values():
        agent.decide_period_strategy()
    #Step2: Apply movement part of strategies   
    if 'move_if_seen' in self.action_names:
      id_list = [k for k in self.agents.keys()]
      shuffle(id_list)
      for aid in id_list:
        agent = self.agents[aid]
        agent.seen_before_move = self.is_agent_seen(agent)
        agent.apply_move_strategy()
    #Step3: Update who sees who
    if 'move_if_seen' in self.action_names:
      for agent in self.agents.values():
        agent.seen = self.is_agent_seen(agent)
    #Step4: Apply grazing part of strategy
    if 'move_if_seen' in self.action_names:
      for agent in self.agents.values():
        agent.apply_sheep_strategy()
    else:
      for agent in self.agents.values():
        agent.apply_sheep_strategy()
    #Step4: Apply grazing part of strategy
    self.calculate_harvests()
    #Step5: Fine players
    self.apply_fines()
    #Step6: Agents evaluate the sucess of their strategy
    for agent in self.agents.values():
      agent.calc_selfish_felicity()
    total_fel = self.calc_total_selfish_felicity()
    for agent in self.agents.values():
      agent.calc_felicity(total_fel)
      agent.update_memory()
    self.calc_social_welfare()
    #Step7: Output and prepping for next round of play
    self.take_stock()
    self.print_output()
    self.update_datafile()
    self.period += 1

def run_model(model, steps):
  for i in range(steps):
    model.step()
  model.update_datafile(last = True)

'''def run_run_model(steps, runs, model_params):
  sw = 0
  for r in range(runs):
    run_r_model = Model(model_params)
    run_r_results = run_model(run_r_model, steps)
    sw += run_r_model.avg_social_welfare
  return sw'''