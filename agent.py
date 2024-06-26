from random import uniform, choice, choices
from numpy.random import normal
from numpy import exp, arange
from copy import copy

def default_felicity_fnc(x):
  return x


class Agent:
  '''
  A base class for all Agents
  '''
  def __init__(self, aid, model):
    self.id = aid
    self.model = model
    self.period_payoff = None
    self.period_felicity = None
    self.period_strategy = {}
    self.period_choices = {}
    self.memory = {}
    self.attraction_snapshots = {}


class Shepherd(Agent):
  """
  Shepherds learning to choose herd size (and where to graze).

  DECISION MAKING:
  Agents choose..
    A random action with prob p
    The action with the highest EU with prob otherwise
  prob to explore (p) shrinks with time

  VARIABLE DETAILS:
  - self.agent_variables ~ a dictionary which holds agent characteristics:
    - 'init_pos' ~ determines the agent's initial position if spatial component.
                   can give a particular init_pos value, or can specify 'random' to draw from U[min,max]
    - 'action_set' ~ a list of dictionaries a name, min, max, discrete, distribution, epsilon.
  """
  def __init__(self, aid, agent_variables, init_pos,  landscape_size, min_payoff, max_payoff, model, altruism = 0):
    super().__init__(aid, model)
    self.id = aid
    self.agent_type = 'Shepherd'
    self.agent_variables = agent_variables
    self.choice_freq = {}
    self.pos = init_pos
    self.landscape_size = landscape_size
    self.min_payoff = min_payoff
    self.max_payoff = max_payoff
    self.init_explore_rate = 1
    self.explore_rate = self.init_explore_rate
    self.min_explore_rate = 0
    self.actions_avail = {}
    self.lifetime_felicity = 0
    #For model experiments:
    if 'recency_bias' in agent_variables:
      self.recency_bias = agent_variables['recency_bias']
    else:
      self.recency_bias = None
    if 'similarity' in agent_variables: #NOTE Similarity may only applied for harvest actions presently, and doesn't work with recency bias.
      self.similarity = agent_variables['similarity']
    else:
      self.similarity = False
    if 'altruism' in agent_variables:
      if type(agent_variables['altruism']) is list:
        self.altruism = agent_variables['altruism'][self.id]
      else:
        self.altruism = agent_variables['altruism']
    else:
      self.altruism = 0
    if 'risk_aversion_fnc' in agent_variables:
      self.risk_aversion_fnc = agent_variables['risk_aversion_fnc']
    else:
      self.risk_aversion_fnc = None
    if 'utility_fnc' in agent_variables:
      self.felicity_fnc = agent_variables['utility_fnc']
    else:
      self.felicity_fnc = default_felicity_fnc
    #Setting up memory and learning:
    if 'explore_decay' in self.agent_variables:
      self.explore_decay = self.agent_variables['explore_decay']
    else:
      self.explore_decay = 0.0005
    for a_k, a_v in self.agent_variables['action_set'].items():
      self.memory[a_k] = []
      self.choice_freq[a_k] = []
      act_options = arange(a_v['min'], a_v['max'] + a_v['grain'], a_v['grain'])
      act_options = act_options.tolist()
      self.actions_avail[a_k] = act_options
      for act in self.actions_avail[a_k]:
        self.memory[a_k].append(self.max_payoff-self.min_payoff)
        self.choice_freq[a_k].append(0)
    self.seen_before_move = None
    self.seen = None

  def _period_prob_explore(self):
    '''Determines with what probability agents will explore a new strategy this turn
    (as opposed to just using the best scoring so far).'''
    if self.explore_rate > self.min_explore_rate:
      self.explore_rate = self.min_explore_rate + (self.init_explore_rate-self.min_explore_rate)*exp(-self.explore_decay*self.model.period)
      self.explore_rate = round(max(self.explore_rate, self.min_explore_rate),3)
      #if self.id == 0 and self.model.period % 100 == 0:
      #  print(f'\nPeriod {self.model.period} explore rate: {self.explore_rate}')
    return self.explore_rate

  def decide_period_strategy(self):
    '''Returns the agents strategy for this period.'''
    self._period_prob_explore()
    roll = uniform(0,1)
    for act, av in self.agent_variables['action_set'].items():
      actions_avail = [i for i in range(len(self.memory[act]))]
      if roll < self.explore_rate: #Explore:
        action = choices(actions_avail, weights = [i**2 for i in self.memory[act]], k = 1)[0] #Weighted exploration
        #action = choice(actions_avail)
      else: #Exploit
        max_score = max(self.memory[act])
        max_positions = []
        for s in range(len(self.memory[act])):
          if self.memory[act][s] == max_score:
            max_positions.append(s)
        max_pos = choice(max_positions)
        action = actions_avail[max_pos]
      if type(av['grain']) == int:
        action = round(action)
      self.period_strategy[act] = action

  def apply_move_strategy(self):
    '''Takes the agents strategy for moving and acts on it.
       1 means move to random unoccupied, 0 means stay.
    '''
    self.period_choices['moved'] = False
    if (self.seen_before_move and self.period_strategy['move_if_seen'] == 1) or (not self.seen_before_move and self.period_strategy['move_if_unseen'] == 1):
      unoccupied_positions = [k for k in range(0,self.landscape_size)]
      for a in self.model.agents.values():
        if a.pos in unoccupied_positions:
          unoccupied_positions.remove(a.pos)
      self.pos = choice(unoccupied_positions)
      self.period_choices['moved'] = True

  def apply_sheep_strategy(self):
    '''Takes the agents strategy for choosing grazing level and acts on it.'''
    if 'move_if_seen' in self.model.action_names and not self.seen:
      self.period_choices['sheep_count'] = self.period_strategy['sheep_count_if_unseen']
    else:
      self.period_choices['sheep_count'] = self.period_strategy['sheep_count_if_seen']

  def calc_selfish_felicity(self):
    '''Calculates felicity attributed to own consumption only.'''
    self.period_selfish_felicity = self.felicity_fnc(self.period_payoff - self.min_payoff)

  def calc_felicity(self, total_fel):
    '''Calculates felicity, accounting for altruism'''
    if self.altruism > 0:
      self.period_felicity = (self.altruism * total_fel) + ((1-self.altruism) * self.period_selfish_felicity)
    else:
      self.period_felicity = self.period_selfish_felicity
    self.lifetime_felicity += self.period_felicity

  def update_memory(self):
    '''Updates action weights based on performance this round.'''
    r = self.period_felicity
    #if self.id == 0:
    #if self.model.period % 100 == 0 or self.model.period <5:
    #  print(f'Got a payoff of {r}')
    #  print(f'Period {self.model.period} action payoffs: {self.memory}')
    #  print(f'Period {self.model.period} action frequencies: {self.choice_freq}')
    if 'move_if_seen' in self.model.action_names:
      #---Updating Move Decision---
      if not self.seen_before_move:
        existing_score = self.memory['move_if_unseen'][self.period_strategy['move_if_unseen']]
        self.choice_freq['move_if_unseen'][self.period_strategy['move_if_unseen']] += 1
        freq = self.choice_freq['move_if_unseen'][self.period_strategy['move_if_unseen']]
        if self.recency_bias:
          self.memory['move_if_unseen'][self.period_strategy['move_if_unseen']] = (1-self.recency_bias)*existing_score + (self.recency_bias)*r
        else:
          self.memory['move_if_unseen'][self.period_strategy['move_if_unseen']] = ((freq-1)/freq)*existing_score + (1/freq)*r

      else:
        existing_score = self.memory['move_if_seen'][self.period_strategy['move_if_seen']]
        self.choice_freq['move_if_seen'][self.period_strategy['move_if_seen']] += 1
        freq = self.choice_freq['move_if_seen'][self.period_strategy['move_if_seen']]
        if self.recency_bias:
          self.memory['move_if_seen'][self.period_strategy['move_if_seen']] = (1-self.recency_bias)*existing_score + (self.recency_bias)*r
        else:
          self.memory['move_if_seen'][self.period_strategy['move_if_seen']] = ((freq-1)/freq)*existing_score + (1/freq)*r
      
      #---Updating Harvest Decision---
      if not self.seen:
        existing_score = self.memory['sheep_count_if_unseen'][self.period_strategy['sheep_count_if_unseen']]
        self.choice_freq['sheep_count_if_unseen'][self.period_strategy['sheep_count_if_unseen']] += 1
        freq = self.choice_freq['sheep_count_if_unseen'][self.period_strategy['sheep_count_if_unseen']]
        if self.recency_bias:
          self.memory['sheep_count_if_unseen'][self.period_strategy['sheep_count_if_unseen']] = (1-self.recency_bias)*existing_score + (self.recency_bias)*r
        else:
          self.memory['sheep_count_if_unseen'][self.period_strategy['sheep_count_if_unseen']] = ((freq-1)/freq)*existing_score + (1/freq)*r
          if self.similarity:
            #Adjust payoffs of actions 1 higher and 1 lower:
            act_grain = self.agent_variables['action_set']['sheep_count_if_unseen']['grain']
            sim_weight = exp(-act_grain)
            #Checking if -1 position is in range:
            act_pos = self.actions_avail['sheep_count_if_unseen'].index(self.period_strategy['sheep_count_if_unseen'])
            if act_pos != 0:
              act_L = self.actions_avail['sheep_count_if_unseen'][act_pos-1]
              freq_L = self.choice_freq['sheep_count_if_unseen'][act_L] + sim_weight
              self.memory['sheep_count_if_unseen'][act_L] = ((freq_L - sim_weight)/freq_L)*existing_score + (sim_weight/freq_L)*r
            if act_pos != len(self.actions_avail['sheep_count_if_unseen']):
              act_H = self.actions_avail['sheep_count_if_unseen'][act_pos + 1]
              freq_H = self.choice_freq['sheep_count_if_unseen'][act_H] + sim_weight
              self.memory['sheep_count_if_unseen'][act_H] = ((freq_H - sim_weight)/freq_H)*existing_score + (sim_weight/freq_H)*r
      else:
        existing_score = self.memory['sheep_count_if_seen'][self.period_strategy['sheep_count_if_seen']]
        self.choice_freq['sheep_count_if_seen'][self.period_strategy['sheep_count_if_seen']] += 1
        freq = self.choice_freq['sheep_count_if_seen'][self.period_strategy['sheep_count_if_seen']]
        if self.recency_bias:
          self.memory['sheep_count_if_seen'][self.period_strategy['sheep_count_if_seen']] = (1-self.recency_bias)*existing_score + (self.recency_bias)*r
        else:
          self.memory['sheep_count_if_seen'][self.period_strategy['sheep_count_if_seen']] = ((freq-1)/freq)*existing_score + (1/freq)*r
          if self.similarity:
            act_grain = self.agent_variables['action_set']['sheep_count_if_seen']['grain']
            sim_weight = exp(-act_grain)
            act_pos = self.actions_avail['sheep_count_if_seen'].index(self.period_strategy['sheep_count_if_seen'])
            if act_pos != 0:
              act_L = self.actions_avail['sheep_count_if_seen'][act_pos-1]
              freq_L = self.choice_freq['sheep_count_if_seen'][act_L] + sim_weight
              self.memory['sheep_count_if_seen'][act_L] = ((freq_L - sim_weight)/freq_L)*existing_score + (sim_weight/freq_L)*r
            if act_pos != len(self.actions_avail['sheep_count_if_seen']):
              act_H = self.actions_avail['sheep_count_if_seen'][act_pos + 1]
              freq_H = self.choice_freq['sheep_count_if_seen'][act_H] + sim_weight
              self.memory['sheep_count_if_seen'][act_H] = ((freq_H - sim_weight)/freq_H)*existing_score + (sim_weight/freq_H)*r
        
    else:
      #---Updating Harvest Decision in Non-Spatial model---
      existing_score = self.memory['sheep_count_if_seen'][self.period_strategy['sheep_count_if_seen']]
      self.choice_freq['sheep_count_if_seen'][self.period_strategy['sheep_count_if_seen']] += 1
      freq = self.choice_freq['sheep_count_if_seen'][self.period_strategy['sheep_count_if_seen']]
      if self.recency_bias:
        self.memory['sheep_count_if_seen'][self.period_strategy['sheep_count_if_seen']] = (1-self.recency_bias)*existing_score + (self.recency_bias)*r
      else:
        self.memory['sheep_count_if_seen'][self.period_strategy['sheep_count_if_seen']] = ((freq-1)/freq)*existing_score + (1/freq)*r
      #if self.id == 0:
        #if self.model.period % 100 == 0 or self.model.period <5:
        #  print(f'freq: {freq}')
        #  print(f'r: {r}')
        #  print(f'weight on new score: {(1/freq)}')
        #  print(f'existing score: {existing_score}')
        #  print(f'weight on existing score: {((freq-1)/freq)}')
        #  print(f"final new score: {self.memory['sheep_count_if_seen'][self.period_strategy['sheep_count_if_seen']]}")
  
  def store_attraction_snapshot(self, period):
    '''Stores probability of taking each action at this period of time.'''
    p_explore = self._period_prob_explore()
    for act in self.agent_variables['action_set'].keys():
      actions_avail = [i for i in range(len(self.memory[act]))]
      #Contribute prob_explore component to prob_choose vector:
      weights = [i**2 for i in self.memory[act]]
      weight_sum = sum(weights)
      if weight_sum == 0:
        prob_choose = [1/len(actions_avail) for aa in range(len(actions_avail))]
      else:
        prob_choose = [w*p_explore/weight_sum for w in weights]
      #Contribute prob_exploit component to prob_choose vector:
      if period > 0:
        max_score = max(self.memory[act])
        max_positions = []
        for s in range(len(self.memory[act])):
          if self.memory[act][s] == max_score:
            max_positions.append(s)
        p_exploit = (1-p_explore)
        for pos in max_positions:
            prob_choose[pos] += p_exploit/len(max_positions)
      #Save to agent
      self.attraction_snapshots[period] = {}
      self.attraction_snapshots[period][act] = {}
      for aa_pos in range(len(actions_avail)):
        self.attraction_snapshots[period][act][actions_avail[aa_pos]] = prob_choose[aa_pos]
    return self.attraction_snapshots[period]
    
        

        
      
