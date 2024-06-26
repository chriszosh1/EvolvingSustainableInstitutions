from model import Model, run_model
from bundleV2A_SPHarvestOnly import run_run_model_harvestOnly
from theoretical_functions import marginal_benefit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

'''
Version summary
   This operates the same a V2. We can use this to do 1 off runs with policies of interest.
'''

gv = {'return_type': 'negative_externality', 'landscape_size':1, 'alpha':.8, 'beta':.05}
av = {'vision':1, 'altruism':0,'similarity':False,
      'action_set':{'sheep_count_if_seen': {'min': 0, 'max': 5, 'grain':1}}}
#vv = ['period','payoff', 'strategies','actions']
vv = []
dto = {'tag': 'SW_Calcs-PolSelfishNoSimFromCompModelSPNoRedistr', 'files':['final']}
#dto = {'tag': 'S2_test', 'files':['per_period']}
#Creating the minimal fine to adjust behavior
fv = [0,0,0,0.92,2.13,2.42]
pv = {'fine_vector':{'type':'vector','min':0, 'max':10, 'size':6}}
store_attraction_snapshots = [0, 999, 1_999, 4_999, 9_999, 49_999]
runs = 25

#Run the model many times:
att_df_list = []
total_sw = 0
for r in range(runs):
    scenario2 = Model(agent_count = 4, store_attraction_snapshots = store_attraction_snapshots, game_variables = gv, agent_variables = av,
                      verbose_variables = vv, fine_vector = fv, social_planner_vars = pv, data_to_output = dto, run = r)
    results = run_model(model = scenario2, steps = 50_000)
    attraction_df = pd.DataFrame.from_dict(scenario2.attraction_snaps, orient='columns', dtype=None, columns=None)
    #print(attraction_df)
    att_df_list.append(attraction_df)
    print(f'Done with run {r}')
    total_sw += scenario2.avg_social_welfare
avg_sw = total_sw / runs
print(f'Social welfare: {avg_sw}')

mean_attractions = pd.concat(att_df_list).reset_index().groupby("index").mean()
print(mean_attractions)

def get_rainbow_colors(N):
    # Generate bar plot colors evenly spaced from rainbow
    hsv_values = np.column_stack([
        np.linspace(0, 1, N),
        np.ones(N),
        np.ones(N)
    ])
    rgb_values = hsv_to_rgb(hsv_values)
    return rgb_values

def create_bar_chart(filename, title, x_label, y_label, df):
    df.index.name = 'index'
    fig, ax = plt.subplots(figsize=(10, 6))
    num_bars = len(df.columns)
    bar_width = 0.15
    bar_positions = np.arange(len(df))
    colors = get_rainbow_colors(num_bars+1)
    for i, column in enumerate(df.columns):
        plt.bar(bar_positions + i * bar_width, df[column], width=bar_width, label=column,  color=colors[i])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0, 1)
    plt.legend(loc='upper left')
    plt.xticks(bar_positions + (bar_width * (num_bars - 1) / 2), df.index)
    plt.savefig(filename)

def create_line_chart(filename, title, x_label, y_label, df):
    plt.ylim(0, 1)
    colors = ['red', 'orange', 'gold', 'green', 'blue', 'purple']
    for i, column in enumerate(df.columns):
        plt.plot(df.index, df[column], label=int(column) + 1, color=colors[i], alpha=0.3, marker='o')
    plt.legend(title="Periods")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(filename)


#create_bar_chart('Choices_democracy.png',"Harvest attractions over Time", "Actions", "Frequency", df = mean_attractions)

# Plot the lines
create_line_chart('Choices_Democracy', 'Harvest level over time','Harvest levels','Frequency',mean_attractions)