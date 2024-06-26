import pandas as pd
from matplotlib import pyplot as plt
import opinionated
plt.style.use("opinionated_rc")

def plot_my_stuff(in_filename, y_var, y_label, title, figname, aid = 'all'):
    df = pd.read_csv(in_filename, sep=',')

    # Define a list of colors for the plot lines. You can change these as you like.
    color_list = ['#686868', '#0f203a', '#f98e31', '#a81a26', 'magenta', 'yellow', 'black']
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create a scatter plot for each agent id
    if aid == 'all':
        for i, agent_id in enumerate(df['agent_id'].unique()):
            plt.scatter(df[df['agent_id'] == agent_id]['period'], df[df['agent_id'] == agent_id][y_var],
                    s = 1,
                    color=color_list[i % len(color_list)], alpha=.1)  # Set line color in a cycling manner
    elif type(aid) == int:
        plt.scatter(df[df['agent_id'] == aid]['period'], df[df['agent_id'] == aid][y_var],
                    s = 1,
                    color=color_list[0], alpha=.5)

    plt.xlabel('Period')
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(figname)

#---For V0 and V1 and V2PolicySandbox---
plot_my_stuff('V0_fullrul_data.txt', 'sheep_count_if_seen', 'harvest level h', 'Altruistic harvest choice', 'V0_choices.png')
#plot_my_stuff('V0_fullrul_data.txt', 'avg_felicity', 'Avg felicity', 'Altruistic average felicity', 'V0_felicity.png', aid = 0)

plot_my_stuff('V1_fullrul_data.txt', 'sheep_count_if_seen', 'harvest level h', 'Selfish harvest choice', 'V1_choices.png')
#plot_my_stuff('V1_fullrul_data.txt', 'avg_felicity', 'Avg felicity', 'Selfish average felicity', 'V1_felicity.png', aid = 0)

plot_my_stuff('V2_SP_Policy_data.txt', 'sheep_count_if_seen', 'harvest level h', 'Selfish harvest choice facing f(.)', 'V2_choices.png')

#---For V2---
#plot_my_stuff('S2_test_data.txt', 'sheep_count_if_seen', 'Harvest choices over time', 'S2 Extraction', 'S2_extraction.png')
#plot_my_stuff('S2_test_data.txt', 'avg_felicity', 'Avg felicity', 'S2 Average felicity', 'S2_avg_payoff.png', aid = 0)

#---For V3---
'''
plot_my_stuff('S3_test_data.txt', 'sheep_count_if_seen', 'Extraction choice | monitored', 'S3 Favored extraction if seen', 'S3_fav_extraction_seen.png')
plot_my_stuff('S3_test_data.txt', 'sheep_count_if_unseen', 'Extraction choice | not monitored', 'S3 Favored extraction if not seen', 'S3_fav_extraction_unseen.png')
plot_my_stuff('S3_test_data.txt', 'move_if_seen', 'Movement choice | monitored', 'S3 Favored movement if seen', 'S3_Fav_movement_seen.png')
plot_my_stuff('S3_test_data.txt', 'move_if_unseen', 'Movement choice | not monitored', 'S3 Favored movement if not seen', 'S3_Fav_movement_unseen.png')
plot_my_stuff('S3_test_data.txt', 'avg_felicity', 'Avg felicity', 'S3 Average payoff over time', 'S3_avg_payoff.png', aid = 0)
plot_my_stuff('S3_test_data.txt', 'monitoring_rate', '% monitored', 'S3 Monitoring over time', 'S3_mon.png', aid = 0)
'''