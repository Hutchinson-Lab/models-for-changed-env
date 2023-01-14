import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plot_descriptions import plots_metadata, selected_plots

output_table_dir = './experiments/tables/'
output_plot_main_dir = './experiments/plots/'
output_plot_dir = './experiments/plots/all/'

def plot_cost_boxplots (df, plot_metadata, ds_keys):
    
    col_names = list(plot_metadata.keys())[1:-1]

    nrow = 3
    ncol = 5
    sns.set_style('darkgrid')
    fig, _ = plt.subplots(nrow, ncol, sharey=False, figsize=(16, 8))
    
    handles = None

    for i, ax in enumerate(fig.axes):
        
        current_df = df[df['Data Set']==ds_keys[i]]

        current_df = pd.melt(
            current_df, 
            id_vars=[plot_metadata['varying']], 
            value_vars=[
                'Optimal Point Cost (Actual)',
                'Optimal Point Cost (ROCCH Method)',
                'Optimal Point Cost (Accuracy-Max)', 
                'Optimal Point Cost (F1-score-Max)',
                ]
        )

        # current_df = current_df.round(4)
        # current_df.to_csv(f'{output_table_dir}current_{plot_metadata["identifier"]}_{ds_keys[i]}.csv')

        # Plot boxplots
        sns.boxplot(y='value', x=plot_metadata['varying'], hue='variable', data=current_df,  orient='v' , ax=ax)
        
        # Format subplot labels and title
        ax.set(xlabel=None, ylabel=None)
        ax.set_title(ds_keys[i], fontsize=14)

        # Save subplot legend hangles and labels for suplegend
        handles,_ = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        
    # ax.set_title(f"{col_names[0]}={plot_metadata[col_names[0]]}\nK=3, Sep. to Train Ratio=0.5\nOrig. To Impr. Ratio=0.5, FN cost=1.0", fontsize=12)

    # Format legend
    fig.legend(
        handles, 
        ['Actual', 'ROCCH Method', 'Accuracy Max', 'F1-score Max'],
        title='Cost Incurred by Optimal Points',
        title_fontsize=16,
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.1), 
        ncol=4, 
        fontsize=14)

    # Format suplabels and title
    fig.supylabel('Expected Cost', x=0.01, fontsize=16)
    fig.supxlabel(plot_metadata['varying'], y=0.02, fontsize=16)
    fig.suptitle(f'Incurred Cost while varying "{plot_metadata["varying"]}"\n{col_names[0]}={plot_metadata[col_names[0]]}, {col_names[1]}={plot_metadata[col_names[1]]}, {col_names[2]}={plot_metadata[col_names[2]]}, {col_names[3]}={plot_metadata[col_names[3]]}', fontsize=18)
    fig.tight_layout()

    fig.savefig(f'{output_plot_dir}cost_{plot_metadata["identifier"]}.png', bbox_inches='tight', dpi=300)
    plt.close()


def plot_dist_boxplots(df, plot_metadata, ds_keys):

    col_names = list(plot_metadata.keys())[1:-1]

    nrow = 3
    ncol = 5
    sns.set_style('darkgrid')
    fig, _ = plt.subplots(nrow, ncol, sharey=False, figsize=(16, 8))
    
    handles = None

    for i, ax in enumerate(fig.axes):
        
        current_df = df[df['Data Set']==ds_keys[i]]

        current_df = pd.melt(
            current_df, 
            id_vars=[plot_metadata['varying']], 
            value_vars=[
                'Distance between ROCCHM and Actual',
                'Distance between Accuracy-Max and Actual',
                'Distance between F1-score-Max and Actual', 
                ]
        )

        # current_df = current_df.round(4)
        # current_df.to_csv(f'{output_table_dir}current_{plot_metadata["identifier"]}_{ds_keys[i]}.csv')

        # Plot boxplots
        palette = {"Distance between ROCCHM and Actual": "orange", "Distance between Accuracy-Max and Actual": "green",  "Distance between F1-score-Max and Actual": "red",}
        sns.boxplot(y='value', x=plot_metadata['varying'], hue='variable', data=current_df,  orient='v' , ax=ax, palette=palette)
        
        # Format subplot labels and title
        ax.set(xlabel=None, ylabel=None)
        ax.set_title(ds_keys[i], fontsize=14)

        # Save subplot legend hangles and labels for suplegend
        handles,_ = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        
    # ax.set_title(f"{col_names[0]}={plot_metadata[col_names[0]]}\nK=3, Sep. to Train Ratio=0.5\nOrig. To Impr. Ratio=0.5, FN cost=1.0", fontsize=12)

    # Format legend
    fig.legend(
        handles, 
        ['ROCCH Method', 'Accuracy Max', 'F1-score Max'],
        title='Distance to Actual Optimal Point',
        title_fontsize=16,
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.1), 
        ncol=4, 
        fontsize=14)

    # Format suplabels and title
    fig.supylabel('Distance', x=0.01, fontsize=16)
    fig.supxlabel(plot_metadata['varying'], y=0.02, fontsize=16)
    fig.suptitle(f'Distance to Actual Optimal Point while varying "{plot_metadata["varying"]}"\n{col_names[0]}={plot_metadata[col_names[0]]}, {col_names[1]}={plot_metadata[col_names[1]]}, {col_names[2]}={plot_metadata[col_names[2]]}, {col_names[3]}={plot_metadata[col_names[3]]}', fontsize=18)
    fig.tight_layout()

    fig.savefig(f'{output_plot_dir}dist_{plot_metadata["identifier"]}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
def plot_results():
    '''
    
    '''
    print("\nPlotting results:")
    
    if not os.path.exists(output_plot_main_dir):
        os.makedirs(output_plot_main_dir)

    if not os.path.exists(output_plot_dir):
        os.makedirs(output_plot_dir)

    dataset_descriptions = pd.read_csv(f'{output_table_dir}dataset_descriptons.csv') # Saved during preprocessing
    ds_keys = list(dataset_descriptions['Data Set'])

    performance_df_summarized = pd.read_csv(f'{output_table_dir}performance_summarized.csv')


    for k in plots_metadata:

        col_names = list(plots_metadata[k].keys())[1:-1]
 
        current_slice_idx = True
        for j in col_names:
            current_slice_idx &=  (performance_df_summarized[j]==plots_metadata[k][j])

        current_df = performance_df_summarized[current_slice_idx].copy()

        # print(current_df.shape)
        # current_df = current_df.round(4)
        # current_df.to_csv(f'{output_table_dir}current_{plots_metadata[k]["identifier"]}.csv')

        plot_cost_boxplots(current_df, plots_metadata[k], ds_keys)
        plot_dist_boxplots(current_df, plots_metadata[k], ds_keys)

    # Save selected plots of interest to a separate directory
    for plot_filename in selected_plots:
        shutil.copyfile(f'{output_plot_dir}{plot_filename}', f'{output_plot_main_dir}{plot_filename}')
    
    arr = os.listdir(f'{output_plot_dir}')
    print(len(arr))
    
    print("Plotting completed.")