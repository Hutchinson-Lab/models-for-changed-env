import os
import shutil
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from .plot_descriptions import general_plots_metadata, selected_general_plots, characteristics_plots_metadata, metrics_plots_metadata

output_table_dir = './experiments/Synthetic/tables/'

output_plot_main_dir = './experiments/Synthetic/plots/'

output_plot_general_dir = './experiments/Synthetic/plots/general/'
output_plot_general_c_dir = './experiments/Synthetic/plots/general/cost/'
output_plot_general_c_n_dir = './experiments/Synthetic/plots/general/cost/norm/'
output_plot_general_c_e_dir = './experiments/Synthetic/plots/general/cost/exp/'
output_plot_general_d_dir = './experiments/Synthetic/plots/general/distance/'
output_plot_general_d_n_dir = './experiments/Synthetic/plots/general/distance/norm/'
output_plot_general_d_e_dir = './experiments/Synthetic/plots/general/distance/exp/'

output_plot_metrics_dir = './experiments/Synthetic/plots/metrics/'
output_plot_metrics_n_dir = './experiments/Synthetic/plots/metrics/norm/' #new
output_plot_metrics_e_dir = './experiments/Synthetic/plots/metrics/exp/' #new


ds_markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', 'P', '*', 'h', 'H']


def plot_general_cost_boxplots (df, plot_metadata, ds_keys, cost_type):
    
    cost_type_abr = None
    save_dir = None

    if (cost_type=='Normalized'):
        cost_type_abr = 'Norm'
        save_dir = output_plot_general_c_n_dir
    elif (cost_type=='Expected'):
        cost_type_abr = 'Exp'
        save_dir = output_plot_general_c_e_dir

    col_names = list(plot_metadata.keys())[1:-1]

    nrow = 3
    ncol = 4
    sns.set_style('darkgrid')
    fig, _ = plt.subplots(nrow, ncol, sharey=True, figsize=(8, 5))
    
    handles = None

    for i, ax in enumerate(fig.axes):
        
        current_df = df[df['Data Set']==ds_keys[i]]

        current_df = pd.melt(
            current_df, 
            id_vars=[plot_metadata['varying']], 
            value_vars=[
                f'Optimal Point {cost_type} Cost (Actual-{cost_type_abr})',
                f'Optimal Point {cost_type} Cost (Norm-Cost-Min)', 
                f'Optimal Point {cost_type} Cost (Exp-Cost-Min)',
                f'Optimal Point {cost_type} Cost (Accuracy-Max)', 
                f'Optimal Point {cost_type} Cost (F1-score-Max)',
                f'Optimal Point {cost_type} Cost (ROCCH Method)',
                ]
        )

        # current_df = current_df.round(4)
        # current_df.to_csv(f'{output_table_dir}current_{plot_metadata["identifier"]}_{ds_keys[i]}.csv')


        # Plot boxplots
        palette = {
            f'Optimal Point {cost_type} Cost (Actual-{cost_type_abr})' : 'limegreen',
            f'Optimal Point {cost_type} Cost (Norm-Cost-Min)' : 'royalblue', 
            f'Optimal Point {cost_type} Cost (Exp-Cost-Min)' : 'orange',
            f'Optimal Point {cost_type} Cost (Accuracy-Max)' : 'sienna', 
            f'Optimal Point {cost_type} Cost (F1-score-Max)' : 'mediumpurple',
            f'Optimal Point {cost_type} Cost (ROCCH Method)' : 'red',

        }
        sns.boxplot(y='value', x=plot_metadata['varying'], hue='variable', data=current_df,  orient='v', linewidth=0.5, fliersize=1, palette=palette, ax=ax)
        
        # Format subplot labels and title
        ax.set(xlabel=None, ylabel=None)
        ax.set_title(ds_keys[i], fontsize=8)

        ax.tick_params(axis='both', which='major', labelsize=6)

        ax.patch.set_edgecolor('black')  

        # ax.patch.set_linewidth('1') 

        # Save subplot legend hangles and labels for suplegend
        handles,_ = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        
    # ax.set_title(f"{col_names[0]}={plot_metadata[col_names[0]]}\nK=3, Sep. to Train Ratio=0.5\nOrig. To Impr. Ratio=0.5, FN cost=1.0", fontsize=12)

    # Format legend
    fig.legend(
        handles, 
        ['Actual',  'Norm-Cost-Min', 'Exp-Cost-Min', 'Accuracy-Max', 'F1-score-Max', 'ROCCH Method',],
        title='Cost Incurred by Selected Points',
        title_fontsize=8,
        loc='lower center', 
        bbox_to_anchor=(0.5, - 0.075), 
        ncol=6, 
        fontsize=7)

    # Format suplabels and title
    fig.supylabel(f'{cost_type} Cost', x=0.02, fontsize=8)
    fig.supxlabel(plot_metadata['varying'], y=0.04, fontsize=8)
    fig.suptitle(f'{cost_type} Cost while varying "{plot_metadata["varying"]}"\n{col_names[0]}={plot_metadata[col_names[0]]}, {col_names[1]}={plot_metadata[col_names[1]]}\n{col_names[2]}={plot_metadata[col_names[2]]}, {col_names[3]}={plot_metadata[col_names[3]]}, {col_names[4]}={plot_metadata[col_names[4]]}, {col_names[5]}={plot_metadata[col_names[5]]}', fontsize=9)
    fig.tight_layout()

    fig.savefig(f'{save_dir}cost_{plot_metadata["identifier"]}.png', bbox_inches='tight', dpi=300)
    plt.close()


def plot_general_dist_boxplots(df, plot_metadata, ds_keys, cost_type):

    cost_type_abr = None
    save_dir = None

    if (cost_type=='Normalized'):
        cost_type_abr = 'Norm'
        save_dir = output_plot_general_d_n_dir
    elif (cost_type=='Expected'):
        cost_type_abr = 'Exp'
        save_dir = output_plot_general_d_e_dir

    col_names = list(plot_metadata.keys())[1:-1]

    nrow = 3
    ncol = 4
    sns.set_style('darkgrid')
    fig, _ = plt.subplots(nrow, ncol, sharey=True, figsize=(8, 5))
    
    handles = None

    for i, ax in enumerate(fig.axes):
        
        current_df = df[df['Data Set']==ds_keys[i]]

        current_df = pd.melt(
            current_df, 
            id_vars=[plot_metadata['varying']], 
            value_vars=[
                f'Distance between Norm-Cost-Min and Actual-{cost_type_abr}',
                f'Distance between Exp-Cost-Min and Actual-{cost_type_abr}',
                f'Distance between Accuracy-Max and Actual-{cost_type_abr}',
                f'Distance between F1-score-Max and Actual-{cost_type_abr}',
                f'Distance between ROCCHM and Actual-{cost_type_abr}', 
                ]
        )

        # current_df = current_df.round(4)
        # current_df.to_csv(f'{output_table_dir}current_{plot_metadata["identifier"]}_{ds_keys[i]}.csv')

        # Plot boxplots
        palette = {
            f'Distance between Norm-Cost-Min and Actual-{cost_type_abr}'  : 'royalblue',
            f'Distance between Exp-Cost-Min and Actual-{cost_type_abr}' : 'orange',
            f'Distance between Accuracy-Max and Actual-{cost_type_abr}' : 'sienna',
            f'Distance between F1-score-Max and Actual-{cost_type_abr}' : 'mediumpurple',
            f'Distance between ROCCHM and Actual-{cost_type_abr}' : 'red',
        }
        sns.boxplot(y='value', x=plot_metadata['varying'], hue='variable', data=current_df,  orient='v' , linewidth=0.5,  fliersize=1, ax=ax, palette=palette)
        
        # Format subplot labels and title
        ax.set(xlabel=None, ylabel=None)
        ax.set_title(ds_keys[i], fontsize=8)

        ax.tick_params(axis='both', which='major', labelsize=6)

        ax.patch.set_edgecolor('black')  

        # ax.patch.set_linewidth('1') 

        # Save subplot legend hangles and labels for suplegend
        handles,_ = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        
    # ax.set_title(f"{col_names[0]}={plot_metadata[col_names[0]]}\nK=3, Sep. to Train Ratio=0.5\nOrig. To Impr. Ratio=0.5, FN cost=1.0", fontsize=12)

    # Format legend
    fig.legend(
        handles, 
        ['Norm-Cost-Min', 'Exp-Cost-Min', 'Accuracy-Max', 'F1-score-Max', 'ROCCH Method',],
        title='Distance to Actual Optimal Point',
        title_fontsize=8,
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.075), 
        ncol=5, 
        fontsize=7)

    # Format suplabels and title
    fig.supylabel('Distance', x=0.02, fontsize=8)
    fig.supxlabel(plot_metadata['varying'], y=0.04, fontsize=8)
    fig.suptitle(f'Distance to Actual Optimal Point based on {cost_type} Cost while varying "{plot_metadata["varying"]}"\n{col_names[0]}={plot_metadata[col_names[0]]}, {col_names[1]}={plot_metadata[col_names[1]]}\n{col_names[2]}={plot_metadata[col_names[2]]}, {col_names[3]}={plot_metadata[col_names[3]]}, {col_names[4]}={plot_metadata[col_names[4]]}, {col_names[5]}={plot_metadata[col_names[5]]}', fontsize=9)
    fig.tight_layout()

    fig.savefig(f'{save_dir}dist_{plot_metadata["identifier"]}.png', bbox_inches='tight', dpi=300)
    plt.close()




        
def plot_metrics_pointplot(df, descriptions_df, plot_metadata, identifier, cost_type):

    cost_type_abr = None
    save_dir = None

    if (cost_type=='Normalized'):
        cost_type_abr = 'Norm'
        save_dir = output_plot_metrics_n_dir
    elif (cost_type=='Expected'):
        cost_type_abr = 'Exp'
        save_dir = output_plot_metrics_e_dir

    
    subplot_row_titles = ['Wasserstein Distance', 'Energy Distance', 'Maximum Mean Discrepancy', 'Area Under ROC Curve (AUC)', "Matthew's Correlation Coefficient", "Cr$\\'{a}$mer-Von Mises Criterion"]
    subplot_row_titles_fmt = ['Wasserstein\nDistance', 'Energy\nDistance', 'Maximum Mean\nDiscrepancy', 'Area Under\nROC Curve\n(AUC)', "Matthew's\nCorrelation\nCoefficient", "Cr$\\'{a}$mer-Von Mises\nCriterion"]

    subplot_column_titles = [0.75, 0.5, 1.0, 1.25]

    df['Cost Difference'] =  df[f'Optimal Point {cost_type} Cost (ROCCH Method)'] - df[f'Optimal Point {cost_type} Cost (Actual-{cost_type_abr})']
    df.reset_index(drop=True)

    col_names = list(plot_metadata.keys())

    nrow = 6
    ncol = 4


    sns.set_style('whitegrid', {"grid.color": "silver", "grid.linestyle": "dotted"})

    matplotlib.rcParams['legend.handlelength'] = 0
    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['legend.borderpad'] = 1.0
    matplotlib.rcParams['legend.handletextpad'] = 1.0
    matplotlib.rcParams['legend.borderaxespad'] = 0.8   
    

    fig, _ = plt.subplots(nrow, ncol, sharey=True, figsize = (6,7))

        
    handles = None

    for i, ax in enumerate(fig.axes):


        r = i // ncol 
        c = i % ncol
        
        df_t = df[df['Test to Train Class Distr. Ratio']==subplot_column_titles[c]].copy()
        
        
        df_t['Avg. '+subplot_row_titles[r]] = 0
        # current_df['Avg. Cost Difference'] = 0
        order = []
        # cost_order = []
        for ds_key in list(descriptions_df['Data Set']):

            order.append(df_t.loc[df['Data Set']==ds_key, subplot_row_titles[r]].mean())
            df_t.loc[df['Data Set']==ds_key, 'Avg. '+subplot_row_titles[r]] = order[-1]


        sns.lineplot(
            y='Cost Difference', 
            x='Avg. '+subplot_row_titles[r], 
            hue='Avg. '+subplot_row_titles[r],
            style='Avg. '+subplot_row_titles[r],
            hue_order= order,
            style_order= order, 
            markers=ds_markers,
            data=df_t,
            palette=sns.color_palette("husl", 12),
            sort=False,
            err_style='bars',
            errorbar='sd',
            err_kws={'elinewidth':0.75},
            dashes=False,
            ax=ax,
            )

        sns.regplot(
            data=df_t, 
            x='Avg. '+subplot_row_titles[r], 
            y="Cost Difference",
            scatter=False,
            line_kws={"lw":0.75, "ls":"--", "color":"grey"},
            ax=ax,
            )

        model = sm.OLS(df_t["Cost Difference"], sm.add_constant(df_t['Avg. '+subplot_row_titles[r]])).fit()

        # print(model.params)
        intercept, slope = round(model.params[0],2), round(model.params[1],2)
        r_squared, p_value = round(model.rsquared, 2), round(model.pvalues.loc['Avg. '+subplot_row_titles[r]], 2) 
        ax.text(
            0.675, 
            0.85, 
            f'y = {slope} x + {intercept}\n$r^{2}$={r_squared}, p-value={p_value}',
            bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', alpha=0.5, linewidth=0.5), 
            horizontalalignment='center', 
            verticalalignment='center', 
            transform=ax.transAxes, 
            fontsize=4)


        
        

        if (r==0):
            ax.set_title('$\\frac{P_{test}(Y=1)}{P_{train}(Y=1)}$='+str(subplot_column_titles[c]), fontsize=5)
        else:
            ax.set(title=None)
        
        if (c==(ncol-1)):
            
            ax.text(
                1.125, 
                0.5, 
                subplot_row_titles_fmt[r],
                # bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', alpha=0.5, linewidth=0.5), 
                horizontalalignment='center', 
                verticalalignment='center', 
                transform=ax.transAxes,
                rotation=90, 
                fontsize=5
            )


    
            
        
        ax.set(xlabel=None,ylabel=None)

        # else:
            # ax.set(ylabel=None)

        ax.tick_params(axis='x', labelsize=3)
        ax.tick_params(axis='y', labelsize=3)

        # Save subplot legend hangles and labels for suplegend
        if handles == None:
            handles,_ = ax.get_legend_handles_labels()
        ax.get_legend().remove()

    # handles,_ = ax.get_legend_handles_labels()
    
    # Format legend
    fig.legend(
        handles, 
        descriptions_df['Data Set'],
        title='Data Sets',
        title_fontsize=6,
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.1), 
        ncol=3, 
        fontsize=6)

    # Format suplabels and title
    fig.supylabel(f'{cost_type} Cost Difference (ROCCH - Actual)', x=0.04, fontsize=8)
    fig.supxlabel('', fontsize=1)
    fig.suptitle(f'Effect of Covariate Shift on Efficacy of ROCCH Method\n{col_names[0]}={plot_metadata[col_names[0]]}, {col_names[1]}={plot_metadata[col_names[1]]}\n {col_names[2]}={plot_metadata[col_names[2]]}, {col_names[3]}={plot_metadata[col_names[3]]}, {col_names[4]}={plot_metadata[col_names[4]]}, {col_names[5]}={plot_metadata[col_names[5]]}, {col_names[6]}={plot_metadata[col_names[6]]}', y =0.9675, fontsize=8)
    fig.tight_layout()

    

    fig.savefig(f'{save_dir}ds_metrics_{identifier}.png', bbox_inches='tight', dpi=300)
    
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    plt.close()







    
def plot_results():
    '''
    
    '''
    print("\nPlotting results:")
    
    if not os.path.exists(output_plot_main_dir):
        os.makedirs(output_plot_main_dir)

    if not os.path.exists(output_plot_general_dir):
        os.makedirs(output_plot_general_dir)

    if not os.path.exists(output_plot_general_c_dir):
        os.makedirs(output_plot_general_c_dir)

    if not os.path.exists(output_plot_general_c_n_dir):
        os.makedirs(output_plot_general_c_n_dir)

    if not os.path.exists(output_plot_general_c_e_dir):
        os.makedirs(output_plot_general_c_e_dir)
    
    if not os.path.exists(output_plot_general_d_dir):
        os.makedirs(output_plot_general_d_dir)

    if not os.path.exists(output_plot_general_d_n_dir):
        os.makedirs(output_plot_general_d_n_dir)
    
    if not os.path.exists(output_plot_general_d_e_dir):
        os.makedirs(output_plot_general_d_e_dir)

    if not os.path.exists(output_plot_metrics_dir):
        os.makedirs(output_plot_metrics_dir)

    if not os.path.exists(output_plot_metrics_n_dir):
        os.makedirs(output_plot_metrics_n_dir)
    
    if not os.path.exists(output_plot_metrics_e_dir):
        os.makedirs(output_plot_metrics_e_dir)


    dataset_descriptions = pd.read_csv(f'{output_table_dir}dataset_descriptons.csv') # Saved during preprocessing
    ds_keys = list(dataset_descriptions['Data Set'])

    performance_df_summarized = pd.read_csv(f'{output_table_dir}performance_summarized.csv')


    # Effects of varying settings/configurations

    for k in general_plots_metadata:

        col_names = list(general_plots_metadata[k].keys())[1:-1]
 
        current_slice_idx = True
        for j in col_names:
            current_slice_idx &=  (performance_df_summarized[j]==general_plots_metadata[k][j])

        current_df = performance_df_summarized[current_slice_idx].copy()
        
        plot_general_cost_boxplots(current_df, general_plots_metadata[k], ds_keys, 'Normalized')
        plot_general_dist_boxplots(current_df, general_plots_metadata[k], ds_keys, 'Normalized')

        plot_general_cost_boxplots(current_df, general_plots_metadata[k], ds_keys, 'Expected')
        plot_general_dist_boxplots(current_df, general_plots_metadata[k], ds_keys, 'Expected')

    # Save selected plots of interest to a separate directory
    # for plot_filename in selected_varying_plots:
    #     shutil.copyfile(f'{output_plot_varying_dir}{plot_filename}', f'{output_plot_main_dir}{plot_filename}')




    # Relationship between prior class probability shift and covariate shift
        

    performance_df_summarized = performance_df_summarized.rename(
        columns={
        
            'Avg. Wasserstein Dist.' : 'Wasserstein Distance',
            'Avg. Energy Dist.' : 'Energy Distance',
            'MMD' : 'Maximum Mean Discrepancy',
            'Avg. AUC (COVSHIFT)' : 'Area Under ROC Curve (AUC)',
            'Avg. Phi' : "Matthew's Correlation Coefficient",
            'Avg. Cramer-von Mises Criterion' : "Cr$\\'{a}$mer-Von Mises Criterion",
        }
    )
 
    for k in metrics_plots_metadata:
        col_names = list(metrics_plots_metadata[k].keys())

        current_slice_idx = True
        for j in col_names:
            current_slice_idx &=  (performance_df_summarized[j]==metrics_plots_metadata[k][j])

        current_df = performance_df_summarized[current_slice_idx].copy()

        plot_metrics_pointplot(current_df, dataset_descriptions, metrics_plots_metadata[k], k, 'Normalized')
        plot_metrics_pointplot(current_df, dataset_descriptions, metrics_plots_metadata[k], k, 'Expected')


    
    print("Plotting completed.")