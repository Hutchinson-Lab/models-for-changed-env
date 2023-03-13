import os
import shutil
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from .plot_descriptions import varying_plots_metadata, selected_varying_plots, ds_plots_metadata, dsdist_plots_metadata

output_table_dir = './experiments/Synthetic/tables/'
output_plot_main_dir = './experiments/Synthetic/plots/'
output_plot_dscomp_dir = './experiments/Synthetic/plots/all_dscomp/'
output_plot_dscomp_r_dir = './experiments/Synthetic/plots/all_dscomp/class_distance_ratio/'
output_plot_dsdist_dir = './experiments/Synthetic/plots/all_dsdist/'
output_plot_dsdist_w_dir = './experiments/Synthetic/plots/all_dsdist/wasserstein/'
output_plot_dsdist_e_dir = './experiments/Synthetic/plots/all_dsdist/energy/'
output_plot_dsdist_mmd_dir = './experiments/Synthetic/plots/all_dsdist/mmd/'
output_plot_dsdist_a_dir = './experiments/Synthetic/plots/all_dsdist/auc/'
output_plot_dsdist_mcc_dir = './experiments/Synthetic/plots/all_dsdist/mcc/'
output_plot_dsdist_c_dir = './experiments/Synthetic/plots/all_dsdist/cramervonmises/'
output_plot_varying_dir = './experiments/Synthetic/plots/all_varying/'


ds_markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', 'P', '*', 'h', 'H', 'X','d', 'D']
ds_markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', 'P', '*', 'h', 'H']        



def plot_ds_cost_cls_ratio_pointplots (df, descriptions_df, plot_metadata, identifier):

    subplot_titles = ['Class Distance Ratio (linear)', 'Class Distance Ratio (poly)', 'Class Distance Ratio (rbf)', 'Class Distance Ratio (sigmoid)']

    df['Cost Difference'] =  df['Avg. Optimal Point Cost (ROCCH Method)'] - df['Avg. Optimal Point Cost (Actual)']
    df.reset_index(drop=True)

    
    for subplot_title in subplot_titles:

        df[subplot_title] = 0
       
    
        for ds_key in descriptions_df['Data Set']:
        
            df.loc[df['Data Set']==ds_key, subplot_title] = float(descriptions_df.loc[descriptions_df['Data Set']==ds_key, subplot_title])


    col_names = list(plot_metadata.keys())

    nrow = 2
    ncol = 2


    sns.set_style('whitegrid', {"grid.color": "silver", "grid.linestyle": "dotted"})

    matplotlib.rcParams['legend.handlelength'] = 0
    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['legend.borderpad'] = 1.0
    matplotlib.rcParams['legend.handletextpad'] = 1.0
    matplotlib.rcParams['legend.borderaxespad'] = 0.8   
    

    fig, _ = plt.subplots(nrow, ncol, sharey=True, figsize = (5,4))

        
    handles = None

    for i, ax in enumerate(fig.axes):
        

        sns.lineplot(
            y='Cost Difference', 
            x=subplot_titles[i], 
            hue=subplot_titles[i],
            style=subplot_titles[i],
            hue_order= descriptions_df[subplot_titles[i]],
            style_order= descriptions_df[subplot_titles[i]], 
            markers=ds_markers,
            data=df,
            palette=sns.color_palette("husl", 12),
            sort=False,
            err_style='bars',
            errorbar='sd',
            err_kws={'elinewidth':0.75},
            dashes=False,
            ax=ax,
            )

        sns.regplot(
            data=df, 
            x=subplot_titles[i], 
            y="Cost Difference",
            scatter=False,
            line_kws={"lw":0.75, "ls":"--", "color":"grey"},
            ax=ax,
            )

        model = sm.OLS(df["Cost Difference"], sm.add_constant(df[subplot_titles[i]])).fit()

        # print(model.params)
        intercept, slope = round(model.params[0],2), round(model.params[1],2)
        r_squared, p_value = round(model.rsquared, 2), round(model.pvalues.loc[subplot_titles[i]], 2) 
        ax.text(
            0.75, 
            0.85, 
            f'y = {slope} x + {intercept}\n$r^{2}$={r_squared}, p-value={p_value}',
            bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', alpha=0.5, linewidth=0.5), 
            horizontalalignment='center', 
            verticalalignment='center', 
            transform=ax.transAxes, 
            fontsize=5)


        
        ax.set(xlabel=None, ylabel=None)
        ax.set_title(subplot_titles[i], fontsize=6)

        ax.tick_params(axis='x', labelsize=5, labelrotation=45)
        ax.tick_params(axis='y', labelsize=5)

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
        bbox_to_anchor=(0.5, - 0.17), 
        ncol=3, 
        fontsize=6)

    # Format suplabels and title
    fig.supylabel('Cost Difference (ROCCH - Actual)', x=0.05, fontsize=7)
    fig.supxlabel('', fontsize=1)
    fig.suptitle(f'Cost Difference\n{col_names[0]}={plot_metadata[col_names[0]]}, {col_names[1]}={plot_metadata[col_names[1]]}\n {col_names[2]}={plot_metadata[col_names[2]]}, {col_names[3]}={plot_metadata[col_names[3]]}, {col_names[4]}={plot_metadata[col_names[4]]}, {col_names[5]}={plot_metadata[col_names[5]]}\n{col_names[6]}={plot_metadata[col_names[6]]}, {col_names[7]}={plot_metadata[col_names[7]]}', y =0.95, fontsize=7)
    fig.tight_layout()


    fig.savefig(f'{output_plot_dscomp_r_dir}ds_cost_{identifier}.png', bbox_inches='tight', dpi=300)
    
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    plt.close()
        







def plot_dsdist_wasserstein_pointplot(df, descriptions_df, plot_metadata, identifier):

    subplot_titles = [1.0, 1.25, 0.75, 0.5]

    df.reset_index(drop=True)

    col_names = list(plot_metadata.keys())

    nrow = 2
    ncol = 2


    sns.set_style('whitegrid', {"grid.color": "silver", "grid.linestyle": "dotted"})

    matplotlib.rcParams['legend.handlelength'] = 0
    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['legend.borderpad'] = 1.0
    matplotlib.rcParams['legend.handletextpad'] = 1.0
    matplotlib.rcParams['legend.borderaxespad'] = 0.8   

    fig, _ = plt.subplots(nrow, ncol, sharey=True, figsize = (5,4))

        
    handles = None

    for i, ax in enumerate(fig.axes):
        
        current_df = df[df['Test to Train Class Distr. Ratio']==subplot_titles[i]].copy()

        current_df['Cost Difference'] =  current_df['Avg. Optimal Point Cost (ROCCH Method)'] - current_df['Avg. Optimal Point Cost (Actual)']
        current_df.reset_index(drop=True)

        
        current_df['Double Avg. Wasserstein Dist.'] = 0
        # current_df['Avg. Cost Difference'] = 0
        dist_order = []
        # cost_order = []
        for ds_key in list(descriptions_df['Data Set']):

            dist_order.append(current_df.loc[df['Data Set']==ds_key, 'Avg. Wasserstein Dist.'].mean())
            current_df.loc[df['Data Set']==ds_key, 'Double Avg. Wasserstein Dist.'] = dist_order[-1]

            # cost_order.append(current_df.loc[df['Data Set']==ds_key, 'Cost Difference'].mean())
            # current_df.loc[df['Data Set']==ds_key, 'Avg. Cost Difference'] = cost_order[-1]


        sns.lineplot(
            y='Cost Difference', 
            x='Double Avg. Wasserstein Dist.', 
            hue='Double Avg. Wasserstein Dist.',
            hue_order=dist_order,
            style='Double Avg. Wasserstein Dist.',
            style_order=dist_order,
            markers=ds_markers,
            data=current_df,
            palette=sns.color_palette("husl", 12), 
            sort=False,
            err_style='bars',
            errorbar='sd',
            err_kws={'elinewidth':0.75},
            dashes=False,
            ax=ax,
            )


        sns.regplot(
            data=current_df, 
            x="Double Avg. Wasserstein Dist.", 
            y="Cost Difference",
            scatter=False,
            line_kws={"lw":0.75, "ls":"--", "color":"grey"},
            ax=ax,
            )

        model = sm.OLS(current_df["Cost Difference"], sm.add_constant(current_df['Double Avg. Wasserstein Dist.'])).fit()

        # print(model.params)
        intercept, slope = round(model.params[0],2), round(model.params[1],2)
        r_squared, p_value = round(model.rsquared, 2), round(model.pvalues.loc['Double Avg. Wasserstein Dist.'], 2) 
        ax.text(
            0.75, 
            0.85, 
            f'y = {slope} x + {intercept}\n$r^{2}$={r_squared}, p-value={p_value}',
            bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', alpha=0.5, linewidth=0.5), 
            horizontalalignment='center', 
            verticalalignment='center', 
            transform=ax.transAxes, 
            fontsize=5)

        
        ax.set(xlabel=None, ylabel=None)
        ax.set_title(f'Test to Train Cls. Distr. = {subplot_titles[i]}', fontsize=6)


        # ax.set_xticklabels([])
        # ax.set_xticks([]) 
        ax.tick_params(axis='y', labelsize=4)
        ax.tick_params(axis='x', labelsize=4)

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
        bbox_to_anchor=(0.5, - 0.19), 
        ncol=3, 
        fontsize=6)

    # Format suplabels and title
    fig.supylabel('Cost Difference (ROCCH - Actual)', x=0.05, fontsize=7)
    fig.supxlabel('Double Avg. Wasserstein Distance between Training and Testing Features', y=0.05,fontsize=7)
    fig.suptitle(f'Effect of Covariate Shift on Efficacy of ROCCH Method\n{col_names[0]}={plot_metadata[col_names[0]]}, {col_names[1]}={plot_metadata[col_names[1]]}\n{col_names[2]}={plot_metadata[col_names[2]]}, {col_names[3]}={plot_metadata[col_names[3]]}, {col_names[4]}={plot_metadata[col_names[4]]}, {col_names[5]}={plot_metadata[col_names[5]]}, {col_names[6]}={plot_metadata[col_names[6]]}', y =0.95, fontsize=7)
    fig.tight_layout()


    fig.savefig(f'{output_plot_dsdist_w_dir}ds_dist_w_{identifier}.png', bbox_inches='tight', dpi=300)

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    plt.close()





def plot_dsdist_energy_pointplot(df, descriptions_df, plot_metadata, identifier):

    subplot_titles = [1.0, 1.25, 0.75, 0.5]

    df.reset_index(drop=True)

    col_names = list(plot_metadata.keys())

    nrow = 2
    ncol = 2


    sns.set_style('whitegrid', {"grid.color": "silver", "grid.linestyle": "dotted"})

    matplotlib.rcParams['legend.handlelength'] = 0
    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['legend.borderpad'] = 1.0
    matplotlib.rcParams['legend.handletextpad'] = 1.0
    matplotlib.rcParams['legend.borderaxespad'] = 0.8   

    fig, _ = plt.subplots(nrow, ncol, sharey=True, figsize = (5,4))

        
    handles = None

    for i, ax in enumerate(fig.axes):
        
        current_df = df[df['Test to Train Class Distr. Ratio']==subplot_titles[i]].copy()

        current_df['Cost Difference'] =  current_df['Avg. Optimal Point Cost (ROCCH Method)'] - current_df['Avg. Optimal Point Cost (Actual)']
        current_df.reset_index(drop=True)

        
        current_df['Double Avg. Energy Dist.'] = 0
        # current_df['Avg. Cost Difference'] = 0
        dist_order = []
        # cost_order = []
        for ds_key in list(descriptions_df['Data Set']):

            dist_order.append(current_df.loc[df['Data Set']==ds_key, 'Avg. Energy Dist.'].mean())
            current_df.loc[df['Data Set']==ds_key, 'Double Avg. Energy Dist.'] = dist_order[-1]

            # cost_order.append(current_df.loc[df['Data Set']==ds_key, 'Cost Difference'].mean())
            # current_df.loc[df['Data Set']==ds_key, 'Avg. Cost Difference'] = cost_order[-1]


        sns.lineplot(
            y='Cost Difference', 
            x='Double Avg. Energy Dist.', 
            hue='Double Avg. Energy Dist.',
            hue_order=dist_order,
            style='Double Avg. Energy Dist.',
            style_order=dist_order,
            markers=ds_markers,
            data=current_df,
            palette=sns.color_palette("husl", 12), 
            sort=False,
            err_style='bars',
            errorbar='sd',
            err_kws={'elinewidth':0.75},
            dashes=False,
            ax=ax,
            )


        sns.regplot(
            data=current_df, 
            x="Double Avg. Energy Dist.", 
            y="Cost Difference",
            scatter=False,
            line_kws={"lw":0.75, "ls":"--", "color":"grey"},
            ax=ax,
            )

        model = sm.OLS(current_df["Cost Difference"], sm.add_constant(current_df['Double Avg. Energy Dist.'])).fit()

        # print(model.params)
        intercept, slope = round(model.params[0],2), round(model.params[1],2)
        r_squared, p_value = round(model.rsquared, 2), round(model.pvalues.loc['Double Avg. Energy Dist.'], 2) 
        ax.text(
            0.75, 
            0.85, 
            f'y = {slope} x + {intercept}\n$r^{2}$={r_squared}, p-value={p_value}',
            bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', alpha=0.5, linewidth=0.5), 
            horizontalalignment='center', 
            verticalalignment='center', 
            transform=ax.transAxes, 
            fontsize=5)

        
        ax.set(xlabel=None, ylabel=None)
        ax.set_title(f'Test to Train Cls. Distr. = {subplot_titles[i]}', fontsize=6)


        # ax.set_xticklabels([])
        # ax.set_xticks([]) 
        ax.tick_params(axis='y', labelsize=4)
        ax.tick_params(axis='x', labelsize=4)

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
        bbox_to_anchor=(0.5, - 0.19), 
        ncol=3, 
        fontsize=6)

    # Format suplabels and title
    fig.supylabel('Cost Difference (ROCCH - Actual)', x=0.05, fontsize=7)
    fig.supxlabel('Double Avg. Energy Distance between Training and Testing Features', y=0.05,fontsize=7)
    fig.suptitle(f'Effect of Covariate Shift on Efficacy of ROCCH Method\n{col_names[0]}={plot_metadata[col_names[0]]}, {col_names[1]}={plot_metadata[col_names[1]]}\n{col_names[2]}={plot_metadata[col_names[2]]}, {col_names[3]}={plot_metadata[col_names[3]]}, {col_names[4]}={plot_metadata[col_names[4]]}, {col_names[5]}={plot_metadata[col_names[5]]}, {col_names[6]}={plot_metadata[col_names[6]]}', y =0.95, fontsize=7)
    fig.tight_layout()


    fig.savefig(f'{output_plot_dsdist_e_dir}ds_dist_e_{identifier}.png', bbox_inches='tight', dpi=300)

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    plt.close()


def plot_dsdist_mmd_pointplot(df, descriptions_df, plot_metadata, identifier):

    subplot_titles = [1.0, 1.25, 0.75, 0.5]

    df.reset_index(drop=True)

    col_names = list(plot_metadata.keys())

    nrow = 2
    ncol = 2


    sns.set_style('whitegrid',  {"grid.color": "silver", "grid.linestyle": "dotted"})

    matplotlib.rcParams['legend.handlelength'] = 0
    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['legend.borderpad'] = 1.0
    matplotlib.rcParams['legend.handletextpad'] = 1.0
    matplotlib.rcParams['legend.borderaxespad'] = 0.8   

    fig, _ = plt.subplots(nrow, ncol, sharey=True, figsize = (5,4))

        
    handles = None

    for i, ax in enumerate(fig.axes):
        
        current_df = df[df['Test to Train Class Distr. Ratio']==subplot_titles[i]].copy()

        current_df['Cost Difference'] =  current_df['Avg. Optimal Point Cost (ROCCH Method)'] - current_df['Avg. Optimal Point Cost (Actual)']
        current_df.reset_index(drop=True)

        
        current_df['Double Avg. MMD'] = 0
        # current_df['Avg. Cost Difference'] = 0
        dist_order = []
        # cost_order = []
        for ds_key in list(descriptions_df['Data Set']):

            dist_order.append(current_df.loc[df['Data Set']==ds_key, 'Avg. MMD'].mean())
            current_df.loc[df['Data Set']==ds_key, 'Double Avg. MMD'] = dist_order[-1]

            # cost_order.append(current_df.loc[df['Data Set']==ds_key, 'Cost Difference'].mean())
            # current_df.loc[df['Data Set']==ds_key, 'Avg. Cost Difference'] = cost_order[-1]


        sns.lineplot(
            y='Cost Difference', 
            x='Double Avg. MMD', 
            hue='Double Avg. MMD',
            hue_order=dist_order,
            style='Double Avg. MMD',
            style_order=dist_order,
            markers=ds_markers,
            data=current_df,
            palette=sns.color_palette("husl", 12), 
            sort=False,
            err_style='bars',
            errorbar='sd',
            err_kws={'elinewidth':0.75},
            dashes=False,
            ax=ax,
            )


        sns.regplot(
            data=current_df, 
            x="Double Avg. MMD", 
            y="Cost Difference",
            scatter=False,
            line_kws={"lw":0.75, "ls":"--", "color":"grey"},
            ax=ax,
            )

        model = sm.OLS(current_df["Cost Difference"], sm.add_constant(current_df['Double Avg. MMD'])).fit()

        # print(model.params)
        intercept, slope = round(model.params[0],2), round(model.params[1],2)
        r_squared, p_value = round(model.rsquared, 2), round(model.pvalues.loc['Double Avg. MMD'], 2) 
        
        ax.text(
            0.75, 
            0.85, 
            f'y = {slope} x + {intercept}\n$r^{2}$={r_squared}, p-value={p_value}',
            bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', alpha=0.5, linewidth=0.5), 
            horizontalalignment='center', 
            verticalalignment='center', 
            transform=ax.transAxes, 
            fontsize=5)

        
        ax.set(xlabel=None, ylabel=None)
        ax.set_title(f'Test to Train Cls. Distr. = {subplot_titles[i]}', fontsize=6)


        # ax.set_xticklabels([])
        # ax.set_xticks([]) 
        ax.tick_params(axis='y', labelsize=4)
        ax.tick_params(axis='x', labelsize=4)

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
        bbox_to_anchor=(0.5, - 0.19), 
        ncol=3, 
        fontsize=6)

    # Format suplabels and title
    fig.supylabel('Cost Difference (ROCCH - Actual)', x=0.05, fontsize=7)
    fig.supxlabel('Double Avg. Maximum Mean Discrepancy', y=0.05,fontsize=7)
    fig.suptitle(f'Effect of Covariate Shift on Efficacy of ROCCH Method\n{col_names[0]}={plot_metadata[col_names[0]]}, {col_names[1]}={plot_metadata[col_names[1]]}\n{col_names[2]}={plot_metadata[col_names[2]]}, {col_names[3]}={plot_metadata[col_names[3]]}, {col_names[4]}={plot_metadata[col_names[4]]}, {col_names[5]}={plot_metadata[col_names[5]]}, {col_names[6]}={plot_metadata[col_names[6]]}', y =0.95, fontsize=7)
    fig.tight_layout()


    fig.savefig(f'{output_plot_dsdist_mmd_dir}ds_dist_mmd_{identifier}.png', bbox_inches='tight', dpi=300)

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    plt.close()


def plot_dsdist_auc_pointplot(df, descriptions_df, plot_metadata, identifier):

    subplot_titles = [1.0, 1.25, 0.75, 0.5]

    df.reset_index(drop=True)

    col_names = list(plot_metadata.keys())

    nrow = 2
    ncol = 2


    sns.set_style('whitegrid', {"grid.color": "silver", "grid.linestyle": "dotted"})

    matplotlib.rcParams['legend.handlelength'] = 0
    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['legend.borderpad'] = 1.0
    matplotlib.rcParams['legend.handletextpad'] = 1.0
    matplotlib.rcParams['legend.borderaxespad'] = 0.8   

    fig, _ = plt.subplots(nrow, ncol, sharey=True, figsize = (5,4))

        
    handles = None

    for i, ax in enumerate(fig.axes):
        
        current_df = df[df['Test to Train Class Distr. Ratio']==subplot_titles[i]].copy()

        current_df['Cost Difference'] =  current_df['Avg. Optimal Point Cost (ROCCH Method)'] - current_df['Avg. Optimal Point Cost (Actual)']
        current_df.reset_index(drop=True)

        
        current_df['Double Avg. AUC'] = 0
        # current_df['Avg. Cost Difference'] = 0
        dist_order = []
        # cost_order = []
        for ds_key in list(descriptions_df['Data Set']):

            dist_order.append(current_df.loc[df['Data Set']==ds_key, 'Avg. AUC (COVSHIFT)'].mean())
            current_df.loc[df['Data Set']==ds_key, 'Double Avg. AUC'] = dist_order[-1]

            # cost_order.append(current_df.loc[df['Data Set']==ds_key, 'Cost Difference'].mean())
            # current_df.loc[df['Data Set']==ds_key, 'Avg. Cost Difference'] = cost_order[-1]


        sns.lineplot(
            y='Cost Difference', 
            x='Double Avg. AUC', 
            hue='Double Avg. AUC',
            hue_order=dist_order,
            style='Double Avg. AUC',
            style_order=dist_order,
            markers=ds_markers,
            data=current_df,
            palette=sns.color_palette("husl", 12), 
            sort=False,
            err_style='bars',
            errorbar='sd',
            err_kws={'elinewidth':0.75},
            dashes=False,
            ax=ax,
            )


        sns.regplot(
            data=current_df, 
            x="Double Avg. AUC", 
            y="Cost Difference",
            scatter=False,
            line_kws={"lw":0.75, "ls":"--", "color":"grey"},
            ax=ax,
            )

        model = sm.OLS(current_df["Cost Difference"], sm.add_constant(current_df['Double Avg. AUC'])).fit()

        # print(model.params)
        intercept, slope = round(model.params[0],2), round(model.params[1],2)
        r_squared, p_value = round(model.rsquared, 2), round(model.pvalues.loc['Double Avg. AUC'], 2) 
        ax.text(
            0.75, 
            0.85, 
            f'y = {slope} x + {intercept}\n$r^{2}$={r_squared}, p-value={p_value}',
            bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', alpha=0.5, linewidth=0.5), 
            horizontalalignment='center', 
            verticalalignment='center', 
            transform=ax.transAxes, 
            fontsize=5)

        
        ax.set(xlabel=None, ylabel=None)
        ax.set_title(f'Test to Train Cls. Distr. = {subplot_titles[i]}', fontsize=6)


        # ax.set_xticklabels([])
        # ax.set_xticks([]) 
        ax.tick_params(axis='y', labelsize=4)
        ax.tick_params(axis='x', labelsize=4)

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
        bbox_to_anchor=(0.5, - 0.19), 
        ncol=3, 
        fontsize=6)

    # Format suplabels and title
    fig.supylabel('Cost Difference (ROCCH - Actual)', x=0.05, fontsize=7)
    fig.supxlabel('Double Avg. AUC on Differentiating Training and Testing ', y=0.05,fontsize=7)
    fig.suptitle(f'Effect of Covariate Shift on Efficacy of ROCCH Method\n{col_names[0]}={plot_metadata[col_names[0]]}, {col_names[1]}={plot_metadata[col_names[1]]}\n{col_names[2]}={plot_metadata[col_names[2]]}, {col_names[3]}={plot_metadata[col_names[3]]}, {col_names[4]}={plot_metadata[col_names[4]]}, {col_names[5]}={plot_metadata[col_names[5]]}, {col_names[6]}={plot_metadata[col_names[6]]}', y =0.95, fontsize=7)
    fig.tight_layout()


    fig.savefig(f'{output_plot_dsdist_a_dir}ds_dist_a_{identifier}.png', bbox_inches='tight', dpi=300)

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    plt.close()


def plot_dsdist_mcc_pointplot(df, descriptions_df, plot_metadata, identifier):

    subplot_titles = [1.0, 1.25, 0.75, 0.5]

    df.reset_index(drop=True)

    col_names = list(plot_metadata.keys())

    nrow = 2
    ncol = 2


    sns.set_style('whitegrid', {"grid.color": "silver", "grid.linestyle": "dotted"})

    matplotlib.rcParams['legend.handlelength'] = 0
    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['legend.borderpad'] = 1.0
    matplotlib.rcParams['legend.handletextpad'] = 1.0
    matplotlib.rcParams['legend.borderaxespad'] = 0.8   

    fig, _ = plt.subplots(nrow, ncol, sharey=True, figsize = (5,4))

        
    handles = None

    for i, ax in enumerate(fig.axes):
        
        current_df = df[df['Test to Train Class Distr. Ratio']==subplot_titles[i]].copy()

        current_df['Cost Difference'] =  current_df['Avg. Optimal Point Cost (ROCCH Method)'] - current_df['Avg. Optimal Point Cost (Actual)']
        current_df.reset_index(drop=True)

        
        current_df['Double Avg. Phi'] = 0
        # current_df['Avg. Cost Difference'] = 0
        dist_order = []
        # cost_order = []
        for ds_key in list(descriptions_df['Data Set']):

            dist_order.append(current_df.loc[df['Data Set']==ds_key, 'Avg. Phi'].mean())
            current_df.loc[df['Data Set']==ds_key, 'Double Avg. Phi'] = dist_order[-1]

            # cost_order.append(current_df.loc[df['Data Set']==ds_key, 'Cost Difference'].mean())
            # current_df.loc[df['Data Set']==ds_key, 'Avg. Cost Difference'] = cost_order[-1]


        sns.lineplot(
            y='Cost Difference', 
            x='Double Avg. Phi', 
            hue='Double Avg. Phi',
            hue_order=dist_order,
            style='Double Avg. Phi',
            style_order=dist_order,
            markers=ds_markers,
            data=current_df,
            palette=sns.color_palette("husl", 12), 
            sort=False,
            err_style='bars',
            errorbar='sd',
            err_kws={'elinewidth':0.75},
            dashes=False,
            ax=ax,
            )


        sns.regplot(
            data=current_df, 
            x="Double Avg. Phi", 
            y="Cost Difference",
            scatter=False,
            line_kws={"lw":0.75, "ls":"--", "color":"grey"},
            ax=ax,
            )

        model = sm.OLS(current_df["Cost Difference"], sm.add_constant(current_df['Double Avg. Phi'])).fit()

        # print(model.params)
        intercept, slope = round(model.params[0],2), round(model.params[1],2)
        r_squared, p_value = round(model.rsquared, 2), round(model.pvalues.loc['Double Avg. Phi'], 2) 
        ax.text(
            0.75, 
            0.85, 
            f'y = {slope} x + {intercept}\n$r^{2}$={r_squared}, p-value={p_value}',
            bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', alpha=0.5, linewidth=0.5), 
            horizontalalignment='center', 
            verticalalignment='center', 
            transform=ax.transAxes, 
            fontsize=5)

        
        ax.set(xlabel=None, ylabel=None)
        ax.set_title(f'Test to Train Cls. Distr. = {subplot_titles[i]}', fontsize=6)


        # ax.set_xticklabels([])
        # ax.set_xticks([]) 
        ax.tick_params(axis='y', labelsize=4)
        ax.tick_params(axis='x', labelsize=4)

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
        bbox_to_anchor=(0.5, - 0.19), 
        ncol=3, 
        fontsize=6)

    # Format suplabels and title
    fig.supylabel('Cost Difference (ROCCH - Actual)', x=0.05, fontsize=7)
    fig.supxlabel('Double Avg. Matthews Correl. Coeff. on Differentiating Training and Testing ', y=0.05,fontsize=7)
    fig.suptitle(f'Effect of Covariate Shift on Efficacy of ROCCH Method\n{col_names[0]}={plot_metadata[col_names[0]]}, {col_names[1]}={plot_metadata[col_names[1]]}\n{col_names[2]}={plot_metadata[col_names[2]]}, {col_names[3]}={plot_metadata[col_names[3]]}, {col_names[4]}={plot_metadata[col_names[4]]}, {col_names[5]}={plot_metadata[col_names[5]]}, {col_names[6]}={plot_metadata[col_names[6]]}', y =0.95, fontsize=7)
    fig.tight_layout()


    fig.savefig(f'{output_plot_dsdist_mcc_dir}ds_dist_mcc_{identifier}.png', bbox_inches='tight', dpi=300)

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    plt.close()




def plot_dsdist_cramervonmises_pointplot(df, descriptions_df, plot_metadata, identifier):

    subplot_titles = [1.0, 1.25, 0.75, 0.5]

    df.reset_index(drop=True)

    col_names = list(plot_metadata.keys())

    nrow = 2
    ncol = 2


    sns.set_style('whitegrid', {"grid.color": "silver", "grid.linestyle": "dotted"})

    matplotlib.rcParams['legend.handlelength'] = 0
    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['legend.borderpad'] = 1.0
    matplotlib.rcParams['legend.handletextpad'] = 1.0
    matplotlib.rcParams['legend.borderaxespad'] = 0.8   

    fig, _ = plt.subplots(nrow, ncol, sharey=True, figsize = (5,4))

        
    handles = None

    for i, ax in enumerate(fig.axes):
        
        current_df = df[df['Test to Train Class Distr. Ratio']==subplot_titles[i]].copy()

        current_df['Cost Difference'] =  current_df['Avg. Optimal Point Cost (ROCCH Method)'] - current_df['Avg. Optimal Point Cost (Actual)']
        current_df.reset_index(drop=True)

        
        current_df['Double Avg. Cramer-von Mises Criterion'] = 0
        # current_df['Avg. Cost Difference'] = 0
        dist_order = []
        # cost_order = []
        for ds_key in list(descriptions_df['Data Set']):

            dist_order.append(current_df.loc[df['Data Set']==ds_key, 'Avg. Cramer-von Mises Criterion'].mean())
            current_df.loc[df['Data Set']==ds_key, 'Double Avg. Cramer-von Mises Criterion'] = dist_order[-1]

            # cost_order.append(current_df.loc[df['Data Set']==ds_key, 'Cost Difference'].mean())
            # current_df.loc[df['Data Set']==ds_key, 'Avg. Cost Difference'] = cost_order[-1]


        sns.lineplot(
            y='Cost Difference', 
            x='Double Avg. Cramer-von Mises Criterion', 
            hue='Double Avg. Cramer-von Mises Criterion',
            hue_order=dist_order,
            style='Double Avg. Cramer-von Mises Criterion',
            style_order=dist_order,
            markers=ds_markers,
            data=current_df,
            palette=sns.color_palette("husl", 12), 
            sort=False,
            err_style='bars',
            errorbar='sd',
            err_kws={'elinewidth':0.75},
            dashes=False,
            ax=ax,
            )


        sns.regplot(
            data=current_df, 
            x="Double Avg. Cramer-von Mises Criterion", 
            y="Cost Difference",
            scatter=False,
            line_kws={"lw":0.75, "ls":"--", "color":"grey"},
            ax=ax,
            )

        model = sm.OLS(current_df["Cost Difference"], sm.add_constant(current_df['Double Avg. Cramer-von Mises Criterion'])).fit()

        # print(model.params)
        intercept, slope = round(model.params[0],2), round(model.params[1],2)
        r_squared, p_value = round(model.rsquared, 2), round(model.pvalues.loc['Double Avg. Cramer-von Mises Criterion'], 2) 
        ax.text(
            0.75, 
            0.85, 
            f'y = {slope} x + {intercept}\n$r^{2}$={r_squared}, p-value={p_value}',
            bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', alpha=0.5, linewidth=0.5), 
            horizontalalignment='center', 
            verticalalignment='center', 
            transform=ax.transAxes, 
            fontsize=5)

        
        ax.set(xlabel=None, ylabel=None)
        ax.set_title(f'Test to Train Cls. Distr. = {subplot_titles[i]}', fontsize=6)


        # ax.set_xticklabels([])
        # ax.set_xticks([]) 
        ax.tick_params(axis='y', labelsize=4)
        ax.tick_params(axis='x', labelsize=4)

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
        bbox_to_anchor=(0.5, - 0.19), 
        ncol=3, 
        fontsize=6)

    # Format suplabels and title
    fig.supylabel('Cost Difference (ROCCH - Actual)', x=0.05, fontsize=7)
    fig.supxlabel('Double Avg. Cramer-Von Mises Distance between Training and Testing Features', y=0.05,fontsize=7)
    fig.suptitle(f'Effect of Covariate Shift on Efficacy of ROCCH Method\n{col_names[0]}={plot_metadata[col_names[0]]}, {col_names[1]}={plot_metadata[col_names[1]]}\n{col_names[2]}={plot_metadata[col_names[2]]}, {col_names[3]}={plot_metadata[col_names[3]]}, {col_names[4]}={plot_metadata[col_names[4]]}, {col_names[5]}={plot_metadata[col_names[5]]}, {col_names[6]}={plot_metadata[col_names[6]]}', y =0.95, fontsize=7)
    fig.tight_layout()


    fig.savefig(f'{output_plot_dsdist_c_dir}ds_dist_c_{identifier}.png', bbox_inches='tight', dpi=300)

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    plt.close()



def plot_ds_distances_pointplot(df, descriptions_df, plot_metadata, identifier):

    plot_dsdist_wasserstein_pointplot(df, descriptions_df, plot_metadata, identifier)
    plot_dsdist_energy_pointplot(df, descriptions_df, plot_metadata, identifier)
    plot_dsdist_mmd_pointplot(df, descriptions_df, plot_metadata, identifier)
    plot_dsdist_auc_pointplot(df, descriptions_df, plot_metadata, identifier)
    plot_dsdist_mcc_pointplot(df, descriptions_df, plot_metadata, identifier)
    plot_dsdist_cramervonmises_pointplot(df, descriptions_df, plot_metadata, identifier)
    
        


def plot_varying_cost_boxplots (df, plot_metadata, ds_keys):
    
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
                'Avg. Optimal Point Cost (Actual)',
                'Avg. Optimal Point Cost (ROCCH Method)',
                'Avg. Optimal Point Cost (Accuracy-Max)', 
                'Avg. Optimal Point Cost (F1-score-Max)',
                ]
        )

        # current_df = current_df.round(4)
        # current_df.to_csv(f'{output_table_dir}current_{plot_metadata["identifier"]}_{ds_keys[i]}.csv')

        # Plot boxplots
        sns.boxplot(y='value', x=plot_metadata['varying'], hue='variable', data=current_df,  orient='v', linewidth=0.5, fliersize=1, ax=ax)
        
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
        ['Actual', 'ROCCH Method', 'Accuracy Max', 'F1-score Max'],
        title='Cost Incurred by Optimal Points',
        title_fontsize=8,
        loc='lower center', 
        bbox_to_anchor=(0.5, - 0.075), 
        ncol=4, 
        fontsize=7)

    # Format suplabels and title
    fig.supylabel('Average Normalized Cost', x=0.02, fontsize=8)
    fig.supxlabel(plot_metadata['varying'], y=0.04, fontsize=8)
    fig.suptitle(f'Average Normalized Cost while varying "{plot_metadata["varying"]}"\n{col_names[0]}={plot_metadata[col_names[0]]}, {col_names[1]}={plot_metadata[col_names[1]]}\n{col_names[2]}={plot_metadata[col_names[2]]}, {col_names[3]}={plot_metadata[col_names[3]]}, {col_names[4]}={plot_metadata[col_names[4]]}, {col_names[5]}={plot_metadata[col_names[5]]}', fontsize=9)
    fig.tight_layout()

    fig.savefig(f'{output_plot_varying_dir}cost_{plot_metadata["identifier"]}.png', bbox_inches='tight', dpi=300)
    plt.close()


def plot_varying_dist_boxplots(df, plot_metadata, ds_keys):

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
                'Distance between ROCCHM and Actual',
                'Distance between Accuracy-Max and Actual',
                'Distance between F1-score-Max and Actual', 
                ]
        )

        # current_df = current_df.round(4)
        # current_df.to_csv(f'{output_table_dir}current_{plot_metadata["identifier"]}_{ds_keys[i]}.csv')

        # Plot boxplots
        palette = {"Distance between ROCCHM and Actual": "orange", "Distance between Accuracy-Max and Actual": "green",  "Distance between F1-score-Max and Actual": "red",}
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
        ['ROCCH Method', 'Accuracy Max', 'F1-score Max'],
        title='Distance to Actual Optimal Point',
        title_fontsize=8,
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.075), 
        ncol=4, 
        fontsize=7)

    # Format suplabels and title
    fig.supylabel('Distance', x=0.02, fontsize=8)
    fig.supxlabel(plot_metadata['varying'], y=0.04, fontsize=8)
    fig.suptitle(f'Distance to Actual Optimal Point while varying "{plot_metadata["varying"]}"\n{col_names[0]}={plot_metadata[col_names[0]]}, {col_names[1]}={plot_metadata[col_names[1]]}\n{col_names[2]}={plot_metadata[col_names[2]]}, {col_names[3]}={plot_metadata[col_names[3]]}, {col_names[4]}={plot_metadata[col_names[4]]}, {col_names[5]}={plot_metadata[col_names[5]]}', fontsize=9)
    fig.tight_layout()

    fig.savefig(f'{output_plot_varying_dir}dist_{plot_metadata["identifier"]}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
def plot_results():
    '''
    
    '''
    print("\nPlotting results:")
    
    if not os.path.exists(output_plot_main_dir):
        os.makedirs(output_plot_main_dir)

    if not os.path.exists(output_plot_varying_dir):
        os.makedirs(output_plot_varying_dir)

    if not os.path.exists(output_plot_dscomp_dir):
        os.makedirs(output_plot_dscomp_dir)

    if not os.path.exists(output_plot_dscomp_r_dir):
        os.makedirs(output_plot_dscomp_r_dir)

    if not os.path.exists(output_plot_dsdist_dir):
        os.makedirs(output_plot_dsdist_dir)

    if not os.path.exists(output_plot_dsdist_w_dir):
        os.makedirs(output_plot_dsdist_w_dir)

    if not os.path.exists(output_plot_dsdist_e_dir):
        os.makedirs(output_plot_dsdist_e_dir)

    if not os.path.exists(output_plot_dsdist_mmd_dir):
        os.makedirs(output_plot_dsdist_mmd_dir)

    if not os.path.exists(output_plot_dsdist_a_dir):
        os.makedirs(output_plot_dsdist_a_dir)

    if not os.path.exists(output_plot_dsdist_mcc_dir):
        os.makedirs(output_plot_dsdist_mcc_dir)

    if not os.path.exists(output_plot_dsdist_c_dir):
        os.makedirs(output_plot_dsdist_c_dir)



    dataset_descriptions = pd.read_csv(f'{output_table_dir}dataset_descriptons.csv') # Saved during preprocessing
    ds_keys = list(dataset_descriptions['Data Set'])

    performance_df_summarized = pd.read_csv(f'{output_table_dir}performance_summarized.csv')


    # Compare dataset-specific metrics
    dataset_descriptions = dataset_descriptions.rename(
        columns={
            'Instances' : 'Data Set Size',
            'Class Balance' : 'Class Distribution',
        }
    )
    

    for k in ds_plots_metadata:
        col_names = list(ds_plots_metadata[k].keys())

        current_slice_idx = True
        for j in col_names:
            current_slice_idx &=  (performance_df_summarized[j]==ds_plots_metadata[k][j])

        current_df = performance_df_summarized[current_slice_idx].copy()


        # print(ds_plots_metadata[k])
        # print(current_df['Avg. Wasserstein Dist.'].head(5))
        # plot_ds_cost_pointplots(current_df, dataset_descriptions.copy(), ds_plots_metadata[k], k)
        plot_ds_cost_cls_ratio_pointplots(current_df, dataset_descriptions.copy(), ds_plots_metadata[k], k)


    # Relationship between prior class probability shift and covariate shift

 
    for k in dsdist_plots_metadata:
        col_names = list(dsdist_plots_metadata[k].keys())

        current_slice_idx = True
        for j in col_names:
            current_slice_idx &=  (performance_df_summarized[j]==dsdist_plots_metadata[k][j])

        current_df = performance_df_summarized[current_slice_idx].copy()

        plot_ds_distances_pointplot(current_df, dataset_descriptions, dsdist_plots_metadata[k], k)



    # Effects of varying settings/configurations
    
    import itertools
    varying_plots_metadata_1 = dict(itertools.islice(varying_plots_metadata.items(), 10))

    for k in varying_plots_metadata_1:

        col_names = list(varying_plots_metadata[k].keys())[1:-1]
 
        current_slice_idx = True
        for j in col_names:
            current_slice_idx &=  (performance_df_summarized[j]==varying_plots_metadata[k][j])

        current_df = performance_df_summarized[current_slice_idx].copy()


        plot_varying_cost_boxplots(current_df, varying_plots_metadata[k], ds_keys)
        plot_varying_dist_boxplots(current_df, varying_plots_metadata[k], ds_keys)

    # # Save selected plots of interest to a separate directory
    # for plot_filename in selected_varying_plots:
    #     shutil.copyfile(f'{output_plot_varying_dir}{plot_filename}', f'{output_plot_main_dir}{plot_filename}')

    
    print("Plotting completed.")