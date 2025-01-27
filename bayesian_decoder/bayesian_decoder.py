#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 29/08/2023 09:32
@author: hheise

"""
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as scistats

from schema import hheise_decoder, hheise_grouping, common_mice, common_img, hheise_placecell, hheise_behav

useful_metrics = ['accuracy', 'accuracy_quad', 'mae', 'mae_quad', 'sensitivity_rz', 'specificity_rz']
#%% Plot accuracy for all mice in prestroke

mice = np.unique(hheise_grouping.BehaviorGrouping().fetch('mouse_id'))
mice = mice[~np.isin(mice, [89, 112, 121])]
mice = mice[~np.isin(mice, [112])]

dfs = []
for mouse in mice:

    # Get day of surgery
    surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' & f'mouse_id={mouse}').fetch(
        'surgery_date')[0].date()

    # Fetch model parameters for all available prestroke sessions
    # dfs.append(pd.DataFrame((hheise_decoder.BayesianDecoderWithinSession & f'mouse_id = {mouse}'
    #                          & f'day <= "{surgery_day}"' & 'bayesian_id = 1').fetch('mouse_id', 'day', *useful_metrics, as_dict=True)))

    dfs.append(pd.DataFrame((hheise_decoder.BayesianDecoderWithinSession * hheise_decoder.BayesianParameter & f'mouse_id = {mouse}'
                             & f'day <= "{surgery_day}"' & 'bayesian_id > 0' & 'bayesian_id < 10').fetch(as_dict=True)))

df = pd.concat(dfs)

plt.figure()
sns.barplot(df[df.bayesian_id == 1], x='mouse_id', y='accuracy')
sns.stripplot(df[df.bayesian_id == 1], x='mouse_id', y='accuracy', color='gray')


#%% Compare differences between neuron subsets
subsets = df.neuron_subset.unique()
subset_comps = list(itertools.combinations(df.neuron_subset.unique(), 2))
# attr = ['accuracy', 'mae', 'mae_quad_std', 'mae_quad', 'sensitivity_rz', 'specificity_rz', 'mcc', 'abs_error']
attr = useful_metrics

# Plot general difference in performance (export for prism)
df.pivot(index=['mouse_id', 'day'], columns='neuron_subset', values='specificity_rz')[['stability', 'si', 'place_cells', 'firing_rate']].to_clipboard(index=False)

df_filt = df[['mouse_id', 'day', 'neuron_subset', *useful_metrics]]
df_filt = df_filt.melt(id_vars=['mouse_id', 'neuron_subset'])
g = sns.FacetGrid(df_filt, col="variable", col_wrap=3, sharey=False)
g.map(sns.boxplot, "neuron_subset", 'value')

# Plot scatter plots
for sub1, sub2 in subset_comps:
    fig, axes = plt.subplots(nrows=2, ncols=4)
    fig.suptitle(f"{sub1} vs {sub2}")
    for i in range(len(attr)):
        ax = axes.flatten()[i]
        sns.scatterplot(x=np.array(df[df['neuron_subset'] == sub1][attr[i]]),
                        y=np.array(df[df['neuron_subset'] == sub2][attr[i]]),
                        ax=ax)
        ax.set_title(attr[i])

        lower = np.min([ax.get_ylim()[0], ax.get_xlim()[0]])
        upper = np.max([ax.get_ylim()[1], ax.get_xlim()[1]])
        ax.set_xlim((lower, upper))
        ax.set_ylim((lower, upper))
        ax.set_xlabel(sub1)
        ax.set_ylabel(sub2)
        diag = ax.axline((upper, upper), slope=1, c='grey', ls='--')
        # ax.fill_between(diag._x, diag._y, transform=ax.get_yaxis_transform())
        ax.set_aspect('equal')


# Plot absolute values
df_abs = df.copy()
df_abs['mae'] = -df_abs['mae']
df_abs['mae_quad'] = -df_abs['mae_quad']
df_long = pd.melt(df_abs, id_vars='neuron_subset', value_vars=[*useful_metrics], var_name='metric')
df_long['value_norm'] = df_long.groupby('metric').transform(lambda x: (x - x.mean()) / x.std())
plt.figure()
ax = sns.boxplot(data=df_long, x='metric', y='value_norm', hue='neuron_subset', showfliers=False)
ax.set_ylabel('Normalized absolute performance')
ax.axhline(0, c='grey', ls='--')

# Compute and plot difference
diff_df = []
for sub1, sub2 in subset_comps:
    for i in range(len(attr)-1):
        diff = (np.array(df[df['neuron_subset'] == sub2][attr[i]]) - np.array(df[df['neuron_subset'] == sub1][attr[i]])) /\
               np.array(df[df['neuron_subset'] == sub1][attr[i]]) * 100
        if 'mae' in attr[i]:
            diff = -diff
        diff_df.append(pd.DataFrame(dict(diff=diff, comp=f'{sub1} vs {sub2}', attr=attr[i])))
diff_df = pd.concat(diff_df)

plt.figure()
# ax = sns.barplot(data=diff_df, x='comp', y='diff', hue='attr')
ax = sns.boxplot(data=diff_df, x='comp', y='diff', hue='attr', showfliers=False)
ax.axhline(0, c='grey', ls='--')

# ax = sns.stripplot(data=diff_df, x='comp', y='diff', hue='attr', ax=ax)
ax.set_ylabel('Performance improvement [%]')

# Compute and plot difference in a matrix
mats = {}
for metric in useful_metrics:
    diff_mat = pd.DataFrame(columns=subsets, index=subsets)
    for sub1 in subsets:
        for sub2 in subsets:
            diff = (np.array(df[df['neuron_subset'] == sub2][metric]) - np.array(df[df['neuron_subset'] == sub1][metric])) /\
                   np.array(df[df['neuron_subset'] == sub1][metric]) * 100
            if metric in ['mae', 'mae_quad']:
                diff = -diff
            diff_mat.loc[sub2, sub1] = np.mean(diff)
    mats[metric] = diff_mat.astype(float)
fig, axes = plt.subplots(2, 3, sharex='all', sharey='all')
for i, ax in enumerate(axes.flatten()):
    if i < len(useful_metrics):
        sns.heatmap(mats[useful_metrics[i]], vmin=-mats[useful_metrics[i]].abs().max().max(),
                    vmax=mats[useful_metrics[i]].abs().max().max(), ax=ax, cbar=False, annot=True, cmap='vlag')
        ax.set_title(useful_metrics[i])
        ax.set_aspect(1)

fig.delaxes(ax)
fig.supxlabel('TO')
fig.supylabel('FROM')

#%% Test normal distribution for error metrics

out = (hheise_decoder.BayesianDecoderWithinSession & 'bayesian_id =1').fetch(format='frame')
numeric_only = out.select_dtypes(include=np.number)
test = numeric_only.apply(scistats.normaltest)

#%% Test if prestroke metrics are above chance level for mice (use stability sorting)

dfs = []
for mouse in mice:

    # Get day of surgery
    surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' & f'mouse_id={mouse}').fetch(
        'surgery_date')[0].date()

    # Fetch model parameters for all available prestroke sessions
    dfs.append(pd.DataFrame((hheise_decoder.BayesianDecoderLastPrestroke & f'mouse_id = {mouse}' & f'day <= "{surgery_day}"' & 'bayesian_id = 1').fetch(as_dict=True)))
df = pd.concat(dfs)

# Fetch chance levels
chance = hheise_decoder.BayesianDecoderErrorChanceLevels().fetch(format='frame').drop(columns=['calculation_procedure']).squeeze()
chance.name = 'chance_levels'
chance = chance[~chance.index.str.contains('std')]
chance = chance.drop(['abs_error', 'abs_error_quad', 'failed_prediction'])
chance = chance[useful_metrics]

# Plot barplots
df_long = pd.melt(df.reset_index(), value_vars=list(chance.index), id_vars=['mouse_id', 'index'], var_name='metric')
# g = sns.FacetGrid(df_long, col="metric", col_wrap=6, sharey=False, hue='mouse_id')
# g.map(sns.barplot, "mouse_id", 'value', order=df_long.mouse_id.unique())
g = sns.FacetGrid(df_long, col="metric", col_wrap=3, sharey=False)
g.map(sns.boxplot, "mouse_id", 'value')
for col, ax in zip(g.col_names, g.axes):
    ax.axhline(chance[col], c='grey', ls='--')

# Compute p-values for one-sample t-test -> is per-mouse prestroke avg above chance level?
# First tried wilcoxon, as samples might not be normally distributed, but Wilcoxon cant handle low sample sizes
t_tests = []
raw_data = []
for metric in chance.index:
    curr_metric = df_long[df_long['metric'] == metric]
    raw_data.append(curr_metric.pivot(index='index', columns='mouse_id', values='value'))
    for mouse_id, mouse_data in curr_metric.groupby('mouse_id'):
        res = scistats.ttest_1samp(mouse_data['value'], chance[metric])
        if res.pvalue <= 0.05:
            direction = 'greater' if (mouse_data['value'] - chance[metric]).mean() > 0 else 'less'
        else:
            direction = 'none'
        t_tests.append(pd.DataFrame([{'metric': metric, 'mouse_id': mouse_id, 'p': res.pvalue, 'direction': direction}]))
t_tests = pd.concat(t_tests)

# Test difference in SDs between mean and chance level
mouse_means = df.groupby('mouse_id')[chance.index].mean()     # Error metric means
mouse_sd = df.groupby('mouse_id')[chance.index].std()         # Error metric SDs
mouse_chance_dif = (mouse_means - chance).abs()                 # Difference between mean and chance level
chance_dif_sd = mouse_chance_dif / mouse_sd                     # Difference in SDs (sigmas)

# Plot error metrics separately
g = sns.FacetGrid(chance_dif_sd.reset_index().melt(id_vars='mouse_id'), col='variable', col_wrap=3, sharey=False)
g.map(sns.barplot, 'mouse_id', 'value')
g.set_ylabels(r"Mean - Chance [$\sigma$'s]")
for ax in g.axes.flatten():
    ax.axhline(2, c='tab:orange', ls='--')
    ax.axhline(3, c='tab:red', ls='--')
    ax.text(len(chance_dif_sd)-0.5, 2, r'$2 \sigma$', verticalalignment='top', c='tab:orange')
    ax.text(len(chance_dif_sd)-0.5, 3, r'$3 \sigma$', verticalalignment='bottom', c='tab:red')

# Average error metrics
plt.figure(figsize=(12,6))
ax = sns.barplot(chance_dif_sd.reset_index().melt(id_vars='mouse_id'), x='mouse_id', y='value')
sns.stripplot(chance_dif_sd.reset_index().melt(id_vars='mouse_id'), x='mouse_id', y='value', ax=ax, c='grey', alpha=0.7)
ax.set_ylabel(r"Mean - Chance [$\sigma$'s]")
ax.axhline(2, c='tab:orange', ls='--')
ax.axhline(3, c='tab:red', ls='--')
# ax.set_ylim(0, 50)
ax.text(len(chance_dif_sd) - 0.5, 2, r'$2 \sigma$', verticalalignment='top', c='tab:orange')
ax.text(len(chance_dif_sd) - 0.5, 3, r'$3 \sigma$', verticalalignment='bottom', c='tab:red')
plt.tight_layout()

mean_chance_dif = chance_dif_sd.mean(axis=1)

#%% Correlate metrics against each other
numeric_only_filt = numeric_only.iloc[:, (~numeric_only.columns.str.contains('shift')) & (~numeric_only.columns.str.contains('std'))]
numeric_only_filt['accuracy'] = 1 - numeric_only_filt['accuracy']
numeric_only_filt['accuracy_quad'] = 1 - numeric_only_filt['accuracy_quad']
numeric_only_filt['acc_max'] = 1 - numeric_only_filt['acc_max']
numeric_only_filt['acc_max_quad'] = 1 - numeric_only_filt['acc_max_quad']
corr = numeric_only_filt.corr()
plt.figure()
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=np.bool)), vmin=-1, vmax=1, annot=True,
            xticklabels=corr.columns, yticklabels=corr.index, cmap='vlag')

corr = numeric_only_filt[useful_metrics].corr()
plt.figure()
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=np.bool)), vmin=-1, vmax=1, annot=True,
            xticklabels=corr.columns, yticklabels=corr.index, cmap='vlag')


#%% Correlate VR performance with metrics

### Session-wise
data = pd.DataFrame((hheise_decoder.BayesianDecoderWithinSession * hheise_behav.VRPerformance
                     & 'bayesian_id=1' & 'day < "2022-09-09"').fetch('mouse_id', 'day', 'si_binned_run', *useful_metrics, as_dict=True))
pks = (hheise_decoder.BayesianDecoderWithinSession * hheise_behav.VRPerformance & 'bayesian_id=1' & 'day < "2022-09-09"').fetch('KEY')
data['binned_lick_ratio'] = [(hheise_behav.VRPerformance & pk).get_mean()[0] for pk in pks]

periods = []
for mouse_id, mouse_data in data.groupby('mouse_id'):
    surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' & f'mouse_id={mouse_id}').fetch(
        'surgery_date')[0].date()

    period = pd.Series(['none'] * len(mouse_data['day']))
    period.iloc[mouse_data['day'] <= surgery_day] = 'pre'
    period.iloc[((mouse_data['day'] - surgery_day).dt.days > 0) & ((mouse_data['day'] - surgery_day).dt.days <= 7)] = 'early'
    period.iloc[(mouse_data['day'] - surgery_day).dt.days > 7] = 'late'

    periods.append(period)
data['period'] = pd.concat(periods).to_numpy()

behav_metric = 'si_binned_run'

# Scatter plots
data_long = data.melt(id_vars=['mouse_id', behav_metric], value_vars=useful_metrics)
g = sns.FacetGrid(data_long, col="variable", col_wrap=3, sharey=False, hue='mouse_id')
g.map(sns.scatterplot, behav_metric, 'value')

export = data.pivot(columns='mouse_id', values='mae')

# Correlate metrics with SI performance
perf_corr = []
for mouse_id, mouse_data in data.groupby('mouse_id'):
    # Pearson
    # for corr_method in ['pearson', 'spearman']:
    corr_method = 'pearson'
    for behav_metric in ['si_binned_run', 'binned_lick_ratio']:
        corr_all = mouse_data[[*useful_metrics, behav_metric]].corr(method=corr_method)[behav_metric].drop(index=behav_metric)
        corr_pre = mouse_data[mouse_data['period'] == 'pre'][[*useful_metrics, behav_metric]].corr(method=corr_method)[behav_metric].drop(index=behav_metric)
        corr_early = mouse_data[mouse_data['period'] == 'early'][[*useful_metrics, behav_metric]].corr(method=corr_method)[behav_metric].drop(index=behav_metric)
        corr_late = mouse_data[mouse_data['period'] == 'late'][[*useful_metrics, behav_metric]].corr(method=corr_method)[behav_metric].drop(index=behav_metric)
        corr_post = mouse_data[mouse_data['period'] != 'pre'][[*useful_metrics, behav_metric]].corr(method=corr_method)[behav_metric].drop(index=behav_metric)

        perf_corr.append(pd.DataFrame([dict(mouse_id=mouse_id, **dict(corr_all), period='all', method=corr_method, metric=behav_metric)]))
        perf_corr.append(pd.DataFrame([dict(mouse_id=mouse_id, **dict(corr_pre), period='pre', method=corr_method, metric=behav_metric)]))
        perf_corr.append(pd.DataFrame([dict(mouse_id=mouse_id, **dict(corr_post), period='post', method=corr_method, metric=behav_metric)]))
        perf_corr.append(pd.DataFrame([dict(mouse_id=mouse_id, **dict(corr_early), period='early', method=corr_method, metric=behav_metric)]))
        perf_corr.append(pd.DataFrame([dict(mouse_id=mouse_id, **dict(corr_late), period='late', method=corr_method, metric=behav_metric)]))
perf_corr = pd.concat(perf_corr)
perf_corr_long = perf_corr[perf_corr.metric == 'si_binned_run'].melt(value_vars=useful_metrics, id_vars=['mouse_id', 'period', 'metric'])

dat = perf_corr_long[(perf_corr_long['variable'] == 'accuracy')]
plt.bar(x=dat['mouse_id'].astype('str'), height=dat['value'])
sns.barplot(dat, x='period', y='value', hue='mouse_id')

for period in perf_corr_long.period.unique():
    g = sns.FacetGrid(perf_corr_long[perf_corr_long['period'] == period], col="variable", col_wrap=3, sharey=True, ylim=(-1, 1))
    g.map(sns.barplot, "mouse_id", 'value')
    g.fig.suptitle(period)


g = sns.FacetGrid(perf_corr_long[perf_corr_long['metric'] == 'si_binned_run'], col="variable", col_wrap=3, sharey=True, ylim=(-1, 1))
for ax in g.axes:
    ax.axhline(0, c='grey', ls='--')
g.map(sns.boxplot, "period", 'value')
g.axes.flatten()[-1].legend()

g = sns.FacetGrid(perf_corr_long[perf_corr_long['metric'] == 'si_binned_run'], col="mouse_id", col_wrap=6, sharey=True, ylim=(-1, 1))
for ax in g.axes:
    ax.axhline(0, c='grey', ls='--')
g.map(sns.boxplot, "variable", 'value')

plt.figure()
sns.boxplot(perf_corr_long[perf_corr_long['variable'] == 'mae_quad'], x='period', y='value', hue='method')


# Does correlation with VR performance depend on model performance?
df_corr = perf_corr_long[(perf_corr_long['period'] == 'both') & (perf_corr_long['metric'] == 'si_binned_run')].rename(columns={'value': 'model_vr_corr'}).drop(columns=['metric', 'period'])
df_score = data_long.drop(columns='si_binned_run').groupby(['mouse_id', 'variable']).mean().reset_index().rename(columns={'value': 'model_score'})
df_merge = pd.merge(df_corr, df_score, on=['mouse_id', 'variable'])

g = sns.FacetGrid(df_merge, col="variable", col_wrap=3, sharey=False, xlim=(-1, 1))
g.map(sns.scatterplot, "model_vr_corr", 'model_score')

################# Trial-wise ############################
data = pd.DataFrame((hheise_decoder.BayesianDecoderWithinSession.Trial * hheise_behav.VRPerformance
                     & 'bayesian_id=1' & 'day < "2022-09-09"').fetch('mouse_id', 'day', 'trial_id', 'si_count', 'binned_lick_ratio',
                                                                     *useful_metrics, as_dict=True))
data['blr'] = data.apply(lambda x: x['binned_lick_ratio'][x['trial_id']], axis=1)
data['si'] = data.apply(lambda x: x['si_count'][x['trial_id']], axis=1)

# Scatter plots
behav_metric = 'si'
data_long = data.melt(id_vars=['mouse_id', behav_metric], value_vars=useful_metrics)
g = sns.FacetGrid(data_long, col="variable", col_wrap=3, sharey=False, hue='mouse_id')
g.map(sns.scatterplot, behav_metric, 'value')

# Middle-ground: Compute trial-wise correlation for each session separately
trial_corr = []
for idx, session in data.groupby(by=['mouse_id', 'day']):
    curr_corr = session[[*useful_metrics, behav_metric]].corr(method='spearman')[behav_metric].drop(index=behav_metric)

    surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' & f'mouse_id={idx[0]}').fetch(
        'surgery_date')[0].date()
    if idx[1] <= surgery_day:
        period = 'pre'
    elif (idx[1] - surgery_day).days <= 7:
        period = 'early'
    else:
        period = 'late'

    trial_corr.append(pd.DataFrame([dict(mouse_id=idx[0], day=idx[1], period=period, **curr_corr)]))
trial_corr = pd.concat(trial_corr)
trial_corr['mae'] = -trial_corr['mae']
trial_corr['mae_quad'] = -trial_corr['mae_quad']
trial_long = trial_corr.melt(id_vars=['mouse_id', 'period'], value_vars=useful_metrics)

g = sns.FacetGrid(trial_long, col="mouse_id", col_wrap=6, sharey=True)
g.map(sns.boxplot, 'variable', 'value', 'period')
for ax in g.axes:
    ax.axhline(0, c='grey', ls='--')

g = sns.FacetGrid(trial_long, col="variable", col_wrap=3, sharey=True)
g.map(sns.boxplot, 'period', 'value')
for ax in g.axes:
    ax.axhline(0, c='grey', ls='--')

#%% Add behavioral grouping

coarse = (hheise_grouping.BehaviorGrouping & 'grouping_id = 4' & 'cluster = "coarse"').get_groups()
fine = (hheise_grouping.BehaviorGrouping & 'grouping_id = 4' & 'cluster = "fine"').get_groups()

# Adjust correlations of MAE and MAE_quad (let all be positive)
perf_corr_long.loc[perf_corr_long['variable'].isin(['mae', 'mae_quad']), 'value'] = -perf_corr_long.loc[perf_corr_long['variable'].isin(['mae', 'mae_quad']), 'value']

merge = perf_corr_long.merge(coarse, how='left', on='mouse_id').rename(columns={'group': 'coarse'})
merge = merge.merge(fine, how='left', on='mouse_id').rename(columns={'group': 'fine'})

# Coarse
merge[merge.period == 'pre'].pivot(index='variable', columns='mouse_id', values='value').to_clipboard(header=False, index=False)
merge[merge.variable == 'accuracy'].pivot(index='period', columns='mouse_id', values='value').loc[['pre', 'post'], :].to_clipboard(header=False)

# Fine
merge[merge.period == 'all'].pivot(index='variable', columns='mouse_id', values='value').to_clipboard(header=False, index=False)
merge[merge.variable == 'accuracy'].pivot(index='period', columns='mouse_id', values='value').loc[['pre', 'early', 'late'], :].to_clipboard(header=False)

#%% Decoder performance over time

data = pd.DataFrame((hheise_decoder.BayesianDecoderLastPrestroke & 'bayesian_id=1' & 'day < "2022-09-09"'
                     & 'mouse_id!=112').fetch('mouse_id', 'day', *useful_metrics, as_dict=True))
data = pd.DataFrame((hheise_decoder.BayesianDecoderWithinSession & 'bayesian_id=1' & 'day < "2022-09-09"'
                     & 'mouse_id!=112').fetch('mouse_id', 'day', *useful_metrics, as_dict=True))

new_dfs = []
for i, (mouse_id, mouse_data) in enumerate(data.groupby('mouse_id')):
    #
    # if i == 1:
    #     break

    surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' & f'mouse_id={mouse_id}').fetch(
        'surgery_date')[0].date()
    rel_day = (mouse_data['day'] - surgery_day).dt.days
    period = pd.Series(['none'] * len(mouse_data['day']))
    period.iloc[mouse_data['day'] <= surgery_day] = 'pre'
    period.iloc[(rel_day > 0) & (rel_day <= 7)] = 'early'
    period.iloc[rel_day > 7] = 'late'
    period.index = mouse_data.index

    phase_sess = np.concatenate([np.arange(period.value_counts()['pre']), np.arange(period.value_counts()['early']),
                                 np.arange(period.value_counts()['late'])])

    if mouse_id in [63, 69]:
        rel_day = rel_day.mask(rel_day.isin([1, 2, 4]))

    rel_day[(rel_day == 2) | (rel_day == 3) | (rel_day == 4)] = 3
    rel_day[(rel_day == 5) | (rel_day == 6) | (rel_day == 7)] = 6
    rel_day[(rel_day == 8) | (rel_day == 9) | (rel_day == 10)] = 9
    rel_day[(rel_day == 11) | (rel_day == 12) | (rel_day == 13)] = 12
    rel_day[(rel_day == 14) | (rel_day == 15) | (rel_day == 16)] = 15
    rel_day[(rel_day == 17) | (rel_day == 18) | (rel_day == 19)] = 18
    rel_day[(rel_day == 20) | (rel_day == 21) | (rel_day == 22)] = 21
    rel_day[(rel_day == 23) | (rel_day == 24) | (rel_day == 25)] = 24
    if 28 not in rel_day:
        rel_day[(rel_day == 26) | (rel_day == 27) | (rel_day == 28)] = 27

    rel_sess = np.arange(len(rel_day)) - np.argmax(np.where(rel_day <= 0, rel_day, -np.inf))
    pre_mask = (-5 < rel_sess) & (rel_sess < 1).astype(int)
    rel_day[pre_mask.astype(bool)] = np.arange(-pre_mask.sum(), 0)

    norm = mouse_data[useful_metrics] / mouse_data[rel_day.isin([-2, -1, 0])][useful_metrics].mean()
    norm = norm.add_suffix('_norm')

    new_dfs.append(pd.DataFrame(dict(mouse_id=mouse_data.mouse_id, day=mouse_data.day,
                                     period=period, rel_days=rel_day, period_day=phase_sess, **norm)))
new_dfs = pd.concat(new_dfs)
data = data.merge(new_dfs, on=['mouse_id', 'day'])
data = data.dropna()
data = data[data['rel_days'] >= -4]
data = data[data['rel_days'] != 1]

data.pivot(index='rel_days', columns='mouse_id', values='specificity_rz_norm').to_clipboard(header=False, index=False)

### Summarize phases (single mice)
merge = data.merge(coarse, how='left', on='mouse_id').rename(columns={'group': 'coarse'})
merge = merge.merge(fine, how='left', on='mouse_id').rename(columns={'group': 'fine'})

# Copy data for Prism
prism_data = merge[merge.fine == 'No Recovery'].pivot(index='period', columns=['mouse_id', 'period_day'],
                                                   values='specificity_rz_norm').loc[['pre', 'early', 'late']]
prism_data.to_clipboard(header=False, index=False)

# Copy subcolumn names
columns = pd.Series(merge[merge.fine == 'Stroke'].pivot(index='period', columns=['mouse_id', 'period_day'],
                                      values='accuracy').loc[['pre', 'early', 'late']].columns.to_flat_index())
columns.apply(lambda x: f'{x[0]}_{x[1]}').to_clipboard(index=False, header=False)

# Average mice
avg = merge.groupby(['mouse_id', 'period'])[[*useful_metrics, *[x+'_norm' for x in useful_metrics]]].mean(numeric_only=True)
avg = avg.join(merge[['mouse_id', 'coarse', 'fine']].drop_duplicates().set_index('mouse_id'), how='inner').reset_index()

avg_prism = avg.pivot(index='period', columns='mouse_id', values='accuracy_quad').loc[['pre', 'early', 'late']]
avg_prism.to_clipboard(header=False, index=False)

avg = merge.groupby(['mouse_id', 'period'])[['mean', 'median']].mean(numeric_only=True)
avg = avg.join(merge[['mouse_id', 'coarse', 'fine']].drop_duplicates().set_index('mouse_id'), how='inner').reset_index()

avg_prism = avg.pivot(index='period', columns='mouse_id', values='mean').loc[['pre', 'early', 'late']]
avg_prism.to_clipboard(header=False, index=False)

#%% Compare stability distribution of cells that the model used across time & mice

# Use "data" from over-time cell above
# data = pd.DataFrame((hheise_decoder.BayesianDecoderWithinSession & 'bayesian_id=1' & 'day < "2022-09-09"'
#                      & 'mouse_id!=112').fetch('mouse_id', 'day', *useful_metrics, as_dict=True))

# Stability
dist = []
for i, row in data.iterrows():
    dist.append(pd.DataFrame([dict(**row, val=(hheise_placecell.SpatialInformation.ROI & dict(row) &
                                                     'place_cell_id=2' & 'corridor_type=0').fetch('stability', order_by='stability desc')[:100])]))
dist = pd.concat(dist)
dist['mean'] = dist.apply(lambda x: np.mean(x['val']), axis=1)
dist['median'] = dist.apply(lambda x: np.median(x['val']), axis=1)

# Spatial Info
# dist = []
# for i, row in data.iterrows():
#     dist.append(pd.DataFrame([dict(**row, val=(hheise_placecell.SpatialInformation.ROI & dict(row) &
#                                                    'place_cell_id=2' & 'corridor_type=0').fetch('si', order_by='si desc')[:100])]))
# dist = pd.concat(dist)
# dist['mean'] = dist.apply(lambda x: np.mean(x['val']), axis=1)
# dist['median'] = dist.apply(lambda x: np.median(x['val']), axis=1)

# Place Cells
# dist = []
# for i, row in data.iterrows():
#     dist.append(pd.DataFrame([dict(**row, val=(hheise_placecell.PlaceCell.ROI & dict(row) & 'place_cell_id=2' &
#                                                'corridor_type=0' & 'is_place_cell=1').fetch('p', order_by='p desc')[:100])]))
# dist = pd.concat(dist)
# dist['mean'] = dist.apply(lambda x: np.mean(x['val']), axis=1)
# dist['median'] = dist.apply(lambda x: np.median(x['val']), axis=1)

# Firing rate
dist = []
for i, row in data.iterrows():
    dist.append(pd.DataFrame([dict(**row, val=(common_img.ActivityStatistics.ROI & dict(row)).fetch('rate_spikes', order_by='rate_spikes desc')[:100])]))
dist = pd.concat(dist)
dist['mean'] = dist.apply(lambda x: np.mean(x['val']), axis=1)
dist['median'] = dist.apply(lambda x: np.median(x['val']), axis=1)


# Plot stability distribution across days
dist_ex = dist.explode('stab')
g = sns.displot(dist_ex, col='mouse_id', col_wrap=4, hue='rel_days', x='val', kind='kde', palette='plasma_r')

### Correlate mean stability of cells in the network with performance
dist_melt = dist.melt(id_vars=['mouse_id', 'rel_days', 'period', 'mean', 'median'],
                      value_vars=useful_metrics, var_name='metric')

g = sns.FacetGrid(dist_melt, col="metric", col_wrap=3, sharey=False, hue='mouse_id')
g.map(sns.scatterplot, 'mean', 'value')

g = sns.FacetGrid(dist_melt, col="metric", col_wrap=3, sharey=False, hue='rel_days', palette='plasma_r')
g.map(sns.scatterplot, 'mean', 'value')
g.add_legend()

g = sns.FacetGrid(dist_melt, col="metric", col_wrap=3, sharey=False, hue='rel_days', palette='plasma_r')
g.map(sns.scatterplot, 'median', 'value')
g.add_legend()

# Pivot for Prism
dist[['median', *useful_metrics]].to_clipboard(index=False)


#%% Draw dummy curve for stroke severity turning point
from scipy.optimize import curve_fit

# Define the sigmoid function
def sigmoid(x, L, k, x0):
    return L / (1 + np.exp(-k*(x-x0)))

# Create the data
x_data = np.linspace(-10, 10, num=22)
y_data = 1 / (1 + np.exp(-x_data))
y_noise = 0.1 * np.random.normal(size=x_data.size)
y_data += y_noise
y_data *= 0.8

# Perform the regression
p0 = [max(y_data), np.median(x_data), 1]
popt, pcov = curve_fit(sigmoid, x_data, y_data, p0)

# Plot the results
plt.figure()
plt.plot(np.linspace(0, 1, len(x_data)), y_data, 'o', label='mice')
plt.plot(np.linspace(0, 1, len(x_data)), sigmoid(x_data, *popt), 'r-')
plt.ylim(top=1)
plt.ylabel('Behavior-Decoder correlation')
plt.xlabel('Stroke severity (sphere count, lesion volume...)')
plt.legend()
plt.show()


#%% Plot heatmap of decoder confidence for example trial

key = dict(mouse_id=114, day='2022-08-09', bayesian_id=1, trial_id=2)   # Good example (Figure 3F)
key = dict(mouse_id=41, day='2020-08-27', bayesian_id=1, trial_id=0)    # Decent accuracy
key = dict(mouse_id=41, day='2020-08-30', bayesian_id=1, trial_id=0)    # Complete failure

pos_true, pos_pred, confidence = (hheise_decoder.BayesianDecoderWithinSession.Trial & key).fetch1('pos_true', 'pos_predict', 'confidence')

conf_rescale = (1/-confidence).T
# conf_norm = (confidence - np.nanmin(confidence, axis=1)[..., np.newaxis]) / (np.nanmax(confidence, axis=1) - np.nanmin(confidence, axis=1))[..., np.newaxis]
# conf_norm = (confidence/ np.nansum(-confidence, axis=1)[..., np.newaxis])
#
# plt.figure()
# ax = sns.heatmap(conf_rescale, vmin=0, vmax=0.06, cmap=white_crest)
# ax = sns.heatmap(conf_rescale, cmap=white_crest)
# ax.invert_yaxis()
#
# plt.figure()
# plt.plot(pos_pred, color='orange')
# plt.plot(pos_true, color='black')

# Customize color map to have small values in white
import matplotlib as mpl


def white_extend_cmap(cmap_name, white_frac, cmap_res=256):

    orig_range = int(np.round(cmap_res-(cmap_res*white_frac)))
    orig = mpl.colormaps[cmap_name].resampled(orig_range)
    start_col = orig(0)
    vals = np.ones((cmap_res-orig_range, 4))
    vals[:, 0] = np.linspace(start_col[0], 1, cmap_res-orig_range)
    vals[:, 1] = np.linspace(start_col[1], 1, cmap_res-orig_range)
    vals[:, 2] = np.linspace(start_col[2], 1, cmap_res-orig_range)
    white_cmap = mpl.colors.ListedColormap(np.flip(vals, 0))

    newcolors = np.vstack((
        white_cmap(np.linspace(0, 1, cmap_res-orig_range)),
        orig(np.linspace(0, 1, orig_range))
    ))
    return mpl.colors.ListedColormap(newcolors, name=f'extended_{cmap_name}')


white_crest = white_extend_cmap(cmap_name='crest', white_frac=0.25)

plt.figure()
ax = sns.heatmap(conf_rescale, vmin=0, vmax=0.06, cmap=white_crest)
ax.invert_yaxis()
plt.savefig(r'C:\Users\hheise.UZH\Desktop\preprint\figure_stability\decoder\M41_20200827_trial0_confidence_white_crest.png')

sns.heatmap(np.random.random((10, 10)), cmap=white_crest)
plt.savefig(r'C:\Users\hheise.UZH\Desktop\preprint\figure_stability\decoder\white_crest_colorbar.svg')


plt.figure()
ax = sns.heatmap(conf_norm.T, cmap='magma')
ax.invert_yaxis()
plt.savefig(r'C:\Users\hheise.UZH\Desktop\preprint\bayesian_decoder\M114_20220809_trial2_confidence_norm.png')
