# LAST TOKEN
python plot_knockout.py --binary_search_results "/home/idg/projects/ssm_analysis/resources/split_results/ZERO_ATTENTION_2.8B_bin_search.csv" --output "/home/idg/projects/ssm_analysis/resources/split_results/plots/zero_attention.html"

# SUBJ TOKEN
# python plot_knockout.py --binary_search_results "/home/idg/projects/ssm_analysis/resources/split_results/ZERO_ATTENTION_2.8B_bin_search_SUBJ.csv" --output "/home/idg/projects/ssm_analysis/resources/split_results/plots/subj_zero_attention.html"

# SSM
python plot_knockout.py --binary_search_results "/home/idg/projects/ssm_analysis/resources/split_results/IGNORE_SSM_2.8B_norm_1_bin_search.csv" --output "/home/idg/projects/ssm_analysis/resources/split_results/plots/ssm_knockout.html" --knockout_mode ssm

# SSM_EARLY
python plot_knockout.py --binary_search_results "/home/idg/projects/ssm_analysis/resources/split_results/IGNORE_SSM_2.8B_norm_1_bin_search_early_layers_focus.csv" --output "/home/idg/projects/ssm_analysis/resources/split_results/plots/ssm_knockout_early.html" --knockout_mode ssm