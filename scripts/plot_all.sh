# LAST TOKEN
python plot_knockout.py --binary_search_results "/home/idg/projects/ssm_analysis/resources/final_plots/ZERO_ATTENTION_2.8B_bin_search.csv" --output "last_token_zero_attention.html"

# SUBJ TOKEN
python plot_knockout.py --binary_search_results "/home/idg/projects/ssm_analysis/resources/final_plots/ZERO_ATTENTION_2.8B_bin_search_SUBJ.csv" --output "subj_zero_attention.html"

# SSM
python plot_knockout.py --binary_search_results "/home/idg/projects/ssm_analysis/resources/final_plots/IGNORE_SSM_2.8B_norm_1_bin_search.csv" --output "ssm_knockout.html" --knockout_mode ssm

# SSM_EARLY
python plot_knockout.py --binary_search_results "/home/idg/projects/ssm_analysis/resources/final_plots/IGNORE_SSM_2.8B_norm_1_bin_search_early_layers_focus.csv" --output "ssm_knockout_early.html" --knockout_mode ssm