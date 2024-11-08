python scripts/evaluate_context_interference.py --model_size 2.8B --interfere_mode ZERO_ATTENTION --output_dir ""

python scripts/evaluate_context_interference.py --model_size 2.8B --interfere_mode IGNORE_SSM --output_dir ""

python scripts/evaluate_context_interference.py --model_size 2.8B --interfere_mode IGNORE_SSM --output_dir "" --early_layers_ssm_knockout

python scripts/evaluate_context_interference.py --model_size 2.8B --interfere_mode INCREASE_DELTA --output_dir "" --delta_factor_root 0.5 --delta_start_layer 40 --delta_end_layer 48 --increase_delta_target LAST

python scripts/evaluate_context_interference.py --model_size 2.8B --interfere_mode INCREASE_DELTA --output_dir "" --delta_factor_root 1.5 --delta_start_layer 40 --delta_end_layer 48 --increase_delta_target LAST

python scripts/evaluate_context_interference.py --model_size 2.8B --interfere_mode INCREASE_DELTA --output_dir "" --delta_factor_root 0.5 --delta_start_layer 56 --delta_end_layer 64 --increase_delta_target LAST

python scripts/evaluate_context_interference.py --model_size 2.8B --interfere_mode INCREASE_DELTA --output_dir "" --delta_factor_root 1.5 --delta_start_layer 56 --delta_end_layer 64 --increase_delta_target LAST