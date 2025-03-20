#!/usr/bin/env sh
cd ..

pydeps src --debug --noshow --rankdir LR -T svg --only src -o scripts/deps_mxb8_md2.svg --cluster --max-bacon 8 --show-cycles --max-module-depth=2

pydeps src --debug --noshow --rankdir LR -T svg --only src -o scripts/deps_mxb2_md2.svg --cluster --max-bacon 2 --show-cycles --max-module-depth=2

pydeps src --debug --noshow --rankdir LR -T svg --only src -o scripts/deps_mxb3_nl3.svg --cluster --max-bacon 3 --show-cycles --noise-level 3

pydeps src --debug --noshow --rankdir LR -T svg --only src -o scripts/deps_mxb2_sc.svg --cluster --max-bacon 2 --show-cycles

pydeps src --debug --noshow --rankdir LR -T svg --only src -o scripts/deps_mxb8_md3.svg --cluster --max-bacon 8 --show-cycles --max-module-depth=3

pydeps src --debug --noshow --rankdir LR -T svg --only src -o scripts/deps_mxb4_nl10.svg --cluster --max-bacon 4 --show-cycles --noise-level 10

pydeps src --debug --noshow --rankdir LR -T svg --only src -o scripts/deps_mxb3_md3.svg --cluster --max-module-depth=3



# Inner modules

# src/analysis/

pydeps src/analysis --debug --noshow --rankdir LR -T svg --only src/ -o scripts/deps_analysis.svg --cluster --max-bacon 8 --show-cycles --max-module-depth=2

# src/app/

pydeps src/app --debug --noshow --rankdir LR -T svg --only src/ -o scripts/deps_app.svg --cluster --max-bacon 8 --show-cycles --max-module-depth=2

# src/utils/

pydeps src/utils --debug --noshow --rankdir LR -T svg --only src/ -o scripts/deps_utils.svg --cluster --max-bacon 8 --show-cycles --max-module-depth=2

# src/core

pydeps src/core --debug --noshow --rankdir LR -T svg --only src/ -o scripts/deps_core.svg --cluster --max-bacon 8 --show-cycles --max-module-depth=2

# src/experiments/

# tests/

