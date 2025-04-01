#!/usr/bin/env sh
cd ..

OUTPUT_DIR="scripts/outputs/new"

echo "Starting to create dependency graphs..."
pydeps src --noshow --rankdir LR -T svg --only src -o $OUTPUT_DIR/deps_mxb8_md2.svg --cluster --max-bacon 8 --show-cycles --max-module-depth=2
echo "Created deps_mxb8_md2.svg"

pydeps src --noshow --rankdir LR -T svg --only src -o $OUTPUT_DIR/deps_mxb2_md2.svg --cluster --max-bacon 2 --show-cycles --max-module-depth=2
echo "Created deps_mxb2_md2.svg"

pydeps src --noshow --rankdir LR -T svg --only src -o $OUTPUT_DIR/deps_mxb3_nl3.svg --cluster --max-bacon 3 --show-cycles --noise-level 3
echo "Created deps_mxb3_nl3.svg"

pydeps src --noshow --rankdir LR -T svg --only src -o $OUTPUT_DIR/deps_mxb2_sc.svg --cluster --max-bacon 2 --show-cycles
echo "Created deps_mxb2_sc.svg"

pydeps src --noshow --rankdir LR -T svg --only src -o $OUTPUT_DIR/deps_mxb8_md3.svg --cluster --max-bacon 8 --show-cycles --max-module-depth=3
echo "Created deps_mxb8_md3.svg"

pydeps src --noshow --rankdir LR -T svg --only src -o $OUTPUT_DIR/deps_mxb4_nl10.svg --cluster --max-bacon 4 --show-cycles --noise-level 10
echo "Created deps_mxb4_nl10.svg"

pydeps src --noshow --rankdir LR -T svg --only src -o $OUTPUT_DIR/deps_mxb3_md3.svg --cluster --max-module-depth=3
echo "Created deps_mxb3_md3.svg"



# Inner modules

# src/analysis/

#pydeps src/analysis --noshow --rankdir LR -T svg --only src/ -o scripts/deps_analysis.svg

#pydeps src/analysis --noshow --rankdir LR -T svg --only src/analysis -o scripts/deps_analysis1.svg --cluster --max-bacon 8 --show-cycles --max-module-depth=2
#
#pydeps src/ --noshow --rankdir LR -T svg --only src/analysis -o scripts/deps_analysis2.svg --cluster --max-bacon 8 --show-cycles --max-module-depth=3



#
## src/app/
#
#pydeps src/app --noshow --rankdir LR -T svg --only src/ -o scripts/deps_app.svg --cluster --max-bacon 8 --show-cycles --max-module-depth=2
#
## src/utils/
#
#pydeps src/utils --noshow --rankdir LR -T svg --only src/ -o scripts/deps_utils.svg --cluster --max-bacon 8 --show-cycles --max-module-depth=2
#
## src/core
#
#pydeps src/core --noshow --rankdir LR -T svg --only src/ -o scripts/deps_core.svg --cluster --max-bacon 8 --show-cycles --max-module-depth=2
#
## src/experiments/
#
## tests/
#
