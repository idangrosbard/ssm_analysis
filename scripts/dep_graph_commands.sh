#!/usr/bin/env sh
cd ..

pydeps src --debug --noshow --rankdir LR -T svg --only src -o scripts/module_dependencies1.svg --cluster --max-bacon 8 --show-cycles --max-module-depth=2

pydeps src --debug --noshow --rankdir LR -T svg --only src -o scripts/module_dependencies2.svg --cluster --max-bacon 2 --show-cycles --max-module-depth=2

pydeps src --debug --noshow --rankdir LR -T svg --only src -o scripts/module_dependencies3.svg --cluster --max-bacon 3 --show-cycles --noise-level 3

pydeps src --debug --noshow --rankdir LR -T svg --only src -o scripts/module_dependencies4.svg --cluster --max-bacon 2 --show-cycles

pydeps src --debug --noshow --rankdir LR -T svg --only src -o scripts/module_dependencies5.svg --cluster --max-bacon 8 --show-cycles --max-module-depth=3

pydeps src --debug --noshow --rankdir LR -T svg --only src -o scripts/module_dependencies7.svg --cluster --max-bacon 4 --show-cycles --noise-level 10

pydeps src --debug --noshow --rankdir LR -T svg --only src -o scripts/module_dependencies8.svg --cluster --max-module-depth=3

