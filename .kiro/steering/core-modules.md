---
description: Core Module Coordination Patterns
inclusion: manual
---


# Core Module Coordination Patterns

## Critical 3-File Coordination Pattern

**HIGHEST PRIORITY** - mistakes here break the entire project.

### File Update Sequence (MANDATORY)
1. **First**: Add enum/name to `src/core/names.py`
2. **Second**: Add type definition to `src/core/types.py` (if needed)
3. **Third**: Add constant value to `src/core/consts.py`

### Absolute Rules
- **NEVER hardcode constants outside `src/core/` module**
- **NEVER modify only one core file when changes affect multiple files**
- **NEVER create constants in individual modules**
- **NEVER bypass the 3-file coordination pattern**

### File Responsibilities
- **names.py**: Enum definitions and name constants
- **types.py**: Type definitions and aliases
- **consts.py**: Centralized constant definitions and configuration values

### Common Patterns
- Adding new model architecture: names.py → types.py → consts.py
- Adding new experiment type: names.py → types.py → consts.py
- Adding new dataset: names.py → types.py → consts.py

### Verification Checklist
- [ ] All three files updated in correct sequence
- [ ] No constants hardcoded outside `src/core/`
- [ ] All imports reference core modules
- [ ] Type safety maintained
- [ ] No circular dependencies created

**Reference**: docs/core-modules.md for complete patterns and examples
