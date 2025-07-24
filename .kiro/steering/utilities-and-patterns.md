---
description: Utilities and Code Organization Patterns
inclusion: fileMatch
fileMatchPattern: 'src/utils/*'
---

# Utilities and Code Organization Patterns

## Utility Organization

### Directory Structure
- **File**: `src/utils/`
- **Purpose**: Utility organization rules and common patterns
- **Usage**: Centralized utilities for common operations

### Utility Placement Rules
- **PLACE utilities in appropriate `src/utils/` subdirectories**
- **NEVER create utilities outside `src/utils/` hierarchy**
- **FOLLOW consistent naming conventions** for utility modules

## Common Patterns

### Utility Categories
- **Infrastructure utilities** in `src/utils/infra/`
- **Type manipulation utilities** in `src/utils/types_utils.py`
- **Additional utility modules** as needed

### Organization Rules
- **CENTRALIZE common functionality** in utilities
- **AVOID code duplication** across modules
- **MAINTAIN clear separation** between utility categories

## Critical Rules

### Utility Rules
- **NEVER create utilities outside `src/utils/`**
- **FOLLOW established directory structure**
- **MAINTAIN consistent patterns** across utility modules

### Best Practices
- **REUSE existing utilities** before creating new ones
- **DOCUMENT utility functions** clearly
- **TEST utility functions** thoroughly

## File Organization

### Primary Directory
- `src/utils/` - Main utilities directory
- `src/utils/infra/` - Infrastructure utilities
- `src/utils/types_utils.py` - Type manipulation utilities

### Organization Standards
- **Clear separation** of utility categories
- **Consistent naming** conventions
- **Proper documentation** for all utilities

**Reference**: docs/utilities-and-patterns.md for complete organization patterns
