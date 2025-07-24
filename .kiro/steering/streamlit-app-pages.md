---
description: Streamlit Page Architecture and Navigation
inclusion: manual
---

# Streamlit Page Architecture and Navigation

## Page Structure

### Page Implementation Pattern
- **Format**: `p{XX}_{page_name}.py` where XX is page order number
- **Location**: `src/app/page/`
- **Entry Point**: Always `src/app/entry_point.py`

### Page Types
- **p01_home.py** - Home page
- **p02_results_bank.py** - Results management
- **p03-p09** - Additional specialized pages

## Navigation Patterns

### Page Navigation
- **CONSISTENT navigation** across all pages
- **PROPER page routing** through entry point
- **STATE management** between page transitions

### Navigation Rules
- **USE consistent navigation patterns**
- **MAINTAIN page state** during navigation
- **HANDLE navigation errors** gracefully

## Page Architecture

### Component Integration
- **INTEGRATE reusable components** from `src/app/components/`
- **USE base classes** for consistent behavior
- **FOLLOW component patterns** for UI elements

### Page Implementation
- **CONSISTENT page structure** across all implementations
- **PROPER error handling** for page operations
- **EFFICIENT rendering** for complex pages

## Critical Rules

### Page Development Rules
- **FOLLOW page naming convention** (p{XX}_{name}.py)
- **USE consistent navigation patterns**
- **INTEGRATE with component architecture**

### Navigation Standards
- **CONSISTENT routing** through entry point
- **PROPER state management** between pages
- **ERROR handling** for navigation failures

**Reference**: docs/streamlit-app-pages.md for complete page architecture patterns
