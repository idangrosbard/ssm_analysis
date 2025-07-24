---
description: Streamlit UI Component Architecture
inclusion: manual
---

# Streamlit UI Component Architecture

## Component-Based Architecture

### Application Structure
- **Entry Point**: `src/app/entry_point.py` - Main application entry
- **Pages**: `src/app/page/` - Individual page implementations
- **Components**: `src/app/components/` - Reusable UI components
- **Data Store**: `src/app/data_store.py` - Session state management

### Page Architecture
- **Format**: `p{XX}_{page_name}.py` where XX is page order number
- **Base Classes**: Inherit from appropriate base classes
- **Navigation**: Consistent navigation patterns across pages

## UI Patterns

### Component Architecture
- **REUSABLE components** for common UI elements
- **BASE classes** for consistent component behavior
- **SESSION state management** through data_store.py

### Page Implementation
- **CONSISTENT page structure** across all pages
- **PROPER navigation** between pages
- **COMPONENT integration** for reusable elements

## Session State Management

### Data Store Pattern
- **File**: `src/app/data_store.py`
- **Purpose**: Centralized session state management
- **Usage**: Manage application state across pages

### State Management Rules
- **USE data_store.py** for all session state
- **MAINTAIN state consistency** across pages
- **HANDLE state transitions** properly

## Critical Rules

### UI Development Rules
- **FOLLOW component-based architecture**
- **USE consistent page patterns**
- **MANAGE session state properly**

### File Organization
- `src/app/entry_point.py` - Main entry point
- `src/app/page/` - Page implementations
- `src/app/components/` - Reusable components
- `src/app/data_store.py` - Session state management
- `src/app/app_consts.py` - Application constants

**Reference**: docs/streamlit-infrastructure.md for complete architecture patterns
