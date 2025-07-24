---
description: Data Object Interfaces and Relationships
inclusion: manual
---


# Data Object Interfaces and Relationships

## Core Data Interface Classes

### DataReqs (Data Requirements)
- **Purpose**: Specify data requirements for experiments
- **File**: `src/data_ingestion/data_defs/data_defs.py`
- **Usage**: Define what data is needed for computation

### ResultBank (Experiment Result Access)
- **Purpose**: Access experiment results and outputs
- **Usage**: Unified interface for accessing experiment outputs
- **Pattern**: Use instead of direct file access

### FulfilledReqs (Fulfilled Data Requirements)
- **Purpose**: Represent fulfilled data requirements
- **Usage**: Track what data has been processed and is available

## Data Object Hierarchy

### Object Relationships
- **DataReqs** → specifies requirements
- **FulfilledReqs** → represents fulfilled requirements
- **ResultBank** → provides access to results

### Interface Patterns
- **ALWAYS use ResultBank** for accessing experiment outputs
- **NEVER access experiment outputs directly** - use data interfaces
- **FOLLOW object-oriented patterns** for data manipulation

### Data Flow Pattern
1. Define requirements with **DataReqs**
2. Process data to create **FulfilledReqs**
3. Access results through **ResultBank**

## Critical Rules

### Data Access Rules
- **NEVER access files directly** - use data interface objects
- **ALWAYS use ResultBank** for experiment result access
- **FOLLOW object-oriented patterns** for all data interactions

### File Location
- **Primary file**: `src/data_ingestion/data_defs/data_defs.py`
- **Contains**: All data interface classes and relationships

**Reference**: docs/data-relationships-interface.md for complete patterns and examples
