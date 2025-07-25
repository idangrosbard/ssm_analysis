---
description: Steering Creation and Best Practices Guidelines
inclusion: manual
---

# Steering Guidelines

Meta-steering document that defines the principles and standards for creating effective steering files in the Kiro system.

## What Makes a Good Steering File

### Core Characteristics

**Short but Information-Dense**
- Steerings should be concise while encoding maximum useful information
- **MAXIMUM 200-300 lines** - longer steerings indicate too much detail
- Focus on essential rules, patterns, and decision-making guidance
- Eliminate verbose explanations in favor of actionable directives
- Aim for high information density per line of content

**Concept and Module Descriptions with Rules**
- Describe concepts clearly with associated operational rules
- Provide specific guidance on how to work with modules and patterns
- Include both "what" (description) and "how" (rules) information
- Focus on practical application rather than theoretical background

**Connection Descriptions to Other Modules**
- Explicitly describe relationships and dependencies between modules
- Use cross-references to guide LLMs to related steering files
- Explain how different components interact and coordinate
- Provide navigation guidance for complex multi-module tasks

**Minimal Code with Expansion Instructions**
- Include minimal code examples that illustrate key patterns (max 5-10 lines)
- **AVOID extensive code blocks** - use file references instead
- Provide clear instructions on how LLMs can expand context when needed
- Reference specific files and line numbers for detailed examples
- Guide LLMs on when and how to read additional source code

## Steering Structure Standards

### Front-Matter Requirements

All steering files must include proper YAML front-matter:

```yaml
---
description: [Concise description of steering purpose]
inclusion: manual
---
```

**Description Guidelines**:
- Use concise, descriptive phrases that explain the steering's purpose
- Focus on what the steering helps LLMs accomplish
- Avoid redundant words like "steering" or "documentation"
- Examples: "Core Module Coordination Patterns", "Infrastructure Base Classes and Patterns"

**Inclusion Configuration**:
- Use `inclusion: manual` for all steering files except the master navigation
- Master navigation uses `inclusion: always` to ensure it's always loaded
- Manual inclusion allows selective loading based on task context

### Content Organization Patterns

**Hierarchical Structure**
- Use clear heading hierarchy (H1, H2, H3) to organize content
- Group related concepts under logical sections
- Maintain consistent heading styles across steering files

**Rule-Based Content**
- Lead with rules and directives rather than explanations
- Use imperative language: "ALWAYS", "NEVER", "MUST", "SHOULD"
- Provide specific, actionable guidance
- Include verification checklists where appropriate

**Cross-Reference Integration**
- Include cross-reference sections that guide to related steerings
- Use the format: "Use #steering-name for [specific guidance]"
- Explain what each referenced steering contains
- Group cross-references by logical categories

## Context Expansion Guidelines

### When LLMs Should Expand Context

**File References**
- When steering mentions specific files, LLMs should read them if implementation details are needed
- Use format: "See: src/path/to/file.py:45-67" for specific line references
- Provide guidance on which files contain the most relevant information

**Pattern Implementation**
- When steering describes patterns, LLMs should examine existing implementations
- Guide LLMs to representative examples in the codebase
- Explain how to identify good vs. poor implementations

**Complex Interactions**
- When multiple modules interact, LLMs should read related steering files
- Provide clear guidance on which steerings to combine for complex tasks
- Explain the order in which to consume related steerings

### How to Provide Expansion Instructions

**Specific File Guidance**
```markdown
For implementation details, examine:
- `src/core/names.py` - Enum definitions and patterns
- `src/core/types.py` - Type alias patterns
- `src/core/consts.py` - Constant organization patterns
```

**Cross-Reference Guidance**
```markdown
For related patterns:
- Use #infrastructure for base class coordination
- Use #experiment-runners for dependency management
- Use #data-relationships-interface for data object patterns
```

**Conditional Expansion**
```markdown
If implementing new experiment types:
1. First read #infrastructure for base patterns
2. Then examine existing runners in src/experiments/runners/
3. Finally check #core-modules for constant coordination
```

## Steering Creation Process

### Step 1: Identify the Concept or Module

**Define Scope**
- Identify the specific concept, module, or pattern the steering will cover
- Determine the boundaries and relationships to other concepts
- Establish the primary use cases and scenarios

**Analyze Existing Patterns**
- Examine existing code implementations
- Identify common patterns and anti-patterns
- Document critical rules and best practices

### Step 2: Structure the Content

**Create Hierarchical Organization**
- Start with overview and core concepts
- Progress to specific rules and patterns
- End with cross-references and expansion guidance

**Focus on Rules and Directives**
- Emphasize actionable guidance over explanatory content
- Use imperative language for clear direction
- Include specific examples and counter-examples

### Step 3: Add Cross-References and Context

**Identify Related Steerings**
- Map relationships to other modules and concepts
- Determine which steerings should be read together
- Create logical groupings for complex tasks

**Provide Expansion Guidance**
- Explain when and how to read additional source code
- Guide LLMs to the most relevant files and examples
- Include conditional guidance for different scenarios

### Step 4: Validate and Refine

**Test Information Density**
- Ensure every line provides actionable value
- Remove redundant or verbose content
- Optimize for quick comprehension and application

**Verify Cross-Reference Accuracy**
- Confirm all referenced steerings exist and are relevant
- Test that cross-reference chains provide complete guidance
- Ensure no circular or broken references

## Steering Maintenance Guidelines

### When to Update Steerings

**Code Changes**
- Update steerings when underlying code patterns change
- Revise rules when new best practices emerge
- Add new cross-references when modules are added or reorganized

**Pattern Evolution**
- Update steerings when architectural patterns evolve
- Revise guidance when new tools or frameworks are adopted
- Maintain consistency when project standards change

### How to Maintain Consistency

**Regular Review Process**
- Periodically review steerings for accuracy and relevance
- Check that cross-references remain valid and useful
- Verify that examples and file references are current

**Coordinated Updates**
- When updating one steering, check related steerings for consistency
- Maintain consistent terminology and patterns across all steerings
- Update cross-reference networks when steerings are reorganized

**Version Control Integration**
- Track steering changes alongside code changes
- Document the reasoning behind steering updates
- Maintain steering history for understanding evolution

## Common Steering Anti-Patterns

### What to Avoid

**Verbose Explanations**
- Don't include lengthy background explanations
- Avoid academic or theoretical discussions
- Don't repeat information available in code comments

**Outdated Information**
- Don't reference deprecated patterns or files
- Avoid including obsolete rules or practices
- Don't maintain steerings that no longer apply

**Circular References**
- Don't create circular cross-reference chains
- Avoid steerings that only reference other steerings
- Don't create dependency loops between steerings

**Implementation Details**
- Don't include extensive code implementations
- **NEVER include more than 10 lines of code per example**
- Avoid duplicating information better found in source code
- Don't provide step-by-step coding instructions
- Use file references instead of code blocks for complex examples

### Best Practices Summary

**Content Quality**
- Prioritize actionable rules over explanatory content
- Maintain high information density
- Focus on practical application guidance

**Structure and Organization**
- Use consistent hierarchical organization
- Provide clear cross-reference navigation
- Include expansion guidance for complex topics

**Maintenance and Evolution**
- Keep steerings current with code changes
- Maintain consistency across related steerings
- Document steering evolution and reasoning

**LLM Guidance**
- Provide clear instructions for context expansion
- Guide LLMs to the most relevant additional resources
- Explain when and how to combine multiple steerings
