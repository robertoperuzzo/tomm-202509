# Architecture Decision Records (ADR)

This directory contains Architecture Decision Records for the Chunking Strategies Demo project. ADRs document important architectural and design decisions made during the development process.

## ADR Index

| ADR                                                          | Title                                                                | Date       | Status   |
| ------------------------------------------------------------ | -------------------------------------------------------------------- | ---------- | -------- |
| [ADR-001](ADR-001-arxiv-download-implementation.md)          | ArXiv Paper Download Implementation                                  | 2025-08-27 | Accepted |
| [ADR-002](ADR-002-preprocessing-pipeline-implementation.md)  | ArXiv Preprocessing Pipeline Implementation                          | 2025-08-27 | Accepted |
| [ADR-003](ADR-003-preprocessing-architecture-refactoring.md) | Preprocessing Architecture Refactoring to Generic Document Processor | 2025-08-27 | Accepted |
| [ADR-004](ADR-004-unit-testing-implementation.md)            | Unit Testing Implementation with pytest Framework                    | 2025-08-27 | Accepted |

## ADR Template

We use the following template for all ADRs:

```markdown
# Title

## Date

## Status

What is the status, such as proposed, accepted, rejected, deprecated, superseded, etc.?

## Context

What is the issue that we're seeing that is motivating this decision or change?

## Decision

What is the change that we're proposing and/or doing?

## Consequences

What becomes easier or more difficult to do because of this change?
```

## Decision Flow

The ADRs represent the evolution of the preprocessing system:

1. **ADR-001**: Established ArXiv paper download capabilities with async processing
2. **ADR-002**: Implemented comprehensive preprocessing pipeline for text extraction
3. **ADR-003**: Refactored to generic document processor following separation of concerns
4. **ADR-004**: Added comprehensive unit testing with pytest framework

Each decision builds upon previous ones while addressing specific architectural needs and user feedback.
