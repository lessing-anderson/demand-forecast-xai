# Architecture Decision Records (ADRs)

## Overview

This directory contains the **Architecture Decision Records (ADRs)** for the **demand-forecast-xai** project. ADRs document significant architectural, technical, and methodological decisions, along with their context, rationale, and consequences.

## Index of ADRs

| ADR | Title | Status | Date |
|---|---|---|---|
| [ADR 001](01_data-model.md) | Layered Parquet Data Model (Star Schema) | Accepted | 2026-07-26 |
| [ADR 002](02_temporal-split.md) | Out-of-Time Cutoff Temporal Split Strategy | Accepted | 2026-07-26 |
| [ADR 003](03_xai-sampling-strategy.md) | Deterministic Error-Tail Sampling for Explanations | Accepted | 2026-07-26 |

## ADR Format

Each ADR document follows the standard structure:
- **Title & Status**: Unique ID, title, and current decision status (`Proposed`, `Accepted`, `Superceded`).
- **Context**: The background, technical requirements, and motivation for the decision.
- **Decision**: The chosen technical solution or architectural approach.
- **Consequences**: Summary of positive outcomes, trade-offs, and managed risks.
