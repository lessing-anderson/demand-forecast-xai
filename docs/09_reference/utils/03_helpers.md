# `src.utils.helpers`

## Purpose

Provides generic pickle serialization helpers.

## Public Functions

- `save_pickle(obj, filepath)`: creates parent directories, serializes `obj`, and reports the destination.
- `load_pickle(filepath)`: deserializes an object and reports the source.

These helpers use Python pickle and must only load artifacts from trusted sources.
