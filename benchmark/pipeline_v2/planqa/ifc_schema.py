"""Compatibility shim: the IFC graph is now the common building-graph/1 schema (planqa/building_schema.py).

IFC specifics kept from ifc-graph/1:
- SI units everywhere (m, m², m³); numbers rounded; empty values and all-null columns omitted.
- Everything that repeats per element type (materials, layers, thickness, fire rating, U-value,
  profile) is stored ONCE in `types`; elements only reference a type id.
- Large element lists are tables (columns + rows): same information, roughly half the tokens.
- The spatial tree (project > site > building > storey > space > element) is encoded by the
  `storey` / `space` columns; the graph edges are in `relations`.
"""
from .building_schema import (SCHEMA_VERSION, Building, BuildingGraph, Relation, Source, Storey,  # noqa: F401
                              Table, TypeInfo, make_table)

IfcGraph = BuildingGraph
