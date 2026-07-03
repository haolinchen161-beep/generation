# Innovation1 v3 Method Summary

## Definition

Innovation1 v3 is defined as a weak B-Rep motif graph extraction method for no-semantic CAD data.

The goal is not to render gray CAD screenshots and not to force ABC or DeepCAD into aerospace-composite labels. The goal is to extract an inspectable motif graph:

B-Rep face / edge / vertex / adjacency / geometry sampling
-> motif graph M = (Vm, Em, Pm)
-> face-group / topology / geometry generation interface

## Relation to DTG

DTG focuses on low-level B-Rep topology and geometry generation. v3 does not modify DTG source code. It uses public STEP data and B-Rep parsing as the experimental base.

The additional v3 layer is:

Structure Motif M -> Face Group -> Topology -> Geometry

Motif node types:

- sheet_like_group
- face_group
- thin_wall_pair
- loop_or_hole
- transition_group
- repeated_feature
- boundary_group

Motif relation types:

- parallel_to
- opposite_to
- orthogonal_to
- coplanar_with
- adjacent_to
- smooth_connected
- embedded_in
- repeated_with
- bounded_by

## Difference from v2

v2: aerospace procedural samples -> Gc -> weak face group -> inferred_Gc.

v3: public no-semantic B-Rep -> motif graph M -> inspectable weak structural prior.

v2 can still be used as the aerospace-composite domain-enhancement layer. v3 is the public-data main line.

## Visualization requirement

v3 visualization must show algorithm results, not only a shaded CAD-like body. Each *__motif_debug.png contains:

1. 3D motif bounding boxes with node ids, motif types and face ids.
2. 2D motif graph with relation labels.
3. Text evidence panel with node and relation counts.

## Key outputs

- outputs/motif_graphs/motif_graphs.jsonl
- outputs/reports/motif_stats.csv
- outputs/reports/motif_relation_evidence.csv
- outputs/reports/motif_metrics_report.txt
- outputs/reports/motif_visualization_report.txt
- outputs/visualizations/*__motif_debug.png

## Boundary statement

Motifs are algorithm-extracted weak structural priors, not manual semantic truth. ABC and DeepCAD are not claimed to be aerospace-composite datasets. Cleaning and single-solid filtering are not the innovation. The innovation is the inspectable motif graph M on top of no-semantic B-Rep data.
