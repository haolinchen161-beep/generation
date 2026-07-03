# Innovation1 v3: Weak B-Rep Motif Graph Extraction

## Research Position

This module implements the first thesis innovation: a weak structural motif graph representation for semantics-free public B-Rep data such as ABC and DeepCAD.

The goal is not to assign aerospace composite semantic labels. Instead, the method extracts general geometric-topological motifs from raw B-Rep faces, edges and adjacency:

```text
STEP B-Rep -> face-level evidence graph -> motif graph M -> structure prior for generation
```

The resulting graph is designed as the upstream representation for a later hierarchy:

```text
M -> face group -> topology -> geometry
```

## Motif Graph Definition

The graph is represented as:

```text
M = (Vm, Em, Pm)
```

`Vm` contains weak structural hypernodes over B-Rep face ids:

- `face_group`
- `sheet_like_group`
- `thin_wall_pair`
- `loop_or_hole`
- `transition_group`
- `repeated_feature`
- `boundary_group`

`Em` contains motif-level relations aggregated from face evidence:

- `adjacent_to`
- `parallel_to`
- `opposite_to`
- `orthogonal_to`
- `coplanar_with`
- `smooth_connected`
- `embedded_in`
- `repeated_with`
- `bounded_by`

`Pm` is stored as `motif_prior`, including node/relation vocabularies, type ids and face-to-motif assignments for downstream neural training.

## Method

1. Parse public STEP files with `occwl + DTG parse_solid` when available. On the DTG backend, `face_count` is counted after `parse_solid` splits closed faces and closed edges, matching DTG's canonical B-Rep counting path.
2. Fall back to a pythonOCC bbox/grid parser when available, and mark such samples as `bbox_fallback_sampling`.
3. Apply strict clean filters: single solid, canonical `face_count <= 50` by default for the v3 motif-ready data path, nonzero edges/vertices, constructible adjacencies, finite WCS fields and valid global scale. The original DTG parse code still has a 70-face parse guard; v3 uses a stricter downstream threshold for structural-prior training.
4. Extract face features: centroid, bbox, bbox dimensions, area proxy, normal proxy, aspect ratio, degree, boundary flag and adjacent faces.
5. Build a face-level evidence graph for parallel, opposite, orthogonal, coplanar, adjacent and smooth-connected relations.
6. Select a sparse key structural prior from the evidence graph. The exported training graph keeps generation-relevant motifs only:
   - base face groups as the local topological support;
   - dominant sheet-like anchors;
   - high-confidence opposed/thin-wall candidates;
   - internal loop/closure candidates;
   - small transition connector candidates;
   - regular repeated-feature clusters;
   - boundary groups and bounded relations.
7. Aggregate the selected evidence into weak motif hypernodes and motif-level relations.
8. Export `motif_graph_index.jsonl`, per-sample motif graph JSON files, reports and visualizations.

Dense pairwise evidence is not used as training edges. It is retained only through summary statistics and per-relation evidence fields. This prevents the motif graph from becoming a full pairwise geometry graph. The exported motif graph also enforces `|Vm| <= canonical face_count`, so motif ids do not grow into a dense face-pair enumeration.

## Difference from DTG

DTG mainly learns B-Rep topology and geometry distributions directly. This v3 module inserts an explicit weak structural prior before topology and geometry generation. The prior captures long-range and repeated geometric relations that are not represented as a separate structure layer in DTG.

The key difference is:

```text
DTG: topology -> geometry
v3/v4 thesis line: M -> face group -> topology -> geometry
```

The geometry filtering policy is therefore two-stage:

1. `clean_manifest.csv` keeps DTG-compatible parse/size filters for fair comparison.
2. `motif_ready_manifest.csv` and `motif_graph_index_ready.jsonl` add v3-specific motif-quality filtering for neural training.

A sample can be DTG-clean but not motif-ready. Motif-ready samples should have observable face-group support, at least one non-base motif family, a sparse structural relation prior, controlled relation density, and explicit geometry sampling quality. This prevents the first innovation from becoming a dense pairwise relation enumerator.

The default downstream interface is the strict motif-ready subset. By default, only high-grade samples are written to `motif_graph_index_ready.jsonl`; medium/low samples remain available for audit and ablation but are not used as the default training subset.

## Difference from v2

The previous v2 parser was built for self-generated aerospace composite samples and used part-level procedural semantics. This v3 method:

- does not depend on `part_type`;
- does not call v2 `_classify_faces`;
- does not force ABC/DeepCAD parts into panel/stiffener/flange/hole semantics;
- treats all extracted structures as weak motif candidates with confidence and evidence.

## Limitations

- `M` is a weak structural motif prior, not manually annotated semantic ground truth.
- `loop_or_hole` is an internal-closure candidate, not a true engineering hole label.
- `transition_group` is a geometric-topological connector candidate and does not guarantee true fillet semantics.
- Public ABC/DeepCAD data are not forcibly mapped to aerospace composite semantics.
