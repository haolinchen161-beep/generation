"""Paper-aligned geometric/CAD metrics and Innovation-1 structural metrics."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from chamferdist import ChamferDistance
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepTools import breptools
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.TopoDS import topods
from OCC.Extend.DataExchange import read_step_file
from scipy.optimize import linear_sum_assignment

from innovation1_v3_brep_motif_graph.brep_loader import parse_step_file
from innovation1_v3_brep_motif_graph.motif_graph_builder import (
    build_motif_graph,
    extract_motif_features,
    make_motif_prior_graph,
)


MOTIF_TYPES = ("sheet_region", "loop_or_hole", "repeated_feature")
RELATION_TYPES = ("embedded_in", "hosted_by", "thin_wall_pair")
SURFACE_TYPES = ("plane", "cylinder", "cone", "sphere", "torus", "freeform_or_other")
HASH_SCHEMA = "brepgen_face_geometry_topology_v2"


def _indexed(shape: Any, kind: int) -> TopTools_IndexedMapOfShape:
    result = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, kind, result)
    return result


def shape_validity(shape: Any) -> Dict[str, Any]:
    shells = _indexed(shape, TopAbs_SHELL)
    closed = sum(
        bool(BRep_Tool.IsClosed(topods.Shell(shells.FindKey(index))))
        for index in range(1, shells.Size() + 1)
    )
    edges = _indexed(shape, TopAbs_EDGE)
    uses = [0] * int(edges.Size())
    faces = _indexed(shape, TopAbs_FACE)
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        edge_explorer = TopExp_Explorer(topods.Face(explorer.Current()), TopAbs_EDGE)
        while edge_explorer.More():
            index = int(edges.FindIndex(edge_explorer.Current())) - 1
            if index >= 0:
                uses[index] += 1
            edge_explorer.Next()
        explorer.Next()
    counts = {
        "faces": int(faces.Size()),
        "edges": int(edges.Size()),
        "vertices": int(_indexed(shape, TopAbs_VERTEX).Size()),
        "shells": int(shells.Size()),
        "closed_shells": int(closed),
        "solids": int(_indexed(shape, TopAbs_SOLID).Size()),
    }
    occ_valid = bool(BRepCheck_Analyzer(shape).IsValid())
    watertight = bool(counts["shells"] >= 1 and closed == counts["shells"])
    manifold = bool(uses and all(value == 2 for value in uses))
    valid = bool(
        occ_valid
        and watertight
        and manifold
        and counts["solids"] == 1
        and counts["shells"] == 1
    )
    return {
        **counts,
        "occ_valid": occ_valid,
        "watertight": watertight,
        "manifold": manifold,
        "valid": valid,
    }


def _surface_grids(shape: Any, resolution: int = 32) -> List[np.ndarray]:
    grids = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        u_min, u_max, v_min, v_max = (float(value) for value in breptools.UVBounds(face))
        adaptor = BRepAdaptor_Surface(face, True)
        values = np.empty((resolution, resolution, 3), dtype=np.float64)
        for ui, u in enumerate(np.linspace(u_min, u_max, resolution)):
            for vi, v in enumerate(np.linspace(v_min, v_max, resolution)):
                point = adaptor.Value(float(u), float(v))
                values[ui, vi] = (point.X(), point.Y(), point.Z())
        grids.append(values)
        explorer.Next()
    if not grids:
        raise ValueError("B-rep contains no faces")
    return grids


def _canonical_grid_bytes(grid: np.ndarray) -> bytes:
    variants = []
    for rotation in range(4):
        rotated = np.rot90(grid, rotation, axes=(0, 1))
        variants.extend([rotated.tobytes(), np.flip(rotated, axis=0).tobytes()])
    return min(variants)


def brep_hash(shape: Any, bits: int = 4) -> str:
    grids = _surface_grids(shape, 32)
    all_points = np.concatenate([grid.reshape(-1, 3) for grid in grids], axis=0)
    center = all_points.mean(axis=0, keepdims=True)
    scale = float(np.max(np.abs(all_points - center)))
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError("degenerate B-rep")
    levels = 2 ** int(bits) - 1
    face_hashes = []
    for grid in grids:
        normalized = (grid - center.reshape(1, 1, 3)) / scale
        quantized = np.rint((np.clip(normalized, -1, 1) + 1) * 0.5 * levels).astype(np.uint8)
        face_hashes.append(hashlib.sha256(_canonical_grid_bytes(quantized)).hexdigest())

    vertex_map, edge_map, face_map = (
        _indexed(shape, TopAbs_VERTEX),
        _indexed(shape, TopAbs_EDGE),
        _indexed(shape, TopAbs_FACE),
    )
    graph = nx.Graph()
    for index in range(face_map.Size()):
        graph.add_node("f%d" % index, kind="face")
    for index in range(edge_map.Size()):
        graph.add_node("e%d" % index, kind="edge")
    for index in range(vertex_map.Size()):
        graph.add_node("v%d" % index, kind="vertex")
    for edge_index in range(1, edge_map.Size() + 1):
        explorer = TopExp_Explorer(topods.Edge(edge_map.FindKey(edge_index)), TopAbs_VERTEX)
        while explorer.More():
            endpoint = vertex_map.FindIndex(explorer.Current()) - 1
            if endpoint >= 0:
                graph.add_edge("e%d" % (edge_index - 1), "v%d" % endpoint)
            explorer.Next()
    for face_index in range(1, face_map.Size() + 1):
        explorer = TopExp_Explorer(topods.Face(face_map.FindKey(face_index)), TopAbs_EDGE)
        while explorer.More():
            edge_index = edge_map.FindIndex(explorer.Current()) - 1
            if edge_index >= 0:
                graph.add_edge("f%d" % (face_index - 1), "e%d" % edge_index)
            explorer.Next()
    topology = nx.weisfeiler_lehman_graph_hash(
        graph, node_attr="kind", iterations=4, digest_size=32
    )
    payload = "%s|bits=%d|F=%d|E=%d|V=%d|%s|%s" % (
        HASH_SCHEMA,
        bits,
        face_map.Size(),
        edge_map.Size(),
        vertex_map.Size(),
        topology,
        "|".join(sorted(face_hashes)),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _triangles(shape: Any, linear: float = 0.02, angular: float = 0.5) -> np.ndarray:
    mesher = BRepMesh_IncrementalMesh(shape, linear, False, angular, True)
    mesher.Perform()
    triangles = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is not None:
            transform = location.Transformation()
            nodes = []
            for index in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(index).Transformed(transform)
                nodes.append([point.X(), point.Y(), point.Z()])
            nodes = np.asarray(nodes)
            for index in range(1, triangulation.NbTriangles() + 1):
                a, b, c = triangulation.Triangle(index).Get()
                triangles.append(nodes[[a - 1, b - 1, c - 1]])
        explorer.Next()
    if not triangles:
        raise ValueError("STEP triangulation is empty")
    return np.asarray(triangles, dtype=np.float64)


def sample_shape_points(shape: Any, count: int, seed: int) -> np.ndarray:
    triangles = _triangles(shape)
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    ) * 0.5
    usable = np.isfinite(areas) & (areas > 1e-14)
    triangles, areas = triangles[usable], areas[usable]
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(triangles), count, replace=True, p=areas / areas.sum())
    selected = triangles[chosen]
    first, second = rng.rand(count), rng.rand(count)
    reflected = first + second > 1
    first[reflected], second[reflected] = 1 - first[reflected], 1 - second[reflected]
    points = (
        selected[:, 0]
        + first[:, None] * (selected[:, 1] - selected[:, 0])
        + second[:, None] * (selected[:, 2] - selected[:, 0])
    ).astype(np.float32)
    points -= points.mean(axis=0, keepdims=True)
    scale = float(np.max(np.abs(points)))
    if scale <= 1e-12:
        raise ValueError("degenerate point cloud")
    return points / scale


def occupancy_counts(clouds: np.ndarray, resolution: int = 28) -> np.ndarray:
    scaled = np.rint((np.clip(clouds, -1, 1) + 1) * 0.5 * (resolution - 1)).astype(np.int64)
    flat = scaled[..., 0] * resolution * resolution + scaled[..., 1] * resolution + scaled[..., 2]
    return np.bincount(flat.reshape(-1), minlength=resolution ** 3).astype(np.float64)


def jsd(left: np.ndarray, right: np.ndarray) -> float:
    left, right = left.astype(np.float64), right.astype(np.float64)
    left, right = left / left.sum(), right / right.sum()
    mean = 0.5 * (left + right)
    def kl(source, target):
        mask = source > 0
        return float(np.sum(source[mask] * np.log2(source[mask] / target[mask])))
    return 0.5 * kl(left, mean) + 0.5 * kl(right, mean)


def pairwise_chamfer(samples: np.ndarray, references: np.ndarray, batch_size: int = 8) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = torch.from_numpy(references.astype(np.float32)).to(device)
    chamfer = ChamferDistance()
    distances = np.empty((len(samples), len(references)), dtype=np.float32)
    with torch.no_grad():
        for sample_index, cloud in enumerate(samples):
            source = torch.from_numpy(cloud.astype(np.float32)).to(device)
            for start in range(0, len(references), batch_size):
                stop = min(start + batch_size, len(references))
                repeated = source.unsqueeze(0).expand(stop - start, -1, -1).contiguous()
                values = chamfer(
                    repeated,
                    reference[start:stop],
                    bidirectional=True,
                    batch_reduction=None,
                    point_reduction="mean",
                )
                distances[sample_index, start:stop] = values.cpu().numpy()
    return distances


def distribution_metrics(samples: np.ndarray, references: np.ndarray) -> Dict[str, float]:
    distances = pairwise_chamfer(samples, references)
    selected = np.argmin(distances, axis=1)
    return {
        "cov": float(len(np.unique(selected)) / len(references)),
        "mmd": float(np.min(distances, axis=0).mean()),
        "jsd": float(jsd(occupancy_counts(samples), occupancy_counts(references))),
    }


def cad_metrics(hashes: Sequence[str], training_hashes: set, requested: int) -> Dict[str, Any]:
    counts = Counter(hashes)
    valid = len(hashes)
    novel = sum(value not in training_hashes for value in hashes)
    return {
        "valid": valid / requested if requested else 0.0,
        "novel": novel / valid if valid else None,
        "unique": len(counts) / valid if valid else None,
        "valid_count": valid,
        "novel_count": novel,
        "unique_hash_count": len(counts),
    }


def _f1(tp: int, predicted: int, expected: int) -> Dict[str, float]:
    precision = tp / predicted if predicted else (1.0 if expected == 0 else 0.0)
    recall = tp / expected if expected else (1.0 if predicted == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def prior_structure(item: Mapping[str, Any]) -> Dict[str, Any]:
    prior = item["prior"]
    nf = int(item["num_faces"])
    instances = prior["motif_instance"][:nf].cpu().numpy()
    confidences = prior["motif_confidence"][:nf].cpu().numpy()
    nodes = []
    node_id_by_instance = {}
    for channel, motif_type in enumerate(MOTIF_TYPES):
        for instance_id in sorted(set(instances[:, channel].tolist()) - {0}):
            faces = np.flatnonzero(instances[:, channel] == instance_id).astype(int).tolist()
            node_id = "p_%d_%d" % (channel, instance_id)
            node_id_by_instance[(channel, int(instance_id))] = node_id
            nodes.append(
                {
                    "id": node_id,
                    "type": motif_type,
                    "face_ids": faces,
                    "confidence": float(confidences[faces, channel].mean()) if faces else 0.0,
                }
            )
    surface = prior["surface_type"][:nf].cpu().numpy().astype(int)
    pair = prior["pair_relations"][:nf, :nf].cpu().numpy()
    relations = set()
    for left in range(nf):
        for right in range(left + 1, nf):
            if pair[left, right, 0] > 0:
                candidates = (
                    (instances[left, 1], instances[right, 0]),
                    (instances[right, 1], instances[left, 0]),
                )
                for loop_instance, sheet_instance in candidates:
                    if loop_instance and sheet_instance:
                        relations.add(
                            (
                                "hosted_by",
                                node_id_by_instance.get((1, int(loop_instance))),
                                node_id_by_instance.get((0, int(sheet_instance))),
                            )
                        )
            if pair[left, right, 1] > 0:
                first, second = instances[left, 0], instances[right, 0]
                if first and second and first != second:
                    endpoints = sorted(
                        [
                            node_id_by_instance.get((0, int(first))),
                            node_id_by_instance.get((0, int(second))),
                        ]
                    )
                    relations.add(("thin_wall_pair", endpoints[0], endpoints[1]))
    relations = sorted(relation for relation in relations if None not in relation)
    return {
        "num_faces": nf,
        "nodes": nodes,
        "relations": relations,
        "motif_counts": np.asarray(prior["motif_counts"].cpu(), dtype=np.float64),
        "relation_counts": np.asarray(
            [
                int((prior["motif_membership"][:nf] > 0).sum().item()),
                int(prior["relation_counts"][0]),
                int(prior["relation_counts"][1]),
            ],
            dtype=np.float64,
        ),
        "relation_confidence": np.asarray(
            [
                float(
                    prior["motif_confidence"][:nf][prior["motif_confidence"][:nf] > 0].mean().item()
                )
                if torch.any(prior["motif_confidence"][:nf] > 0)
                else 0.0,
                float(pair[..., 0][pair[..., 0] > 0].mean()) if np.any(pair[..., 0] > 0) else 0.0,
                float(pair[..., 1][pair[..., 1] > 0].mean()) if np.any(pair[..., 1] > 0) else 0.0,
            ]
        ),
        "surface": surface,
    }


def generated_structure(step_path: Path) -> Dict[str, Any]:
    parsed = parse_step_file(str(step_path))
    parsed["dtg_train_compatible"] = 1
    features = extract_motif_features(parsed)
    raw = build_motif_graph(parsed, raw_mode=True, features=features)
    graph = make_motif_prior_graph(raw)
    nodes = [
        {
            "id": str(node.get("id")),
            "type": str(node.get("type")),
            "face_ids": [int(value) for value in node.get("face_ids", [])],
            "confidence": float(node.get("confidence", 0.0)),
        }
        for node in graph.get("motif_nodes", [])
        if str(node.get("type")) in MOTIF_TYPES
    ]
    relation_counts = np.asarray(
        [
            sum(len(node["face_ids"]) for node in nodes),
            sum(str(rel.get("type")) == "hosted_by" for rel in graph.get("motif_relations", [])),
            sum(str(rel.get("type")) == "thin_wall_pair" for rel in graph.get("motif_relations", [])),
        ],
        dtype=np.float64,
    )
    relation_confidence = np.asarray(
        [
            float(np.mean([node["confidence"] for node in nodes])) if nodes else 0.0,
            float(np.mean([
                float(rel.get("confidence", 0.0))
                for rel in graph.get("motif_relations", [])
                if str(rel.get("type")) == "hosted_by"
            ])) if any(str(rel.get("type")) == "hosted_by" for rel in graph.get("motif_relations", [])) else 0.0,
            float(np.mean([
                float(rel.get("confidence", 0.0))
                for rel in graph.get("motif_relations", [])
                if str(rel.get("type")) == "thin_wall_pair"
            ])) if any(str(rel.get("type")) == "thin_wall_pair" for rel in graph.get("motif_relations", [])) else 0.0,
        ]
    )
    raw_surface = np.asarray(parsed["face_surface_type"], dtype=np.int64)
    node_ids = {node["id"] for node in nodes}
    relations = []
    for relation in graph.get("motif_relations", []):
        relation_type = str(relation.get("type"))
        source, target = str(relation.get("source")), str(relation.get("target"))
        if relation_type not in ("hosted_by", "thin_wall_pair"):
            continue
        if source not in node_ids or target not in node_ids:
            continue
        if relation_type == "thin_wall_pair" and source > target:
            source, target = target, source
        relations.append((relation_type, source, target))
    surface = np.full(raw_surface.shape, 5, dtype=np.int64)
    for value in range(5):
        surface[raw_surface == value] = value
    return {
        "num_faces": int(parsed["face_count"]),
        "nodes": nodes,
        "relations": sorted(set(relations)),
        "motif_counts": np.asarray(
            [sum(node["type"] == motif_type for node in nodes) for motif_type in MOTIF_TYPES],
            dtype=np.float64,
        ),
        "relation_counts": relation_counts,
        "relation_confidence": relation_confidence,
        "surface": surface,
    }


def structural_scores(prior: Mapping[str, Any], generated: Mapping[str, Any]) -> Dict[str, Any]:
    prior_nodes, generated_nodes = prior["nodes"], generated["nodes"]
    matched = 0
    node_mapping = {}
    prior_node_by_id = {node["id"]: node for node in prior_nodes}
    generated_node_by_id = {node["id"]: node for node in generated_nodes}
    for motif_type in MOTIF_TYPES:
        expected = [node for node in prior_nodes if node["type"] == motif_type]
        predicted = [node for node in generated_nodes if node["type"] == motif_type]
        if not expected or not predicted:
            continue
        costs = np.ones((len(predicted), len(expected)), dtype=np.float64)
        for i, left in enumerate(predicted):
            for j, right in enumerate(expected):
                left_size = len(left["face_ids"]) / max(generated["num_faces"], 1)
                right_size = len(right["face_ids"]) / max(prior["num_faces"], 1)
                size_similarity = 1.0 - min(abs(left_size - right_size), 1.0)
                confidence_similarity = 1.0 - min(abs(left["confidence"] - right["confidence"]), 1.0)
                costs[i, j] = 1.0 - (0.75 * size_similarity + 0.25 * confidence_similarity)
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows, columns):
            if costs[row, column] <= 0.5:
                matched += 1
                node_mapping[predicted[row]["id"]] = expected[column]["id"]
    motif = _f1(matched, len(generated_nodes), len(prior_nodes))

    expected_relations = set(tuple(value) for value in prior.get("relations", []))
    relation_tp = 0
    for relation_type, source, target in generated.get("relations", []):
        if source not in node_mapping or target not in node_mapping:
            continue
        mapped_source, mapped_target = node_mapping[source], node_mapping[target]
        if relation_type == "thin_wall_pair" and mapped_source > mapped_target:
            mapped_source, mapped_target = mapped_target, mapped_source
        if (relation_type, mapped_source, mapped_target) in expected_relations:
            relation_tp += 1
    # embedded_in is evaluated after type-constrained motif matching because
    # generated STEP face IDs are not comparable with template face IDs.
    for generated_id, prior_id in node_mapping.items():
        relation_tp += min(
            len(generated_node_by_id[generated_id]["face_ids"]),
            len(prior_node_by_id[prior_id]["face_ids"]),
        )
    relation = _f1(
        relation_tp,
        len(generated.get("relations", []))
        + sum(len(node["face_ids"]) for node in generated_nodes),
        len(prior.get("relations", []))
        + sum(len(node["face_ids"]) for node in prior_nodes),
    )
    per_surface = []
    active_surface = []
    for surface_type in range(6):
        expected = int(np.sum(prior["surface"] == surface_type))
        predicted = int(np.sum(generated["surface"] == surface_type))
        if expected == 0 and predicted == 0:
            per_surface.append(None)
        else:
            value = _f1(min(expected, predicted), predicted, expected)["f1"]
            per_surface.append(value)
            active_surface.append(value)
    surface_macro = float(np.mean(active_surface)) if active_surface else 1.0

    prior_coverage = np.asarray(
        [
            sum(
                len(node["face_ids"])
                for node in prior_nodes
                if node["type"] == motif_type
            ) / max(prior["num_faces"], 1)
            for motif_type in MOTIF_TYPES
        ]
    )
    generated_coverage = np.asarray(
        [
            sum(
                len(node["face_ids"])
                for node in generated_nodes
                if node["type"] == motif_type
            ) / max(generated["num_faces"], 1)
            for motif_type in MOTIF_TYPES
        ]
    )
    def surface_ratio(value):
        return np.asarray([np.mean(value == index) for index in range(6)])
    left = np.concatenate(
        [
            np.clip(prior["motif_counts"] / 10.0, 0, 1),
            np.clip(prior_coverage, 0, 1),
            np.clip(prior["relation_counts"] / max(prior["num_faces"], 1), 0, 1),
            np.clip(prior["relation_confidence"], 0, 1),
            surface_ratio(prior["surface"]),
            [min(prior["num_faces"] / 30.0, 1.0)],
        ]
    )
    right = np.concatenate(
        [
            np.clip(generated["motif_counts"] / 10.0, 0, 1),
            np.clip(generated_coverage, 0, 1),
            np.clip(generated["relation_counts"] / max(generated["num_faces"], 1), 0, 1),
            np.clip(generated["relation_confidence"], 0, 1),
            surface_ratio(generated["surface"]),
            [min(generated["num_faces"] / 30.0, 1.0)],
        ]
    )
    signature_similarity = float(np.clip(1.0 - np.mean(np.abs(left - right)), 0.0, 1.0))
    return {
        "motif": motif,
        "relation": relation,
        "surface_macro_f1": surface_macro,
        "surface_per_class_f1": per_surface,
        "signature_similarity": signature_similarity,
        "prior_signature": left.astype(float).tolist(),
        "generated_signature": right.astype(float).tolist(),
    }
