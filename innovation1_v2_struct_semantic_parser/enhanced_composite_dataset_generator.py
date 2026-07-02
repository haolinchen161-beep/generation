# -*- coding: utf-8 -*-
"""
enhanced_composite_dataset_generator.py

Generate enhanced aviation-composite thin-wall/stiffened B-Rep samples.

Outputs under --outdir:
  enhanced_dataset/*.step, *.stl, *.json
  reports/enhanced_manifest.csv
  reports/enhanced_dataset_report.txt

Run in your project:
  cd /d "F:\\开题答辩\\a中期答辩专用\\DTGBrepGen-master"
  "F:/pytorch_cuda12/python.exe" enhanced_composite_dataset_generator.py ^
    --outdir innovation1_v2_struct_semantic_parser/outputs ^
    --num_per_type 50 ^
    --seed 42

Notes:
  - Requires pythonOCC / OCC.
  - JSON configuration_graph is procedural_Gc.
  - procedural_face_groups keep face_ids empty; fill/alignment should be done by the later B-Rep weak semantic parser.
"""

import argparse
import csv
import json
import math
import os
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from OCC.Core.gp import gp_Pnt, gp_Vec
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_MakePolygon
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.StlAPI import StlAPI_Writer
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    OCC_OK = True
except Exception as _e:
    OCC_OK = False
    OCC_ERR = repr(_e)

PARAMETER_KEYS = [
    "length", "width", "thickness", "height", "flange_width",
    "rib_width", "rib_height", "rib_count", "fillet_radius",
    "hole_radius", "hole_width", "hole_height", "hole_count",
    "taper_ratio", "curvature_radius", "runout_length", "notch_depth",
]

PART_TYPES = [
    "panel_with_circular_cutout",
    "panel_with_rectangular_cutout",
    "stiffened_panel_with_cutout",
    "multi_stiffened_panel",
    "asymmetric_stiffened_panel",
    "tapered_c_channel",
    "tapered_hat_stiffener",
    "curved_panel",
    "curved_stiffened_panel",
    "stiffener_runout_panel",
]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Dict[str, Any]):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict[str, Any]], headers: List[str]):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def box(x, y, z, dx, dy, dz):
    return BRepPrimAPI_MakeBox(gp_Pnt(float(x), float(y), float(z)), float(dx), float(dy), float(dz)).Shape()


def cylinder_z(cx, cy, z0, radius, height):
    cyl = BRepPrimAPI_MakeCylinder(float(radius), float(height)).Shape()
    from OCC.Core.gp import gp_Trsf
    tr = gp_Trsf(); tr.SetTranslation(gp_Vec(float(cx), float(cy), float(z0)))
    return BRepBuilderAPI_Transform(cyl, tr, True).Shape()


def fuse(a, b):
    op = BRepAlgoAPI_Fuse(a, b); op.Build()
    if not op.IsDone():
        raise RuntimeError("Fuse failed")
    return op.Shape()


def fuse_many(shapes):
    s = shapes[0]
    for t in shapes[1:]:
        s = fuse(s, t)
    return s


def cut(a, b):
    op = BRepAlgoAPI_Cut(a, b); op.Build()
    if not op.IsDone():
        raise RuntimeError("Cut failed")
    return op.Shape()


def export_shape(shape, step_path: Path, stl_path: Path, mesh_deflection=0.8):
    ensure_dir(step_path.parent); ensure_dir(stl_path.parent)
    sw = STEPControl_Writer(); sw.Transfer(shape, STEPControl_AsIs)
    status = sw.Write(str(step_path))
    if status != 1:
        raise RuntimeError(f"STEP write failed, status={status}")
    BRepMesh_IncrementalMesh(shape, float(mesh_deflection))
    StlAPI_Writer().Write(shape, str(stl_path))


def bbox(x, y, z, dx, dy, dz):
    return [round(x, 4), round(y, 4), round(z, 4), round(x + dx, 4), round(y + dy, 4), round(z + dz, 4)]


def node(i, t):
    return {"id": i, "type": t}


def rel(s, t, r):
    return {"source": s, "target": t, "type": r}


def fg(i, t, b=None, component=None):
    return {
        "config_node": i,
        "node_type": t,
        "semantic_component": component or i,
        "bbox_hint": b or [],
        "face_ids": [],
        "source": "procedural_expected_group_without_face_ids",
    }


def params(pt: str, rng: random.Random):
    p = {k: 0.0 for k in PARAMETER_KEYS}
    p["length"] = round(rng.uniform(180, 480), 3)
    p["width"] = round(rng.uniform(80, 240), 3)
    p["thickness"] = round(rng.uniform(2.0, 4.0), 3)
    p["height"] = round(rng.uniform(35, 120), 3)
    p["flange_width"] = round(rng.uniform(18, 55), 3)
    p["rib_width"] = round(rng.uniform(8, 18), 3)
    p["rib_height"] = round(rng.uniform(15, 55), 3)
    p["rib_count"] = rng.choice([1, 2, 3, 4, 5])
    p["fillet_radius"] = round(rng.uniform(3, 8), 3)
    p["hole_radius"] = round(rng.uniform(8, min(35, p["width"] * 0.2)), 3)
    p["hole_width"] = round(rng.uniform(18, min(70, p["length"] * 0.18)), 3)
    p["hole_height"] = round(rng.uniform(12, min(55, p["width"] * 0.35)), 3)
    p["hole_count"] = rng.choice([1, 2, 3])
    p["taper_ratio"] = round(rng.uniform(0.70, 1.25), 3)
    p["curvature_radius"] = round(rng.uniform(250, 800), 3)
    p["runout_length"] = round(rng.uniform(35, 120), 3)
    p["notch_depth"] = round(rng.uniform(5, 25), 3)
    if pt == "multi_stiffened_panel": p["rib_count"] = rng.choice([2, 3, 4, 5])
    if pt == "asymmetric_stiffened_panel": p["rib_count"] = rng.choice([2, 3, 4])
    if pt == "stiffener_runout_panel": p["rib_count"] = 1
    if "rectangular_cutout" in pt: p["hole_radius"] = 0.0
    if "circular_cutout" in pt: p["hole_width"] = p["hole_height"] = 0.0
    return p


def closed_wire(points_xy: List[Tuple[float, float]], z=0.0):
    poly = BRepBuilderAPI_MakePolygon()
    for x, y in points_xy:
        poly.Add(gp_Pnt(float(x), float(y), float(z)))
    poly.Close()
    return poly.Wire()


def loft(points0, points1, length):
    l = BRepOffsetAPI_ThruSections(True, True, 1e-6)
    l.AddWire(closed_wire(points0, 0.0))
    l.AddWire(closed_wire(points1, length))
    l.Build()
    if not l.IsDone():
        raise RuntimeError("Loft failed")
    return l.Shape()

# ------------------------------------------------------------
# Geometry builders
# ------------------------------------------------------------

def panel_circular(p, rng):
    L,W,t = p["length"],p["width"],p["thickness"]
    s = box(0,0,0,L,W,t)
    nodes=[node("panel_0","panel")]; rels=[]; groups=[fg("panel_0","panel",bbox(0,0,0,L,W,t),"base_panel")]
    comps=[{"id":"panel_0","type":"panel","bbox_hint":bbox(0,0,0,L,W,t)}]
    hc=int(p["hole_count"]); r=p["hole_radius"]
    for i in range(hc):
        cx=L*(i+1)/(hc+1); cy=W*0.5+rng.uniform(-0.1*W,0.1*W)
        s=cut(s,cylinder_z(cx,cy,-t,r,3*t))
        hid=f"hole_{i}"; nodes.append(node(hid,"hole")); rels.append(rel(hid,"panel_0","hole_of"))
        groups.append(fg(hid,"hole",bbox(cx-r,cy-r,-t,2*r,2*r,3*t),f"circular_hole_{i}"))
        comps.append({"id":hid,"type":"hole","center":[round(cx,3),round(cy,3)],"radius":r})
    return s,nodes,rels,groups,comps,["hole","inner_loop","circular_cutout"]


def panel_rect(p, rng):
    L,W,t = p["length"],p["width"],p["thickness"]
    s=box(0,0,0,L,W,t)
    nodes=[node("panel_0","panel")]; rels=[]; groups=[fg("panel_0","panel",bbox(0,0,0,L,W,t),"base_panel")]
    comps=[{"id":"panel_0","type":"panel","bbox_hint":bbox(0,0,0,L,W,t)}]
    hc=max(1,min(2,int(p["hole_count"]))); hw=p["hole_width"]; hh=p["hole_height"]
    for i in range(hc):
        cx=L*(i+1)/(hc+1); cy=W*0.5+rng.uniform(-0.08*W,0.08*W)
        x0=cx-hw/2; y0=cy-hh/2
        s=cut(s,box(x0,y0,-t,hw,hh,3*t))
        cid=f"cutout_{i}"; nodes.append(node(cid,"cutout")); rels.append(rel(cid,"panel_0","cutout_of"))
        groups.append(fg(cid,"cutout",bbox(x0,y0,-t,hw,hh,3*t),f"rectangular_cutout_{i}"))
        comps.append({"id":cid,"type":"cutout","bbox_hint":bbox(x0,y0,-t,hw,hh,3*t)})
    return s,nodes,rels,groups,comps,["hole","inner_loop","rectangular_cutout"]


def stiffened_cutout(p, rng):
    shape,nodes,rels,groups,comps,mechs = panel_circular(p, rng)
    L,W,t=p["length"],p["width"],p["thickness"]; rw=p["rib_width"]; rh=p["rib_height"]
    rib_count=max(2,min(5,int(p["rib_count"])))
    shapes=[shape]
    for i in range(rib_count):
        y=W*(i+1)/(rib_count+1)
        shapes.append(box(0,y-rw/2,t,L,rw,rh))
        sid=f"stiffener_{i}"; nodes.append(node(sid,"stiffener")); rels.append(rel(sid,"panel_0","attached_to"))
        groups.append(fg(sid,"stiffener",bbox(0,y-rw/2,t,L,rw,rh),f"longitudinal_stiffener_{i}"))
        comps.append({"id":sid,"type":"stiffener","bbox_hint":bbox(0,y-rw/2,t,L,rw,rh)})
    return fuse_many(shapes),nodes,rels,groups,comps,mechs+["stiffener","multi_body_fuse"]


def multi_stiffened(p, rng, asymmetric=False):
    L,W,t=p["length"],p["width"],p["thickness"]; rw=p["rib_width"]; rh=p["rib_height"]
    rc=max(2,min(5,int(p["rib_count"])))
    shapes=[box(0,0,0,L,W,t)]
    nodes=[node("panel_0","panel")]; rels=[]; groups=[fg("panel_0","panel",bbox(0,0,0,L,W,t),"base_panel")]
    comps=[{"id":"panel_0","type":"panel","bbox_hint":bbox(0,0,0,L,W,t)}]
    ys=sorted([rng.uniform(0.15*W,0.85*W) for _ in range(rc)]) if asymmetric else [W*(i+1)/(rc+1) for i in range(rc)]
    for i,y in enumerate(ys):
        h = rh*(rng.uniform(0.65,1.35) if asymmetric else 1.0)
        shapes.append(box(0,y-rw/2,t,L,rw,h))
        sid=f"stiffener_{i}"; nodes.append(node(sid,"stiffener")); rels.append(rel(sid,"panel_0","attached_to"))
        if i>0: rels.append(rel(f"stiffener_{i-1}",sid,"parallel_to"))
        groups.append(fg(sid,"stiffener",bbox(0,y-rw/2,t,L,rw,h),sid))
        comps.append({"id":sid,"type":"stiffener","bbox_hint":bbox(0,y-rw/2,t,L,rw,h)})
    mechs=["multi_stiffener","variable_node_count"]
    if asymmetric: mechs += ["asymmetric_layout","nonuniform_stiffener_height"]
    return fuse_many(shapes),nodes,rels,groups,comps,mechs


def c_points(W,H,t):
    return [(0,0),(W,0),(W,H),(0,H),(0,H-t),(W-t,H-t),(W-t,t),(0,t)]


def hat_points(W,H,t,F):
    x0=0; x1=F; x2=F+0.25*W; x3=F+0.75*W; x4=F+W; x5=F+W+F
    return [(x0,0),(x1,0),(x2,H),(x3,H),(x4,0),(x5,0),(x5,t),(x4+0.2*t,t),(x3-0.2*t,H-t),(x2+0.2*t,H-t),(x1-0.2*t,t),(x0,t)]


def tapered_c(p,rng):
    L,W,H,t,ta=p["length"],p["width"],p["height"],p["thickness"],p["taper_ratio"]
    s=loft(c_points(W,H,t),c_points(W*ta,H*(0.85+0.3*rng.random()),t),L)
    nodes=[node("web_0","web"),node("flange_0","flange"),node("flange_1","flange"),node("transition_0","transition"),node("transition_1","transition")]
    rels=[rel("flange_0","web_0","attached_to"),rel("flange_1","web_0","attached_to"),rel("transition_0","flange_0","smooth_connected"),rel("transition_1","flange_1","smooth_connected")]
    groups=[fg(n["id"],n["type"],component=n["id"]) for n in nodes]
    return s,nodes,rels,groups,[{"id":n["id"],"type":n["type"]} for n in nodes],["taper","non_prismatic","open_section"]


def tapered_hat(p,rng):
    L,W,H,t,F,ta=p["length"],p["width"],p["height"],p["thickness"],p["flange_width"],p["taper_ratio"]
    s=loft(hat_points(W,H,t,F),hat_points(W*ta,H*(0.85+0.3*rng.random()),t,F*(0.85+0.3*rng.random())),L)
    nodes=[node("cap_0","panel"),node("web_0","web"),node("web_1","web"),node("flange_0","flange"),node("flange_1","flange"),node("transition_0","transition"),node("transition_1","transition")]
    rels=[rel("web_0","cap_0","attached_to"),rel("web_1","cap_0","attached_to"),rel("flange_0","web_0","attached_to"),rel("flange_1","web_1","attached_to"),rel("web_0","web_1","symmetric_to")]
    groups=[fg(n["id"],n["type"],component=n["id"]) for n in nodes]
    return s,nodes,rels,groups,[{"id":n["id"],"type":n["type"]} for n in nodes],["taper","hat_section","non_prismatic"]


def curved_panel(p,rng,stiff=False):
    L,W,t,R=p["length"],p["width"],p["thickness"],p["curvature_radius"]
    nseg=16; shapes=[]
    for i in range(nseg):
        x0=L*i/nseg; x1=L*(i+1)/nseg; xm=(x0+x1)/2; relx=xm-L/2
        z=R-math.sqrt(max(R*R-relx*relx,1.0))
        shapes.append(box(x0,0,z,x1-x0,W,t))
    nodes=[node("panel_0","panel")]; rels=[]; groups=[fg("panel_0","panel",bbox(0,0,0,L,W,t+L*L/(8*R)),"curved_panel")]
    comps=[{"id":"panel_0","type":"panel","curvature_radius":R}]
    mechs=["curved_surface","segmented_curvature_approximation"]
    if stiff:
        rw=p["rib_width"]; rh=p["rib_height"]; rc=max(2,min(4,int(p["rib_count"])))
        for j in range(rc):
            y=W*(j+1)/(rc+1)
            for i in range(nseg):
                x0=L*i/nseg; x1=L*(i+1)/nseg; xm=(x0+x1)/2; relx=xm-L/2
                z=R-math.sqrt(max(R*R-relx*relx,1.0))+t
                shapes.append(box(x0,y-rw/2,z,x1-x0,rw,rh))
            sid=f"stiffener_{j}"; nodes.append(node(sid,"stiffener")); rels.append(rel(sid,"panel_0","attached_to"))
            groups.append(fg(sid,"stiffener",component=f"curved_stiffener_{j}")); comps.append({"id":sid,"type":"stiffener","rib_y":y})
        mechs += ["curved_stiffener","multi_stiffener"]
    return fuse_many(shapes),nodes,rels,groups,comps,mechs


def runout_panel(p,rng):
    L,W,t=p["length"],p["width"],p["thickness"]; rw=p["rib_width"]; rh=p["rib_height"]; ro=p["runout_length"]
    shapes=[box(0,0,0,L,W,t)]; y=W/2; nseg=14
    for i in range(nseg):
        x0=L*i/nseg; x1=L*(i+1)/nseg; xm=(x0+x1)/2
        factor=min(max(xm/max(ro,1),0),max((L-xm)/max(ro,1),0),1)
        h=max(0.8,rh*factor)
        shapes.append(box(x0,y-rw/2,t,x1-x0,rw,h))
    nodes=[node("panel_0","panel"),node("stiffener_0","stiffener"),node("runout_0","runout"),node("runout_1","runout")]
    rels=[rel("stiffener_0","panel_0","attached_to"),rel("runout_0","stiffener_0","runout_of"),rel("runout_1","stiffener_0","runout_of"),rel("runout_0","panel_0","attached_to"),rel("runout_1","panel_0","attached_to")]
    groups=[fg("panel_0","panel",bbox(0,0,0,L,W,t),"base_panel"),fg("stiffener_0","stiffener",bbox(ro,y-rw/2,t,max(L-2*ro,1),rw,rh),"main_stiffener"),fg("runout_0","runout",bbox(0,y-rw/2,t,ro,rw,rh),"left_runout"),fg("runout_1","runout",bbox(L-ro,y-rw/2,t,ro,rw,rh),"right_runout")]
    return fuse_many(shapes),nodes,rels,groups,[{"id":n["id"],"type":n["type"]} for n in nodes],["runout","local_transition","variable_stiffener_height"]

def build(pt,p,rng):
    if pt=="panel_with_circular_cutout": return panel_circular(p,rng)
    if pt=="panel_with_rectangular_cutout": return panel_rect(p,rng)
    if pt=="stiffened_panel_with_cutout": return stiffened_cutout(p,rng)
    if pt=="multi_stiffened_panel": return multi_stiffened(p,rng,False)
    if pt=="asymmetric_stiffened_panel": return multi_stiffened(p,rng,True)
    if pt=="tapered_c_channel": return tapered_c(p,rng)
    if pt=="tapered_hat_stiffener": return tapered_hat(p,rng)
    if pt=="curved_panel": return curved_panel(p,rng,False)
    if pt=="curved_stiffened_panel": return curved_panel(p,rng,True)
    if pt=="stiffener_runout_panel": return runout_panel(p,rng)
    raise ValueError(pt)


def write_sample(pt,idx,out_dataset,rng,mesh_deflection):
    uid=f"{pt}_{idx:06d}"; p=params(pt,rng)
    step=out_dataset/f"{uid}.step"; stl=out_dataset/f"{uid}.stl"; js=out_dataset/f"{uid}.json"
    row={"uid":uid,"part_type":pt,"step_file":step.name,"stl_file":stl.name,"json_file":js.name,"generation_status":"FAILED","error":"","num_nodes":0,"num_relations":0,"topology_mechanisms":""}
    try:
        shape,nodes,rels,groups,comps,mechs=build(pt,p,rng)
        export_shape(shape,step,stl,mesh_deflection)
        write_json(js,{
            "uid":uid,
            "part_type":pt,
            "source":"procedural_enhanced",
            "parameters":p,
            "configuration_graph":{"nodes":nodes,"relations":rels},
            "procedural_face_groups":groups,
            "procedural_components":comps,
            "topology_mechanisms":mechs,
            "semantic_label_note":"configuration_graph/procedural_face_groups are procedural supervision labels. face_ids are empty and should be inferred/aligned by the later B-Rep weak semantic parser."
        })
        row.update({"generation_status":"PASS","num_nodes":len(nodes),"num_relations":len(rels),"topology_mechanisms":"|".join(mechs)})
    except Exception as e:
        row["error"]=f"{type(e).__name__}: {e}"
        write_json(js,{"uid":uid,"part_type":pt,"source":"procedural_enhanced","parameters":p,"generation_status":"FAILED","error":row["error"],"traceback":traceback.format_exc()})
    return row


def report(rows,path,num_per_type):
    total=len(rows); succ=sum(1 for r in rows if r["generation_status"]=="PASS")
    by={}; mech={}; node_sum=0; rel_sum=0
    for r in rows:
        pt=r["part_type"]; by.setdefault(pt,{"total":0,"pass":0,"fail":0}); by[pt]["total"]+=1
        if r["generation_status"]=="PASS":
            by[pt]["pass"]+=1; node_sum+=int(r["num_nodes"]); rel_sum+=int(r["num_relations"])
            for m in r["topology_mechanisms"].split("|"):
                if m: mech[m]=mech.get(m,0)+1
        else: by[pt]["fail"]+=1
    lines=["="*78,"Enhanced Composite Dataset Generation Report","="*78,
           f"total_requested: {total}",f"num_per_type: {num_per_type}",f"success_step_stl_json: {succ}",f"failures: {total-succ}",
           f"avg_nodes_success_samples: {node_sum/max(succ,1):.3f}",f"avg_relations_success_samples: {rel_sum/max(succ,1):.3f}","","[By Part Type]"]
    for pt in sorted(by):
        d=by[pt]; lines.append(f"  {pt}: total={d['total']}, pass={d['pass']}, fail={d['fail']}")
    lines += ["","[Topology Mechanism Coverage]"]
    for k in sorted(mech): lines.append(f"  {k}: {mech[k]}")
    lines += ["","[Note]","  This enhanced dataset is still procedurally generated.","  procedural_Gc is supervision metadata, not automatically discovered engineering annotation.","  Later B-Rep weak semantic parser should infer inferred_Gc from STEP/PKL and compare it with procedural_Gc.","="*78]
    ensure_dir(path.parent); path.write_text("\n".join(lines),encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir",default="innovation1_v2_struct_semantic_parser/outputs")
    ap.add_argument("--num_per_type",type=int,default=50)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--types",default=",".join(PART_TYPES))
    ap.add_argument("--mesh_deflection",type=float,default=0.8)
    args=ap.parse_args()
    if not OCC_OK:
        raise RuntimeError("pythonOCC/OCC import failed. Run with project Python. Original error: "+OCC_ERR)
    rng=random.Random(args.seed)
    out_root=Path(args.outdir); data_dir=out_root/"enhanced_dataset"; rep_dir=out_root/"reports"; aux_dir=rep_dir/"auxiliary"
    ensure_dir(data_dir); ensure_dir(rep_dir); ensure_dir(aux_dir)
    types=[x.strip() for x in args.types.split(",") if x.strip()]
    for t in types:
        if t not in PART_TYPES: raise ValueError(f"Unknown type {t}. Available: {PART_TYPES}")
    rows=[]
    for pt in types:
        print(f"[Generate] {pt}: {args.num_per_type}")
        for i in range(1,args.num_per_type+1):
            r=write_sample(pt,i,data_dir,rng,args.mesh_deflection); rows.append(r)
            if r["generation_status"]!="PASS": print("  FAILED",r["uid"],r["error"])
    headers=["uid","part_type","step_file","stl_file","json_file","generation_status","error","num_nodes","num_relations","topology_mechanisms"]
    write_csv(aux_dir/"enhanced_manifest.csv",rows,headers)
    report(rows,rep_dir/"enhanced_dataset_report.txt",args.num_per_type)
    succ=sum(1 for r in rows if r["generation_status"]=="PASS")
    print("="*78); print(f"Enhanced dataset generation finished: {succ}/{len(rows)} PASS"); print(f"Dataset: {data_dir}"); print(f"Report : {rep_dir/'enhanced_dataset_report.txt'}"); print("="*78)


if __name__=="__main__":
    main()
