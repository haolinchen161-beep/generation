# -*- coding: utf-8 -*-
"""
程序名称：utils_io.py
程序功能：
    本程序提供文件读写、日志保存以及利用 OpenCascade 几何读回器进行最终 STEP 生成文件质量检验的辅助工具。
    在最终生成阶段 (generate_class / generate_uncond / generate_batch) 自动调用回读逻辑，
    验证重建的几何模型完整性。

主要模块功能：
    1. verify_occ_step: 利用 STEPControl_Reader 对生成的 STEP 文件进行读取验证，检查 Shape 是否非空或受损。
    2. verify_stl: 检查生成的 STL 表面网格模型文件尺寸与物理存在性。
    3. utils: 通用的 json、csv 辅助写盘函数。

使用方法：
    由 run_innovation2.py 在后评估、测试及模型生成阶段导入调用。
"""

import os
import json
import csv

# OpenCascade 用于回读测试的导入
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

def verify_occ_step(filepath):
    """
    使用 OpenCascade 的 STEP 读取接口，尝试载入 STEP 文件。
    检查读取状态是否正常，且几何拓扑 Shape 是否非空。
    """
    if not os.path.exists(filepath):
        return "FILE_MISSING"
    try:
        reader = STEPControl_Reader()
        status = reader.ReadFile(filepath)
        if status == IFSelect_RetDone:
            reader.TransferRoots()
            shape = reader.OneShape()
            if shape.IsNull():
                return "NULL_SHAPE"
            return "SUCCESS"
        else:
            return f"READ_ERROR_CODE_{status}"
    except Exception as e:
        return f"EXCEPTION_{type(e).__name__}"

def verify_stl(filepath):
    """
    检查生成的 STL 表面网格文件是否成功保存且非空。
    """
    if not os.path.exists(filepath):
        return "FILE_MISSING"
    try:
        size = os.path.getsize(filepath)
        if size > 100:
            return "SUCCESS"
        return "EMPTY_OR_CORRUPT"
    except Exception as e:
        return f"EXCEPTION_{type(e).__name__}"

def write_json(data, filepath):
    """通用写 JSON 文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def write_csv(rows, filepath, headers):
    """通用写 CSV 报表"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
