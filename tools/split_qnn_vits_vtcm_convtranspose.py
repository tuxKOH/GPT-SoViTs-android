#!/usr/bin/env python3
"""Split a static stride-2 one-dimensional VITS ConvTranspose along time.

For the final VITS upsampler (kernel=2, stride=2), input windows have no halo, so slicing
the input and concatenating the independently computed output chunks is numerically exact.
This is conversion-only and exists to keep each SM8750 HTP operation below its VTCM limit.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--node", default="/dec/ups.4/ConvTranspose")
    p.add_argument("--chunks", type=int, default=16)
    args = p.parse_args()
    model = onnx.load(str(args.source), load_external_data=True)
    inferred = onnx.shape_inference.infer_shapes(model)
    node = next((n for n in model.graph.node if n.name == args.node), None)
    if node is None or node.op_type != "ConvTranspose": raise ValueError("target ConvTranspose missing")
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    if attrs.get("strides") != [2] or attrs.get("kernel_shape") != [2] or attrs.get("pads", [0,0]) != [0,0]:
        raise ValueError("only kernel=2 stride=2 no-padding ConvTranspose is supported")
    values = {v.name: v for v in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]}
    shape = [d.dim_value for d in values[node.input[0]].type.tensor_type.shape.dim]
    # ONNX shape inference leaves the fixed batch as zero for this exported graph.
    if shape and shape[0] == 0: shape[0] = 1
    if len(shape) != 3 or not all(shape): raise ValueError(f"input shape is not static: {shape}")
    length = shape[2]
    if args.chunks < 2 or args.chunks > length: raise ValueError("invalid chunk count")
    index = next(i for i,n in enumerate(model.graph.node) if n is node)
    prefix = f"gsv_vtcm_deconv_split_{index}"
    axes = f"{prefix}_axes"; steps = f"{prefix}_steps"
    model.graph.initializer.extend([
        numpy_helper.from_array(np.array([2], dtype=np.int64), axes),
        numpy_helper.from_array(np.array([1], dtype=np.int64), steps),
    ])
    replacement=[]; outputs=[]
    for chunk in range(args.chunks):
        start = length * chunk // args.chunks; end = length * (chunk+1) // args.chunks
        sn=f"{prefix}_{chunk}_start"; en=f"{prefix}_{chunk}_end"; sliced=f"{prefix}_{chunk}_input"; out=f"{prefix}_{chunk}_output"
        model.graph.initializer.extend([
            numpy_helper.from_array(np.array([start], dtype=np.int64), sn),
            numpy_helper.from_array(np.array([end], dtype=np.int64), en),
        ])
        replacement.append(helper.make_node("Slice", [node.input[0],sn,en,axes,steps], [sliced], name=f"{node.name}/VtcmSlice{chunk}"))
        replacement.append(helper.make_node("ConvTranspose", [sliced,*node.input[1:]], [out], name=f"{node.name}/VtcmConvTranspose{chunk}", **attrs))
        outputs.append(out)
    replacement.append(helper.make_node("Concat", outputs, list(node.output), name=f"{node.name}/VtcmConcat", axis=2))
    model.graph.node.remove(node)
    for offset,value in enumerate(replacement): model.graph.node.insert(index+offset,value)
    model.metadata_props.add(key=f"gsv.qnn.vtcm_deconv_split.{index}", value=f"{node.name};time={length};chunks={args.chunks}")
    args.output.parent.mkdir(parents=True,exist_ok=True); onnx.save(model,str(args.output))
    print(f"split {node.name}: time={length}, chunks={args.chunks} -> {args.output}")
if __name__ == "__main__": main()
