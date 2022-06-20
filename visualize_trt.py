import argparse
import re
from typing import NamedTuple, List, Optional, Dict, Any, Tuple

import pydot
import json

# 该脚本用于生成tensorrt engine的layer可视化图
# 借鉴于https://github.com/jerryzh168/pytorch/blob/fb09fd4ab4ba618db148f9dfc035be589efb9355/torch/fx/experimental/fx2trt/tools/engine_layer_visualize.py
# 不同之处在于本脚本面向trt inspector输出的详细profile信息，不支持trtexec.因此完全使用profile的详细数据生成，不受log打印数据限制 

style = {
    "shape": "record",
    "fillcolor": "Salmon",
    "style": '"filled,rounded"',
    "fontcolor": "#000000",
}

class LayerInfo(NamedTuple):
    kernel_name: str
    layer_name: str
    tactic: str
    input_names: Optional[List[str]]
    input_types: Optional[List[str]]
    output_names: Optional[List[str]]
    output_types: Optional[List[str]]
    time: str

    @classmethod
    def from_string(cls, string, tactic_names, layer_times=None):
        input_names = []
        input_types = []
        kernel_name, layer_name, tactic, inputs, output_name, output_type = re.findall(
            "Layer\\((.+)\\): (.+), Tactic: (-?\\d+), (.+)? -> (.+)\\[(.+)\\]", string
        )[0]

        if kernel_name != "Constant":
            inputs = re.findall("[, ]*(.+?)\\[([Half|Float|Int8]+\\(\\d[,\\d]*\\))\\]", inputs)
            for input_name, input_type in inputs:
                input_names.append(input_name)
                input_types.append(input_type)

            if layer_name in tactic_names:
                kernel_name = tactic_names[layer_name]
        else:
            input_names = input_types = None  # type:ignore[assignment]

        return cls(
            kernel_name,
            layer_name,
            tactic,
            input_names,
            input_types,
            output_name,
            output_type,
            layer_times[layer_name] if layer_times else "NA",
        )

    @classmethod
    def get_layer(cls, layer_dic, layer_times=None):
        input_names = []
        input_types = []
        output_names = []
        output_types = []
        kernel_name = layer_dic["LayerType"]
        layer_name = layer_dic["Name"].replace(":", "-")
        tactic = layer_dic["TacticValue"]
        for input_dict in layer_dic["Inputs"]:
            input_names.append(input_dict["Name"])
            input_types.append(input_dict["Format/Datatype"])
        for output_dict in layer_dic["Outputs"]:
            output_names.append(output_dict["Name"])
            output_types.append(output_dict["Format/Datatype"])

        return cls(
            kernel_name,
            layer_name,
            tactic,
            input_names,
            input_types,
            output_names,
            output_types,
            layer_times[layer_name] if layer_times else "NA",
        )



def build_node(layer):
    layer_name = layer.layer_name.replace("|", "\\|")
    label = f"{{{layer_name}|kernel: {layer.kernel_name}\\l|tactic: {layer.tactic}\\l|time: {layer.time}\\l}}"
    label = label.replace(">", "\\>")
    return pydot.Node(layer.layer_name, label=label, **style)


def build_edge(layer, graph, output_name2node, layer_name2node):
    if layer.input_names is None:
        return
    print("=======")
    print(layer.layer_name)
    print("=======")
    for input_name, input_type in zip(layer.input_names, layer.input_types):
        print(input_name)
        if input_name in output_name2node:
            from_node = output_name2node[input_name]
        else:
            from_node = input_name

        edge_name = input_name.replace(">", "\\>")
        graph.add_edge(
            pydot.Edge(
                from_node,
                layer_name2node[layer.layer_name],
                label=f"{edge_name}\\l{input_type}\\l",
            )
        )

def main():
    f = open("/backup/wfs/dasou/infer/nlg/base_mp1/engine_info.json")
    layers = []
    data = json.load(f)
    for layer_dict in data["Layers"]:
        layers.append(LayerInfo.get_layer(layer_dic=layer_dict))



    output_name2node = {}
    layer_name2node = {}
    dot_graph = pydot.Dot("Layer Graph")

    for layer in layers:
        node = build_node(layer)
        dot_graph.add_node(node)
        for output_name in layer.output_names:
            output_name2node[output_name] = node
        layer_name2node[layer.layer_name] = node

    for layer in layers:
        build_edge(layer, dot_graph, output_name2node, layer_name2node)

    dot_graph.write_pdf("trt_engine.pdf")

    # dot_graph.write_raw(f"EngineLayers.dot")

    # graphs = pydot.graph_from_dot_file("EngineLayers.dot")
    # graph = graphs[0]
    # graph.write_png("trt_engine.png")

if __name__ == "__main__":
    main()