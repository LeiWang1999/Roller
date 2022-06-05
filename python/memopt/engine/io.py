from typing import Dict, List, Tuple
from memopt.graph import OutputNode, IRNode, Node
from .engine import FusionGroup
import json

# internal name for debug
def get_node_name(id, op_type):
    return "_".join([op_type, str(id)])

def get_id_by_name(name):
    return int(name.split("_")[-1])

def load_model(fname: str) -> List[Node]:
    with open(fname) as f:
        a = json.load(f)

    node_map = {}
    ordered_nodes = []
    for node_id, ir, op_type, inputs in a:
        input_list = []
        if op_type == "Pad":
            inputs.pop(-1)
        for src_node, src_id in inputs:
            if src_node not in node_map:
                input_list.append(None)
            else:
                input_list.append([node_map[src_node], src_id])
        if op_type == "Result":
            node = OutputNode(*input_list[0])
        else:
            node = IRNode(input_list, ir, get_node_name(node_id, op_type))
            if op_type == "Softmax":
                node.add_tag("skip")
        node_map[node_id] = node
        ordered_nodes.append(node)
    return ordered_nodes, node_map

def dump(fusion_groups: List[FusionGroup]):
    obj = []
    for group in fusion_groups:
        group_desc = {}
        node_names = [node.name for node in group.nodes]
        group_desc["nodes"] = [get_id_by_name(name) for name in node_names]
        group_desc["group_id"] = group.group_id
        if group.cpresult is not None:
            cpresult = group.cpresult
            group_desc["code"] = cpresult.code
            group_desc["block_size"] = cpresult.block_size
            group_desc["grid_size"] = cpresult.grid_size
            group_desc["latency"] = cpresult.latency
            group_desc["input_desc"] = [[get_id_by_name(name), id] for name, id in cpresult.input_desc]
            group_desc["output_desc"] = [[get_id_by_name(name), id] for name, id in cpresult.output_desc]
        obj.append(group_desc)
    return obj

def save_results(fusion_groups: List[FusionGroup], fname: str):
    obj = dump(fusion_groups)
    with open(fname, "w") as f:
        json.dump(obj, f, indent=2)
    return None