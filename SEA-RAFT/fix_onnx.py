
import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

INPUT_SHAPE = (1, 3, 432, 960)

model = onnx.load("raft.onnx")

inputs = {
    "image1": np.random.rand(*INPUT_SHAPE).astype(np.float32),
    "image2": np.random.rand(*INPUT_SHAPE).astype(np.float32)
}


# 1. 替换 Mod 节点
new_nodes = []

for node in model.graph.node:
    if node.op_type == "Mod":
        print(f"Replace Mod: {node.name}")

        zero = numpy_helper.from_array(
            np.array(0, dtype=np.int64),
            name=node.output[0] + "_value"
        )

        new_nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=list(node.output),
                value=zero,
                name=node.name + "_constant"
            )
        )
    else:
        new_nodes.append(node)

model.graph.ClearField("node")
model.graph.node.extend(new_nodes)


# 2. 检查所有 Pad 节点
pad_values = {}

for node in list(model.graph.node):
    if node.op_type != "Pad":
        continue

    pads_name = node.input[1]

    debug_model = onnx.load_from_string(
        model.SerializeToString()
    )

    debug_model.graph.output.append(
        helper.make_tensor_value_info(
            pads_name,
            onnx.TensorProto.INT64,
            None
        )
    )

    session = ort.InferenceSession(
        debug_model.SerializeToString(),
        providers=["CPUExecutionProvider"]
    )

    pads = session.run([pads_name], inputs)[0]
    pad_values[node.name] = pads

    print(f"Pad: {node.name}, values: {pads}")


# 3. 删除 Padding 全为 0 的 Pad
replacements = {}

for node in model.graph.node:
    if node.op_type == "Pad" and node.name in pad_values:
        if np.all(pad_values[node.name] == 0):
            replacements[node.output[0]] = node.input[0]
            print(f"Remove Pad: {node.name}")


# 4. 删除 Identity
initializers = {
    tensor.name for tensor in model.graph.initializer
}

for node in model.graph.node:
    if node.op_type == "Identity":
        source = node.input[0]

        # 仅删除输入为 initializer 的 Identity
        if source in initializers:
            replacements[node.output[0]] = source
            print(f"Remove Identity: {node.name}")


# 5. 统一替换节点引用
def resolve(name):
    visited = set()

    while name in replacements:
        if name in visited:
            raise RuntimeError(f"Replacement cycle detected: {name}")

        visited.add(name)
        name = replacements[name]

    return name


for node in model.graph.node:
    for i, name in enumerate(node.input):
        node.input[i] = resolve(name)

for output in model.graph.output:
    output.name = resolve(output.name)


# 6. 删除已经替换的 Pad 和 Identity
kept_nodes = []

for node in model.graph.node:
    if node.op_type == "Pad" and node.output[0] in replacements:
        continue

    if node.op_type == "Identity" and node.output[0] in replacements:
        continue

    kept_nodes.append(node)

model.graph.ClearField("node")
model.graph.node.extend(kept_nodes)


# 7. 检查并保存
onnx.checker.check_model(model)
onnx.save(model, "raft_fixed.onnx")

print("Saved raft_fixed.onnx")
