# ONNX
* 基于`https://github.com/Kololu777/onnx-sea-raft`导出onnx
# TensorRT(TODO)

# Ascend(TODO)
* atc --model=./raft.onnx --framework=5 --input_shape="image1:1,3,432,960;image2:1,3,432,960 " --insert_op_conf=./insert_op.cfg --output=sea-raft --soc_version=Ascend310P3 --precision_mode_v2=mixed_float16
