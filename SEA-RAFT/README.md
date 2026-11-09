# ONNX
* 基于`https://github.com/Kololu777/onnx-sea-raft`导出onnx
* SEA-RAFT输入格式:RGB

# TensorRT
* TensorRT版本：8.4、8.5、10.4

1. 编译插件(只有TensorRT8.4需要，TensorRT8.5和TensorRT10.4也需要编译，但是并没有真正使用-仅仅是链接动态库，也可以在Makefile_trt中把-lgrid_sample_plugin去掉)
* cd plugin
* mkdir build && cd build
* export PATH=$PATH:/usr/local/cuda/bin && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
* cmake -DTRT_PATH=<path_to_TensorRT> ..

2. 模型转换
* python fix_onnx.py(只有TensorRT8.4需要)
* TensorRT8.4
    * trtexec --onnx=raft_fixed.onnx --saveEngine=sea-raft.engine --plugins=./plugin/build/libgrid_sample_plugin.so --fp16
* TensorRT8.5/TensorRT10.4
    * trtexec --onnx=raft.onnx --saveEngine=sea-raft.engine --fp16

3. 推理
* make -f Makefile_trt TRT_VERSION=<TRT_10;TRT_84_85> TRT_PATH=</PATH/TensorRT-10.4.0.26;PATH/TensorRT-8.5.1.7;PATH/TensorRT-8.4.1.5>
* opencv路径(替换为实际的路径)：`export LD_LIBRARY_PATH=/usr/local/opencvgpu/lib/`
* export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/plugin/build/
* ./infer_trt sea-raft.engine ../video/test.mp4 video && ./infer_cann sea-raft.engine ../images picture

# Ascend
* cann版本：8.2.RC1

1. 模型转换
* atc --model=./raft.onnx --framework=5 --input_shape="image1:1,3,432,960;image2:1,3,432,960 " --insert_op_conf=./insert_op.cfg --output=sea-raft --soc_version=Ascend310P3 --precision_mode_v2=mixed_float16

2. 推理
* make -f Makefile_cann
* ./infer_cann sea-raft_linux_aarch64.om ../video/test.mp4 video && ./infer_cann sea-raft_linux_aarch64.om ../images picture
