# ONNX
* 基于`https://github.com/ibaiGorordo/NeuFlow_v2-Pytorch-Inference`导出onnx
# TensorRT
* TensorRT版本：8.4、8.5、10.4

1. 编译插件(只有TensorRT8.4需要，TensorRT8.5和TensorRT10.4也需要编译，但是并没有真正使用-仅仅是链接动态库，也可以在Makefile_trt中把-lgrid_sample_plugin去掉)
* cd plugin
* mkdir build && cd build
* export PATH=$PATH:/usr/local/cuda/bin && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
* cmake -DTRT_PATH=<path_to_TensorRT> ..

2. 模型转换
* TensorRT8.4
    * trtexec --onnx=neuflow_mixed.onnx --saveEngine=neuflow_mixed.engine --plugins=./plugin/build/libgrid_sample_plugin.so --fp16 --noTF32
* TensorRT8.5/TensorRT10.4
    * trtexec --onnx=neuflow_mixed.onnx --saveEngine=neuflow_mixed.engine --fp16

3. 推理
* make -f Makefile_trt TRT_VERSION=<TRT_10;TRT_84_85> TRT_PATH=</PATH/TensorRT-10.4.0.26;PATH/TensorRT-8.5.1.7;PATH/TensorRT-8.4.1.5>
* opencv路径(替换为实际的路径)：`export LD_LIBRARY_PATH=/usr/local/opencvgpu/lib/`
* export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/plugin/build/
* ./infer_trt neuflow_mixed.engine ../video/test.mp4 video && ./infer_cann neuflow_mixed.engine ../images picture

# Ascend
* cann版本：8.2.RC1

1. 模型转换
* atc --model=./neuflow_mixed.onnx --framework=5 --input_shape="input1:1,3,480,640;input2:1,3,480,640 " --insert_op_conf=./insert_op.cfg --output=neuflow_mixed --soc_version=Ascend310P3 --precision_mode_v2=mixed_float16

2. 推理
* make -f Makefile_cann
* ./infer_cann neuflow_mixed.om ../video/test.mp4 video && ./infer_cann neuflow_mixed.om ../images picture