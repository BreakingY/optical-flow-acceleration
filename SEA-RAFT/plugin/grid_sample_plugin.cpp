#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <string>

using namespace nvinfer1;

// CUDA implementation
void launch_grid_sample(
    const float* input,
    const float* grid,
    float* output,
    int batch,
    int channels,
    int inputHeight,
    int inputWidth,
    int outputHeight,
    int outputWidth,
    cudaStream_t stream);


// ============================================================
// GridSample Plugin
//
// input : [N, C, H, W]       NCHW
// grid  : [N, Hout, Wout, 2] NHW2
// output: [N, C, Hout, Wout] NCHW
//
// Fixed ONNX attributes:
//   mode          = bilinear
//   padding_mode  = zeros
//   align_corners = 1
// ============================================================

class GridSamplePlugin : public IPluginV2DynamicExt
{
public:
    GridSamplePlugin() = default;

    GridSamplePlugin(const void* serialData, size_t serialLength)
    {
        // No serialized parameters.
    }

    const char* getPluginType() const noexcept override
    {
        return "GridSample";
    }

    const char* getPluginVersion() const noexcept override
    {
        return "1";
    }

    int getNbOutputs() const noexcept override
    {
        return 1;
    }

    DimsExprs getOutputDimensions(
        int outputIndex,
        const DimsExprs* inputs,
        int nbInputs,
        IExprBuilder& exprBuilder) noexcept override
    {
        // input: [N, C, H, W]
        // grid:  [N, Hout, Wout, 2]
        // output:[N, C, Hout, Wout]

        DimsExprs output{};
        output.nbDims = 4;

        output.d[0] = inputs[0].d[0];
        output.d[1] = inputs[0].d[1];
        output.d[2] = inputs[1].d[1];
        output.d[3] = inputs[1].d[2];

        return output;
    }

    size_t getWorkspaceSize(
        const PluginTensorDesc* inputs,
        int nbInputs,
        const PluginTensorDesc* outputs,
        int nbOutputs) const noexcept override
    {
        return 0;
    }

    int enqueue(
        const PluginTensorDesc* inputDesc,
        const PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override
    {
        const int batch = inputDesc[0].dims.d[0];
        const int channels = inputDesc[0].dims.d[1];
        const int inputHeight = inputDesc[0].dims.d[2];
        const int inputWidth = inputDesc[0].dims.d[3];

        const int outputHeight = inputDesc[1].dims.d[1];
        const int outputWidth = inputDesc[1].dims.d[2];

        launch_grid_sample(
            static_cast<const float*>(inputs[0]),
            static_cast<const float*>(inputs[1]),
            static_cast<float*>(outputs[0]),
            batch,
            channels,
            inputHeight,
            inputWidth,
            outputHeight,
            outputWidth,
            stream);

        return 0;
    }

    int initialize() noexcept override
    {
        return 0;
    }

    void terminate() noexcept override
    {
    }

    size_t getSerializationSize() const noexcept override
    {
        return 0;
    }

    void serialize(void* buffer) const noexcept override
    {
    }

    void destroy() noexcept override
    {
        delete this;
    }

    IPluginV2DynamicExt* clone() const noexcept override
    {
        auto* plugin = new GridSamplePlugin();
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }

    void setPluginNamespace(const char* pluginNamespace) noexcept override
    {
        mNamespace = pluginNamespace ? pluginNamespace : "";
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

    DataType getOutputDataType(
        int index,
        const DataType* inputTypes,
        int nbInputs) const noexcept override
    {
        return inputTypes[0];
    }

    void configurePlugin(
        const DynamicPluginTensorDesc* inputs,
        int nbInputs,
        const DynamicPluginTensorDesc* outputs,
        int nbOutputs) noexcept override
    {
    }

    bool supportsFormatCombination(
        int pos,
        const PluginTensorDesc* inOut,
        int nbInputs,
        int nbOutputs) noexcept override
    {
        // Only FP32 + LINEAR is supported.
        return inOut[pos].type == DataType::kFLOAT &&
               inOut[pos].format == TensorFormat::kLINEAR;
    }

private:
    std::string mNamespace;
};


// ============================================================
// GridSample Plugin Creator
// ============================================================

class GridSamplePluginCreator : public IPluginCreator
{
public:
    GridSamplePluginCreator()
    {
        // The ONNX node contains these attributes. The implementation
        // is intentionally specialized for their values.
        mFields[0] = PluginField(
            "align_corners", nullptr, PluginFieldType::kINT32, 1);
        mFields[1] = PluginField(
            "mode", nullptr, PluginFieldType::kCHAR, 1);
        mFields[2] = PluginField(
            "padding_mode", nullptr, PluginFieldType::kCHAR, 1);

        mFC.nbFields = 3;
        mFC.fields = mFields;
    }

    const char* getPluginName() const noexcept override
    {
        return "GridSample";
    }

    const char* getPluginVersion() const noexcept override
    {
        return "1";
    }

    const PluginFieldCollection* getFieldNames() noexcept override
    {
        return &mFC;
    }

    IPluginV2* createPlugin(
        const char* name,
        const PluginFieldCollection* fc) noexcept override
    {
        // This plugin implements exactly:
        // bilinear + zeros + align_corners=1.
        // The ONNX graph being converted has these fixed attributes.
        return new GridSamplePlugin();
    }

    IPluginV2* deserializePlugin(
        const char* name,
        const void* serialData,
        size_t serialLength) noexcept override
    {
        return new GridSamplePlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char* pluginNamespace) noexcept override
    {
        mNamespace = pluginNamespace ? pluginNamespace : "";
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace;
    PluginField mFields[3]{};
    PluginFieldCollection mFC{};
};


// Register creator. Keep name/version/namespace consistent with
// the ONNX GridSample fallback-plugin lookup.
REGISTER_TENSORRT_PLUGIN(GridSamplePluginCreator);
