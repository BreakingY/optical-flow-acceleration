#include <cuda_runtime.h>

#include <cstddef>
#include <cmath>

namespace {

// ============================================================
// GridSample CUDA Kernel
//
// input : [N, C, H, W]       NCHW
// grid  : [N, Hout, Wout, 2] NHW2, coordinate order (x, y)
// output: [N, C, Hout, Wout] NCHW
//
// Semantics:
//   interpolation = bilinear
//   padding       = zeros
//   align_corners = true
//
// For align_corners=true:
//   ix = (grid_x + 1) * (W - 1) / 2
//   iy = (grid_y + 1) * (H - 1) / 2
//
// Each out-of-bound neighbor contributes zero independently.
// No extra validity-mask threshold is applied.
// ============================================================

__global__ void kernel_GridSample(
    const float* __restrict__ input,
    const float* __restrict__ grid,
    float* __restrict__ output,
    int channels,
    int inputHeight,
    int inputWidth,
    int outputHeight,
    int outputWidth,
    size_t totalElements)
{
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= totalElements) {
        return;
    }

    const size_t outputSpatialSize =
        static_cast<size_t>(outputHeight) * outputWidth;

    const size_t outputBatchSize =
        static_cast<size_t>(channels) * outputSpatialSize;

    const int batch =
        static_cast<int>(index / outputBatchSize);

    const size_t batchOffset =
        index % outputBatchSize;

    const int channel =
        static_cast<int>(batchOffset / outputSpatialSize);

    const size_t outputSpatialIndex =
        batchOffset % outputSpatialSize;

    const int outputY =
        static_cast<int>(outputSpatialIndex / outputWidth);

    const int outputX =
        static_cast<int>(outputSpatialIndex % outputWidth);

    // Grid is stored as [N, Hout, Wout, 2], with (x, y).
    const size_t gridOffset =
        (static_cast<size_t>(batch) * outputSpatialSize
         + outputSpatialIndex) * 2;

    const float gridX = grid[gridOffset + 0];
    const float gridY = grid[gridOffset + 1];

    // ONNX GridSample: align_corners = 1.
    const float sourceX =
        (gridX + 1.0f) *
        (static_cast<float>(inputWidth - 1) * 0.5f);

    const float sourceY =
        (gridY + 1.0f) *
        (static_cast<float>(inputHeight - 1) * 0.5f);

    // Avoid unsafe float-to-int conversion for extreme coordinates.
    // Outside this range all four bilinear neighbors are out of bounds.
    if (!isfinite(sourceX) || !isfinite(sourceY) ||
        sourceX < -1.0f || sourceX > static_cast<float>(inputWidth) ||
        sourceY < -1.0f || sourceY > static_cast<float>(inputHeight)) {
        output[index] = 0.0f;
        return;
    }

    const int x0 = static_cast<int>(floorf(sourceX));
    const int y0 = static_cast<int>(floorf(sourceY));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float wx = sourceX - static_cast<float>(x0);
    const float wy = sourceY - static_cast<float>(y0);

    const bool valid00 =
        x0 >= 0 && x0 < inputWidth &&
        y0 >= 0 && y0 < inputHeight;

    const bool valid01 =
        x1 >= 0 && x1 < inputWidth &&
        y0 >= 0 && y0 < inputHeight;

    const bool valid10 =
        x0 >= 0 && x0 < inputWidth &&
        y1 >= 0 && y1 < inputHeight;

    const bool valid11 =
        x1 >= 0 && x1 < inputWidth &&
        y1 >= 0 && y1 < inputHeight;

    const size_t inputSpatialSize =
        static_cast<size_t>(inputHeight) * inputWidth;

    const size_t inputBase =
        (static_cast<size_t>(batch) * channels + channel)
        * inputSpatialSize;

    float value00 = 0.0f;
    float value01 = 0.0f;
    float value10 = 0.0f;
    float value11 = 0.0f;

    if (valid00) {
        value00 = input[
            inputBase + static_cast<size_t>(y0) * inputWidth + x0];
    }

    if (valid01) {
        value01 = input[
            inputBase + static_cast<size_t>(y0) * inputWidth + x1];
    }

    if (valid10) {
        value10 = input[
            inputBase + static_cast<size_t>(y1) * inputWidth + x0];
    }

    if (valid11) {
        value11 = input[
            inputBase + static_cast<size_t>(y1) * inputWidth + x1];
    }

    const float value =
        value00 * (1.0f - wx) * (1.0f - wy) +
        value01 * wx * (1.0f - wy) +
        value10 * (1.0f - wx) * wy +
        value11 * wx * wy;

    output[index] = value;
}

} // namespace


// ============================================================
// Launch
// ============================================================

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
    cudaStream_t stream)
{
    const size_t totalElements =
        static_cast<size_t>(batch) *
        channels *
        outputHeight *
        outputWidth;

    constexpr int blockSize = 256;

    const int gridSize =
        static_cast<int>((totalElements + blockSize - 1) / blockSize);

    kernel_GridSample<<<
        gridSize,
        blockSize,
        0,
        stream>>>(
            input,
            grid,
            output,
            channels,
            inputHeight,
            inputWidth,
            outputHeight,
            outputWidth,
            totalElements);
}
