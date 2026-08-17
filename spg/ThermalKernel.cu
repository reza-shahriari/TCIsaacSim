// SPG CUDA kernel -- docs/isaacsim_implementation_plan.md SS1.
// Function name must match ThermalKernel.cu.lua and ThermalKernel.usda's subIdentifier.
extern "C" __global__ void thermal_ir(
    int width, int height,
    float netdSigma,
    cudaTextureObject_t inputSignal,        // emission-encoded temperature AOV -- SS2
    cudaSurfaceObject_t outputThermalIR)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float4 px = tex2D<float4>(inputSignal, x, y);
    float signal = px.x;  // TODO: decode encoded temperature, then run
                           // atmosphere -> optics -> detector -> noise -> colormap
                           // (docs/thermal_camera_model.md SS3-SS11) here.

    uchar4 out = { (unsigned char)(signal * 255.0f), 0, 0, 255 };
    surf2Dwrite<uchar4>(out, outputThermalIR, x * sizeof(uchar4), y);
}
