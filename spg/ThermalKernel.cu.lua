-- SPG Lua launch script -- docs/isaacsim_implementation_plan.md SS1.
-- Validates inputs, allocates outputs, configures the kernel launch.
-- Does NOT do pixel math -- that's ThermalKernel.cu's job.
function thermal_ir(inputs, outputs)
    local height = inputs["Signal"].shape[1]
    local width  = inputs["Signal"].shape[2]
    outputs["ThermalIR"] = cuda.image(width, height, cuda.uchar4)
    return cuda.kernel({
        args = {
            cuda.int(width), cuda.int(height),
            cuda.float(inputs["netdSigma"]),
            cuda.TextureObject(inputs["Signal"]),   -- NOTE: TextureObject, not texture()
            cuda.SurfaceObject(outputs["ThermalIR"]), -- NOTE: SurfaceObject, not surface()
        },
        block = { 32, 32 },   -- NOTE: block/grid, not blockDim/gridDim
        grid  = { math.ceil(width / 32), math.ceil(height / 32) },
    })
end
