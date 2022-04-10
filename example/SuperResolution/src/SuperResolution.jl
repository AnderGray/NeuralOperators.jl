module SuperResolution

using WaterLily, LinearAlgebra, ProgressMeter, MLUtils
using NeuralOperators, Flux
using CUDA, FluxTraining, BSON

function circle(n, m; Re=250) # copy from [WaterLily](https://github.com/weymouth/WaterLily.jl)
    # Set physical parameters
    U, R, center = 1., m/8., [m/2, m/2]
    ν = U * R / Re

    body = AutoBody((x,t) -> LinearAlgebra.norm2(x .- center) - R)
    Simulation((n+2, m+2), [U, 0.], R; ν, body)
end

function gen_data(ts::AbstractRange; resolution=2)
    @info "gen data with $(resolution)x resolution... "
    p = Progress(length(ts))

    n, m = resolution * 3(2^5), resolution * 2^6
    circ = circle(n, m)

    𝐩s = Array{Float32}(undef, 1, n, m, length(ts))
    for (i, t) in enumerate(ts)
        sim_step!(circ, t)
        𝐩s[1, :, :, i] .= Float32.(circ.flow.p)[2:end-1, 2:end-1]

        next!(p)
    end

    return 𝐩s
end

function get_dataloader(; ts::AbstractRange=LinRange(100, 11000, 10000), ratio::Float64=0.95, batchsize=100)
    data = gen_data(ts, resolution=1)
    data_train, data_validate = splitobs(shuffleobs((𝐱=data[:, :, :, 1:end-1], 𝐲=data[:, :, :, 2:end])), at=ratio)

    data = gen_data(ts, resolution=2)
    data_test = (𝐱=data[:, :, :, 1:end-1], 𝐲=data[:, :, :, 2:end])

    loader_train = DataLoader(data_train, batchsize=batchsize, shuffle=true)
    loader_validate = DataLoader(data_validate, batchsize=batchsize, shuffle=false)
    loader_test = DataLoader(data_test, batchsize=batchsize, shuffle=false)

    return (training=loader_train, validation=loader_validate, testing=loader_test)
end

struct TestPhase<:FluxTraining.AbstractValidationPhase end

FluxTraining.phasedataiter(::TestPhase) = :testing

function FluxTraining.step!(learner, phase::TestPhase, batch)
    xs, ys = batch
    FluxTraining.runstep(learner, phase, (xs=xs, ys=ys)) do _, state
        state.ŷs = learner.model(state.xs)
        state.loss = learner.lossfn(state.ŷs, state.ys)
    end
end

function fit!(learner, nepochs::Int, (trainiter, validiter, testiter))
    for i in 1:nepochs
        epoch!(learner, TrainingPhase(), trainiter)
        epoch!(learner, ValidationPhase(), validiter)
        epoch!(learner, TestPhase(), testiter)
    end
end

function fit!(learner, nepochs::Int)
    fit!(learner, nepochs, (learner.data.training, learner.data.validation, learner.data.testing))
end

function train(; epochs=50)
    if has_cuda()
        @info "CUDA is on"
        device = gpu
        CUDA.allowscalar(false)
    else
        device = cpu
    end

    model = MarkovNeuralOperator(ch=(1, 64, 64, 64, 64, 64, 1), modes=(24, 24), σ=gelu)
    data = get_dataloader()
    optimiser = Flux.Optimiser(WeightDecay(1f-4), Flux.ADAM(1f-3))
    loss_func = l₂loss

    learner = Learner(
        model, data, optimiser, loss_func,
        ToDevice(device, device),
        # Checkpointer(joinpath(@__DIR__, "../model/"))
    )

    fit!(learner, epochs)

    return learner
end

function get_model()
    model_path = joinpath(@__DIR__, "../model/")
    model_file = readdir(model_path)[end]

    return BSON.load(joinpath(model_path, model_file), @__MODULE__)[:model]
end

# using NeuralOperators
# using Flux
# using Flux.Losses: mse
# using Flux.Data: DataLoader
# using GeometricFlux
# using Graphs
# using CUDA
# using JLD2
# using ProgressMeter: Progress, next!

# include("data.jl")
# include("models.jl")

# function update_model!(model_file_path, model)
#     model = cpu(model)
#     jldsave(model_file_path; model)
#     @info "model updated!"
# end

# function get_model()
#     f = jldopen(joinpath(@__DIR__, "../model/model.jld2"))
#     model = f["model"]
#     close(f)

#     return model
# end

# loss(m, 𝐱, 𝐲) = mse(m(𝐱), 𝐲)
# loss(m, loader::DataLoader, device) = sum(loss(m, 𝐱 |> device, 𝐲 |> device) for (𝐱, 𝐲) in loader)/length(loader)

end # module
