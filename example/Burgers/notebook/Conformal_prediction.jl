using IntervalArithmetic, StatsBase, Distributions

struct CP_model{M, T} <: AbstractOperatorModel
    m :: M
    non_conformity_scores :: Vector{T}
    modulation :: Matrix{T}
end

function calibrate_modulation_max(model, calibration_X, calibration_y)

    calibration_prediction = model(calibration_X)[1, :, :]

    cal_mean = mean(calibration_y, dims = 2)
    modulation = std(calibration_y, dims = 2)
    non_conformity_scores = maximum(abs.((cal_mean - calibration_prediction) ./modulation), dims = 1)

    return CP_model(model, sort(non_conformity_scores[:]), modulation)
 end

function calibrate_modulation_error(model, calibration_X, calibration_y)

    calibration_prediction = model(calibration_X)[1,:,:]
    modulation = Matrix(std(calibration_y .- calibration_prediction, dims = 2))
    non_conformity_scores = maximum(abs.((calibration_y .- calibration_prediction) ./ modulation), dims = 1)

    return CP_model(model, sort(non_conformity_scores[:]), modulation)
end

function compute_prediction_sets(cpmodel :: CP_model, input_prediction, alpha=0.9)

    Y_predictions = cpmodel.m(input_prediction)[1, :, :]
    n = length(cpmodel.non_conformity_scores)
    qhat = quantile(cpmodel.non_conformity_scores, ceil.(n+1) .*(1 .-alpha)/n)

    prediction_sets = fill( interval(0), size(Y_predictions)[1], size(Y_predictions)[2], length(qhat))

    for (j, q) in enumerate(qhat)
        prediction_sets[:,:,j] =  Y_predictions .+ interval.( .- q .* cpmodel.modulation,  q .* cpmodel.modulation)
    end

    return prediction_sets
end

function emirical_coverage(cpmodel :: CP_model, input_test, output_test, n_alpha = 100)

    alphas = range(0, 1, n_alpha+ 2)[2:end-1]
    prediction_sets = compute_prediction_sets(cpmodel, input_test, alphas)
    samples_in = output_test[1,:,:] .∈ prediction_sets

    coverage = sum(all(samples_in, dims = 1), dims = 2)[:]/size(input_test)[3]

    return coverage, 1 .- alphas
end

function plot_prediction(cpmodel, input, ground_truth, n_alpha = 10)

    alphas = range(0, 1, length=n_alpha+2)[2:end-1]
    prediction_sets = compute_prediction_sets(cpmodel, input, alphas)
    
    colours = colormap("RdBu", n_alpha)
    p1 = plot()
    for (i, alpha) in enumerate(alphas)
        p1 = plot!(input[1, :, 1], inf.(prediction_sets[:,:,i]), fill_between = sup.(prediction_sets[:,:,i]), color = colours[i], label = false)
    end

    p1 = plot!(input[1, :, 1], ground_truth, label = "ground truth", linewidth= 1, color = "yellow")
    return p1
end