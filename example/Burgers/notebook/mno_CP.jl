using Burgers, Plots
using DataDeps, MAT, MLUtils
using NeuralOperators, Flux
using CUDA, BSON

include("notebook/Conformal_prediction.jl")

dataset = Burgers.get_data();

m = Burgers.get_model();
input_data, ground_truth = dataset[1], dataset[2];

n_cal = 1000

input_cal = input_data[:,:, 1:n_cal]
output_cal = ground_truth[:,:, 1:n_cal]

input_test = input_data[:,:, n_cal+1:end]
output_test = ground_truth[:,:, n_cal+1:end]

###
#    Make a cp Neural operator model
###
my_CP_model = calibrate_modulation_error(m, input_cal, output_cal[1, : ,:])


# Plot a random prediction
i = rand(1:size(input_test)[3])
prediction_sets = compute_prediction_sets(my_CP_model, view(input_test, :, :, i:i), 0.1);

plot(input_data[1, :, 1], output_test[1, :, i ], label = "ground_truth");
plot!(input_data[1, :, 1], m(view(input_test, :, :, i:i))[1, :, 1], label = "predict")

plot!(input_data[1, :, 1], sup.(prediction_sets[:,:,1]), label = "upperbound");
plot!(input_data[1, :, 1], inf.(prediction_sets[:,:,1]), label = "lower bounds")


###
#    Plot emirial coverage
###
coverge, alphas = emirical_coverage(my_CP_model, input_test, output_test, 100 )

plot(coverage, alphas)
plot!([0, 1], [0,1], linewidth = 2, label = "exact")
xlabel!("1 - α")
ylabel!("empirial coverage")


i = rand(1:size(input_test)[3])
p1 = plot_prediction(my_CP_model, view(input_test, :, :, i:i), output_test[1,:,i])
i = rand(1:size(input_test)[3])
p2 = plot_prediction(my_CP_model, view(input_test, :, :, i:i), output_test[1,:,i])
i = rand(1:size(input_test)[3])
p3 = plot_prediction(my_CP_model, view(input_test, :, :, i:i), output_test[1,:,i])
i = rand(1:size(input_test)[3])
p4 = plot_prediction(my_CP_model, view(input_test, :, :, i:i), output_test[1,:,i])

p = plot(p1, p2, p3, p4)



#### Test backward prop of data

# project_layer = m.project_net

# last_layer = project_layer[2]
# first_layer = project_layer[1]

# X_init = rand(Float32, 64)

# forward = first_layer.weight * X_init .+ first_layer.bias
# backwards = first_layer.weight \ (forward .- first_layer.bias)

# println(mean(X_init .- backwards))

