using HierarchicalMetric
using HierarchicalMetric.Mill
using HierarchicalMetric.Plots

function paramsImportance(n)

    """
    Trains the parameters trainNumber-times. After every finished training the function
    decides which parameters are important (+1 on counts[numOfParam]).
    The result is displayed and the importance of each parameter is evident.

    Params:
    trainNumber (Int64): the number of trainings

    """
    X, y = load("data/mutagenesis.json")
    distances = pairwiseDistance(X)
    metric = reflectmetric(X[1], weight_sampler=randn, weight_transform=softplus) 

    counts = zeros(Int64, 13)
    for _ in 1:n
        ps, h = train(SelectHard(), X, y, distances; max_iter=25);
        h = reduce(hcat, h)'
        flat = softplus.(Flux.destructure(metric)[1])
        println(flat)
        for i in 1:length(flat)
            (flat[i] >= 0.1) && (counts[i] += 1)
        end
    end

    bar(1:length(counts), counts, xlabel="parameters", ylabel="parameter counts")
    xticks!(1:length(counts))
end

function plotProcess(ps, h)

    p = plot(reduce(hcat, h)', 
        xlabel="number of iteations", 
        ylabel="values of the parameters", 
        title="Parameters learning with lasso regularization"
    )
    display(p)

    return vcat(softplus.(ps)[1])
end

"""
EXAMPLE OF A TRAINING EXECUTION:

X, y = load("data/mutagenesis.json")
distances = pairwiseDistance(X)

heatmap(distances, aspect_ratio = 1)

ps, h = train(SelectRandom(), X, y, distances);
plotProcess(ps, h)

"""