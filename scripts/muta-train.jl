using HierarchicalMetric
using HierarchicalMetric.Mill
using HierarchicalMetric.Plots

function params_importance(n)

    """
    Trains the parameters trainNumber-times. After every finished training the function
    decides which parameters are important (+1 on counts[numOfParam]).
    The result is displayed and the importance of each parameter is evident.

    Params:
    trainNumber (Int64): the number of trainings

    """
    X, y = load("data/mutagenesis.json")
    distances = pairwise_distance(X)

    counts = zeros(Int64, 13)
    for _ in 1:n
        ps, _ = train(SelectHard(), X, y, distances; max_iter=20);
        for i in 1:length(ps)
            (ps[i] >= 0.1) && (counts[i] += 1)
        end
    end

    bar(1:length(counts), counts, title="Importance of each parameter", xlabel="parameters", ylabel="parameter counts", label="importance")
    xticks!(1:length(counts))
end

"""
EXAMPLE OF A TRAINING EXECUTION:

X, y = load("data/mutagenesis.json")
distances = pairwiseDistance(X)

heatmap(distances, aspect_ratio = 1)

ps, h = train(SelectRandom(), X, y, distances);
plotProcess(ps, h)

"""