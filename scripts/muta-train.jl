using Plots

function paramsImportance(n)

    """
    Trains the parameters trainNumber-times. After every finished training the function
    decides which parameters are important (+1 on counts[numOfParam]).
    The result is displayed and the importance of each parameter is evident.

    Params:
    trainNumber (Int64): the number of trainings

    """

    counts = zeros(Int64, 13)
    for _ in 1:n
        ps, h = train(SelectRandom());
        h = reduce(hcat, h)'
        flat = vcat(softplus.(ps)...)
        for i in 1:length(flat)
            (flat[i] >= 0.1) && (counts[i] += 1)
        end
    end

    bar(1:length(counts), counts, xlabel="parameters", ylabel="parameter counts")
    xticks!(1:length(counts))
end

function trainMutagenesis()

    function plotProcess(ps, h)

        plot(reduce(hcat, h)', xlabel="number of iterations", ylabel="values of the parameters", title="Parameters learning with lasso regularization")
        vcat(softplus.(ps)...)
    end

    # loading the dataset and its visualisation
    X, y = load("data/mutagenesis.json")
    distances = pairwiseDistance(X)
    heatmap(distances, aspect_ratio = 1)

    # training the parameters once and plot the result
    ps, h = train(SelectRandom(), X, y, distances);

    # plot process of training
    plotProcess(ps, h)
end