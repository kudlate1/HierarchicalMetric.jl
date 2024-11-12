
data = load_dataset("Mutagenesis"; to_mill=true, to_pad_leafs=false);
idx =     eachindex(data[2]);
data = (data[1][idx], data[2][idx])
train, val, test = preprocess(data...; ratios=(0.6,0.2,0.2), procedure=:clf, filter_under=10);

bag_m = getfield(HSTreeDistance, Symbol("ChamferDistance"))
card_m = getfield(HSTreeDistance, Symbol("ScaleOne"))
# metric

Random.seed!(1234) # rondom initialization of weights with fiexed seed
_metric = reflectmetric(data[1][1]; set_metric=bag_m, card_metric=card_m, weight_sampler=randn, weight_transform=softplus)
metric = mean ∘ _metric;
