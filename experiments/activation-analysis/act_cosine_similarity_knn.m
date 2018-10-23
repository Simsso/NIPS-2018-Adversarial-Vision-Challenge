% sort label vector based on similarity
[cos_sim_sorted, cos_sim_order] = sort(cos_sim, 'descend');
labels_sorted = labels(cos_sim_order);

sample_label
k = 100
h1 = histogram(labels_sorted(1:k))
h1.BinWidth = 1
h1.BinLimits = [0, 200]
hold on
h2 = histogram(sample_label);
h2.BinWidth = 1
h2.BinLimits = [0, 200]