act = act5_block3;
labels = transpose(target_labels);
sample_index = 4

act_shape = size(act)
vec_size = act_shape(2)*act_shape(3)*act_shape(4)
num_samples = act_shape(1)

act = reshape(act, num_samples, vec_size)*pca_out;

% normalize vectors
act = normr(act);

sample_vec = transpose(act(sample_index:sample_index,:))
sample_label = labels(sample_index)
cos_sim = act*sample_vec;
h1 = histogram(cos_sim);
h1.BinWidth = 0.01
hold on

same_class_act = act(labels==sample_label,:);
cos_sim_same_class = same_class_act*sample_vec;
h2 = histogram(cos_sim_same_class);
h2.BinWidth = 0.01

export_path = '/Users/timodenk/.data/activations/lesci.mat' % LESCI
save(export_path, 'act', 'labels')