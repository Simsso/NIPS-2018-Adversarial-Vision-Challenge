act = act5_block3;
act_shape = size(act)

num_samples = act_shape(1)
channels = reshape(act, num_samples, act_shape(2)*act_shape(3)*act_shape(4));

pca_out = pca(channels);
dim_out = 3968
pca_out = pca_out(:,1:dim_out);

export_path = '/Users/timodenk/.data/activations/pca.mat'
save(export_path, 'pca_out')
