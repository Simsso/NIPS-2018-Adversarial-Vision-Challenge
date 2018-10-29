%{
Takes a tensor of shape [n,m,o,p], which is a common shape for CNN
activations, where n is the batch, m and o are spatial, and p are the
channels. The tensor is flattened, such that the actiations of the layer
are a single vector of size m*o*p. A PCA is then being computed on the 
resulting matrix. The first dim_out columns of the PCA output matrix, which
can be used e.g. for compression, is then written to a file at export_path.
%}

act = act6_block4; % CNN activation tensor (n, m, o, p)
labels = transpose(target_labels);
export_path = '/Users/timodenk/.data/activations/baseline/pca.mat' % PCA output file
export_path_act_reduced = '/Users/timodenk/.data/activations/baseline/lesci.mat'
dim_out = 256 % number of PCA output vectors to use, must be <= m*o*p

act_shape = size(act)

num_samples = act_shape(1)
vec_size = act_shape(2)*act_shape(3)*act_shape(4)

% flatten the activations as numpy / TF would do it
act_np_reshape = permute(act, [1, 4, 3, 2]);
channels = reshape(act_np_reshape, num_samples, vec_size);

pca_out = pca(channels);  % compute PCA
pca_out = pca_out(:,1:dim_out); % grab top dim_out eigenvectors

act_compressed = channels * pca_out; % compressed

save(export_path, 'pca_out')
save(export_path_act_reduced, 'act_compressed', 'labels')
