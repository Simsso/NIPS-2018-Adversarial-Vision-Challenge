%{
Takes a tensor of shape [n,m,o,p], which is a common shape for CNN
activations, where n is the batch, m and o are spatial, and p are the
channels. The tensor is flattened, such that the actiations of the layer
are a single vector of size m*o*p. A PCA is then being computed on the 
resulting matrix. The first dim_out columns of the PCA output matrix, which
can be used e.g. for compression, is then written to a file at export_path.
%}

act = act5_block3; % CNN activation tensor (n, m, o, p)
export_path = '/Users/timodenk/.data/activations/pca.mat' % PCA output file
dim_out = 3968 % number of PCA output vectors to use, must be <= m*o*p

act_shape = size(act)

num_samples = act_shape(1)
vec_size = act_shape(2)*act_shape(3)*act_shape(4)
channels = reshape(act, num_samples, vec_size); % flatten the activations

pca_out = pca(channels);  % compute PCA
pca_out = pca_out(:,1:dim_out); % grab top dim_out eigenvectors

save(export_path, 'pca_out')
