%{
Writes a random matrix to the file at export_path. Created for testing
purposes, namely importing to Python.
%}

export_path = '/Users/timodenk/.data/activations/test_export.mat'
mat = rand(64, 61);
save(export_path, 'mat')