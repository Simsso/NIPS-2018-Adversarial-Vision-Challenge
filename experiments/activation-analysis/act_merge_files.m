%{
Loads a number of .mat files and merges the matrices contained.
%}

num_files = 20
base_path = '/Volumes/timodenk/activations/baseline/act6_block4_%03d.mat'

act_name = 'act6_block4'
labels_name = 'target_labels'

act_collected = []
labels_collected = []

for i = 0:num_files-1
    file_path = sprintf(base_path, i)
    imported = load(file_path, act_name, labels_name)
    size(labels_collected)
    size(getfield(imported,labels_name))
    act_collected = cat(1, act_collected, getfield(imported,act_name));
    labels_collected = cat(2, labels_collected, getfield(imported,labels_name));
end
