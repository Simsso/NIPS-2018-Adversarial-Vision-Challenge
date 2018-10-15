%{
 Shows a randomly chosen input image and prints its label.
%}

act1_shape = size(act1_input)
rand_img_index = randi([1 act1_shape(1)],1,1)
img = squeeze(act1_input(rand_img_index,:,:,:));
image(img)
labels(1,rand_img_index)
