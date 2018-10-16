%{
Shows a randomly chosen input image (input), the next layer's activations,
and prints its label.
%}

act1_shape = size(act1_input);
rand_img_index = randi([1 act1_shape(1)],1,1);

img = squeeze(act1_input(rand_img_index,:,:,:));
image(img)

channels_first_conv = squeeze(act2_first_conv(rand_img_index,:,:,:));
act2_shape = size(act2_first_conv);
img_tiled_dim = [act2_shape(2)*sqrt(act2_shape(4)) act2_shape(3)*sqrt(act2_shape(4))]
first_conv_stacked = reshape(channels_first_conv, img_tiled_dim);
image(first_conv_stacked*128)

labels(1,rand_img_index)
