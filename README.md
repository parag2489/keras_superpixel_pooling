# keras_superpixel_pooling
Implementation of superpixel pooling layer in Keras. See keras.layers.pooling for implemenation.

The concept of superpixel pooling layer can be found in the paper: "Weakly Supervised Semantic Segmentation Using Superpixel Pooling Network", AAAI 2017. This layer takes two inputs, a superpixel map (size `M x N`) and a feature map (size `K x M x N`). It pools the features (in this implementation, average-pool) belonging to the same superpixel and forms a `1 x K` vector where `K` is the feature map depth/channels. 

A naive implementation will require three for loops: one iterating over batch, another over row and the last one iterating over columns of the feature map and pooling it on-the-fly. However, this gives "maximum recursion depth exceeded" error in Theano whenever you try to compile a model containing this layer. This error occurs even when the feature map width and height is only 32.

To overcome this problem, I thought that passing all the things as parameters to this layer will get rid of at least two for loops. Eventually, I was able to create a one-liner to implement the core of the entire average-pooling operation. You need to pass:

1. Number of superpixels in the image
2. Feature map depth/channels
3. Batch size
4. Shape of feature map and superpixel map
5. An `N x 3` matrix that contains all the possible combination of indices corresponding to `(batch_size, row, column)` called `positions`. This only needs to be generated once during training provided your input image size and batch size remains constant.
6. An `N x 2` matrix called `superpixel_positions`. The row i contains the superpixel index corresponding to the indices in the row `i` of matrix `positions`. For example, if row `i` of the matrix `positions` contains `(12, 10, 20)`, then the same row of superpixel positions will contain `(12, sp_i)` where `sp_i = superpixel_map[12, 10, 20]`.
7. An `N x S` matrix - `superpixel_hist` - where `S` are the nubmer of superpixels in that image. As the name suggests, this matrix keeps a histogram of superpixels present in the current image.

The shortcoming of this implementation is that these parameters will have to be changed per image (specifically, parameters mentioned in points 6 and 7). This is impractical when GPU processes an entire batch at a time. I think this can be solved by passing all these parameters as inputs to the layer externally. Basically, they can be read from (say) HDF5 files. I plan to do that shortly. I will update this when that's done.
