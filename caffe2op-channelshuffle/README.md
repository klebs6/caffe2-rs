## Caffe2Op-ChannelShuffle

This rust crate implements a mathematical operator
used in deep learning computations called Channel
Shuffle. The Channel Shuffle operation shuffles
the channels of a tensor to increase the
representational capacity of neural networks.

**Note: This crate is currently being translated from C++ to Rust, and some function bodies may still be in the process of translation.**

### Channel Shuffle Operation

The Channel Shuffle operation is defined as follows:

Given a tensor `X` with shape `[N, C, H, W]` or
`[N, H, W, C]` (depending on the dimension order
used), and a shuffle group size `G`, the output
`Y` is computed as:

```
Y[i, j, k, l] = X[(i*G + j) % C, k, l, (i*G + j) / C] if input_shape is [N, C, H, W]
Y[i, j, k, l] = X[k, l, (i*G + j) % C, (i*G + j) / C] if input_shape is [N, H, W, C]
```

where `i, j, k, l` are indices that iterate over the dimensions of `Y`.

The Channel Shuffle operation is used to create
stronger feature representations by shuffling the
channels of a tensor. This helps to improve the
generalization power of deep learning models and
reduce overfitting.

### Implementation

This rust crate provides two implementations of
the Channel Shuffle operation: `shuffleNCHW` and
`shuffleNHWC`. The `shuffleNCHW` implementation
shuffles the channels of the input tensor in the
order of `NCHW` dimension, while the `shuffleNHWC`
implementation shuffles the channels in the order
of `NHWC` dimension. Both implementations take as
input the input tensor and the shuffle group size
`G`, and return the shuffled tensor.

### Example Usage

```rust
use caffe2op_channelshuffle::*;

fn main() {
    let input = vec![1., 2., 3., 4., 5., 6., 7., 8.];
    let shuffle_size = 2;
    let output = shuffleNCHW(&input, shuffle_size);
    println!("{:?}", output); // Prints [1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]
}
```

In this example, the `shuffleNCHW` function
shuffles the channels of the input tensor `input`
in the order of `NCHW` dimension with a shuffle
group size of 2, and returns the shuffled tensor
`output`.
