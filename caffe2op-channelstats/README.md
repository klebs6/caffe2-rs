
## Description for `caffe2op-channelstats` Rust crate

The `caffe2op-channelstats` crate defines
a mathematical operator used in DSP and machine
learning computations. The main purpose of this
crate is to compute the mean and standard
deviation of a set of input channels for a given
tensor.

### ChannelStatsOp

The `ChannelStatsOp` is the main operator provided
by this crate. It takes as input a tensor and
returns its mean and standard deviation across
a specified set of channels. The operator can be
applied in two different ways:
`ComputeChannelStatsNCHW` and
`ComputeChannelStatsNHWC`.

`ComputeChannelStatsNCHW` computes the
channel-wise statistics of a tensor with
dimensions `[batch_size, channels, height,
width]`, while `ComputeChannelStatsNHWC` computes
the channel-wise statistics of a tensor with
dimensions `[batch_size, height, width,
channels]`.

### Mathematical Analysis

Given a tensor `X` with dimensions `[batch_size,
channels, height, width]`, the mean and standard
deviation for the `i`th channel can be computed as
follows:

```
mean_i = sum(X[:,i,:,:]) / (batch_size * height * width)
std_i = sqrt(sum((X[:,i,:,:] - mean_i)**2) / (batch_size * height * width))
```

Similarly, for a tensor `X` with dimensions
`[batch_size, height, width, channels]`, the mean
and standard deviation for the `i`th channel can
be computed as follows:

```
mean_i = sum(X[:,:,:,i]) / (batch_size * height * width)
std_i = sqrt(sum((X[:,:,:,i] - mean_i)**2) / (batch_size * height * width))
```

The `ChannelStatsOp` applies these computations
channel-wise to produce the mean and standard
deviation for the specified set of channels.

### Usage

The `ChannelStatsOp` can be used in conjunction
with other operators in deep learning frameworks
such as Caffe2 and PyTorch to perform
normalization of input data. By computing the
channel-wise statistics of a tensor, it is
possible to subtract the mean and divide by the
standard deviation, which can help to improve the
performance of deep neural networks.

This crate provides both NCHW and NHWC
implementations of the operator to ensure
compatibility with different deep learning
frameworks.

### Example

```rust
use caffe2op_channelstats::{ChannelStatsOp, ComputeChannelStatsNCHW};

let input_tensor = // some tensor with dimensions [batch_size, channels, height, width]
let num_channels = // number of channels to compute statistics for
let op = ChannelStatsOp::new(ComputeChannelStatsNCHW(num_channels));
let (mean, std) = op.forward(&[input_tensor]);
```
