crate::ix!();

/**
  | `LpPool` consumes an input blob and
  | applies max pooling across the the blob
  | according to kernel sizes, stride sizes,
  | pad lengths and dilation. $L_p$ pooling
  | consists of taking the $L_p$ norm of
  | a subset of the input tensor according
  | to the kernel size and downsampling
  | the data into the output blob for further
  | processing.
  | 
  | Pooling layers reduce the spatial dimensionality
  | of the input blob. Each of the output
  | blob's dimensions will reduce according
  | to:
  | 
  | $$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$
  | 
  | Github Links: - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lp_pool_op.cc
  |
  */
#[derive(Default)]
pub struct LpPoolFunctor { }

register_cpu_operator!{LpPool, PoolOp<f32, CPUContext, LpPoolFunctor>}

num_inputs!{LpPool, 1}

num_outputs!{LpPool, 1}

inputs!{LpPool, 
    0 => ("X",          "(*Tensor`<float>`*): input tensor")
}

outputs!{LpPool, 
    0 => ("Y",          "(*Tensor`<float>`*): output tensor")
}

args!{LpPool, 
    0 => ("p",          "(*float*): type of $L_p$ norm to use (default=2.0)"),
    1 => ("kernel",     "(*int*): the size of the window to take a max over"),
    2 => ("stride",     "(*int*): the stride of the window"),
    3 => ("pad",        "(*int*): implicit zero padding to be added on both sides"),
    4 => ("dilation",   "(*int*): parameter that controls the stride of elements in the window"),
    5 => ("order",      "(*string*): order of blob dimensions (default=NCHW)")
}
