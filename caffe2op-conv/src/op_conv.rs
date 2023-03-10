crate::ix!();

#[test] fn conv_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Conv",
        ["X", "filter", "bias"],
        ["Y"],
        kernel=5,
        pad=1,
        stride=2
    )

    // Create X: (N,C,H,W)
    data = np.random.randn(1,1,8,8).astype(np.float32)
    print("Data shape: ",data.shape)

    // Create W: (M,C,Kh,Kw)
    filters = np.random.randn(3,1,5,5).astype(np.float32)
    print("Filter shape: ",filters.shape)

    // Create b: M
    bias = np.array([1.,1.,1.]).astype(np.float32)
    print("Bias shape: ",bias.shape)

    // Put the inputs into the workspace
    workspace.FeedBlob("X", data)
    workspace.FeedBlob("filter", filters)
    workspace.FeedBlob("bias", bias)

    // Run the operator
    workspace.RunOperatorOnce(op)
    print("Y:\n", workspace.FetchBlob("Y"))

    Data shape:  (1, 1, 8, 8)
    Filter shape:  (3, 1, 5, 5)
    Bias shape:  (3,)
    Y:
     [[[[  0.6406407    0.8620521    0.56461596]
       [ -1.5042953   -0.79549205 -10.683343  ]
       [ -0.5240259    3.4538248   -3.9564204 ]]

      [[  0.6876496    4.8328524   -1.9525816 ]
       [  1.2995434   -2.3895378    7.2670045 ]
       [  3.9929862    1.8126237    5.4699917 ]]

      [[  3.55949      4.7934155    0.76086235]
       [  3.9588015   -1.3251319    4.413117  ]
       [ -1.5296054   -1.4924102   -3.2552304 ]]]]

    */
}

/**
  | The Conv2D operator computes a 2D convolution
  | operation over an input blob $(X)$,
  | with a filter blob $(filter)$ and a bias
  | blob $(bias)$, and outputs a single
  | output blob $(Y)$.
  | 
  | Although there are several options
  | for order, the convention is that the
  | input $(X)$ is a blob of shape $(N,C_{in},H_{in},W_{in})$
  | and the output $(Y)$ is a blob of shape
  | $(N,C_{out},H_{out},W_{out})$.
  | Here, $N$ is the batch size, $C$ is the
  | number of channels, $H$ is the spatial
  | height, and $W$ is the spatial width.
  | For example, if your input data was a
  | batch of five, 100x120pixel RGB images,
  | $X$ would have shape $(5,3,120,100)$.
  | 
  | The $filter$ input blob may contain
  | multiple filters and has shape $(M,
  | C_{in}, K_H, K_W)$.
  | 
  | Here, $M$ is the number of individual
  | filters contained in the blob, $C_{in}$
  | is the number of channels of each filter
  | (by convention in 2D convolution it
  | is the same as the number of channels
  | in the input), $K_H$ is the spatial height
  | of the kernel, and $K_W$ is the spatial
  | width of the kernel.
  | 
  | The $bias$ blob is a vector of length
  | $M$, where there is one bias for each
  | filter in the $filter$ blob.
  | 
  | Given the shape of the input blob and
  | the filter blob, we can calculate the
  | shape of the output blob as follows.
  | The number of items in the batch $N$ will
  | stay the same. The number of channels
  | in the output will equal the number of
  | kernels in the filter blob, so $C_{out}
  | = M.$ With stride and pad defined below,
  | the spatial height and width of the output
  | ($H_{out}$ and $W_{out}$) are calculated
  | as
  | 
  | $$H_{out} = \left \lfloor{\frac{H_{in}
  | - K_H + 2*pad}{stride}+1}\right \rfloor$$
  | 
  | $$W_{out} = \left \lfloor{\frac{W_{in}
  | - K_W + 2*pad}{stride}+1}\right \rfloor$$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.cc
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h
  |
  */
pub struct ConvOp<T, Context> {

    //USE_CONV_POOL_BASE_FUNCTIONS(Context);
    base: ConvPoolOpBase<Context>,

    col_buffer:              Tensor, // {Context::GetDeviceType()};
    bias_multiplier:         Tensor, // {Context::GetDeviceType()};
    img_shape_device:        Tensor, // {Context::GetDeviceType()};
    col_buffer_shape_device: Tensor, // {Context::GetDeviceType()};
    phantom: PhantomData<T>,
}

impl<T,Context> ConvOp<T,Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<Context>(operator_def, ws) 

        // Since this is the default convolution implementation, we will
        // use CAFFE_ENFORCE instead of OPERATOR_NEEDS_FEATURE.
        CAFFE_ENFORCE(
            (group_ == 1 || order_ == StorageOrder::NCHW ||
             std::is_same<Context, CPUContext>::value),
            "Group convolution only supports NCHW order or CPUContext right now.");

        // Create shared buffer mutex in the constructor
        // to avoid race-condition in DAGNet.
        if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
          createSharedBuffer<Context>(ws_);
        }
        */
    }
}

/// Input: X, W, b
/// Output: Y
input_tags!{
    ConvOp {
        Input,
        Filter,
        Bias
    }
}

pub struct ConvGradientOp<T, Context> {

    //USE_CONV_POOL_BASE_FUNCTIONS(Context);

    base: ConvPoolOpBase<Context>,

    col_buffer:              Tensor,
    bias_multiplier:         Tensor,
    img_shape_device:        Tensor, //{Context::GetDeviceType()};
    col_buffer_shape_device: Tensor, //{Context::GetDeviceType()};
    no_bias:                 bool,
    phantom: PhantomData<T>,

    // input: X, W, dY
    // output: dW, db, and optionally dX
}

input_tags!{
    ConvGradientOp {
        Input,
        Filter,
        OutputGrad
    }
}

output_tags!{
    ConvGradientOp {
        FilterGrad,
        BiasOrInputGrad,
        InputGrad
    }
}

impl<T,Context> ConvGradientOp<T,Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<Context>(operator_def, ws),
            no_bias_(this->template GetSingleArgument<int>("no_bias", 0)) 

        CAFFE_ENFORCE(
            !(no_bias_ && OutputSize() == 3),
            "If bias is not present, you should not have 3 grad output.");
        CAFFE_ENFORCE(
            (group_ == 1 || order_ == StorageOrder::NCHW ||
             std::is_same<Context, CPUContext>::value),
            "Group convolution only supports NCHW order or CPUContext right now.");
        */
    }
}

type doc_fn = fn(s: &OpSchema) -> ();

pub fn conv_doc_generator(dim: *const u8) -> doc_fn {

    todo!();
    /*
  return [=](OpSchema& schema) {
    string doc = R"DOC(
The convolution operator consumes an input vector, a {dim}filter blob
and a bias blob and computes the output. {conv_doc})DOC";
    c10::ReplaceAll(doc, "{dim}", dim);
    c10::ReplaceAll(doc, "{conv_doc}", kConvDoc);
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data blob, of shape $(N, C_{in}, H_{in}, W_{in})$, to be convolved with the kernels in the filter blob."
      );
    schema.Input(
        1,
        "filter",
        "The filter blob, of shape $(M, C_{in}, K_H, K_W)$, containing the filters to be convolved with the data."
      );
    schema.Input(
        2,
        "bias",
        "The bias blob, of length $M$, containing the biases for the convolution, one bias per filter."
      );
    schema.Output(
        0,
        "Y",
        "Output data blob, of shape $(N, C_{out}, H_{out}, W_{out})$, that contains the result of the convolution."
      );
      /*
    schema.Arg(
        "kernel",
        "*(type: int; default: 0)* Desired kernel size. If left at default the kernel size will be inferred from the input $filter$ blob.",
        0
    );
    schema.Arg(
        "stride",
        "*(type: int; default: 1)* Controls the stride of the kernel as it traverses the input blob.",
        0
    );
    schema.Arg(
        "dilation",
        "*(type: int; default: 1)* Controls spacing between kernel points. If dilation is greater than one, the kernel does not operate on a contiguous spatial region. For a visualization click [here](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md).",
        0
    );
    schema.Arg(
        "pad",
        "*(type: int; default: 0)* Controls the amount of padding to apply to the input feature map before computing the convolution.",
        0
    );
    schema.Arg(
        "float16_compute",
        "*(type: bool; default: False)* Whether to use float-16 compute kernel.",
        0
    );
    schema.Arg(
        "group",
        "*(type: int; default: 1)* Controls level of group convolution. For more info click [here](https://blog.yani.io/filter-group-tutorial/).",
        0
    );
    schema.Arg(
        "order",
        "*(type: string; default: \"NCHW\")* Specifies the order of the input data blob, where $N$ is batch size, $C$ is number of channels, $H$ is spatial height, and $W$ is spatial width. The only other valid option is \"NHWC\".",
        0
    );
    schema.Arg(
        "shared_buffer",
        "*(type: int; default: 0)*",
        0
    );
    */
  };

    */
}

///----------------------------
register_cpu_operator!{Conv, ConvOp<float, CPUContext>}
num_inputs!{Conv, (2,3)}
num_outputs!{Conv, 1}
tensor_inference_function!{Conv, ConvPoolOpBase::<CPUContext>::TensorInferenceForConv}
cost_inference_function!{Conv, OpSchema::CostInferenceFunctionType(ConvPoolOpBase::<CPUContext>::CostInferenceForConv)}
inherit_onnx_schema!{Conv}


///----------------------------
register_cpu_operator!{Conv1D, ConvOp::<f32, CPUContext>}
num_inputs!{Conv1D, (2,3)}
num_outputs!{Conv1D, 1}
inherit_onnx_schema!{Conv1D}
tensor_inference_function!{Conv1D, ConvPoolOpBase::<CPUContext>::TensorInferenceForConv}


///----------------------------
register_cpu_operator!{Conv2D, ConvOp<f32, CPUContext>}
num_inputs!{Conv2D, (2,3)}
num_outputs!{Conv2D, 1}
inherit_onnx_schema!{Conv2D}
cost_inference_function!{Conv2D, /* OpSchema::CostInferenceFunctionType( ConvPoolOpBase<CPUContext>::CostInferenceForConv) */}
tensor_inference_function!{
    Conv2D, 
    ConvPoolOpBase::<CPUContext>::TensorInferenceForConv
}


///----------------------------
register_cpu_operator!{Conv3D, ConvOp<f32, CPUContext>}
num_inputs!{Conv3D, (2,3)}
num_outputs!{Conv3D, 1}
inherit_onnx_schema!{Conv3D}
cost_inference_function!{Conv3D, /* OpSchema::CostInferenceFunctionType( ConvPoolOpBase<CPUContext>::CostInferenceForConv) */}
tensor_inference_function!{Conv3D, ConvPoolOpBase::<CPUContext>::TensorInferenceForConv}

