crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorDef,
    Tensor,
    OperatorStorage,
};

/**
  | The FC operator computes an output $(Y)$
  | as a linear combination of the input
  | data blob $(X)$ with a weight blob $(W)$
  | and bias blob $(b)$. More formally,
  | 
  | $$Y = XW^T+b$$
  | 
  | Here, $X$ is a matrix of shape $(M,K)$,
  | $W$ is a matrix of shape $(N,K)$, $b$
  | is a vector of length $N$, and $Y$ is a
  | matrix of shape $(M,N)$. $N$ can be thought
  | of as the number of nodes in the layer,
  | $M$ is the batch size, and $K$ is the number
  | of features in an input observation.
  | 
  | -----------
  | @note
  | 
  | $X$ does not need to explicitly be a 2-dimensional
  | matrix, however, if it is not it will
  | be coerced into one. For an arbitrary
  | $n$-dimensional tensor $X$, e.g. $[a_0,
  | a_1, \ldots ,a_{k-1}, a_k, \ldots ,
  | a_{n-1}]$, where $a_i$ in $N$, and $k$
  | is the $axis$ arg provided, then $X$
  | will be coerced into a 2-dimensional
  | tensor with dimensions $[a_0 * \ldots
  | * a_{k-1}, a_k * \ldots * a_{n-1}]$.
  | For the default case where axis=1, this
  | means the $X$ tensor will be coerced
  | into a 2D tensor of dimensions $[a_0,
  | a_1 \ldots * a_{n-1}]$, where $a_0$
  | is often the batch size. In this situation,
  | we must have $a_0 = M$ and $a_1 * \ldots
  | * a_{n-1} = K$. Lastly, even though $b$
  | is a vector of length $N$, it is copied
  | and resized to shape $(M x N)$ implicitly,
  | then added to each vector in the batch.*
  | 
  | This is Caffe's InnerProductOp, with
  | a name that fits its purpose better.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/fully_connected_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/fully_connected_op.cc
  |
  */
pub struct FullyConnectedOp<Context, Engine, const TransposeWeight: bool> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:         OperatorStorage,
    context:         Context,

    axis:            usize, //{1};
    axis_w:          usize, //{1};

    /**
      | A local vector to cache the output shape
      | so we don't need to recreate a vector
      | object every time we run Run().
      |
      */
    y_shape_cache:   Vec<i64>,
    bias_multiplier: Option<Tensor>,
    float16_compute: bool,
    phantomE:        PhantomData<Engine>,
}

#[test] fn fully_connected_op_example() {

    todo!();

    /*
    // In this example, our batch size is 1 (M=1), the input observation will have
    //   6 features (K=6), and the layer will have one hidden node (N=1). The
    //   expected output is Y=7.
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "FC",
        ["X", "W", "b"],
        ["Y"]
    )

    // Create X: MxK
    data = np.array([1,2,3,4,5,6]).astype(np.float32)
    data = data[np.newaxis,:]

    // Create W: NxK
    weights = np.array(np.array([1,1/2.,1/3.,1/4.,1/5.,1/6.])).astype(np.float32)
    weights = weights[np.newaxis,:]

    // Create b: N
    bias = np.array([1.]).astype(np.float32)

    // Put the inputs into the workspace
    workspace.FeedBlob("X", data)
    workspace.FeedBlob("W", weights)
    workspace.FeedBlob("b", bias)

    // Run the operator
    workspace.RunOperatorOnce(op)
    print("Y:\n", workspace.FetchBlob("Y"))

    Y:
     [[7.]]

    */
}

num_inputs!{FC, 3}

num_outputs!{FC, 1}

inputs!{FC, 
    0 => ("X", "Input blob to be coerced into a 2D matrix of shape $(M,K)$, where $M$ is the batch size and $K$ is the number of features in a single observation."),
    1 => ("W", "Input blob to be coerced into a 2D matrix of shape $(N,K)$ describing a fully connected weight matrix. Here, $K$ is the number of features in a single observation and $N$ is the number of nodes in the FC layer."),
    2 => ("b", "Input blob containing vector of length $N$ which describes one bias for each node in the layer.")
}

outputs!{FC, 
    0 => ("Y", "Output blob containing a 2D output matrix of shape $(M,N)$, where $M$ is the batch size and $N$ is the number of nodes in the layer. The output is calculated as $Y=XW^T+b$.")
}

args!{FC, 
    0 => ("axis", "*(type: int; default: 1)* Describes the axis of the input data $X$. Defaults to one because in the common case when the input $X$ has shape $(M,K)$, the first axis encodes the batch size."),
    1 => ("axis_w", "*(type: int; default: 1)* Describes the axis of the input weight matrix $W$. Defaults to one because the first axis most likely describes the batch_size."),
    2 => ("float16_compute", "*(type: bool; default: False)* Whether to use float-16 compute kernel.")
}

inherit_onnx_schema!{FC, "Gemm"}

tensor_inference_function!{FC, /* std::bind(FCShapeInference, _1, _2, false) */}

cost_inference_function!{FC, /* std::bind(CostInferenceForFC, _1, _2, false) */}

impl<Context, Engine, const TransposeWeight: bool> 
FullyConnectedOp<Context, Engine, TransposeWeight> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axis_(this->template GetSingleArgument<int32_t>("axis", 1)),
            axis_w_(this->template GetSingleArgument<int32_t>("axis_w", 1)),
            float16_compute_(
                this->template GetSingleArgument<bool>("float16_compute", false))
        */
    }

    #[inline] pub fn do_run_with_type<T_X, T_W, T_B, T_Y, MATH>(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(0);
            const auto& W = Input(1);
            const auto& b = Input(2);

            CAFFE_ENFORCE(b.dim() == 1, b.dim());
            // batch size
            const auto canonical_axis = X.canonical_axis_index(axis_);
            const auto M = X.size_to_dim(canonical_axis);
            const auto K = X.size_from_dim(canonical_axis);
            const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
            const int N = TransposeWeight ? W.size_to_dim(canonical_axis_w)
                                          : W.size_from_dim(canonical_axis_w);

            auto dimErrorString = [&]() {
              return c10::str(
                  "Dimension mismatch: ",
                  "X: ",
                  X.sizes(),
                  ", W: ",
                  W.sizes(),
                  ", b: ",
                  b.sizes(),
                  ", axis: ",
                  axis_,
                  ", M: ",
                  M,
                  ", N: ",
                  N,
                  ", K: ",
                  K);
            };

            // Error checking
            CAFFE_ENFORCE(M == X.numel() / K, dimErrorString());
            CAFFE_ENFORCE(K == W.numel() / N, dimErrorString());
            CAFFE_ENFORCE(N == b.dim32(0), dimErrorString());
            CAFFE_ENFORCE(N == b.numel(), dimErrorString());

            Y_shape_cache_ = X.sizes().vec();
            // This is an invariant of canonical_axis, so we can DCHECK.
            DCHECK_LE(canonical_axis + 1, Y_shape_cache_.size());
            Y_shape_cache_.resize(canonical_axis + 1);
            Y_shape_cache_[canonical_axis] = N;
            auto* Y = Output(0, Y_shape_cache_, at::dtype<T_Y>());
            CAFFE_ENFORCE(M * N == Y->numel(), dimErrorString());

            if (X.numel() == 0) {
              // skip the rest of the computation if X is empty
              Y->template mutable_data<T_Y>();
              return true;
            }

            // default to FLOAT as math.h does.
            TensorProto::DataType math_type = TensorProto_DataType_FLOAT;
            if (fp16_type<MATH>()) {
              math_type = TensorProto_DataType_FLOAT16;
            }

            // W * x
            math::Gemm<T_X, Context, Engine>(
                CblasNoTrans,
                TransposeWeight ? CblasTrans : CblasNoTrans,
                M,
                N,
                K,
                1,
                X.template data<T_X>(),
                W.template data<T_W>(),
                0,
                Y->template mutable_data<T_Y>(),
                &context_,
                math_type);

            // Add bias term
            if (!bias_multiplier_.has_value()) {
              bias_multiplier_ =
                  caffe2::empty({M}, at::dtype<T_B>().device(Context::GetDeviceType()));
              math::Set<T_B, Context>(
                  M,
                  convert::To<float, T_B>(1),
                  bias_multiplier_->template mutable_data<T_B>(),
                  &context_);
            } else if (bias_multiplier_->numel() != M) {
              bias_multiplier_->Resize(M);
              math::Set<T_B, Context>(
                  M,
                  convert::To<float, T_B>(1),
                  bias_multiplier_->template mutable_data<T_B>(),
                  &context_);
            }

            math::Gemm<T_B, Context, Engine>(
                CblasNoTrans,
                CblasNoTrans,
                M,
                N,
                1,
                1,
                bias_multiplier_->template data<T_B>(),
                b.template data<T_B>(),
                1,
                Y->template mutable_data<T_Y>(),
                &context_,
                math_type);

            return true;
        */
    }

    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DoRunWithType<
            float, // X
            float, // W
            float, // B
            float, // Y
            float>(); // Math
        */
    }
}

///-----------------------------------------
pub struct FullyConnectedGradientOp<Context, Engine, const TransposeWeight: bool> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:         OperatorStorage,
    context:         Context,

    axis:            usize, //{1};
    axis_w:          usize, //{1};
    bias_multiplier: Option<Tensor>,
    float16_compute: bool,

    phantomE:        PhantomData<Engine>,
}

num_inputs!{FCGradient, 3}

num_outputs!{FCGradient, (2,3)}

tensor_inference_function!{FCGradient, /* std::bind(FCGradientShapeInference, _1, _2, false) */}

cost_inference_function!{FCGradient, /* std::bind(CostInferenceForFCGradient, _1, _2, false) */}

impl<Context, Engine,  const TransposeWeight: bool> 
FullyConnectedGradientOp<Context, Engine, TransposeWeight> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axis_(this->template GetSingleArgument<int32_t>("axis", 1)),
            axis_w_(this->template GetSingleArgument<int32_t>("axis_w", 1)),
            float16_compute_(
                this->template GetSingleArgument<bool>("float16_compute", false))
        */
    }

    #[inline] pub fn do_run_with_type<T_X, T_W, T_DY, T_B, T_DX, T_DW, T_DB, MATH>(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(0);
            const auto& W = Input(1);
            const auto& dY = Input(2);
            // batch size
            const auto canonical_axis = X.canonical_axis_index(axis_);
            const int M = X.size_to_dim(canonical_axis);
            const int K = X.size_from_dim(canonical_axis);
            const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
            const int N = TransposeWeight ? W.size_to_dim(canonical_axis_w)
                                          : W.size_from_dim(canonical_axis_w);

            auto dimErrorString = [&]() {
              return c10::str(
                  "Dimension mismatch: ",
                  "X: ",
                  X.sizes(),
                  ", W: ",
                  W.sizes(),
                  ", dY: ",
                  dY.sizes(),
                  ", axis: ",
                  axis_,
                  ", M: ",
                  M,
                  ", N: ",
                  N,
                  ", K: ",
                  K);
            };

            CAFFE_ENFORCE(M * K == X.numel(), dimErrorString());
            CAFFE_ENFORCE(K * N == W.numel(), dimErrorString());

            auto* dW = Output(0, W.sizes(), at::dtype<T_DW>());
            auto* db = Output(1, {N}, at::dtype<T_DB>());

            if (X.numel() == 0) {
              // generate a zero blob for db and dW when X is empty
              math::Set<T_DB, Context>(
                  db->numel(),
                  convert::To<float, T_DB>(0),
                  db->template mutable_data<T_DB>(),
                  &context_);
              math::Set<T_DW, Context>(
                  dW->numel(),
                  convert::To<float, T_DW>(0),
                  dW->template mutable_data<T_DW>(),
                  &context_);

              if (OutputSize() == 3) {
                Output(2, X.sizes(), at::dtype<T_DX>());
              }

              return true;
            }

            // default to FLOAT as math.h does.
            TensorProto::DataType math_type = TensorProto_DataType_FLOAT;
            if (fp16_type<MATH>()) {
              math_type = TensorProto_DataType_FLOAT16;
            }

            // Compute dW
            math::Gemm<T_DY, Context, Engine>(
                CblasTrans,
                CblasNoTrans,
                TransposeWeight ? N : K,
                TransposeWeight ? K : N,
                M,
                1,
                TransposeWeight ? dY.template data<T_DY>() : X.template data<T_X>(),
                TransposeWeight ? X.template data<T_X>() : dY.template data<T_DY>(),
                0,
                dW->template mutable_data<T_DW>(),
                &context_,
                math_type);
            if (!bias_multiplier_.has_value()) {
              bias_multiplier_ =
                  caffe2::empty({M}, at::dtype<T_B>().device(Context::GetDeviceType()));
              math::Set<T_B, Context>(
                  M,
                  convert::To<float, T_B>(1),
                  bias_multiplier_->template mutable_data<T_B>(),
                  &context_);
            } else if (bias_multiplier_->numel() != M) {
              bias_multiplier_->Resize(M);
              math::Set<T_B, Context>(
                  M,
                  convert::To<float, T_B>(1),
                  bias_multiplier_->template mutable_data<T_B>(),
                  &context_);
            }
            // Compute dB
            math::Gemv<T_DY, Context>(
                CblasTrans,
                M,
                N,
                1,
                dY.template data<T_DY>(),
                bias_multiplier_->template data<T_B>(),
                0,
                db->template mutable_data<T_DB>(),
                &context_);

            // Compute dX
            if (OutputSize() == 3) {
              auto* dX = Output(2, X.sizes(), at::dtype<T_DX>());
              math::Gemm<T_DX, Context, Engine>(
                  CblasNoTrans,
                  TransposeWeight ? CblasNoTrans : CblasTrans,
                  M,
                  K,
                  N,
                  1,
                  dY.template data<T_DY>(),
                  W.template data<T_W>(),
                  0,
                  dX->template mutable_data<T_DX>(),
                  &context_,
                  math_type);
            }
            return true;
        */
    }

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DoRunWithType<
            float, //  X
            float, //  W
            float, // dY
            float, //  B
            float, // dX
            float, // dW
            float, // dB
            float>(); // Math
        */
    }
}

register_cpu_operator!{
    FC, 
    FullyConnectedOp<CPUContext>
}

register_cpu_gradient_operator!{
    FCGradient,
    FullyConnectedGradientOp<CPUContext>
}

/**
  | Same as FC, but weight matrix is supposed
  | to be already pretransposed.
  | 
  | FCTransposed stands for calling blass
  | with no noTrans, noTrans
  |
  */
register_cpu_operator!{
    FCTransposed,
    FullyConnectedOp<
        CPUContext,
        DefaultEngine,
        false /* don't transpose weight */>
}

register_cpu_gradient_operator!{
    FCTransposedGradient,
    FullyConnectedGradientOp<
        CPUContext,
        DefaultEngine,
        DontTransposeWeight>
}

pub struct GetFCGradient;

impl GetGradientDefs for GetFCGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(def_.input_size(), 3);
        CAFFE_ENFORCE(def_.type() == "FC" || def_.type() == "FCTransposed");
        return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(1), GI(2), GI(0)});
        */
    }
}

register_gradient!{FC, GetFCGradient}

register_gradient!{FCTransposed, GetFCGradient}
