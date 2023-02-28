crate::ix!();

use crate::{
    SameTypeAsInput,
    CPUContext,
    OperatorStorage,
    TensorTypes,
    Tensor
};

pub type NumericTypes = TensorTypes<(i32, i64, f32, f64)>;
pub type IntTypes     = TensorTypes<(i32, i64)>;
pub type BoolTypes    = TensorTypes<(bool)>;
pub type IntBoolTypes = TensorTypes<(i32, i64, bool)>; // discrete types

///-------------------------------------------------
pub struct UnaryElementwiseWithArgsOp<InputTypes, Context, Functor, OutputTypeMap=SameTypeAsInput> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:    OperatorStorage,
    context:    Context,

    functor:    Functor,
    phantomIT:  PhantomData<InputTypes>,
    phantomOTM: PhantomData<OutputTypeMap>,
}

impl<InputTypes,Context,Functor> UnaryElementwiseWithArgsOp<InputTypes,Context,Functor> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...), functor_(*this)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<InputTypes>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>() -> bool {
        todo!();
        /*
            const auto& X = Input(0);

            auto* Y = Output(
                0, X.sizes(), at::dtype<typename OutputTypeMap::template type<T>>());
            return functor_(
                X.numel(),
                X.template data<T>(),
                Y->template mutable_data<typename OutputTypeMap::template type<T>>(),
                &context_);
        */
    }
}

/**
  | UnaryFunctorWithDefaultCtor is a functor that
  | can be used as the functor of an
  | UnaryElementwiseWithArgsOp.
  |
  | It simply forwards the operator() call into
  | another functor that doesn't accept arguments
  | in its constructor.
  */
pub struct UnaryFunctorWithDefaultCtor<Functor> {
    functor: Functor,
}

impl<Functor> UnaryFunctorWithDefaultCtor<Functor> {

    #[inline] pub fn invoke<TIn, TOut, Context>(
        size: i32, 
        x: *const TIn, 
        y: *mut TOut, 
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            return functor(size, X, Y, context);
        */
    }
}

/**
  | UnaryElementwiseOp is a wrapper around
  | UnaryElementwiseWithArgsOp, with the
  | difference that it takes a functor with
  | default constructor, e.g. that does not need
  | to take into consideration any arguments
  | during operator creation.
  */
pub type UnaryElementwiseOp<InputTypes, Context, Functor, OutputTypeMap> = 
UnaryElementwiseWithArgsOp<InputTypes, Context, UnaryFunctorWithDefaultCtor<Functor>, OutputTypeMap>;

///-------------------------------------------------
pub struct BinaryElementwiseWithArgsOp<InputTypes, Context, Functor, OutputTypeMap> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:          OperatorStorage,
    context:          Context,

    legacy_broadcast: bool,
    axis:             i32,
    axis_str:         String,
    order:            String,
    functor:          Functor,
    phantomIT:        PhantomData<InputTypes>,
    phantomOTM:       PhantomData<OutputTypeMap>,
}

impl<InputTypes,Context,Functor,OutputTypeMap> BinaryElementwiseWithArgsOp<InputTypes, Context, Functor, OutputTypeMap> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(bool, "broadcast", legacy_broadcast_, false),
            OP_SINGLE_ARG(int, "axis", axis_, -1),
            OP_SINGLE_ARG(string, "axis_str", axis_str_, string("")),
            OP_SINGLE_ARG(string, "order", order_, "NCHW"),
            functor_(*this) 

        if (legacy_broadcast_) {
          if (axis_ != -1) {
            // Get axis from an explicit axis argument.
            CAFFE_ENFORCE_EQ(
                axis_str_.size(),
                0U,
                "Args axis and axis_str cannot be used simultaneously.");
          } else if (axis_str_.size()) {
            // Get the axis index semantically.
            CAFFE_ENFORCE_EQ(
                axis_str_.size(), 1U, "Unsupported axis string", axis_str_);
            const size_t semantic_axis_ = order_.find(axis_str_);
            CAFFE_ENFORCE_NE(
                semantic_axis_,
                string::npos,
                "Unrecognizable axis string ",
                axis_str_,
                " from order string ",
                order_);
            axis_ = semantic_axis_;
          } else {
            CAFFE_ENFORCE(
                axis_ == -1 && axis_str_.empty(),
                "Do not specify axis or axis_str if broadcast is not enabled.");
          }
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<InputTypes>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>() -> bool {
        todo!();
        /*
            const auto& A = Input(0);
            const auto& B = Input(1);

            const T* A_data = A.template data<T>();
            const T* B_data = B.template data<T>();
            std::vector<int> A_dims;
            std::vector<int> B_dims;
            std::vector<int64_t> C_dims;

            if (legacy_broadcast_) {
              CAFFE_ENFORCE(
                  !IsInputOutputAlias(1, 0),
                  "In-place is allowed only with the first tensor when "
                  "legacy-broadcasting");
              C_dims = A.sizes().vec();
              if (B.numel() == 1) {
                A_dims = {static_cast<int>(A.numel())};
                B_dims = {1};
              } else {
                size_t pre, n, post;
                std::tie(pre, n, post) =
                    elementwise_ops_utils::ComputeLegacyBroadcastSizes(A, B, axis_);
                A_dims = {
                    static_cast<int>(pre), static_cast<int>(n), static_cast<int>(post)};
                B_dims = {static_cast<int>(n), 1};
              }
            } else {
              std::copy(
                  A.sizes().cbegin(), A.sizes().cend(), std::back_inserter(A_dims));
              std::copy(
                  B.sizes().cbegin(), B.sizes().cend(), std::back_inserter(B_dims));
              // TODO: change the types to vector<int64_t>
              auto C_dims_int =
                  elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
                      A_dims, B_dims);
              std::copy(
                  C_dims_int.cbegin(), C_dims_int.cend(), std::back_inserter(C_dims));
              if (IsInputOutputAlias(0, 0)) {
                CAFFE_ENFORCE_EQ(C_dims_int, A_dims);
              } else if (IsInputOutputAlias(1, 0)) {
                CAFFE_ENFORCE_EQ(C_dims_int, B_dims);
              }
            }

            auto* C = Output(
                0, C_dims, at::dtype<typename OutputTypeMap::template type<T>>());
            auto* C_data =
                C->template mutable_data<typename OutputTypeMap::template type<T>>();
            return functor_.Forward(A_dims, B_dims, A_data, B_data, C_data, &context_);
        */
    }
}

///--------------------------------------
pub struct BinaryElementwiseWithArgsGradientOp<InputTypes, Context, Functor, OutputTypeMap, GradientTypeMap> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:                OperatorStorage,
    context:                Context,

    legacy_broadcast:       bool,
    axis:                   i32,
    axis_str:               String,
    order:                  String,
    functor:                Functor,
    phantomIT:              PhantomData<InputTypes>,
    phantomOTM:             PhantomData<OutputTypeMap>,
    phantomGradientTypeMap: PhantomData<GradientTypeMap>,
}

impl<InputTypes, Context, Functor, OutputTypeMap, GradientTypeMap> 
BinaryElementwiseWithArgsGradientOp<InputTypes, Context, Functor, OutputTypeMap, GradientTypeMap> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(bool, "broadcast", legacy_broadcast_, false),
            OP_SINGLE_ARG(int, "axis", axis_, -1),
            OP_SINGLE_ARG(string, "axis_str", axis_str_, ""),
            OP_SINGLE_ARG(string, "order", order_, "NCHW"),
            functor_(*this) 

        if (legacy_broadcast_) {
          if (axis_ != -1) {
            // Get axis from an explicit axis argument.
            CAFFE_ENFORCE_EQ(
                axis_str_.size(),
                0U,
                "Args axis and axis_str cannot be used simultaneously.");
          } else if (axis_str_.size()) {
            // Get the axis index semantically.
            CAFFE_ENFORCE_EQ(
                axis_str_.size(), 1U, "Unsupported axis string", axis_str_);
            const size_t semantic_axis_ = order_.find(axis_str_);
            CAFFE_ENFORCE_NE(
                semantic_axis_,
                string::npos,
                "Unrecognizable axis string ",
                axis_str_,
                " from order string ",
                order_);
            axis_ = semantic_axis_;
          } else {
            CAFFE_ENFORCE(
                axis_ == -1 && axis_str_.empty(),
                "Do not specify axis or axis_str if broadcast is not enabled.");
          }
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<InputTypes>::call(this, Input(1));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& dC = Input(0);
            const auto& A = Input(1);
            const auto& B = Input(2);

            vector<int> A_dims;
            vector<int> B_dims;
            if (legacy_broadcast_) {
              if (B.numel() == 1) {
                A_dims = {static_cast<int>(A.numel())};
                B_dims = {1};
              } else {
                size_t pre, n, post;
                std::tie(pre, n, post) =
                    elementwise_ops_utils::ComputeLegacyBroadcastSizes(A, B, axis_);
                A_dims = {
                    static_cast<int>(pre), static_cast<int>(n), static_cast<int>(post)};
                B_dims = {static_cast<int>(n), 1};
              }
            } else {
              std::copy(
                  A.sizes().cbegin(), A.sizes().cend(), std::back_inserter(A_dims));
              std::copy(
                  B.sizes().cbegin(), B.sizes().cend(), std::back_inserter(B_dims));
            }
            const typename OutputTypeMap::template type<T>* C_data = nullptr;
            if (InputSize() == 4) {
              const auto& C = Input(3);
              C_data = C.template data<typename OutputTypeMap::template type<T>>();
            }
            const auto* dC_data =
                dC.template data<typename GradientTypeMap::template type<T>>();
            const T* A_data = A.template data<T>();
            const T* B_data = B.template data<T>();
            auto* dA = Output(
                0, A.sizes(), at::dtype<typename GradientTypeMap::template type<T>>());
            auto* dB = Output(
                1, B.sizes(), at::dtype<typename GradientTypeMap::template type<T>>());
            auto* dA_data =
                dA->template mutable_data<typename GradientTypeMap::template type<T>>();
            auto* dB_data =
                dB->template mutable_data<typename GradientTypeMap::template type<T>>();
            return functor_.Backward(
                A_dims,
                B_dims,
                dC_data,
                A_data,
                B_data,
                C_data,
                dA_data,
                dB_data,
                &context_);
        */
    }
}

///-------------------------------------------------
pub struct BinaryFunctorWithDefaultCtor<Functor> {
    functor: Functor,
}

impl<Functor> BinaryFunctorWithDefaultCtor<Functor> {

    #[inline] pub fn forward<TIn, TOut, Context>(
        &mut self,
        a_dims:   &Vec<i32>,
        b_dims:   &Vec<i32>,
        a_data:   *const TIn,
        b_data:   *const TIn,
        c_data:   *mut TOut,
        context:  *mut Context) -> bool 
    {
        todo!();
        /*
            return functor.Forward(A_dims, B_dims, A_data, B_data, C_data, context);
        */
    }

    #[inline] pub fn backward<TGrad, TIn, TOut, Context>(
        &mut self,
        a_dims:     &Vec<i32>,
        b_dims:     &Vec<i32>,
        dC_data:    *const TGrad,
        a_data:     *const TIn,
        b_data:     *const TIn,
        c_data:     *const TOut,
        dA_data:    *mut TGrad,
        dB_data:    *mut TGrad,
        context:    *mut Context) -> bool 
    {
        todo!();
        /*
           return functor.Backward(
                A_dims,
                B_dims,
                dC_data,
                A_data,
                B_data,
                C_data,
                dA_data,
                dB_data,
                context);
        */
    }
}

/**
  | BinaryElementwiseOp is a wrapper around
  | BinaryElementwiseWithArgsOp, with the
  | difference that it takes a functor with
  | default constructor, e.g. that does not need
  | to take into consideration any arguments
  | during operator creation.
  */
pub type BinaryElementwiseOp<InputTypes, Context, Functor, TypeMap = SameTypeAsInput> 
= BinaryElementwiseWithArgsOp<
InputTypes,
Context,
BinaryFunctorWithDefaultCtor<Functor>,
TypeMap>;

/**
  | BinaryElementwiseGradientOp is a wrapper
  | around BinaryElementwiseGradientWithArgsOp,
  | with the difference that it takes a functor
  | with default constructor, e.g. that does not
  | need to take into consideration any arguments
  | during operator creation.
  */
pub type BinaryElementwiseGradientOp<InputTypes, Context, Functor, OutputTypeMap, GradientTypeMap> 
= BinaryElementwiseWithArgsGradientOp<
InputTypes,
Context,
BinaryFunctorWithDefaultCtor<Functor>,
OutputTypeMap,
GradientTypeMap>;

/// Forward-only Unary Functors.
#[test] fn not_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
    "Not",
    ["X"],
    ["Y"],
    )

    workspace.FeedBlob("X", (np.random.rand(3, 3) > 0.5))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X:
    [[ True False False]
    [False False False]
    [ True  True  True]]
    Y:
    [[False  True  True]
    [ True  True  True]
    [False False False]]

    */
}

/**
  | Performs element-wise negation on
  | input tensor `X`.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc
  |
  */
pub struct NotFunctor<Context> {
    phantom: PhantomData<Context>,
}

num_inputs!{Not, 1}

num_outputs!{Not, 1}

inputs!{Not, 
    0 => ("X", "*(Tensor`<bool>`)* Input tensor.")
}

outputs!{Not, 
    0 => ("Y", "*(Tensor`<bool>`)* Negated output tensor.")
}

identical_type_and_shape_of_input!{Not, 0}

inherit_onnx_schema!{Not}

should_not_do_gradient!{Not}

impl<Context> NotFunctor<Context> {
    
    #[inline] pub fn invoke(
        &self, 
        n:       i32,
        x:       *const bool,
        y:       *mut bool,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Not(N, X, Y, context);
        return true;
        */
    }
}

///------------------------------------------
#[test] fn sign_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
    "Sign",
    ["X"],
    ["Y"],
    )

    workspace.FeedBlob("X", (np.random.rand(3, 3).astype(np.float32) - np.random.rand(3, 3).astype(np.float32)))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    **Result**

    X:
    [[ 0.02816287  0.22408086 -0.30342305]
    [-0.18481976  0.03948995  0.39698976]
    [-0.63304734 -0.6919183  -0.31524038]]
    Y:
    [[ 1.  1. -1.]
    [-1.  1.  1.]
    [-1. -1. -1.]]
    */
}

/**
  | Computes sign for each element of the
  | input: -1, 0 or 1.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc
  |
  */
pub struct SignFunctor<Context> {
    phantom: PhantomData<Context>,
}

num_inputs!{Sign, 1}

num_outputs!{Sign, 1}

inputs!{Sign, 
    0 => ("X", "*(type: Tensor`<float>`)* Input data tensor.")
}

outputs!{Sign, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor.")
}

identical_type_and_shape_of_input!{Sign, 0}

should_not_do_gradient!{Sign}

impl<Context> SignFunctor<Context> {

    #[inline] pub fn invoke<T>(
        &mut self,
        n: i32, 
        x: *const T, 
        y: *mut T, 
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Sign(N, X, Y, context);
            return true;
        */
    }
}

// Forward-only Binary Functors.
#[macro_export] macro_rules! declare_forward_only_binary_functor {
    ($FunctorName:ident) => {
        /*
        template <class Context>                                   
            struct FunctorName##Functor {                              
                template <typename TIn, typename TOut>                   
                    bool Forward(                                            
                        const std::vector<int>& A_dims,                      
                        const std::vector<int>& B_dims,                      
                        const TIn* A,                                        
                        const TIn* B,                                        
                        TOut* C,                                             
                        Context* context) const {                            
                        math::FunctorName(                                     
                            A_dims.size(),                                     
                            A_dims.data(),                                     
                            B_dims.size(),                                     
                            B_dims.data(),                                     
                            A,                                                 
                            B,                                                 
                            C,                                                 
                            context);                                          
                        return true;                                           
                    }                                                        
            };
        */
    }
}

// Compare functors.
declare_forward_only_binary_functor!{EQ}
declare_forward_only_binary_functor!{NE}
declare_forward_only_binary_functor!{LT}
declare_forward_only_binary_functor!{LE}
declare_forward_only_binary_functor!{GT}
declare_forward_only_binary_functor!{GE}

// Logical functors.
declare_forward_only_binary_functor!{And}
declare_forward_only_binary_functor!{Or}
declare_forward_only_binary_functor!{Xor}

// Bitwise functors.
declare_forward_only_binary_functor!{BitwiseAnd}
declare_forward_only_binary_functor!{BitwiseOr}
declare_forward_only_binary_functor!{BitwiseXor}

///-------------------

/**
 | SumReduceLike operator takes 2 tensors as
 | input. It performs reduce sum to the first input
 | so that the output looks like the second one.
 |
 | It assumes that the first input has more
 | dimensions than the second, and the dimensions of
 | the second input is the contiguous subset of the
 | dimensions of the first.
 |
 | For example, the following tensor shapes are
 | supported:
 |
 |   shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
 |   shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
 |   shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
 |   shape(A) = (2, 3, 2, 5), shape(B) = (2), with axis=0
 |
 | Sum reduction operator that is used for computing
 | the gradient in cases where the forward op is in
 | broadcast mode.
 */
pub struct SumReduceLikeOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    axis:     i32,
    axis_str: String,
    order:    String,

    ones:       Tensor, // {Context::GetDeviceType()};
    sum_buffer: Tensor, // {Context::GetDeviceType()};
}

num_inputs!{SumReduceLike, 2}

num_outputs!{SumReduceLike, 1}

inputs!{SumReduceLike, 
    0 => ("A", "First operand, should share the type with the second operand."),
    1 => ("B", "Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.")
}

outputs!{SumReduceLike, 
    0 => ("C", "Result, has same dimensions and type as B")
}

args!{SumReduceLike, 
    0 => ("axis",      "If set, defines the starting dimension for reduction. Args `axis` and `axis_str` cannot be used simultaneously."),
    1 => ("axis_str",  "If set, it could only be N or C or H or W. `order` arg should also be provided. It defines the reduction dimensions on NCHW or NHWC. Args `axis` and `axis_str` cannot be used simultaneously."),
    2 => ("order",     "Either NHWC or HCWH")
}

identical_type_and_shape_of_input!{SumReduceLike, 0}

impl<Context> SumReduceLikeOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "axis", axis_, -1),
            OP_SINGLE_ARG(string, "axis_str", axis_str_, ""),
            OP_SINGLE_ARG(string, "order", order_, "NCHW") 

        if (axis_ != -1) {
          // Get axis from an explicit axis argument.
          CAFFE_ENFORCE_EQ(
              axis_str_.size(),
              0U,
              "Args axis and axis_str cannot be used simultaneously.");
        } else if (axis_str_.size()) {
          // Get the axis index semantically.
          CAFFE_ENFORCE_EQ(
              axis_str_.size(), 1U, "Unsupported axis string", axis_str_);
          size_t semantic_axis = order_.find(axis_str_);
          CAFFE_ENFORCE_NE(
              semantic_axis,
              string::npos,
              "Unrecognizable axis string ",
              axis_str_,
              " from order string ",
              order_);
          axis_ = semantic_axis;
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
        */
    }
}

register_cpu_operator!{
    Not,
    UnaryElementwiseOp<BoolTypes, CPUContext, NotFunctor<CPUContext>>
}

register_cpu_operator!{
    Sign,
    UnaryElementwiseOp<NumericTypes, CPUContext, SignFunctor<CPUContext>>
}

///----------------------------------------
#[macro_export] macro_rules! register_cpu_compare_operator {
    ($Op:ident) => {
        /*
        REGISTER_CPU_OPERATOR(                                      
            Op,                                                     
            BinaryElementwiseOp<                                    
            TensorTypes<bool, int32_t, int64_t, float, double>, 
            CPUContext,                                         
            Op##Functor<CPUContext>,                            
            FixedType<bool>>)
        */
    }
}

register_cpu_compare_operator!{EQ}
register_cpu_compare_operator!{NE}
register_cpu_compare_operator!{LT}
register_cpu_compare_operator!{LE}
register_cpu_compare_operator!{GT}
register_cpu_compare_operator!{GE}

#[test] fn lt_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LT",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3]))
    workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8]))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A: [ 1  5  2  9 12  3]
    B: [ 1  3  4  9 12  8]
    C: [False False  True False False  True]

    */
}

#[test] fn le_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LE",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3]))
    workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8]))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A: [ 1  5  2  9 12  3]
    B: [ 1  3  4  9 12  8]
    C: [ True False  True  True  True  True]

    */
}

#[test] fn gt_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "GT",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3]))
    workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8]))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A: [ 1  5  2  9 12  3]
    B: [ 1  3  4  9 12  8]
    C: [False  True False False False False]

    */
}

#[test] fn ge_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "GE",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3]))
    workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8]))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A: [ 1  5  2  9 12  3]
    B: [ 1  3  4  9 12  8]
    C: [ True  True False  True  True False]
    
    */
}

#[test] fn eq_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "EQ",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3]))
    workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8]))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A: [ 1  5  2  9 12  3]
    B: [ 1  3  4  9 12  8]
    C: [ True False False  True  True False]
    */
}

#[test] fn ne_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "NE",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3]))
    workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8]))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A: [ 1  5  2  9 12  3]
    B: [ 1  3  4  9 12  8]
    C: [False  True  True False False  True]
    */
}

///----------------------------------------
#[macro_export] macro_rules! register_cpu_logical_binary_operator {
    ($Op:ident) => {
        /*
        REGISTER_CPU_OPERATOR(                         \
            Op, BinaryElementwiseOp<BoolTypes, CPUContext, Op##Functor<CPUContext>>)
        */
    }
}

register_cpu_logical_binary_operator!{And}
register_cpu_logical_binary_operator!{Or}
register_cpu_logical_binary_operator!{Xor}

#[test] fn and_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "And",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", (np.random.rand(3, 3) > 0.5))
    workspace.FeedBlob("B", (np.random.rand(3, 3) > 0.5))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A:
     [[ True False False]
     [False  True False]
     [False False  True]]
    B:
     [[ True False  True]
     [False False False]
     [False False False]]
    C:
     [[ True False False]
     [False False False]
     [False False False]]

    */
}

#[test] fn or_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Or",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", (np.random.rand(3, 3) > 0.5))
    workspace.FeedBlob("B", (np.random.rand(3, 3) > 0.5))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A:
    [[False  True  True]
     [False  True  True]
     [ True  True  True]]
    B:
    [[False  True False]
     [ True  True  True]
     [False  True False]]
    C:
    [[False  True  True]
     [ True  True  True]
     [ True  True  True]]

    */
}

#[test] fn xor_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Xor",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", (np.random.rand(3, 3) > 0.5))
    workspace.FeedBlob("B", (np.random.rand(3, 3) > 0.5))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A:
    [[ True  True  True]
     [False False  True]
     [False  True False]]
    B:
    [[False False False]
     [ True  True  True]
     [False False False]]
    C:
    [[ True  True  True]
     [ True  True False]
     [False  True False]]
    */
}

///---------------------------------------------
#[macro_export] macro_rules! register_cpu_bitwise_binary_operator {
    ($Op:ident) => {
        /*
        REGISTER_CPU_OPERATOR(                         \
            Op,                                        \
            BinaryElementwiseOp<IntBoolTypes, CPUContext, Op##Functor<CPUContext>>)
        */
    }
}

register_cpu_bitwise_binary_operator!{BitwiseAnd}
register_cpu_bitwise_binary_operator!{BitwiseOr}
register_cpu_bitwise_binary_operator!{BitwiseXor}

pub mod srl_helper {

    use super::*;

    #[inline] pub fn sum2one<T>(x: *const T, y: *mut T, n: usize) {
        todo!();
        /*
            *y = ConstEigenArrayMap<T>(x, n, 1).sum();
        */
    }

    #[inline] pub fn run_with_broadcast_front<T>(
        x: *const T,
        y: *mut T,
        pre: usize,
        n: usize,
        context: *mut CPUContext) 
    {
        todo!();
        /*
            EigenArrayMap<T>(y, n, 1) = ConstEigenArrayMap<T>(x, n, pre).rowwise().sum();
        */
    }


    #[inline] pub fn run_with_broadcast_back<T>(
        x:       *const T,
        y:       *mut T,
        post:    usize,
        n:       usize,
        context: *mut CPUContext) 
    {
        todo!();
        /*
            EigenArrayMap<T>(y, 1, n) = ConstEigenArrayMap<T>(x, post, n).colwise().sum();
        */
    }

    #[inline] pub fn run_with_broadcast2<T>(
        a:        *const T,
        y:        *mut T,
        pre:      usize,
        n:        usize,
        post:     usize,
        context:  *mut CPUContext) 
    {
        todo!();
        /*
            for (auto i = 0U; i < n; ++i) {
            y[i] = 0;
            for (auto j = 0U; j < pre; ++j) {
              for (auto k = 0U; k < post; ++k) {
                y[i] += a[(j * n + i) * post + k];
              }
            }
          }
        */
    }
}

impl SumReduceLikeOp<CPUContext> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& A = Input(0);
          const auto& B = Input(1);

          CAFFE_ENFORCE(!IsInputOutputAlias(1, 0), "In-place is not allowed.");
          auto* C = Output(0, B.sizes(), at::dtype<T>());
          const T* Adata = A.template data<T>();
          auto* Cdata = C->template mutable_data<T>();
          if (B.numel() == 1) {
            auto count = A.numel();
            SRLHelper::sum2one<T>(Adata, Cdata, count);
          } else {
            size_t pre, n, post;
            std::tie(pre, n, post) =
                elementwise_ops_utils::ComputeLegacyBroadcastSizes(A, B, axis_);
            if (post == 1) {
              SRLHelper::RunWithBroadcastFront<T>(Adata, Cdata, pre, n, &context_);
            } else if (pre == 1) {
              SRLHelper::RunWithBroadcastBack<T>(Adata, Cdata, post, n, &context_);
            } else {
              SRLHelper::RunWithBroadcast2<T>(Adata, Cdata, pre, n, post, &context_);
            }
          }
          return true;
        */
    }
}

register_cpu_operator!{
    SumReduceLike, 
    SumReduceLikeOp<CPUContext>
}
