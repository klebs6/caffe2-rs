crate::ix!();

use crate::{
    GradientMakerBase,
    Workspace,
    OperatorStorage,
    CPUContext,
    OpSchemaCost,
    OperatorDef,
    TensorShape,
    TensorPrinter,
    TensorProto_DataType,
    Tensor,
};

/**
  | Identity operator, but checks all values
  | for nan or inf
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct NanCheckOp<Context, W: Write> {

    storage:        OperatorStorage,
    context:        Context,

    tensor_printer: TensorPrinter<W>,
    scratch:        Tensor,
}

register_cpu_operator!{NanCheck, NanCheckOp<CPUContext>}

register_gradient!{NanCheck,     GetNanCheckGradient}

num_inputs!{NanCheck, (1,INT_MAX)}

num_outputs!{NanCheck, 1}

inputs!{NanCheck, 
    0 => ("tensor", "Tensor to check for nan/inf")
}

outputs!{NanCheck, 
    0 => ("output", "Tensor to copy input into if no NaNs or inf. Can be in-place")
}

identical_type_and_shape_of_input!{NanCheck, 0}

allow_inplace!{NanCheck, vec![(0, 0)]}

impl<Context,W: Write> NanCheckOp<Context,W> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

impl<CPUContext,W: Write> NanCheckOp<CPUContext,W> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      auto* Y = Output(0);
      const int D = X.numel();
      const float* data = X.data<float>();
      ConstEigenVectorMap<float> input_data(data, D);

      bool all_finite = input_data.allFinite();

      if (!all_finite) {
        std::cerr << "Tensor contained NaN or inf: [" << this->debug_def().input(0)
                  << "]" << std::endl;

        for (int j = 0; j < InputSize(); j++) {
          std::cerr << "Tensor name: " << this->debug_def().input(j) << std::endl;
          std::cerr << "Input tensor:" << std::endl;
          tensorPrinter_.Print<float>(Input(j));
          std::cerr << "NaN idxs:" << std::endl;
          const float* x = Input(j).data<float>();
          for (size_t i = 0; i < Input(j).numel(); ++i) {
            if (std::isnan(x[i]) || std::isinf(x[i])) {
              std::cerr << i << " ";
            }
          }
          std::cerr << std::endl;
        }
        return false;
      }

      if (&X != Y) {
        Y->CopyFrom(X);
      }
      return true;
        */
    }
}

pub struct GetNanCheckGradient;

impl GetGradientDefs for GetNanCheckGradient {
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return {CreateOperatorDef(
            "NanCheck",
            "",
            std::vector<string>{GO(0)},
            std::vector<string>{GI(0)})};
        */
    }
}

/**
  | Returns a new tensor with boolean elements
  | representing if each element is NaN
  | or not.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct IsNanOp<Context> {

    storage: OperatorStorage,
    context: Context,
}

register_cpu_operator!{IsNaN, IsNanOp<CPUContext>}

num_inputs!{IsNaN, 1}

num_outputs!{IsNaN, 1}

inputs!{IsNaN, 
    0 => ("tensor", "Tensor to check for nan")
}

outputs!{IsNaN, 
    0 => ("output", "Tensor containing a 1 at each location of NaN elements.")
}


impl<Context> IsNanOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self, ) -> bool {
        todo!();
        /*
            auto& X = Input(0);
        auto* Y = Output(0, X.sizes(), at::dtype<uint8_t>());
        const auto* X_data = X.template data<T>();
        uint8_t* Y_data = Y->template mutable_data<uint8_t>();
        for (size_t i = 0; i < X.numel(); i++) {
          Y_data[i] = (uint8_t)(std::isnan(X_data[i]));
        }
        return true;
        */
    }
}

/**
  | Time since epoch in nanoseconds.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct WallClockTimeOp<Context> {

    storage: OperatorStorage,
    context: Context,
}

num_inputs!{WallClockTime, 0}

num_outputs!{WallClockTime, 1}

outputs!{WallClockTime, 
    0 => ("time", "The time in nanoseconds.")
}

should_not_do_gradient!{WallClockTime}

impl<Context> WallClockTimeOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            int64_t nanoseconds = static_cast<long int>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
                .count());

        TensorCPU* output = Output(0);
        output->Resize();
        *output->template mutable_data<int64_t>() = nanoseconds;

        return true;
        */
    }
}

///-----------------------------------------
pub const kPrintFileExtension: &'static str = ".log";

/**
  | Logs shape and contents of input tensor
  | to stderr or to a file.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct PrintOp<Context, W: Write> {

    //USE_DISPATCH_HELPER;
    storage:           OperatorStorage,
    context:           Context,

    tensor_printer:    TensorPrinter<W>,
    every_n:           i32,
    occurrences_mod_n: i32, // default = 0
}

should_not_do_gradient!{Print}

num_inputs!{Print, 1}

num_outputs!{Print, 0}

inputs!{Print, 
    0 => ("tensor", "The tensor to print.")
}

args!{Print, 
    0 => ("to_file", "(bool) if 1, saves contents to the root folder of the current workspace, appending the tensor contents to a file named after the blob name. Otherwise, logs to stderr."),
    1 => ("limit",   "(int, default 0) If set, prints the first `limit` elements of tensor. If 0, prints the first `k_limit_default`(1000) elements of tensor"),
    2 => ("every_n", "(int, default 1) Print tensor every `every_n` runs")
}

impl<Context, W: Write> PrintOp<Context,W> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            tensor_printer_(
                operator_def.input(0),
                this->template GetSingleArgument<int>("to_file", 0)
                    ? ws->RootFolder() + "/" + operator_def.input(0) +
                        kPrintFileExtension
                    : "",
                this->template GetSingleArgument<int>("limit", 0)),
            every_n_(this->template GetSingleArgument<int>("every_n", 1)) 

        CAFFE_ENFORCE_GE(every_n_, 1);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (++occurrences_mod_n_ > every_n_) {
          occurrences_mod_n_ -= every_n_;
        }
        if (occurrences_mod_n_ != 1) {
          return true;
        }

        if (!this->InputIsTensorType(0, Context::GetDeviceType()) &&
            !this->InputIsTensorType(0, CPU)) {
          LOG(INFO) << "Blob of type: "
                    << OperatorStorage::Inputs().at(0)->meta().name();
          return true;
        }
        // special-case empty tensors since they may have no meta()
        if (Input(0).numel() == 0) {
          tensor_printer_.PrintMeta(Input(0));
          return true;
        }

        using Types = TensorTypes<
            float,
            double,
            int,
            long,
            bool,
            char,
            unsigned char,
            std::string>;

        if (this->InputIsTensorType(0, CPU)) {
          return DispatchHelper<Types>::call(
              this, this->template Input<Tensor>(0, CPU));
        } else {
          return DispatchHelper<Types>::call(this, Input(0));
        }
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            // A simple strategy to copy tensor if needed, and have the tensor pointer
        // pointing to the right instantiation. Note that tensor_copy_if_needed
        // will handle memory deallocation itself so no smart pointer is needed.
        const TensorCPU* tensor;
        Tensor tensor_copy_if_needed(CPU);
        if (this->InputIsTensorType(0, CPU)) {
          tensor = &this->template Input<Tensor>(0, CPU);
        } else {
          // sync copy
          tensor_copy_if_needed.CopyFrom(Input(0));
          tensor = &tensor_copy_if_needed;
        }
        tensor_printer_.Print<T>(*tensor);
        return true;
        */
    }
}

/**
  | -----------
  | @brief
  | 
  | Alias op makes the output and the input
  | share the same underlying storage.
  | 
  | WARNING: in general, in caffe2's operator
  | interface different tensors should
  | have different underlying storage,
  | which is the assumption made by components
  | such as the dependency engine and memory
  | optimization.
  | 
  | Thus, in normal situations you should
  | not use the AliasOp, especially in a
  | normal forward-backward pass.
  | 
  | The Alias op is provided so one can achieve
  | true asynchrony, such as
  | 
  | Hogwild, in a graph.
  | 
  | But make sure you understand all the
  | implications similar to multi-thread
  | computation before you use it explicitly.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AliasOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{Alias, 1}

num_outputs!{Alias, 1}

inputs!{Alias, 
    0 => ("input", "Input tensor whose storage will be shared.")
}

outputs!{Alias, 
    0 => ("output", "Tensor of same shape as input, sharing its storage.")
}

identical_type_and_shape!{Alias}

impl<Context> AliasOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        CAFFE_ENFORCE_GE(input.numel(), 0, "Tensor is not initialized");
        OutputTensorAlias(0, input);
        return true;
        */
    }
}

/**
  | This operator converts dense or sparse
  | gradients to dense ones.
  | 
  | Therefore, sparse gradient can be back
  | propagated to Operators that consume
  | dense gradients only (e.g., FCGradient).
  | 
  | The operator's behaviors:
  | 
  | - In forward, simply pass in place or
  | copy input to the output.
  | 
  | - In backward, if the gradient passed-in
  | is sparse gradient, change it to dense
  | gradient in linear time; otherwise,
  | simply pass the dense gradient.
  | 
  | -----------
  | @brief
  | 
  | Pass inputs to outputs.
  | 
  | Input:
  |     DATA - dense tensor.
  | 
  | Output:
  |     DATA - same tensor as input.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct EnsureDenseOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{EnsureDense, 1}

num_outputs!{EnsureDense, 1}

inputs!{EnsureDense, 
    0 => ("input", "Input tensors.")
}

outputs!{EnsureDense, 
    0 => ("output", "Output tensor. Same dimension as inputs.")
}

identical_type_and_shape!{EnsureDense}

allow_inplace!{EnsureDense, vec![(0, 0)]}

register_gradient!{EnsureDense, GetEnsureDenseGradient}

impl<Context> EnsureDenseOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& input = Input(0);
        auto* output = Output(0);
        CAFFE_ENFORCE_GT(input.dim(), 0, "Input has to be at least a vector.");
        // it is allowed to have the output inplace overwrite the input but also
        // allow the output to be copied from the input
        if (&input != output) {
          output->ResizeLike(input);
          output->CopyFrom(input, true /*async*/);
        }
        return true;
        */
    }
}

///-----------------------------------------
#[test] fn flatten_to_vec_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "FlattenToVec",
        ["input"],
        ["output"],
    )

    workspace.FeedBlob("input", np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]).astype(np.float32))
    print("input:\n", workspace.FetchBlob("input"))

    workspace.RunOperatorOnce(op)
    print("output: \n", workspace.FetchBlob("output"))

    input:
     [[ 1.  2.  3.]
     [ 4.  5.  6.]
     [ 7.  8.  9.]
     [10. 11. 12.]]
    output:
     [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12.]

    */
}

/**
  | The *FlattenToVec* op flattens the
  | input tensor into a 1-D vector.
  | 
  | The op accepts a single input tensor
  | and returns a single output tensor.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FlattenToVecOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{FlattenToVec, 1}

num_outputs!{FlattenToVec, 1}

inputs!{FlattenToVec, 
    0 => ("input", "A tensor of rank >= 1.")
}

outputs!{FlattenToVec, 
    0 => ("output", "A tensor of rank 1 (vector) with the contents of the input tensor.")
}

tensor_inference_function!{
    FlattenToVec, 
    /* [](const OperatorDef& /*def*/,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      int total = 1;
      for (auto d : in[0].dims()) {
        total *= d;
      }
      out[0].set_data_type(in[0].data_type());
      out[0].add_dims(total);
      return out;
    } */
}

impl<Context> FlattenToVecOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        auto* output = Output(0);
        CAFFE_ENFORCE_GE(input.dim(), 1, "The rank of the tensor must be >= 1.");
        output->Resize(input.numel());

        context_.CopyItemsSameDevice(
            input.dtype(),
            input.numel(),
            input.raw_data(),
            output->raw_mutable_data(input.dtype()));
        return true;
        */
    }
}

/**
  | Produces tensor containing data of
  | first input and shape of second input.
  | 
  | Output gets the data of input(0), but
  | reshapes it like input(1).
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ResizeLikeOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{ResizeLike, 2}

num_outputs!{ResizeLike, 1}

inputs!{ResizeLike, 
    0 => ("data",         "Tensor whose data will be copied into the output."),
    1 => ("shape_tensor", "Tensor whose shape will be applied to output.")
}

outputs!{ResizeLike, 
    0 => ("output",       "Tensor with data of input 0 and shape of input 1.")
}

tensor_inference_function!{
    ResizeLike, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out(1);
          out.at(0) = in.at(1);
          out.at(0).set_data_type(in.at(0).data_type());
          return out;
        */
    }
}

impl<Context> ResizeLikeOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input0 = Input(0);
        auto& input1 = Input(1);
        auto* output = Output(0);
        CAFFE_ENFORCE_EQ(input0.numel(), input1.numel());
        output->ResizeLike(Input(1));
        context_.CopyItemsSameDevice(
            input0.dtype(),
            input0.numel(),
            input0.raw_data(),
            output->raw_mutable_data(input0.dtype()));
        return true;
        */
    }
}

///-----------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SumOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{SumInt, (1,INT_MAX)}

num_outputs!{SumInt, 1}

inputs_can_cross_devices!{SumInt}

tensor_inference_function!{SumInt, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out(1);
          out.push_back(in[0]);
          out[0].set_data_type(TensorProto::INT32);
          return out;
        */
    }
}

allow_inplace!{SumInt, vec![(0, 0)]}

impl<Context> SumOp<Context> {
    
    #[inline] pub fn do_run_with_type<T>(&mut self, ) -> bool {
        todo!();
        /*
            auto& input0 = Input(0);

        if (InputSize() == 1) {
          // TODO: better TensorOptions argument passing(e.g. default argument)
          OutputTensorCopyFrom(
              0,
              // I'll change the order of argument in another diff, so that we don't
              // need to write this
              at::dtype(input0.dtype()),
              input0,
              true /*async*/);
          return true;
        }
        auto* output = Output(0, input0.sizes(), at::dtype<T>());
        T* output_data = output->template mutable_data<T>();
        // Dimension checking
        for (int i = 1; i < InputSize(); ++i) {
          if (output->sizes() != Input(i).sizes()) {
            CAFFE_THROW(
                "Check failed: output->sizes() == Input(i).sizes().",
                "Description: Input #",
                i,
                ", input dimension:",
                Input(i).sizes(),
                " should match output dimension: ",
                output->sizes());
          }
        }

        // Add the first two - works if in-place or not.
        math::Add(
            output->numel(),
            input0.template data<T>(),
            Input(1).template data<T>(),
            output_data,
            &context_);
        // Add remaining.
        for (int i = 2; i < InputSize(); ++i) {
          math::Add(
              output->numel(),
              output_data,
              Input(i).template data<T>(),
              output_data,
              &context_);
        }
        return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double, int32_t, int64_t>>::call(
            this, Input(0));
        */
    }
}

#[inline] pub fn cost_inference_for_sum(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> OpSchemaCost {
    
    todo!();
    /*
        struct OpSchema::Cost cost = PointwiseCostInference<1>(def, in);
      cost.flops *= (in.size() - 1);
      cost.params_bytes = 0;
      return cost;
    */
}

/**
  | WeightedSumOp computes the weighted
  | sum of several tensors.
  | 
  | The input should be in the form X_0, weight_0,
  | X_1, weight_1, ... where X_i all have
  | the same shape, and weight_i are size
  | 1 tensors that specifies the weight
  | of each vector.
  | 
  | -----------
  | @note
  | 
  | if one wants to do in-place computation,
  | it could only be done with X_0 also as
  | the output, but not other X_i.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct WeightedSumOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_outputs!{WeightedSum, 1}

inputs!{WeightedSum, 
    0 => ("data_0", "First of the input tensors."),
    1 => ("weight_0", "Weight of the first input in the sum.")
}

outputs!{WeightedSum, 
    0 => ("output", "Result containing weighted elem-wise sum of inputs.")
}

num_inputs!{WeightedSum, 
    |n: i32| {
        n > 0 && n % 2 == 0
    }
}

tensor_inference_function!{WeightedSum, 
    WeightedSumShapeInference 
}

cost_inference_function!{WeightedSum, 
    CostInferenceForWeightedSum 
}

allow_inplace!{WeightedSum, vec![(0, 0)]}

identical_type_and_shape_of_input!{WeightedSum, 0}

impl<Context> WeightedSumOp<Context> {
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            // the code is written this way because of 10.1 + gcc 7.3.1 compiler bug
        // as discussed at
        // https://devtalk.nvidia.com/default/topic/1048037/linux/cuda-10-1-nvidia-you-re-now-quot-fixing-quot-gcc-bugs-that-gcc-doesn-t-even-have/
        const int input_size = (*this).InputSize();
        CAFFE_ENFORCE_EQ(input_size % 2, 0);
        const auto& X0 = Input(0);
        const auto& weight0 = Input(1);
        CAFFE_ENFORCE_EQ(weight0.numel(), 1);
        const int size = X0.numel();
        // Note: removed Aliasing check, since Output already has
        // caching capability
        auto* Y = Output(0, X0.sizes(), at::dtype<T>());
        T* Y_data = Y->template mutable_data<T>();
        if (X0.numel() == 0) {
          return true;
        }
        CAFFE_ENFORCE_GT(X0.numel(), 0);
        if (input_size == 2) {
          math::Scale<float, T>(
              size,
              weight0.template data<float>(),
              X0.template data<T>(),
              Y_data,
              &context_);
          return true;
        }
        const auto& X1 = Input(2);
        CAFFE_ENFORCE(
            !IsInputOutputAlias(2, 0),
            "Input #2 is the same as output. If you want to do in-place updates, "
            "put the output as input #0.");
        const auto& weight1 = Input(3);
        CAFFE_ENFORCE_EQ(X1.numel(), size);
        CAFFE_ENFORCE_EQ(weight1.numel(), 1);
        if (!IsInputOutputAlias(0, 0)) {
          context_.template CopySameDevice<T>(size, X0.template data<T>(), Y_data);
        }
        math::Axpby<float, T, Context>(
            size,
            weight1.template data<float>(),
            X1.template data<T>(),
            weight0.template data<float>(),
            Y_data,
            &context_);
        for (int i = 4; i < input_size; i += 2) {
          const auto& Xi = Input(i);
          // Do a check: if the input is the same as output, we have a problem -
          // in-place update should always only happen with the zeroth input.
          const std::string err_msg = "Input #" + to_string(i) +
              " is the same as output. If you want to do in-place updates, "
              "put the output as input #0.";
          CAFFE_ENFORCE(!IsInputOutputAlias(i, 0), err_msg);
          const auto& weighti = Input(i + 1);
          CAFFE_ENFORCE_EQ(Xi.numel(), size);
          CAFFE_ENFORCE_EQ(weighti.numel(), 1);
          math::Axpy<float, T, Context>(
              size,
              weighti.template data<float>(),
              Xi.template data<T>(),
              Y_data,
              &context_);
        }
        return true;
        */
    }
}

///-----------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct WeightedSumGradientOp<Context> {
    storage: OperatorStorage,
    context: Context,
    grad_on_w: bool,
}

num_outputs!{WeightedSumGradient, (1,INT_MAX)}

num_inputs!{WeightedSumGradient, 
    |n: i32| {
        n > 0 && n % 2 == 1
    }
}

impl<Context> WeightedSumGradientOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            grad_on_w_(this->template GetSingleArgument<bool>("grad_on_w", false))
        */
    }
    
    #[inline] pub fn do_run_with_type<DstType>(&mut self) -> bool {
        todo!();
        /*
            CAFFE_ENFORCE_EQ(InputSize() % 2, 1);
        auto output_size = grad_on_w_ ? InputSize() - 1 : InputSize() / 2;
        CAFFE_ENFORCE_EQ(OutputSize(), output_size);

        auto& dY = Input(0);
        const auto* dY_data = dY.template data<DstType>();
        int size = dY.numel();

        // The input size should be the input size of the forward op plus 1
        for (int i = 0; i < InputSize() / 2; i++) {
          auto& cur_w = Input(2 * i + 2);
          CAFFE_ENFORCE_EQ(cur_w.numel(), 1);

          auto* cur_dX = Output(i, dY.sizes(), at::dtype<DstType>());

          math::Scale<float, DstType, Context>(
              size,
              cur_w.template data<float>(),
              dY_data,
              cur_dX->template mutable_data<DstType>(),
              &context_);

          if (grad_on_w_) {
            auto& cur_X = Input(2 * i + 1);
            CAFFE_ENFORCE_EQ(cur_X.numel(), size);
            auto* cur_dw = Output(i + output_size / 2);
            cur_dw->Resize(1);
            math::Dot<DstType, Context>(
                size,
                dY_data,
                cur_X.template data<DstType>(),
                cur_dw->template mutable_data<float>(),
                &context_);
          }
        }

        return true;
        */
    }
}

/**
  | Similar to WeightedSum, computes the
  | weighted sum of several tensors, with
  | the difference that inputs are sliced
  | tensors.
  | 
  | The first tensor has to be in-place and
  | only slices of it on the first dimension
  | as indexed by
  | 
  | INDICES will be updated.
  | 
  | -----------
  | @brief
  | 
  | Update slices of the tensor in-place
  | with weighted sum.
  | 
  | ScatterWeightedSumOp is similar to
  | WeightedSum and computes the weighted
  | sum of several tensors.
  | 
  | The first tensor has to be in-place and
  | only slices of it on the first dimension
  | as indexed by
  | 
  | INDICES will be updated.
  | 
  | Input:
  | 
  | X_0 - tensor to be updated
  | 
  | weight_0 - scalar weight for X_0, applied
  | only to slices affected,
  | 
  | INDICES - 1-D list of indices on the
  | first dimension of X_0 that need to be
  | updated
  | 
  | X_1 - update slices, has to have shape
  | of len(INDICES) + shape(X_0)[1:]
  | 
  | weight_1 - scalar weight for X_1 update
  | X_2, weight_2, ...
  | 
  | Output:
  | 
  | X_0 - has to be exactly the same tensor
  | as the input 0
  | 
  | -----------
  | @note
  | 
  | The op pretty much ignores the exact
  | shapes of the input arguments and cares
  | only about sizes.
  | 
  | It's done for performance consideration
  | to avoid unnecessary reshapes.
  | 
  | Only first dimension of X_0 is important,
  | let's call it N.
  | 
  | If M is the total size of X_0 and K is the
  | size of
  | 
  | INDICES then X_i is assumed to be of shape
  | K x (M / N) regardless of the real shape.
  | ----------
  | @note
  | 
  | Each update in INDICES is applied independently
  | which means that if duplicated elements
  | are present in INDICES the corresponding
  | slice of X_0 will be scaled multiple
  | times.
  | 
  | Manual collapsing of INDICES is required
  | beforehand if necessary.
  | ----------
  | @note
  | 
  | Updates are applied sequentially by
  | inputs which might have undesired consequences
  | if the input tensor is accessed concurrently
  | by different op (e.g. when doing Hogwild).
  | 
  | Other threads might see intermediate
  | results even on individual slice level,
  | e.g. X_0 scaled by weight_0 but without
  | any updates applied.
  | 
  | Currently only works on CPU because
  | of access to
  | 
  | INDICES.
  | ----------
  | @note
  | 
  | The op pretty much ignores the exact
  | shapes of the input arguments and cares
  | only about sizes.
  | 
  | It's done for performance consideration
  | to avoid unnecessary reshapes.
  | 
  | Only first dimension of X_0 is important,
  | let's call it N.
  | 
  | If M is the total size of X_0 and K is the
  | size of INDICES then X_i is assumed to
  | be of shape K x (M / N) regardless of the
  | real shape.
  | ----------
  | @note
  | 
  | Each update in INDICES is applied independently
  | which means that if duplicated elements
  | are present in INDICES the corresponding
  | slice of X_0 will be scaled multiple
  | times.
  | 
  | Manual collapsing of INDICES is required
  | beforehand if necessary.
  | ----------
  | @note
  | 
  | Updates are applied sequentially by
  | inputs which might have undesired consequences
  | if the input tensor is accessed concurrently
  | by different op (e.g. when doing Hogwild).
  | 
  | Other threads might see intermediate
  | results even on individual slice level,
  | 
  | e.g. X_0 scaled by weight_0 but without
  | any updates applied.
  | 
  | For now really works only on CPU because
  | of INDICES access
  |
  */
  #[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ScatterWeightedSumOp<T,Context> {

    //USE_DISPATCH_HELPER
    storage:        OperatorStorage,
    context:        Context,

    x_data_host:    Tensor,
    weights_host:   Tensor,
    x_data_device:  Tensor,
    weights_device: Tensor,
    phantom: PhantomData<T>,
}

num_outputs!{ScatterWeightedSum, 1}

inputs!{ScatterWeightedSum, 
    0 => ("X_0",      "Tensor to be updated."),
    1 => ("Weight_0", "Scalar weight for X_0, applied only to slices affected."),
    2 => ("INDICES",  "1-D list of indices on the first dimension of X_0 that need to be updated"),
    3 => ("X_1",      "Update slices, with shape len(INDICES) + shape(X_0)[1:]"),
    4 => ("Weight_1", "Scalar weight for X_1 update")
}

outputs!{ScatterWeightedSum, 
    0 => ("X_0", "Has to be exactly the same tensor as the input 0")
}

enforce_inplace!{ScatterWeightedSum, vec![(0, 0)]}

num_inputs!{ScatterWeightedSum, 
    |n: i32| {
        n > 3 && (n - 3) % 2 == 0
    }
}

impl<T,Context> ScatterWeightedSumOp<T, Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(2));
        */
    }
    
    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
        todo!();
        /*
            int64_t block_size = Input(0).size_from_dim(1);
        return DispatchHelper<FixedValues<1>, Index>::call(this, block_size);
        */
    }
    
    #[inline] pub fn do_run_with_value<Index, const FixedSize: i32>(&mut self) -> bool {
        todo!();
        /*
            CAFFE_ENFORCE_EQ(InputSize() % 2, 1);
        auto& X0 = Input(0);
        auto& weight0 = Input(1);
        auto& indices = Input(2);
        auto* output = Output(0);
        CAFFE_ENFORCE_EQ(&X0, output, "In place operation is required");

        if (X0.numel() == 0) {
          return true;
        }
        CAFFE_ENFORCE_GT(X0.numel(), 0);
        CAFFE_ENFORCE_GT(X0.dim(), 0, "X0 has to be at least the vector");
        CAFFE_ENFORCE_EQ(weight0.numel(), 1);
        int64_t M = X0.numel();
        int64_t N = X0.size(0);
        int64_t K = indices.numel();
        int64_t block_size = M / N;
        T* data = output->template mutable_data<T>();
        const Index* idxs = indices.template data<Index>();
        T w0 = *weight0.template data<T>();
        // It's most likely a constant so exact comparison is fine
        if (w0 != 1.0) {
          for (int i = 0; i < K; ++i) {
            Index idx = idxs[i];
            CAFFE_ENFORCE(
                0 <= idx && idx < N,
                "Index out of bounds: ",
                idx,
                ", range 0 to ",
                N);
            math::ScaleFixedSize<T, Context, FixedSize>(
                block_size,
                w0,
                data + block_size * idx,
                data + block_size * idx,
                &context_);
          }
        }
        for (int inp = 3; inp < InputSize(); inp += 2) {
          auto& X = Input(inp);
          auto& weight = Input(inp + 1);
          CAFFE_ENFORCE_EQ(X.numel(), block_size * K);
          CAFFE_ENFORCE_EQ(weight.numel(), 1);
          const T* x_data = X.template data<T>();
          T w = *weight.template data<T>();
          for (int i = 0; i < K; ++i) {
            Index idx = idxs[i];
            // double-checking the indices, but it's fine as it's DCHECK only
            DCHECK(0 <= idx && idx < N)
                << "Index out of bounds: " << idx << ", range 0 to " << N;
            math::AxpyFixedSize<T, Context, FixedSize>(
                block_size,
                w,
                x_data + block_size * i,
                data + block_size * idx,
                &context_);
          }
        }
        return true;
        */
    }
}

///-----------------------------------------
type RunnerType = fn() -> ();
type RunnerMap  = HashMap<(TensorProto_DataType, TensorProto_DataType), RunnerType>;

/**
  | @brief
  | 
  | Update slices of the tensor in-place
  | by overriding.
  | 
  | Input:
  | 
  | DATA - tensor to be updated
  | 
  | INDICES - 1-D list of indices on the first
  | dimension of X_0 that need to be updated
  | 
  | SLICES - update slices, has to have shape
  | of len(INDICES) + shape(X_0)[1:]
  | 
  | Output:
  | 
  | DATA - has to be exactly the same tensor
  | as the input 0
  | 
  | -----------
  | @note
  | 
  | The op pretty much ignores the exact
  | shapes of the input arguments and cares
  | only about sizes. It's done for performance
  | consideration to avoid unnecessary
  | reshapes. Only first dimension of X_0
  | is important, let's call it
  | 
  | N. If M is the total size of X_0 and K is
  | the size of INDICES then X_i is assumed
  | to be of shape K x (M / N) regardless of
  | the real shape.
  | ----------
  | @note
  | 
  | Each update in INDICES is applied independently
  | which means that if duplicated elements
  | are present in INDICES arbitrary one
  | will win.
  | 
  | For now really works only on CPU because
  | of INDICES access
  | 
  | Update slices of the tensor in-place
  | by overriding current value.
  | ----------
  | @note
  | 
  | The op pretty much ignores the exact
  | shapes of the input arguments and cares
  | only about sizes. It's done for performance
  | consideration to avoid unnecessary
  | reshapes. Only first dimension of X_0
  | is important, let's call it
  | 
  | N. If M is the total size of X_0 and K is
  | the size of INDICES then X_i is assumed
  | to be of shape K x (M / N) regardless of
  | the real shape.
  | ----------
  | @note
  | 
  | Each update in INDICES is applied independently
  | which means that if duplicated elements
  | are present in INDICES arbitrary one
  | will win.
  | 
  | Currently only works on CPU because
  | of access to INDICES.
  |
  */
  #[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ScatterAssignOp<Context> {
    
    storage: OperatorStorage,
    context: Context,
    runners: RunnerMap,
}

num_inputs!{ScatterAssign, 3}

num_outputs!{ScatterAssign, 1}

inputs!{ScatterAssign, 
    0 => ("DATA",    "Tensor to be updated."),
    1 => ("INDICES", "1-D list of indices on the first dimension of X_0 that need to be updated"),
    2 => ("SLICES",  "Update slices, with shape len(INDICES) + shape(X_0)[1:]")
}

outputs!{ScatterAssign, 
    0 => ("DATA", "Has to be exactly the same tensor as the input 0")
}

tensor_inference_function!{ScatterAssign, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out(1);
          out[0] = in[0];
          return out;
        */
    }
}

enforce_inplace!{ScatterAssign, vec![(0, 0)]}

input_tags!{
    ScatterAssignOp {
        Data,
        Indices,
        Slices
    }
}

impl<Context> ScatterAssignOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            runners_({{{TensorProto_DataType_INT32, TensorProto_DataType_FLOAT},
                       &ScatterAssignOp::DoRun<int32_t, float>},
                      {{TensorProto_DataType_INT32, TensorProto_DataType_FLOAT16},
                       &ScatterAssignOp::DoRun<int32_t, at::Half>},
                      {{TensorProto_DataType_INT32, TensorProto_DataType_UINT8},
                       &ScatterAssignOp::DoRun<int32_t, uint8_t>},
                      {{TensorProto_DataType_INT32, TensorProto_DataType_INT32},
                       &ScatterAssignOp::DoRun<int32_t, int32_t>},
                      {{TensorProto_DataType_INT32, TensorProto_DataType_INT64},
                       &ScatterAssignOp::DoRun<int32_t, int64_t>},
                      {{TensorProto_DataType_INT32, TensorProto_DataType_DOUBLE},
                       &ScatterAssignOp::DoRun<int32_t, double>},
                      {{TensorProto_DataType_INT64, TensorProto_DataType_FLOAT},
                       &ScatterAssignOp::DoRun<int64_t, float>},
                      {{TensorProto_DataType_INT64, TensorProto_DataType_FLOAT16},
                       &ScatterAssignOp::DoRun<int64_t, at::Half>},
                      {{TensorProto_DataType_INT64, TensorProto_DataType_UINT8},
                       &ScatterAssignOp::DoRun<int64_t, uint8_t>},
                      {{TensorProto_DataType_INT64, TensorProto_DataType_INT32},
                       &ScatterAssignOp::DoRun<int64_t, int32_t>},
                      {{TensorProto_DataType_INT64, TensorProto_DataType_INT64},
                       &ScatterAssignOp::DoRun<int64_t, int64_t>},
                      {{TensorProto_DataType_INT64, TensorProto_DataType_DOUBLE},
                       &ScatterAssignOp::DoRun<int64_t, double>}})
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& data = Input(DATA);
        const auto& slices = Input(SLICES);
        auto& indices = Input(INDICES);

        const auto dataType = TypeMetaToDataType(data.dtype());
        const auto slicesType = TypeMetaToDataType(slices.dtype());
        const auto indicesType = TypeMetaToDataType(indices.dtype());
        auto* output = Output(0);

        auto runner = GetRunner(dataType, slicesType, indicesType);
        (this->*runner)();
        return true;
        */
    }
    
    #[inline] pub fn get_runner(
        &mut self, 
        data_type:    TensorProto_DataType,
        slices_type:  TensorProto_DataType,
        indices_type: TensorProto_DataType) -> RunnerType 
    {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(dataType, slicesType, "Data and slice types must match");
        auto it = runners_.find({indicesType, dataType});
        CAFFE_ENFORCE(
            it != runners_.end(),
            "Could not find the runner corresponding to indicesType, dataType = ",
            indicesType,
            " ",
            dataType);
        return it->second;
        */
    }
    
    #[inline] pub fn do_run<Index, T>(&mut self) {
        todo!();
        /*
            auto& input = Input(DATA);
        auto& indices = Input(INDICES);
        auto& slices = Input(SLICES);
        auto* output = Output(0);
        CAFFE_ENFORCE_EQ(&input, output, "In place operation is required");

        CAFFE_ENFORCE_GT(input.dim(), 0, "X0 has to be at least the vector");
        int64_t M = input.numel();
        int64_t N = input.size(0);
        int64_t K = indices.numel();
        int64_t block_size = M / N;
        CAFFE_ENFORCE_EQ(slices.numel(), block_size * K);
        // TODO(dzhulgakov): it can be made to work with arbitrary data type by
        // using raw_mutable_data
        T* data = output->template mutable_data<T>();
        const Index* idxs = indices.template data<Index>();
        const T* slicesData = slices.template data<T>();
        DoScatterAssign(data, idxs, slicesData, N, K, block_size);
        */
    }
    
    #[inline] pub fn do_scatter_assign<Index, T>(
        &mut self, 
        data:          *mut T,
        idxs:          *const Index,
        slices_data:   *const T,
        n:             i64,
        k:             i64,
        block_size:    i64) 
    {
        todo!();
        /*
            for (int i = 0; i < K; ++i) {
          Index idx = idxs[i];
          // double-checking the indices, but it's fine as it's DCHECK only
          DCHECK(0 <= idx && idx < N)
              << "Index out of bounds: " << idx << ", range 0 to " << N;
          context_.template CopySameDevice<T>(
              block_size, slicesData + block_size * i, data + block_size * idx);
        }
        */
    }
}

/**
  | Update values of the tensor by overriding
  | current value specified by indices.
  | 
  | Writes all values from the tensor UPDATES
  | into DATA at the indices specified in
  | the INDICES tensor.
  | 
  | For each value in DATA, its output index
  | is specified by its index in UPDATES
  | and by the corresponding value in INDICES
  | for the specified axis.
  | 
  | For a 3-D tensor, DATA is updated as:
  | 
  | DATA[INDICES[i][j][k]][j][k] = UPDATES[i][j][k]
  | # if axis == 0
  | 
  | DATA[i][INDICES[i][j][k]][k] = UPDATES[i][j][k]
  | # if axis == 1
  | 
  | DATA[i][j][INDICES[i][j][k]] = UPDATES[i][j][k]
  | # if axis == 2
  | 
  | Currently only works on CPU because
  | of access to INDICES.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ScatterOp<Context> {

    storage: OperatorStorage,
    context: CPUContext,
    axis:    i32,
    phantom: PhantomData<Context>,
}

num_inputs!{Scatter, 3}

num_outputs!{Scatter, 1}

inputs!{Scatter, 
    0 => ("DATA", "Tensor to be updated."),
    1 => ("INDICES", "1-D list of indices on the first dimension of X_0 that need to be updated"),
    2 => ("UPDATES", "Update slices, with shape len(INDICES) + shape(X_0)[1:]")
}

outputs!{Scatter, 
    0 => ("OUTPUT", "The updated output.")
}

args!{Scatter, 
    0 => ("axis", "*(type: int; default: 1)* Which dimension to scatter on.")
}

allow_inplace!{Scatter, vec![(0, 0)]}

input_tags!{
    ScatterOp
    {
        Data,
        Indices,
        Updates
    }
}

impl<Context> ScatterOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "axis", axis_, 1)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            TORCH_CHECK(
            Context::GetDeviceType() == kCPU,
            "ScatterOp currently only supports CPU.")

        return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, this->template Input<Tensor>(INDICES, CPU));
        */
    }
    
    #[inline] pub fn do_run_with_type<IndexType>(&mut self) -> bool {
        todo!();
        /*
            const Tensor& data = Input(DATA);
        const Tensor& indices = Input(INDICES);
        const Tensor& updates = Input(UPDATES);
        const TypeMeta dataType = data.dtype();
        size_t item_bytesize = dataType.itemsize();

        // ONNX allows negative axis to index from the back, valid range: [-r, r].
        axis_ = data.canonical_axis_index(axis_);

        CAFFE_ENFORCE_GE(
            data.dim(), axis_ + 1, "DATA should be at least [axis+1]-D");
        CAFFE_ENFORCE_GE(axis_, 0, "Axis should be non-negative");
        CAFFE_ENFORCE_LT(axis_, data.dim(), "Axis out of range");

        Tensor* output = Output(0, data.sizes().vec(), at::dtype(dataType));
        output->CopyFrom(data);
        char* out = static_cast<char*>(output->raw_mutable_data(dataType));

        // Succeed if size of output is zero, which can happen for empty batch which
        // would have data dimension size of 0.
        // This *must* be done AFTER output->raw_mutable_data() above as that has
        // important allocation side effect that we must see.
        if (output->numel() == 0) {
          return true;
        }

        const IndexType* idxs = indices.template data<IndexType>();
        const char* src_base = static_cast<const char*>(updates.raw_data());

        const int64_t outer_dims_product = indices.size_to_dim(axis_);

        const int64_t dst_indexing_axis_dim = data.size(axis_);

        const int64_t idxs_block_size = indices.size_from_dim(axis_ + 1);
        const int64_t src_block_size = updates.size_from_dim(axis_ + 1);
        const int64_t dst_block_size = data.size_from_dim(axis_ + 1);

        const int64_t idxs_batch_size = indices.size_from_dim(axis_);
        const int64_t src_batch_size = updates.size_from_dim(axis_);
        const int64_t dst_batch_size = data.size_from_dim(axis_);

        const int64_t N = indices.size(axis_);

        check_indexarray_range<IndexType>(idxs, N, dst_indexing_axis_dim);

        // For a 3-D tensor, dst is updated as:
        //    dst[i][idxs[i][j][k]][k] = src[i][j][k]  # if dim == 1
        // where i, j, k are iterating over their corresponding axis I, J, K.
        // For a given i, j, k tuple.
        // idxs offset can be computed as i * J_src * K + j * K + k.
        // src offset can be computed as i * J_src * K + j * K + k.
        // dst offset can be computed as i * J_dst * K + idxs[idxs_offset] * K + K
        // Note that idxs and src should have the same rank and shape.
        // dst should have the same rank as idxs and src, but the dimension of dim
        // axis can be different. That is why in the above equation, there is the
        // difference of J_src and J_dst.
        for (int64_t outer_batch = 0; outer_batch < outer_dims_product;
             ++outer_batch) {
          for (int64_t i = 0; i < N; ++i) {
            for (int64_t inner_batch = 0; inner_batch < idxs_block_size;
                 ++inner_batch) {
              auto idxs_elem_idx =
                  outer_batch * idxs_batch_size + i * idxs_block_size + inner_batch;
              auto src_elem_idx =
                  outer_batch * src_batch_size + i * src_block_size + inner_batch;
              auto dst_elem_idx = outer_batch * dst_batch_size +
                  idxs[idxs_elem_idx] * dst_block_size + inner_batch;

              auto src = src_base + src_elem_idx * item_bytesize;
              auto dst = out + dst_elem_idx * item_bytesize;
              context_.CopyItemsSameDevice(dataType, 1, src, dst);
            }
          }
        }
        return true;
        */
    }

    /**
      | Check that indices fall within dimension
      | array size with CAFFE_ENFORCE.
      |
      */
    #[inline] pub fn check_indexarray_range<IndexType>(
        &mut self, 
        indices:           *const IndexType,
        n:                 i64,
        indexing_axis_dim: IndexType) 
    {
        todo!();
        /*
            for (auto i = 0; i < n; ++i) {
          auto idx = indices[i];
          CAFFE_ENFORCE(
              0 <= idx && idx < indexing_axis_dim,
              "INDICES element is out of DATA bounds, id=",
              idx,
              " axis_dim=",
              indexing_axis_dim);
        }
        */
    }
}

///-----------------------------------------
#[test] fn lengths_to_segment_ids_op() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LengthsToSegmentIds",
        ["lengths"],
        ["segment_ids"],
    )

    workspace.FeedBlob("lengths", np.array([1, 3, 0, 2]).astype(np.int32))
    print("lengths:\n", workspace.FetchBlob("lengths"))

    workspace.RunOperatorOnce(op)
    print("segment_ids: \n", workspace.FetchBlob("segment_ids"))

    lengths:
     [1 3 0 2]
    segment_ids:
     [0 1 1 1 3 3]

    */
}

/**
  | Given a vector of segment lengths (*lengths*)
  | the *LengthsToSegmentIds* op returns
  | a zero-based, consecutive vector of
  | segment ids (*segment_ids*).
  | 
  | For example, *lengths=[1, 3, 0, 2]*
  | will produce segment_ids=[0, 1, 1,
  | 1, 3, 3]*.
  | 
  | In general, the inverse operation is
  | *SegmentIdsToLengths*.
  | 
  | Notice though that trailing empty sequence
  | lengths can't be properly recovered
  | from segment ids.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LengthsToSegmentIdsOp<Context> {

    storage: OperatorStorage,
    context: Context,
}

disallow_input_fillers!{LengthsToSegmentIdsOp}

num_inputs!{LengthsToSegmentIds, 1}

num_outputs!{LengthsToSegmentIds, 1}

inputs!{LengthsToSegmentIds, 
    0 => ("lengths", "1D tensor of int32 or int64 segment lengths.")
}

outputs!{LengthsToSegmentIds, 
    0 => ("segment_ids", "1D tensor of length *sum(lengths)*")
}

impl<Context> LengthsToSegmentIdsOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        auto* output = Output(0);
        auto* input_data = input.template data<int32_t>();

        CAFFE_ENFORCE(input.sizes().size() == 1, "Input must be a vector.");
        auto total_length =
            std::accumulate(input_data, input_data + input.numel(), 0);

        output->Resize(total_length);
        auto* output_data = output->template mutable_data<int32_t>();

        for (int i = 0; i < input.numel(); ++i) {
          auto len = input_data[i];
          std::fill(output_data, output_data + len, i);
          output_data += len;
        }
        return true;
        */
    }
}

/**
  | Given a vector of segment lengths, calculates
  | offsets of each segment and packs them
  | next to the lengths.
  | 
  | For the input vector of length N the output
  | is a Nx2 matrix with (offset, lengths)
  | packaged for each segment.
  | 
  | For example, `[1, 3, 0, 2]` transforms
  | into `[[0, 1], [1, 3], [4, 0], [4, 2]]`.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LengthsToRangesOp<Context> {

    storage: OperatorStorage,
    context: Context,
}

num_inputs!{LengthsToRanges, 1}

num_outputs!{LengthsToRanges, 1}

inputs!{LengthsToRanges, 
    0 => ("lengths", "1D tensor of int32 segment lengths.")
}

outputs!{LengthsToRanges, 
    0 => ("ranges", "2D tensor of shape len(lengths) X 2 and the same type as `lengths`")
}

tensor_inference_function!{LengthsToRanges, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<int> out_shape(in[0].dims().begin(), in[0].dims().end());
          out_shape.push_back(2);
          return vector<TensorShape>{ 
              CreateTensorShape(out_shape, in[0].data_type())};
        */
    }
}

impl<Context> LengthsToRangesOp<Context> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        auto* output = Output(0);
        auto* input_data = input.template data<int32_t>();

        CAFFE_ENFORCE(input.sizes().size() == 1, "Input must be a vector.");
        auto size = input.numel();

        output->Resize(size, 2);
        auto* output_data = output->template mutable_data<int32_t>();

        int32_t offset = 0;
        for (int i = 0; i < size; ++i) {
          auto len = input_data[i];
          output_data[i * 2] = offset;
          output_data[i * 2 + 1] = len;
          offset += len;
        }
        return true;
        */
    }
}

/**
  | Given a vector of segment lengths, returns
  | a vector of offsets from these lengths,
  | which will have the same size as the input
  | vector.
  | 
  | Output is going to have the same type
  | as input.
  | 
  | For long tensors explicit casting from
  | int32 to int64 might be necessary prior
  | to this op.
  | 
  | For example, `[1, 3, 0, 2]` transforms
  | into `[0, 1, 4, 4]`.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LengthsToOffsetsOp<Context> {

    storage:             OperatorStorage,
    context:             Context,
    include_last_offset: bool,
}

num_inputs!{LengthsToOffsets, 1}

num_outputs!{LengthsToOffsets, 1}

inputs!{LengthsToOffsets, 
    0 => ("lengths", "1D tensor of int32 or int64 segment lengths.")
}

outputs!{LengthsToOffsets, 
    0 => ("offsets", "1D tensor of the same shape and type as `lengths`")
}

tensor_inference_function!{LengthsToOffsets, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          const ArgumentHelper args(def);
          bool include_last_offset =
              args.GetSingleArgument<bool>("include_last_offset", false);
          vector<int> out_shape(in[0].dims().begin(), in[0].dims().end());
          out_shape[0] += include_last_offset ? 1 : 0;
          return vector<TensorShape>{
              CreateTensorShape(out_shape, in[0].data_type())};
        */
    }
}

impl<Context> LengthsToOffsetsOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            include_last_offset_(this->template GetSingleArgument<bool>( "include_last_offset", false))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        auto* output = Output(0);
        auto* input_data = input.template data<int32_t>();

        CAFFE_ENFORCE(input.sizes().size() == 1, "Input must be a vector.");
        auto size = input.numel();

        output->Resize(size + (include_last_offset_ ? 1 : 0));
        auto* output_data = output->template mutable_data<int32_t>();

        int32_t offset = 0;
        for (int i = 0; i < size; ++i) {
          auto len = input_data[i];
          output_data[i] = offset;
          offset += len;
        }
        if (include_last_offset_) {
          output_data[size] = offset;
        }
        return true;
        */
    }
}

/**
  | Transfers a vector of segment ids to
  | a vector of segment lengths.
  | 
  | This operation supports non-consecutive
  | segment ids.
  | 
  | Segments not appearing in the input
  | vector will have length 0.
  | 
  | If the second input is provided, the
  | number of segments = the size of its first
  | dimension.
  | 
  | Otherwise, the number of segments =
  | the last index in the first input vector
  | + 1.
  | 
  | In general, for consecutive, zero-based
  | segment IDs, this is the inverse operation
  | of LengthsToSegmentIds, except that
  | a vector of segment IDs cannot represent
  | empty segments at the end (if the second
  | input is absent).
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SegmentIdsToLengthsOp<Context> {
    
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{SegmentIdsToLengths, (1,2)}

num_outputs!{SegmentIdsToLengths, 1}

//todo, enable the filler
disallow_input_fillers!{SegmentIdsToLengths}

inputs!{SegmentIdsToLengths, 
    0 => ("segment_ids", "1-D int32_t or int64_t tensor of segment ids"),
    1 => ("data (optional)", "if provided, number of segments = the size of its first dimension")
}

outputs!{SegmentIdsToLengths, 
    0 => ("lengths", "1-D int64_t tensor of segment lengths")
}

impl<Context> SegmentIdsToLengthsOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
        todo!();
        /*
            auto& input = Input(0);
        if (input.dim() == 2) {
          CAFFE_ENFORCE(
              input.dim32(0) == 1 || input.dim32(1) == 1,
              "Input must be a vector.");
        } else {
          CAFFE_ENFORCE_EQ(input.dim(), 1, "Input must be a vector.");
        }
        auto* input_data = input.template data<Index>();
        auto input_size = input.numel();
        auto* output = Output(0);
        // segment id starts from 0
        auto num_segments = input_size ? input_data[input_size - 1] + 1 : 0;
        if (InputSize() > 1) {
          CAFFE_ENFORCE_GE(Input(1).dim(), 1);
          CAFFE_ENFORCE_LE(
              num_segments,
              Input(1).size(0),
              "The number of segments inferred should *NOT* be larger "
              "than the size of Input(1)'s first dimension");
          num_segments = Input(1).size(0);
        }
        CAFFE_ENFORCE(0 <= num_segments, "Indices must be in 0..K-1 range");
        output->Resize(num_segments);
        auto* output_data = output->template mutable_data<int32_t>();
        if (num_segments == 0) {
          return true;
        }
        std::fill(output_data, output_data + num_segments, 0);
        Index prev = 0; // Assume that segment_id >= 0.
        for (int64_t i = 0; i < input_size; i++) {
          CAFFE_ENFORCE(
              prev <= input_data[i],
              "Segment ids must be sorted: ",
              prev,
              " vs ",
              input_data[i]);
          prev = input_data[i];
          output_data[input_data[i]] += 1;
        }

        return true;
        */
    }
}

/**
  | Transfers a vector of segment ids to
  | a vector of segment ranges.
  | 
  | This operation supports non-consecutive
  | segment ids.
  | 
  | Segments not appearing in the input
  | vector will have length 0.
  | 
  | If the second input is provided, the
  | number of segments = the size of its first
  | dimension.
  | 
  | Otherwise, the number of segments =
  | the last index in the first input vector
  | + 1.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SegmentIdsToRangesOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

disallow_input_fillers!{SegmentIdsToRangesOp}

num_inputs!{SegmentIdsToRanges, (1,2)}

num_outputs!{SegmentIdsToRanges, 1}

inputs!{SegmentIdsToRanges, 
    0 => ("segment_ids",     "1-D int32_t or int64_t tensor of segment ids"),
    1 => ("data (optional)", "if provided, number of segments = the size of its first dimension")
}

outputs!{SegmentIdsToRanges, 
    0 => ("lengths", "1-D int64_t tensor of segment lengths")
}

impl<Context> SegmentIdsToRangesOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
        todo!();
        /*
            auto& input = Input(0);
        CAFFE_ENFORCE(input.sizes().size() == 1, "Input must be a vector.");
        auto* input_data = input.template data<Index>();
        auto input_size = input.numel();
        auto* output = Output(0);
        // segment id starts from 0
        auto num_segments = input_size ? input_data[input_size - 1] + 1 : 0;
        if (InputSize() > 1) {
          CAFFE_ENFORCE_GE(Input(1).dim(), 1);
          CAFFE_ENFORCE_LE(
              num_segments,
              Input(1).size(0),
              "The number of segments inferred should *NOT* be larger "
              "than the size of Input(1)'s first dimension");
          num_segments = Input(1).size(0);
        }
        CAFFE_ENFORCE(0 <= num_segments, "Indices must be in 0..K-1 range");
        output->Resize(num_segments, 2);
        auto* output_data = output->template mutable_data<int32_t>();
        if (num_segments == 0) {
          return true;
        }
        std::fill(output_data, output_data + num_segments * 2, 0);
        Index prev = input_data[0];
        for (int64_t i = 0; i < input_size; i++) {
          CAFFE_ENFORCE(
              prev <= input_data[i],
              "Segment ids must be sorted: ",
              prev,
              " vs ",
              input_data[i]);
          while (prev != input_data[i]) {
            ++prev;
            output_data[prev * 2] = i;
          }
          output_data[input_data[i] * 2 + 1] += 1;
        }

        return true;
        */
    }
}

/**
  | Similar as LengthsToSegmentIds but
  | output vector of segment weights derived
  | by lengths. i.e 1/pow(length, power)
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LengthsToWeightsOp<Context> {
    storage: OperatorStorage,
    context: Context,
    power:   f32,
}

num_inputs!{LengthsToWeights, 1}

num_outputs!{LengthsToWeights, 1}

inputs!{LengthsToWeights, 
    0 => ("lengths", "1-D int32_t or int64_t tensor of lengths")
}

outputs!{LengthsToWeights, 
    0 => ("a vector of weights", "1-D float tensor of weights by length")
}

args!{LengthsToWeights, 
    0 => ("power", "n of 1/pow(length,n) for normalization")
}

impl<Context> LengthsToWeightsOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            power_(this->template GetSingleArgument<float>("power", 0.5))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
        todo!();
        /*
            auto& input = Input(0);
        CAFFE_ENFORCE(input.sizes().size() == 1, "Input must be a vector.");
        auto* input_data = input.template data<Index>();
        auto input_size = input.numel();
        auto* output = Output(0);

        int64_t output_size = 0;
        for (auto i = 0; i < input_size; i++) {
          CAFFE_ENFORCE_GE(input_data[i], 0, "unexpected negative length value");
          output_size += input_data[i];
        }

        std::function<float(const int64_t& length, const float& power)> getWeight;
        if (power_ == 0.5) {
          getWeight = [](const int64_t& length, const float& /*power*/) {
            return 1.0 / std::sqrt(length);
          };
        } else if (power_ == 1) {
          getWeight = [](const int64_t& length, const float& /*power*/) {
            return 1.0 / length;
          };
        } else {
          getWeight = [](const int64_t& length, const float& power) {
            return 1.0 / std::pow(length, power);
          };
        }

        output->Resize(output_size);
        auto* output_data = output->template mutable_data<float>();
        int64_t cnt = 0;
        for (auto i = 0; i < input_size; i++) {
          auto len = input_data[i];
          if (len == 0) {
            continue;
          }
          CAFFE_ENFORCE_LE(cnt + len, output_size, "unexpected lengths value");

          float weight_value = getWeight(len, power_);
          std::fill(output_data + cnt, output_data + cnt + len, weight_value);
          cnt += len;
        }

        return true;
        */
    }
}

/**
  | The *HasElements* op accepts a single
  | or multiple input tensors, and produces
  | a single boolean output $has\_elements$.
  | The output is *True* if and only if any
  | of the input tensor has size > 0.
  | 
  | Note, this op is the opposite of the *IsEmpty*
  | op.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct HasElementsOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

should_not_do_gradient!{IsEmpty}

should_not_do_gradient!{HasElements}

num_inputs!{HasElements, (1,INT_MAX)}

num_outputs!{HasElements, 1}

inputs!{HasElements, 
    0 => ("X1, X2, ...", "List of input data tensors to check for elements.")
}

outputs!{HasElements, 
    0 => ("has_elements", "Output scalar boolean tensor. True if input has size > 0.")
}

#[test] fn has_elements_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "HasElements",
        ["tensor"],
        ["has_elements"],
    )

    // Use a not-empty tensor
    workspace.FeedBlob("tensor", np.random.randn(2, 2).astype(np.float32))
    print("tensor:\n", workspace.FetchBlob("tensor"))

    workspace.RunOperatorOnce(op)
    print("has_elements: ", workspace.FetchBlob("has_elements"),"\n")

    // Use an empty tensor
    workspace.FeedBlob("tensor", np.empty(0))
    print("tensor:\n", workspace.FetchBlob("tensor"))

    workspace.RunOperatorOnce(op)
    print("has_elements: ", workspace.FetchBlob("has_elements"))

    tensor:
     [[ 0.6116506  -0.54433197]
     [ 0.19406661 -0.7338629 ]]
    has_elements:  True

    tensor:
     []
    has_elements:  False
    */
}

impl<Context> HasElementsOp<Context> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            bool res = false;
        for (auto i = 0; i < InputSize(); ++i) {
          const auto& input = Input(i);
          res = res || input.numel() > 0;
        }
        auto* output = Output(0);
        output->Resize(std::vector<int64_t>{});
        *output->template mutable_data<bool>() = res;
        return true;
        */
    }
}

///-----------------------------------------
#[test] fn size_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Size",
        ["X"],
        ["size"],
    )

    workspace.FeedBlob("X", (np.random.randint(10, size=(3,3))))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("size:", workspace.FetchBlob("size"))

    workspace.ResetWorkspace()

    workspace.FeedBlob("X", (np.random.rand(6,4)))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("size:", workspace.FetchBlob("size"))

    X:
    [[3 7 0]
     [0 1 6]
     [5 0 8]]
    size: 9
    X:
    [[0.92017884 0.32115368 0.68692035 0.64135016]
     [0.8723328  0.77830265 0.80688656 0.25524236]
     [0.37970216 0.76407047 0.85689564 0.30692883]
     [0.69352573 0.42531502 0.16415212 0.59209324]
     [0.52684188 0.37094846 0.60670079 0.6489272 ]
     [0.94715906 0.34800557 0.61898769 0.28947359]]
    size: 24

    */
}

/**
  | Return a 1D tensor of type *int64* that
  | contains the number of elements of the
  | input tensor.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc
  | 
  | Return the size of a tensor
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SizeOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{Size, 1}

num_outputs!{Size, 1}

inputs!{Size, 
    0 => ("X", "*(type: Tensor)* Input tensor to calculate number of elements.")
}

outputs!{Size, 
    0 => ("size", "*(type: Tensor)* 1D tensor of type int64 that contains the number of elements in the input tensor *X*.")
}

register_cpu_operator!{Size, SizeOp<CPUContext>}

no_gradient!{Size}

impl<Context> SizeOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);

        auto* output = Output(0, vector<int64_t>(), at::dtype<int64_t>());
        auto* output_data = output->template mutable_data<int64_t>();

        auto size = input.numel();
        math::Set<int64_t, Context>(
            1, static_cast<int64_t>(size), output_data, &context_);

        return true;
        */
    }
}

///-----------------------------------------
#[test] fn lengths_to_shape_op_example() {
    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LengthsToShape",
        ["X"],
        ["Y"]
    )

    // Create X: Sample softmax output for 5-class model
    X = np.array([2,2,2,2,2,2,2,2,2,2])
    print("X:\n",X)

    // Feed X into workspace
    workspace.FeedBlob("X", X.astype(np.int32))

    // Run op
    workspace.RunOperatorOnce(op)

    // Collect Output
    print("Y:\n", workspace.FetchBlob("Y"))

    X:
     [2 2 2 2 2 2 2 2 2 2]
    Y:
     [10  2]

    */
}

/**
  | This operator takes a list of $N$ equal
  | integers as input which represent the
  | lengths of $N$ vectors.
  | 
  | The output is the calculated shape of
  | the matrix if the $N$ integers were combined
  | into a single matrix.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc
  | 
  | returns a shape to be passed to Reshape
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LengthsToShapeOp<Context> {

    storage: OperatorStorage,
    context: Context,
}

num_inputs!{LengthsToShape, 1}

num_outputs!{LengthsToShape, 1}

inputs!{LengthsToShape, 
    0 => ("X", "List, of length $N$, of equal integers representing the lengths of several vectors.")
}

outputs!{LengthsToShape, 
    0 => ("Y", "Vector of length 2 describing the dimensions of the data if the $N$ vectors from the input were combined to a single matrix.")
}

should_not_do_gradient!{LengthsToShape}

impl<Context> LengthsToShapeOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);

        CAFFE_ENFORCE(input.sizes().size() == 1, "Input must be a vector.");
        auto* output = Output(0);
        auto* input_data = input.template data<int32_t>();

        auto size = input.numel();
        auto first = input_data[0];

        for (int i = 1; i < size; i++) {
          CAFFE_ENFORCE(
              input_data[i] == first, "All elements of input must be same ");
        }

        output->Resize(2);
        auto* output_data = output->template mutable_data<int32_t>();
        output_data[0] = size;
        output_data[1] = first;

        return true;
        */
    }
}

///-----------------------------------------
#[test] fn gather_ranges_op_example() {

    todo!();

    /*
    RANGES dimentions description:
    1: represents list of examples within a batch
    2: represents list features
    3: two values which are start and length or a range (to be applied on DATA)

    Another output LENGTHS represents each example length within OUTPUT

    Example:
      DATA  = [1, 2, 3, 4, 5, 6]
      RANGES = [
        [
          [0, 1],
          [2, 2],
        ],
        [
          [4, 1],
          [5, 1],
        ]
      ]
      OUTPUT = [1, 3, 4, 5, 6]
      LENGTHS = [3, 2]
    */
}

/**
  | Given DATA tensor of rank 1, and RANGES
  | tensor of rank 3, gather corresponding
  | ranges into a 1-D tensor OUTPUT.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct GatherRangesOp<Context> {

    storage: OperatorStorage,
    context: Context,
}

num_inputs!{GatherRanges, 2}

num_outputs!{GatherRanges, 2}

inputs!{GatherRanges, 
    0 => ("DATA",   "Tensor of rank 1."),
    1 => ("RANGES", "Tensor of int32/int64 ranges, of dims (N, M, 2). Where N is number of examples and M is a size of each example. Last dimension represents a range in the format (start, lengths)")
}

outputs!{GatherRanges, 
    0 => ("OUTPUT",  "1-D tensor of size sum of range lengths"),
    1 => ("LENGTHS", "1-D tensor of size N with lengths over gathered data for each row in a batch. sum(LENGTHS) == OUTPUT.size()")
}

tensor_inference_function!{
    GatherRanges, 
    OpSchema::NeedsAllInputShapes(
        |def: &OperatorDef, input: &Vec<TensorShape>| {
            todo!();
            /*
              std::vector<TensorShape> out(2);

              int total = 1;
              for (auto d : in[0].dims()) {
                total *= d;
              }
              out[0].add_dims(total);
              out[0].set_data_type(in[0].data_type());
              out[1].add_dims(in[1].dims(0));
              out[1].set_data_type(in[1].data_type());
              return out;
            */
        }
    )
}

input_tags!{
    GatherRangesOp
    {
        Data,
        Ranges,
        Lengths
    }
}

impl<Context> GatherRangesOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, this->template Input<Tensor>(RANGES, CPU));
        */
    }
    
    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
        todo!();
        /*
            auto& data = Input(DATA);
        auto& ranges = Input(RANGES);
        auto* outputData = Output(0);
        auto* outputLengths = Output(1);

        auto batchSize = ranges.size(0);
        CAFFE_ENFORCE(data.dim() == 1, "Data has to be 1-D");
        CAFFE_ENFORCE(ranges.dim() == 3, "Ranges must be 3-D");
        CAFFE_ENFORCE(ranges.size(1) > 0, "There has to be at least one range");
        CAFFE_ENFORCE_EQ(
            ranges.size(2), 2, "Ranges last dimension should be of size 2");

        auto* rawData = static_cast<const char*>(data.raw_data());
        auto* rangesData = ranges.template data<Index>();

        outputLengths->Resize(batchSize);
        auto* outputLengthsPtr = outputLengths->template mutable_data<int32_t>();
        size_t start = 0;
        size_t blockSize = ranges.size_from_dim(1);
        for (size_t i = 0; i < batchSize; ++i) {
          auto end = start + blockSize;
          outputLengthsPtr[i] = accumulate(rangesData, start, end);
          start = end;
        }

        size_t outputSize = accumulate(rangesData, 0, ranges.numel());
        outputData->Resize(outputSize);

        auto outputRawData =
            static_cast<char*>(outputData->raw_mutable_data(data.dtype()));
        VLOG(1) << "Copying data";
        size_t outputOffsetBytes = 0;
        auto itemsize = data.dtype().itemsize();
        for (int i = 0; i < ranges.numel(); i += 2) {
          auto rangeStart = rangesData[i];
          auto rangeLength = rangesData[i + 1];
          if (!rangeLength) {
            continue;
          }
          auto rangeSizeBytes = rangeLength * itemsize;
          CAFFE_ENFORCE(outputOffsetBytes < outputSize * itemsize);
          CAFFE_ENFORCE(rangeStart + rangeLength <= data.numel());
          context_.CopyItemsSameDevice(
              data.dtype(),
              rangeLength,
              rawData + rangeStart * itemsize,
              outputRawData + outputOffsetBytes);
          outputOffsetBytes += rangeSizeBytes;
        }
        CAFFE_ENFORCE(outputOffsetBytes == outputSize * itemsize);
        return true;
        */
    }
    
    #[inline] pub fn accumulate<Index>(
        &mut self, 
        ranges: *mut Index,
        start:  usize,
        end:    usize) -> usize 
    {
        todo!();
        /*
            size_t result = 0;
        for (size_t i = start + 1; i < end; i += 2) {
          result += ranges[i];
        }
        return result;
        */
    }
}

/**
  | Gather items from sparse tensor.
  | 
  | Sparse tensor is described by items
  | and lengths.
  | 
  | This operator gathers items corresponding
  | to lengths at the given indices.
  | 
  | This deliberately doesn't return lengths
  | of OUTPUTS so that both lists and maps
  | can be supported without special cases.
  | 
  | If you need lengths tensor for
  | 
  | OUTPUT, use `Gather`.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LengthsGatherOp<Context> {

    storage: OperatorStorage,
    context: Context,

    offsets: Vec<i64>,
}

#[test] fn lengths_gather_op_example() {

    todo!();

    /*
    Example:
      ITEMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
      LENGTHS = [0, 2, 3, 1, 4]
      INDICES = [0, 2, 4]

      OUTPUT = [2, 3, 4, 6, 7, 8, 9]
    */
}

num_inputs!{LengthsGather, 3}

num_outputs!{LengthsGather, 1}

inputs!{LengthsGather, 
    0 => ("ITEMS", "items tensor"),
    1 => ("LENGTHS", "lengths tensor"),
    2 => ("INDICES", "indices into LENGTHS where items should be gathered")
}

outputs!{LengthsGather, 
    0 => ("OUTPUT", "1-D tensor containing gathered items")
}

input_tags!{
    LengthsGatherOp
    {
        Items,
        Lengths,
        Indices
    }
}

impl<Context> LengthsGatherOp<Context> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, this->template Input<Tensor>(INDICES, CPU));
        */
    }
    
    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
        todo!();
        /*
            auto& items = Input(ITEMS);
        auto& lengths = Input(LENGTHS);
        auto& indices = Input(INDICES);
        auto* output = Output(0);

        CAFFE_ENFORCE_GE(items.dim(), 1, "ITEMS should be at least 1-D");
        CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTHS should be 1-D");
        CAFFE_ENFORCE_EQ(indices.dim(), 1, "INDICES should be 1-D");

        const auto* lengths_data = lengths.template data<int32_t>();
        const auto* indices_data = indices.template data<Index>();

        int64_t total_length = 0;
        for (size_t i = 0; i < indices.numel(); ++i) {
          auto idx = indices_data[i];
          CAFFE_ENFORCE_LT(idx, lengths.numel());
          total_length += lengths_data[idx];
        }
        auto shape = items.sizes().vec();
        shape[0] = total_length;
        output->Resize(shape);

        offsets_.clear();
        int64_t running_offset = 0;
        offsets_.reserve(lengths.numel());
        for (size_t i = 0; i < lengths.numel(); ++i) {
          offsets_.push_back(running_offset);
          running_offset += lengths_data[i];
        }
        CAFFE_ENFORCE_EQ(
            items.size(0),
            running_offset,
            "LENGTHS must match the first dimension of ITEMS");

        auto src_base = static_cast<const char*>(items.raw_data());
        auto block_size = items.size_from_dim(1);
        auto block_bytesize = block_size * items.itemsize();
        auto out = static_cast<char*>(output->raw_mutable_data(items.dtype()));

        for (size_t i = 0; i < indices.numel(); ++i) {
          auto idx = indices_data[i];
          auto length = lengths_data[idx];
          context_.CopyItemsSameDevice(
              items.dtype(),
              length * block_size,
              src_base + offsets_[idx] * block_bytesize,
              out);
          out += length * block_bytesize;
        }
        return true;
        */
    }
}

/**
  | This operator calculate thes histogram
  | of values in input tensor.
  | 
  | There're 2 outputs, one for histogram
  | of current input tensor, and another
  | for histogram of the all input tensors
  | accumulated through history.
  | 
  | The output would contain num_buckets
  | + 2 values. index[1 ... num_buckets]
  | for values in [lower_bound, upper_bound)
  | interval. And the rest 2 for values smaller
  | than lower_bound or greater than upper_bound
  | respectively.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AccumulateHistogramOp<T,Context> {

    storage:            OperatorStorage,
    context:            Context,

    lower_bound:        f32,
    upper_bound:        f32,
    num_buckets:        i32,
    num_output_buckets: i32,
    accumulate_hist:    Vec<i64>,
    phantom:            PhantomData<T>,
}

num_inputs!{AccumulateHistogram, 1}

num_outputs!{AccumulateHistogram, 2}

inputs!{AccumulateHistogram, 
    0 => ("X", "Input tensor.")
}

outputs!{AccumulateHistogram, 
    0 => ("CurHist", "Output histogram of the current tensor."),
    1 => ("AccHist", "Accumulated histogram of the history tensor.")
}

args!{AccumulateHistogram, 
    0 => ("lower_bound", "the lower bound value"),
    1 => ("upper_bound", "the upper bound value"),
    2 => ("num_buckets", "number of buckets to use in [lower_bound, upper_bound)")
}

input_tags!{
    AccumulateHistogramOp
    {
        XIn
    }
}

output_tags!{
    AccumulateHistogramOp
    {
        CurHist,
        AccHist
    }
}

impl<T,Context> AccumulateHistogramOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            lower_bound_(
                this->template GetSingleArgument<float>("lower_bound", 0.0)),
            upper_bound_(
                this->template GetSingleArgument<float>("upper_bound", 1.0)),
            num_buckets_(this->template GetSingleArgument<int>("num_buckets", 1)) 

        CAFFE_ENFORCE_GT(num_buckets_, 0);
        // 2 more for histograms < lower_bound, >= upper_bound respectively
        num_output_buckets_ = num_buckets_ + 2;
        accumulate_hist_ = std::vector<int64_t>(num_output_buckets_, 0);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(X_IN);
        auto* X_data = X.template data<T>();
        int N = X.numel();
        auto* cur_hist = Output(CUR_HIST);
        auto* acc_hist = Output(ACC_HIST);
        cur_hist->Resize(num_output_buckets_);
        acc_hist->Resize(num_output_buckets_);
        auto* cur_hist_data = cur_hist->template mutable_data<int64_t>();
        auto* acc_hist_data = acc_hist->template mutable_data<int64_t>();
        auto segment = (upper_bound_ - lower_bound_) / num_buckets_;
        math::Set<int64_t, Context>(
            num_output_buckets_, 0, cur_hist_data, &context_);

        for (int i = 0; i < N; i++) {
          int bucket_index = -1;
          if (X_data[i] < lower_bound_) {
            bucket_index = 0;
          } else if (X_data[i] >= upper_bound_) {
            bucket_index = num_buckets_ + 1;
          } else {
            bucket_index = (int)((X_data[i] - lower_bound_) / segment) + 1;
          }
          cur_hist_data[bucket_index] += 1;
          accumulate_hist_[bucket_index] += 1;
        }

        for (int i = 0; i < num_output_buckets_; i++) {
          acc_hist_data[i] = accumulate_hist_[i];
        }

        return true;
        */
    }
}

///-----------------------------------------
#[test] fn range_op_example() {
    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Range",
        ["start", "stop", "step"],
        ["output"]
    )

    workspace.FeedBlob("start", np.array(4, dtype=np.int32))
    workspace.FeedBlob("stop", np.array(17, dtype=np.int32))
    workspace.FeedBlob("step", np.array(2, dtype=np.int32))
    print("start:", workspace.FetchBlob("start"))
    print("stop:", workspace.FetchBlob("stop"))
    print("step:", workspace.FetchBlob("step"))
    workspace.RunOperatorOnce(op)
    print("output:", workspace.FetchBlob("output"))

    start: 4
    stop: 17
    step: 2
    output: [ 4  6  8 10 12 14 16]
    */
}

/**
  | Generates an output tensor within the
  | half-open interval $[start, stop)$
  | (the interval including start but excluding
  | stop).
  | 
  | - The `start` input is optional, and
  | defaults to 0 when not set.
  | 
  | - The `step` input is optional, and defaults
  | to 1 when not set.
  | 
  | - The type of the `output` tensor is determined
  | by the types of inputs used.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RangeOp<Context> {

    storage: OperatorStorage,
    context: Context,

    /// local CPU tensor for copying constants.
    local:   Tensor, // default = CPU
}

num_inputs!{Range, (1,3)}

num_outputs!{Range, 1}

inputs!{Range, 
    0 => ("start",  "(*Tensor*): [OPTIONAL] scalar or 1-element tensor containing the start of the interval (inclusive) (default=0)"),
    1 => ("stop",   "(*Tensor*): scalar or 1-element tensor containing the end of the interval (exclusive)"),
    2 => ("step",   "(*Tensor*): [OPTIONAL] scalar or 1-element tensor specifying the spacing between values (default=1)")
}

outputs!{Range, 
    0 => ("output", "(*Tensor*): 1D tensor of same type as inputs that contains the sequence")
}

register_cpu_operator!{Range, RangeOp<CPUContext>}

no_gradient!{Range}

impl<Context> RangeOp<Context> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t, float, double>>::call(
            this, Input(0));
        */
    }
    
    #[inline] pub fn read_scalar_input<T>(&mut self, index: i32) -> T {
        todo!();
        /*
            if (std::is_same<Context, TensorCPU>::value) {
          return Input(index).template data<T>()[0];
        } else {
          local_.CopyFrom(Input(index));
          return local_.template data<T>()[0];
        }
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            T stop = 0;
        T start = 0;
        T step = 1;

        for (int i = 0; i < InputSize(); ++i) {
          CAFFE_ENFORCE_EQ(
              Input(i).numel(), 1, "All inputs must be scalar/1D tensor.");
        }

        switch (InputSize()) {
          case 1:
            stop = readScalarInput<T>(0);
            break;
          case 2:
            start = readScalarInput<T>(0);
            stop = readScalarInput<T>(1);
            break;
          case 3:
            step = readScalarInput<T>(2);
            start = readScalarInput<T>(0);
            stop = readScalarInput<T>(1);
            break;
        }
        CAFFE_ENFORCE_NE(step, 0, "Step size cannot be 0.");
        int length;
        auto diff = stop - start;
        if (std::is_integral<T>::value) {
          // Avoid casting to and from floats in case it introduces rounding and
          // avoid mod because the compiler doesn't strip unused code until later.
          length = diff / step;
          if (length * step < diff) {
            length += 1;
          }
        } else {
          length = static_cast<int>(ceil(diff / step));
        }

        // Match numpy's behavior here.
        if (length <= 0) {
          Output(0, {0}, at::dtype<T>());
          return true;
        } else {
          auto* output = Output(0, {length}, at::dtype<T>());
          return DoRunOnDevice<T>(start, step, output);
        }
        */
    }
}

impl RangeOp<CPUContext> {
    
    #[inline] pub fn do_run_on_device<T>(
        &mut self, 
        start:  &T,
        step:   &T,
        output: *mut Tensor) -> bool 
    {
        todo!();
        /*
            auto* output_data = output->template mutable_data<T>();
      for (int i = 0; i < output->numel(); ++i) {
        output_data[i] = i * step + start;
      }
      return true;
        */
    }
}

///-----------------------------------------
pub struct ThrowExceptionOp {
    storage: OperatorStorage,
    context: CPUContext,
    message: String,
}

register_cpu_operator!{ThrowException, ThrowExceptionOp}

num_inputs!{ThrowException, 0}

num_outputs!{ThrowException, 0}

should_not_do_gradient!{ThrowException}

impl ThrowExceptionOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
            message_(GetSingleArgument<std::string>(
                "message",
                "Exception from ThrowExceptionOp"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_THROW(message_);
        */
    }
}

///-----------------------------------------
pub struct ThrowChildThreadExceptionOp {
    storage: OperatorStorage,
    context: CPUContext,
    message: String,
}

register_cpu_operator!{ThrowChildThreadException, ThrowChildThreadExceptionOp}

num_inputs!{ThrowChildThreadException, 0}

num_outputs!{ThrowChildThreadException, 0}

should_not_do_gradient!{ThrowChildThreadException}

impl ThrowChildThreadExceptionOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
            message_(GetSingleArgument<std::string>(
                "message",
                "Exception from ThrowChildThreadExceptionOp"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            std::thread t([this]() { CAFFE_THROW(this->message_); });

        t.join();
        return true;
        */
    }
}


///-----------------------------------------
pub struct LogFatalOp {
    storage: OperatorStorage,
    context: CPUContext,
    message: String,
}

register_cpu_operator!{LogFatal, LogFatalOp}

num_inputs!{LogFatal, 0}

num_outputs!{LogFatal, 0}

should_not_do_gradient!{LogFatal}

impl LogFatalOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
            message_(GetSingleArgument<std::string>(
                "message",
                "Logging from LogFatalOp"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            LOG(FATAL) << message_;
        return true;
        */
    }
}

///-----------------------------------------
pub struct FailOp {
    storage: OperatorStorage,
    context: CPUContext,
}

register_cpu_operator!{Fail, FailOp}

num_inputs!{Fail, 0}

num_outputs!{Fail, 0}

should_not_do_gradient!{Fail}

impl FailOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}


impl WeightedSumOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DoRunWithType<float>();
        */
    }
}

impl WeightedSumGradientOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DoRunWithType<float>();
        */
    }
}


#[inline] pub fn weighted_sum_shape_inference(
    unused: &OperatorDef,
    input:  &Vec<TensorShape>) -> Vec<TensorShape> {
    
    todo!();
    /*
        vector<TensorShape> out(1);
      out[0] = in[0];
      return out;
    */
}

#[inline] pub fn cost_inference_for_weighted_sum(
    unused: &OperatorDef,
    input:  &Vec<TensorShape>) -> OpSchemaCost 
{
    todo!();
    /*
        CAFFE_ENFORCE_EQ(
          in.size() % 2, 0, "WeightedSum requires an even number of inputs");
      struct OpSchema::Cost c;

      const auto& X0 = in[0];
      const auto& nElem = nElemFromDim(X0);
      const auto& nInputs = in.size();
      c.flops = (nInputs - 1) * nElem;
      c.bytes_read = (nInputs / 2) * (nElem + 1) * sizeof(X0.data_type());
      c.bytes_written = nElem * sizeof(X0.data_type());
      c.params_bytes = (nInputs / 2) * sizeof(X0.data_type());
      return c;
    */
}

register_cpu_operator!{WallClockTime, WallClockTimeOp<CPUContext>}
register_cpu_operator!{Print, PrintOp<CPUContext>}
register_cpu_operator!{FlattenToVec, FlattenToVecOp<CPUContext>}
register_cpu_operator!{Alias, AliasOp<CPUContext>}
register_cpu_operator!{ResizeLike, ResizeLikeOp<CPUContext>}
register_cpu_operator!{SumInt, SumOp<CPUContext>}
register_cpu_operator!{WeightedSum, WeightedSumOp<CPUContext>}
register_cpu_operator!{WeightedSumGradient, WeightedSumGradientOp<CPUContext>}
register_cpu_operator!{ScatterWeightedSum, ScatterWeightedSumOp<float, CPUContext>}
register_cpu_operator!{ScatterAssign, ScatterAssignOp<CPUContext>}
register_cpu_operator!{Scatter, ScatterOp<CPUContext>}

register_cpu_operator!{LengthsToShape, LengthsToShapeOp<CPUContext>}
register_cpu_operator!{HasElements, HasElementsOp<CPUContext>}
register_cpu_operator!{GatherRanges, GatherRangesOp<CPUContext>}
register_cpu_operator!{LengthsGather, LengthsGatherOp<CPUContext>}
register_cpu_operator!{LengthsToSegmentIds, LengthsToSegmentIdsOp<CPUContext>}
register_cpu_operator!{LengthsToRanges, LengthsToRangesOp<CPUContext>}
register_cpu_operator!{LengthsToOffsets, LengthsToOffsetsOp<CPUContext>}
register_cpu_operator!{SegmentIdsToLengths, SegmentIdsToLengthsOp<CPUContext>}
register_cpu_operator!{SegmentIdsToRanges, SegmentIdsToRangesOp<CPUContext>}
register_cpu_operator!{LengthsToWeights, LengthsToWeightsOp<CPUContext>}
register_cpu_operator!{EnsureDense, EnsureDenseOp<CPUContext>}
register_cpu_operator!{AccumulateHistogram, AccumulateHistogramOp<float, CPUContext>}

pub struct GetEnsureDenseGradient;

impl GetGradientDefs for GetEnsureDenseGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            GradOut(0).IsSparse() || GradOut(0).IsDense(),
            "Input gradient ",
            O(0),
            " should be either sparse or dense.");

        if (GradOut(0).IsDense()) {
          SetDense(0, GO(0));
          return vector<OperatorDef>();
        } else {
          return SingleGradientDef(
              "SparseToDense",
              "",
              vector<string>{GO_I(0), GO_V(0), I(0)},
              vector<string>{GI(0)});
        }
        */
    }
}

pub struct GetAliasGradient;

impl GetGradientDefs for GetAliasGradient {
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            // We will simply pass-along the gradient. Nothing needs to
        // be calculated.
        SetDense(0, GO(0));
        return vector<OperatorDef>();
        */
    }
}

register_gradient!{Alias, GetAliasGradient}

should_not_do_gradient!{ResizeLike}

pub struct GetSumGradient;

impl GetGradientDefs for GetSumGradient {
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            for (auto i = 0; i < def_.input_size(); ++i) {
          SetDense(i, GO(0));
        }
        return vector<OperatorDef>();
        */
    }
}

register_gradient!{Sum, GetSumGradient}

should_not_do_gradient!{ScatterWeightedSum}
should_not_do_gradient!{ScatterAssign}
should_not_do_gradient!{Scatter}

pub struct GetWeightedSumGradient;

impl GetGradientDefs for GetWeightedSumGradient {
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            ArgumentHelper argsHelper(def_);
        const bool grad_on_w = argsHelper.GetSingleArgument<bool>("grad_on_w", 0);

        auto inputs = vector<string>{GO(0)};
        auto outputs = vector<string>();
        for (int i = 0; i < def_.input_size(); i += 2) {
          inputs.push_back(I(i));
          inputs.push_back(I(i + 1));
          outputs.push_back(GI(i));
        }

        if (grad_on_w) {
          for (int i = 0; i < def_.input_size(); i += 2) {
            outputs.push_back(GI(i + 1));
          }
        }

        return SingleGradientDef("WeightedSumGradient", "", inputs, outputs);
        */
    }
}

pub struct GetFlattenToVecGradient;

impl GetGradientDefs for GetFlattenToVecGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "ResizeLike", "", vector<string>{GO(0), I(0)}, vector<string>{GI(0)});
        */
    }
}

register_gradient!{WeightedSum, GetWeightedSumGradient}
register_gradient!{FlattenToVec, GetFlattenToVecGradient}

should_not_do_gradient!{LengthsToSegmentIds}
should_not_do_gradient!{SegmentIdsToLengths}
should_not_do_gradient!{SegmentIdsToRanges}
should_not_do_gradient!{SegmentIdsToLengthWeights}
should_not_do_gradient!{GatherRangesOp}
should_not_do_gradient!{LengthsGather}
should_not_do_gradient!{AccumulateHistogram}
