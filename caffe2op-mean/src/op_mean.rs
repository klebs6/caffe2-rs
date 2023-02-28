crate::ix!();

use crate::{
    OperatorStorage,
    OperatorDef,
    GradientMakerBase,
};

#[test] fn mean_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Mean",
        ["X", "Y", "Z"],
        ["X"],
    )

    workspace.FeedBlob("X", (np.random.rand(3,3)).astype(np.float32))
    workspace.FeedBlob("Y", (np.random.rand(3,3)).astype(np.float32))
    workspace.FeedBlob("Z", (np.random.rand(3,3)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    print("Y:", workspace.FetchBlob("Y"))
    print("Z:", workspace.FetchBlob("Z"))
    workspace.RunOperatorOnce(op)
    print("Mean:", workspace.FetchBlob("X"))

    X:
    [[0.6035237  0.5305746  0.6298913 ]
     [0.9169737  0.01280353 0.16286302]
     [0.6017664  0.9946255  0.05128575]]
    Y:
    [[0.07544111 0.45371833 0.08460239]
     [0.9708728  0.7422064  0.7933344 ]
     [0.97671497 0.3411384  0.73818344]]
    Z:
    [[0.08837954 0.90187573 0.46734726]
     [0.6308827  0.8719029  0.39888734]
     [0.90059936 0.92883426 0.5695987 ]]
    Mean:
    [[0.25578147 0.6287229  0.39394698]
     [0.8395764  0.5423043  0.45169494]
     [0.8263602  0.75486606 0.45302266]]

    */
}

/**
  | Element-wise mean of an arbitrary number
  | of input tensors. This operation can
  | be performed in-place, by using the
  | first input blob as the output blob.
  | All inputs must have the same shape and
  | data type, and the output will have the
  | same shape as the inputs.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/mean_op.cc
  |
  */
pub struct MeanOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{Mean, (1,INT_MAX)}

num_outputs!{Mean, 1}

inputs!{Mean, 
    0 => ("X, Y, ...", "*(type: Tensor`<Ord>`)* List of input tensors with the same shape.")
}

outputs!{Mean, 
    0 => ("M", "*(type: Tensor`<Ord>`)* Output tensor with the same dimensions as inputs. Contains the mean values of the input tensors calculated element-wise.")
}

identical_type_and_shape_of_input!{Mean, 0}

allow_inplace!{Mean, vec![(0, 0)]}

impl<Context> MeanOp<Context> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& input0 = Input(0);

        auto* output = Output(0, input0.sizes(), at::dtype<T>());
        output->CopyFrom(input0, true /*async*/);

        if (InputSize() == 1) {
          return true;
        }

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

        T* output_data = output->template mutable_data<T>();
        for (int i = 1; i < InputSize(); ++i) {
          math::Add(
              output->numel(),
              output_data,
              Input(i).template data<T>(),
              output_data,
              &context_);
        }

        math::Scale(
            output->numel(),
            1.0f / InputSize(),
            output_data,
            output_data,
            &context_);

        return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (Input(0).template IsType<float>()) {
          return DoRunWithType<float>();
        } else if (Input(0).template IsType<double>()) {
          return DoRunWithType<double>();
        } else {
          CAFFE_THROW(
              "Mean operator only supports 32-bit float or 64-bit double, but",
              " input was of type ",
              Input(0).dtype().name());
        }
        */
    }
}

///-----------------------------------------
pub struct MeanGradientOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{MeanGradient, 1}

num_outputs!{MeanGradient, (1,INT_MAX)}

allow_inplace!{MeanGradient, vec![(0, 0)]}

impl<Context> MeanGradientOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& dY = Input(0);
        const auto* dY_data = dY.template data<T>();
        int size = dY.numel();

        int num_inputs = OutputSize();
        float scale = 1.0f / num_inputs;

        // dX0 = scale * dY

        auto* dX0 = Output(0, dY.sizes(), at::dtype<T>());
        math::Scale(
            size, scale, dY_data, dX0->template mutable_data<T>(), &context_);

        // Copy the rest dX
        for (int i = 1; i < num_inputs; i++) {
          auto* cur_dX = Output(i);
          cur_dX->ResizeLike(dY);
          cur_dX->CopyFrom(*dX0, true /*async*/);
        }

        return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (Input(0).template IsType<float>()) {
          return DoRunWithType<float>();
        } else if (Input(0).template IsType<double>()) {
          return DoRunWithType<double>();
        } else {
          CAFFE_THROW(
              "Mean operator only supports 32-bit float or 64-bit double, but",
              " input was of type ",
              Input(0).dtype().name());
        }
        */
    }
}

register_cpu_operator!{Mean,         MeanOp<CPUContext>}

register_cpu_operator!{MeanGradient, MeanGradientOp<CPUContext>}

pub struct GetMeanGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetMeanGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            auto outputs = std::vector<string>();
        for (int i = 0; i < def_.input_size(); i++) {
          outputs.push_back(GI(i));
        }
        return SingleGradientDef(
            "MeanGradient", "", std::vector<string>{GO(0)}, outputs);
        */
    }
}

register_gradient!{Mean, GetMeanGradient}
