crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorDef,
    OperatorStorage
};

#[test] fn mat_mul_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "MatMul",
        ["A", "B"],
        ["Y"],
    )

    workspace.FeedBlob("A", np.random.randint(10, size=(3,3)).astype(np.float32))
    workspace.FeedBlob("B", np.random.randint(10, size=(3,3)).astype(np.float32))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    A: [[1. 8. 3.]
     [6. 4. 4.]
     [5. 4. 7.]]
    B: [[4. 0. 3.]
     [3. 1. 1.]
     [8. 5. 8.]]
    Y: [[52. 23. 35.]
     [68. 24. 54.]
     [88. 39. 75.]]

    */
}

/**
  | Matrix multiplication $Y = A * B$, where
  | `A` has size (M x K), `B` has size (K x N),
  | and `Y` will have a size (M x N).
  | 
  | To transpose `A` or `B` before multiplication,
  | pass 1 to the `trans_a` and/or `trans_b`
  | arguments, which separate the first
  | and second dimensions of the respective
  | matrices using `axis_a` and `axis_b`.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/matmul_op.cc
  |
  */
pub struct MatMulOp<T, Context, Engine> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    /**
      | A local vector to cache the output shape
      | so we don't need to recreate a vector
      | object every time we run Run().
      |
      */
    y_shape_cache: Vec<i64>, // default = {0, 0};

    axis_a:   i32, // default = 1
    axis_b:   i32, // default = 1
    trans_a:  bool,
    trans_b:  bool,
    phantom: PhantomData<T>,
    phantomE: PhantomData<Engine>,
}

num_inputs!{MatMul, (2,3)}

num_outputs!{MatMul, 1}

inputs!{MatMul, 
    0 => ("A", "*(type: Tensor`<float>`)* 2D matrix of size (M x K)."),
    1 => ("B", "*(type: Tensor`<float>`)* 2D matrix of size (K x N).")
}

outputs!{MatMul, 
    0 => ("Y", "*(type: Tensor`<float>`)* 2D matrix of size (M x N).")
}

args!{MatMul, 
    0 => ("axis_a", "*(type: int; default: 1)* Exclusive axis that divides the first and second dimension of matrix `A`."),
    1 => ("axis_b", "*(type: int; default: 1)* Exclusive axis that divides the first and second dimension of matrix `B`."),
    2 => ("trans_a", "*(type: int; default: 0)* Pass 1 to transpose `A` before multiplication and after the dimension adjustment using `axis_a`."),
    3 => ("trans_b", "*(type: int; default: 0)* Pass 1 to transpose `B` before multiplication and after the dimension adjustment using `axis_b`.")
}

tensor_inference_function!{MatMul, /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out[0].set_data_type(in[0].data_type());
      ArgumentHelper arg_helper(def);
      int axis_a = arg_helper.GetSingleArgument<int>("axis_a", 1);
      int axis_b = arg_helper.GetSingleArgument<int>("axis_b", 1);
      int trans_a = arg_helper.GetSingleArgument<bool>("trans_a", false);
      int trans_b = arg_helper.GetSingleArgument<bool>("trans_b", false);
      int canonical_axis_a = canonical_axis_index_(axis_a, in[0].dims().size());
      int canonical_axis_b = canonical_axis_index_(axis_b, in[0].dims().size());

      int M = size_to_dim_(canonical_axis_a, GetDimsVector(in[0]));
      int N = size_from_dim_(canonical_axis_b, GetDimsVector(in[1]));
      if (trans_a) {
        M = size_from_dim_(canonical_axis_a, GetDimsVector(in[0]));
      }
      if (trans_b) {
        N = size_to_dim_(canonical_axis_b, GetDimsVector(in[1]));
      }

      out[0].add_dims(M);
      out[0].add_dims(N);

      return out;
    } */
}

impl<T,Context,Engine> MatMulOp<T,Context,Engine> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axis_a_(this->template GetSingleArgument<int>("axis_a", 1)),
            axis_b_(this->template GetSingleArgument<int>("axis_b", 1)),
            trans_a_(this->template GetSingleArgument<int>("trans_a", 0)),
            trans_b_(this->template GetSingleArgument<int>("trans_b", 0))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& A = Input(0);
        const auto& B = Input(1);

        const auto canonical_axis_a = A.canonical_axis_index(axis_a_);
        const auto canonical_axis_b = B.canonical_axis_index(axis_b_);
        int A_dim0 = A.size_to_dim(canonical_axis_a);
        int A_dim1 = A.size_from_dim(canonical_axis_a);
        int B_dim0 = B.size_to_dim(canonical_axis_b);
        int B_dim1 = B.size_from_dim(canonical_axis_b);

        int a_dim0, a_dim1, b_dim0, b_dim1;

        if (trans_a_) {
          a_dim0 = A_dim1;
          a_dim1 = A_dim0;
        } else {
          a_dim0 = A_dim0;
          a_dim1 = A_dim1;
        }

        if (trans_b_) {
          b_dim0 = B_dim1;
          b_dim1 = B_dim0;
        } else {
          b_dim0 = B_dim0;
          b_dim1 = B_dim1;
        }

        auto dimErrorString = [&]() {
          return c10::str(
              "Dimension mismatch: ",
              trans_a_ ? "trans(A): " : "A: ",
              a_dim0,
              " ",
              a_dim1,
              trans_b_ ? ", trans(B): " : ", B: ",
              b_dim0,
              " ",
              b_dim1);
        };
        // Error checking
        CAFFE_ENFORCE(a_dim1 == b_dim0, dimErrorString());

        Y_shape_cache_[0] = a_dim0;
        Y_shape_cache_[1] = b_dim1;
        auto* Y = Output(0, Y_shape_cache_, at::dtype<T>());
        CAFFE_ENFORCE(a_dim0 * b_dim1 == Y->numel(), dimErrorString());
        // Y = A * B
        math::Gemm<T, Context, Engine>(
            trans_a_ ? CblasTrans : CblasNoTrans,
            trans_b_ ? CblasTrans : CblasNoTrans,
            a_dim0,
            b_dim1,
            a_dim1,
            1,
            A.template data<T>(),
            B.template data<T>(),
            0,
            Y->template mutable_data<T>(),
            &context_);

        if (InputSize() == 3) {
          // In gradient op, resize to input
          Y->ResizeLike(Input(2));
        }
        return true;
        */
    }
}

register_cpu_operator!{MatMul, MatMulOp<f32, CPUContext>}

pub struct GetMatMulGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetMatMulGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE(def_.input_size() == 2 || def_.input_size() == 3);

        bool axis_a = 1;
        bool axis_b = 1;
        bool trans_a = 0;
        bool trans_b = 0;

        if (ArgumentHelper::HasArgument(Def(), "trans_a")) {
          trans_a = GetArgument(Def(), "trans_a").i();
        }
        if (ArgumentHelper::HasArgument(Def(), "trans_b")) {
          trans_b = GetArgument(Def(), "trans_b").i();
        }
        if (ArgumentHelper::HasArgument(Def(), "axis_a")) {
          axis_a = GetArgument(Def(), "axis_a").i();
        }
        if (ArgumentHelper::HasArgument(Def(), "axis_b")) {
          axis_b = GetArgument(Def(), "axis_b").i();
        }

        if (trans_a) {
          if (trans_b) {
            // A'B':
            // dA = B'G', dB = G'A'
            return vector<OperatorDef>{
                CreateOperatorDef(
                    "MatMul",
                    "",
                    vector<string>{I(1), GO(0), I(0)},
                    vector<string>{GI(0)},
                    vector<Argument>{MakeArgument<int>("trans_a", 1),
                                     MakeArgument<int>("trans_b", 1),
                                     MakeArgument<int>("axis_a", axis_b)}),
                CreateOperatorDef(
                    "MatMul",
                    "",
                    vector<string>{GO(0), I(0), I(1)},
                    vector<string>{GI(1)},
                    vector<Argument>{MakeArgument<int>("trans_a", 1),
                                     MakeArgument<int>("trans_b", 1),
                                     MakeArgument<int>("axis_b", axis_a)})};
          } else {
            // A'B:
            // dA = BG', dB = AG
            return vector<OperatorDef>{
                CreateOperatorDef(
                    "MatMul",
                    "",
                    vector<string>{I(1), GO(0), I(0)},
                    vector<string>{GI(0)},
                    vector<Argument>{MakeArgument<int>("trans_b", 1),
                                     MakeArgument<int>("axis_a", axis_b)}),
                CreateOperatorDef(
                    "MatMul",
                    "",
                    vector<string>{I(0), GO(0), I(1)},
                    vector<string>{GI(1)},
                    vector<Argument>{MakeArgument<int>("axis_a", axis_a)})};
          }
        } else {
          if (trans_b) {
            // AB':
            // dA = GB, dB = G'A
            return vector<OperatorDef>{
                CreateOperatorDef(
                    "MatMul",
                    "",
                    vector<string>{GO(0), I(1), I(0)},
                    vector<string>{GI(0)},
                    vector<Argument>{MakeArgument<int>("axis_b", axis_b)}),
                CreateOperatorDef(
                    "MatMul",
                    "",
                    vector<string>{GO(0), I(0), I(1)},
                    vector<string>{GI(1)},
                    vector<Argument>{MakeArgument<int>("trans_a", 1),
                                     MakeArgument<int>("axis_b", axis_a)})};
          } else {
            // AB:
            // dA = GB', dB = A'G
            return vector<OperatorDef>{
                CreateOperatorDef(
                    "MatMul",
                    "",
                    vector<string>{GO(0), I(1), I(0)},
                    vector<string>{GI(0)},
                    vector<Argument>{MakeArgument<int>("trans_b", 1),
                                     MakeArgument<int>("axis_b", axis_b)}),
                CreateOperatorDef(
                    "MatMul",
                    "",
                    vector<string>{I(0), GO(0), I(1)},
                    vector<string>{GI(1)},
                    vector<Argument>{MakeArgument<int>("trans_a", 1),
                                     MakeArgument<int>("axis_a", axis_a)})};
          }
        }
        */
    }
}

impl<'a> CopyArguments for GetMatMulGradient<'a> {
    
    #[inline] fn copy_arguments(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}

register_gradient!{MatMul, GetMatMulGradient}

register_cuda_operator!{MatMul, MatMulOp<float, CUDAContext>}
