crate::ix!();

// ReduceFrontSum: columnwise sum
impl SumReduceDimsOp<CPUContext, true, false> {

    #[inline] pub fn compute<T>(&mut self, 
        rows:         i32,
        cols:         i32,
        in_data:      *const T,
        lengths_data: *const i32,
        out_data:     *mut T)  {
    
        todo!();
        /*
            for (int j = 0; j < cols; j++) {
        T sum = in_data[j];
        int length = lengths_data == nullptr ? rows : lengths_data[j];
        for (int i = 1; i < length; i++) {
          sum += in_data[i * cols + j];
        }
        out_data[j] = sum;
      }
        */
    }
}

// ReduceBackSum: rowwise sum
impl SumReduceDimsOp<CPUContext, false, false> {

    #[inline] pub fn compute<T>(&mut self, 
        rows:         i32,
        cols:         i32,
        in_data:      *const T,
        lengths_data: *const i32,
        out_data:     *mut T)  {
    
        todo!();
        /*
            for (int i = 0; i < rows; i++) {
        int offset = i * cols;
        T sum = in_data[offset];
        int length = lengths_data == nullptr ? cols : lengths_data[i];
        for (int j = 1; j < length; j++) {
          sum += in_data[offset + j];
        }
        out_data[i] = sum;
      }
        */
    }
}

/// ReduceFrontSumGradient
impl SumReduceDimsGradientOp<CPUContext, true, false> {

    #[inline] pub fn compute<T>(&mut self, 
        rows:         i32,
        cols:         i32,
        d_ydata:      *const T,
        lengths_data: *const i32,
        d_xdata:      *mut T)  {
    
        todo!();
        /*
            for (int i = 0; i < rows * cols; i++) {
        int row = i / cols;
        int col = i % cols;
        if (lengths_data == nullptr || row < lengths_data[col]) {
          dXdata[i] = dYdata[col];
        } else {
          dXdata[i] = 0;
        }
      }
        */
    }
}

// ReduceBackSumGradient
impl SumReduceDimsGradientOp<CPUContext, false, false> {

    #[inline] pub fn compute<T>(&mut self, 
        rows:         i32,
        cols:         i32,
        d_ydata:      *const T,
        lengths_data: *const i32,
        d_xdata:      *mut T)  {
    
        todo!();
        /*
            for (int i = 0; i < rows * cols; i++) {
        int row = i / cols;
        int col = i % cols;
        if (lengths_data == nullptr || col < lengths_data[row]) {
          dXdata[i] = dYdata[row];
        } else {
          dXdata[i] = 0;
        }
      }
        */
    }
}

/**
  | Reduces the input tensor along the last
  | dimension of the by applying **sum**.
  | 
  | Can reduce more than one of the "first"
  | dimensions by setting `num_reduce_dim`.
  | 
  | A second (optional) input, `lengths`,
  | can be passed, which enforces that only
  | a subset of the elements are considered
  | in the sum operation.
  | 
  | - If input tensor `X` has shape $(d_0,
  | d_1, d_2, ..., d_n)$, `lengths` must
  | have shape $(d_1 * d_2 * ... * d_{n})$.
  | 
  | - The values of the `lengths` tensor
  | determine how many of the values to consider
  | for each vector in the $d_{0}$ dimension.
  | 
  | For example, if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$
  | and $lengths = [2,3,1,2]$, then $Y =
  | [sum(1,4), sum(5,1,7), sum(2), sum(9,2)]
  | = [2.5, 4.333, 2, 5.5]$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_front_back_sum_ops.cc
  |
  */
register_cpu_operator!{ReduceFrontSum, SumReduceDimsOp<CPUContext, true, false>}

num_inputs!{ReduceFrontSum, (1,2)}

num_outputs!{ReduceFrontSum, 1}

inputs!{ReduceFrontSum, 
    0 => ("X",       "(*Tensor`<float>`*): input tensor"),
    1 => ("lengths", "(*Tensor`<int>`*): number of elements in each sample")
}

outputs!{ReduceFrontSum, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{ReduceFrontSum, 
    0 => ("num_reduce_dims", "(*int*): number of dimensions to reduce (default=1)")
}

tensor_inference_function!{ReduceFrontSum, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
           REDUCTION_OP_SHAPE_INFERENCE(true)
        */
    }
}

#[test] fn reduce_front_sum() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ReduceFrontSum",
        ["X"],
        ["Y"],
        num_reduce_dim=2
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(2,3,3)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X:
    [[[4. 1. 1.]
      [0. 6. 7.]
      [7. 8. 6.]]

     [[5. 7. 7.]
      [0. 1. 6.]
      [2. 9. 0.]]]
    Y: [18. 32. 27.]
    */
}

///--------------------------------

register_cpu_operator!{ReduceFrontSumGradient,
    SumReduceDimsGradientOp<CPUContext, true, false>
}

///-----------------------
pub struct GetReduceFrontSumGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReduceFrontSumGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> grad_in = {GO(0), I(0)};
        if (def_.input_size() == 2) {
          grad_in.push_back(I(1));
        }
        return SingleGradientDef(
            "ReduceFrontSumGradient", "", grad_in, vector<string>{GI(0)});
        */
    }
}

num_inputs!{ReduceFrontSumGradient, (2,3)}

num_outputs!{ReduceFrontSumGradient, 1}

register_gradient!{ReduceFrontSum, GetReduceFrontSumGradient}

///----------------------

#[test] fn reduce_back_sum_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ReduceBackSum",
        ["X"],
        ["Y"],
        num_reduce_dim=2
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(1,2,3,3)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X:
    [[[[2. 7. 7.]
       [1. 1. 0.]
       [9. 7. 2.]]

      [[6. 6. 4.]
       [1. 2. 6.]
       [6. 6. 3.]]]]
    Y: [[36. 40.]]
    */
}

/**
  | Reduces the input tensor along the last
  | dimension of the by applying **sum**.
  | 
  | Can reduce more than one of the "last"
  | dimensions by setting `num_reduce_dim`.
  | 
  | A second (optional) input, `lengths`,
  | can be passed, which enforces that only
  | a subset of the elements are considered
  | in the sum operation.
  | 
  | - If input tensor `X` has shape $(d_0,
  | d_1, d_2, ..., d_n)$, `lengths` must
  | have shape $(d_0 * d_1 * d_2 * ... * d_{n-1})$.
  | 
  | - The values of the `lengths` tensor
  | determine how many of the values to consider
  | for each vector in the $d_{n-1}$ dimension.
  | 
  | For example if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$
  | and $lengths = [2,3,1]$, then $Y = [sum(1,5),
  | sum(4,1,8), sum(2)] = [6, 13, 2]$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_front_back_sum_ops.cc
  |
  */
register_cpu_operator!{ReduceBackSum,         SumReduceDimsOp<CPUContext, false, false>}

num_inputs!{ReduceBackSum, (1,2)}

num_outputs!{ReduceBackSum, 1}

inputs!{ReduceBackSum, 
    0 => ("X", "(*Tensor`<float>`*): input tensor"),
    1 => ("lengths", "(*Tensor`<int>`*): number of elements in each sample")
}

outputs!{ReduceBackSum, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{ReduceBackSum, 
    0 => ("num_reduce_dims", "(*int*): number of dimensions to reduce (default=1)")
}

tensor_inference_function!{ReduceBackSum, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
           REDUCTION_OP_SHAPE_INFERENCE(false)
        */
    }
}

///------------------------
register_cpu_operator!{ReduceBackSumGradient, SumReduceDimsGradientOp<CPUContext, false, false>}

num_inputs!{ReduceBackSumGradient, (2,3)}

num_outputs!{ReduceBackSumGradient, 1}

pub struct GetReduceBackSumGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReduceBackSumGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> grad_in = {GO(0), I(0)};
        if (def_.input_size() == 2) {
          grad_in.push_back(I(1));
        }
        return SingleGradientDef(
            "ReduceBackSumGradient", "", grad_in, vector<string>{GI(0)});
        */
    }
}

register_gradient!{ReduceBackSum, GetReduceBackSumGradient}

