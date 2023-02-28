crate::ix!();

use crate::{
    OperatorDef,
    SumReduceDimsOp,
    GradientMakerBase,
    SumReduceDimsGradientOp,
    CPUContext
};

macro_rules! reduction_op_shape_inference {
    ($is_front_reducer:ident) => {
        todo!();
        /*
        
          CAFFE_ENFORCE_LE(1, in.size());                                           
          CAFFE_ENFORCE_GE(2, in.size());                                           
          ArgumentHelper helper(def);                                               
          int num_reduce_dims = helper.GetSingleArgument<int>("num_reduce_dim", 1); 
          int start_index = is_front_reducer ? num_reduce_dims : 0;                 
          int end_index = is_front_reducer ? in[0].dims_size()                      
                                           : in[0].dims_size() - num_reduce_dims;   
          vector<int> output_shape;                                                 
          for (int i = start_index; i < end_index; ++i) {                           
            output_shape.push_back(in[0].dims(i));                                  
          }                                                                         
          return vector<TensorShape>{                                               
              CreateTensorShape(output_shape, in[0].data_type())};
        */
    }
}

/// ReduceFrontMean: columnwise mean
impl SumReduceDimsOp<CPUContext, true, true> {

    #[inline] pub fn compute_on_cpu<T>(&mut self, 
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
        out_data[j] = sum / length;
      }
        */
    }
}

// ReduceBackMean: rowwise mean
impl SumReduceDimsOp<CPUContext, false, true> {

    #[inline] pub fn compute_on_cpu<T>(&mut self, 
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
        out_data[i] = sum / length;
      }
        */
    }
}

// ReduceFrontMeanGradient
impl SumReduceDimsGradientOp<CPUContext, true, true> {

    #[inline] pub fn compute_on_cpu<T>(&mut self, 
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
        if (lengths_data == nullptr) {
          dXdata[i] = dYdata[col] / rows;
        } else if (row < lengths_data[col]) {
          dXdata[i] = dYdata[col] / lengths_data[col];
        } else {
          dXdata[i] = 0;
        }
      }
        */
    }
}

impl SumReduceDimsGradientOp<CPUContext, false, true> {

    // ReduceBackMeanGradient
    #[inline] pub fn compute_on_cpu<T>(&mut self, 
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
        if (lengths_data == nullptr) {
          dXdata[i] = dYdata[row] / cols;
        } else if (col < lengths_data[row]) {
          dXdata[i] = dYdata[row] / lengths_data[row];
        } else {
          dXdata[i] = 0;
        }
      }
        */
    }
}

///---------------------------------------------------
#[test] fn reduce_front_mean_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ReduceFrontMean",
        ["X"],
        ["Y"],
        num_reduce_dim=2
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(2,3,3)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X:
    [[[5. 0. 9.]
      [4. 1. 1.]
      [9. 0. 8.]]

     [[2. 6. 7.]
      [6. 2. 6.]
      [0. 4. 5.]]]
    Y: [4.3333335    2.1666667     6.]

    */
}

/**
  | Reduces the input tensor along the last
  | dimension of the by applying **mean**.
  | 
  | Can reduce more than one of the "first"
  | dimensions by setting `num_reduce_dim`.
  | 
  | A second (optional) input, `lengths`,
  | can be passed, which enforces that only
  | a subset of the elements are considered
  | in the mean operation.
  | 
  | - If input tensor `X` has shape $(d_0,
  | d_1, d_2, ..., d_n)$, `lengths` must
  | have shape $(d_1 * d_2 * ... * d_{n})$.
  | 
  | - The values of the `lengths` tensor
  | determine how many of the values to consider
  | for each vector in the $d_{0}$ dimension.
  | 
  | For example if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$
  | and $lengths = [2,3,1,2]$, then $Y =
  | [mean(1,4), mean(5,1,7), mean(2),
  | mean(9,2)] = [2.5, 4.333, 2, 5.5]$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_front_back_mean_ops.cc
  |
  */
register_cpu_operator!{ReduceFrontMean, SumReduceDimsOp<CPUContext, true, true>}

num_inputs!{ReduceFrontMean, (1,2)}

num_outputs!{ReduceFrontMean, 1}

inputs!{ReduceFrontMean, 
    0 => ("X", "(*Tensor`<float>`*): input tensor"),
    1 => ("lengths", "(*Tensor`<int>`*): number of elements in each sample")
}

outputs!{ReduceFrontMean, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{ReduceFrontMean, 
    0 => ("num_reduce_dims", "(*int*): number of dimensions to reduce (default=1)")
}

tensor_inference_function!{ReduceFrontMean, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
           REDUCTION_OP_SHAPE_INFERENCE(true)
        */

    }
}

inherit_onnx_schema!{ReduceFrontMean, "ReduceMean"}

register_cpu_operator!{ReduceFrontMeanGradient, SumReduceDimsGradientOp<CPUContext, true, true>}

num_inputs!{ReduceFrontMeanGradient, (2,3)}

num_outputs!{ReduceFrontMeanGradient, 1}

pub struct GetReduceFrontMeanGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReduceFrontMeanGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> grad_in = {GO(0), I(0)};
        if (def_.input_size() == 2) {
          grad_in.push_back(I(1));
        }
        return SingleGradientDef(
            "ReduceFrontMeanGradient", "", grad_in, vector<string>{GI(0)});
        */
    }
}

register_gradient!{ReduceFrontMean, GetReduceFrontMeanGradient}

/**
  | Reduces the input tensor along the last
  | dimension of the by applying **mean**.
  | 
  | Can reduce more than one of the "last"
  | dimensions by setting `num_reduce_dim`.
  | 
  | A second (optional) input, `lengths`,
  | can be passed, which enforces that only
  | a subset of the elements are considered
  | in the mean operation.
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
  | and $lengths = [2,3,1]$, then $Y = [mean(1,5),
  | mean(4,1,8), mean(2)] = [3, 4.333,
  | 2]$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_front_back_mean_ops.cc
  |
  */
#[test] fn reduce_back_mean_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ReduceBackMean",
        ["X"],
        ["Y"],
        num_reduce_dim=2
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(1,2,3,3)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X:
    [[[[5. 9. 0.]
       [8. 4. 0.]
       [2. 2. 4.]]

      [[9. 0. 9.]
       [7. 9. 7.]
       [1. 0. 2.]]]]
    Y: [[3.7777777 4.888889 ]]

    */
}

num_inputs!{ReduceBackMean, (1,2)}

num_outputs!{ReduceBackMean, 1}

inputs!{ReduceBackMean, 
    0 => ("X", "(*Tensor`<float>`*): input tensor"),
    1 => ("lengths", "(*Tensor`<int>`*): number of elements in each sample")
}

outputs!{ReduceBackMean, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{ReduceBackMean, 
    0 => ("num_reduce_dims", "(*int*): number of dimensions to reduce (default=1)")
}

tensor_inference_function!{ReduceBackMean, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
           REDUCTION_OP_SHAPE_INFERENCE(false)
        */
    }
}

inherit_onnx_schema!{ReduceBackMean, "ReduceMean"}

register_cpu_operator!{ReduceBackMean,         SumReduceDimsOp<CPUContext, false, true>}

register_cpu_operator!{ReduceBackMeanGradient, SumReduceDimsGradientOp<CPUContext, false, true>}

pub struct GetReduceBackMeanGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReduceBackMeanGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> grad_in = {GO(0), I(0)};
        if (def_.input_size() == 2) {
          grad_in.push_back(I(1));
        }
        return SingleGradientDef(
            "ReduceBackMeanGradient", "", grad_in, vector<string>{GI(0)});
        */
    }
}

register_gradient!{ReduceBackMean, GetReduceBackMeanGradient}

num_inputs!{ReduceBackMeanGradient, (2,3)}

num_outputs!{ReduceBackMeanGradient, 1}

