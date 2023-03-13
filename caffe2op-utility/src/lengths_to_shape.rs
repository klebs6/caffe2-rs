crate::ix!();

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
