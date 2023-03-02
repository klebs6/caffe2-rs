crate::ix!();

/**
  | Collect tensor into tensor vector by
  | reservoir sampling, argument num_to_collect
  | indicates the max number of tensors
  | that will be collected.
  | 
  | The first half of the inputs are tensor
  | vectors, which are also the outputs.
  | The second half of the inputs are the
  | tensors to be collected into each vector
  | (in the same order).
  | 
  | The input tensors are collected in all-or-none
  | manner. If they are collected, they
  | will be placed at the same index in the
  | output vectors.
  |
  */
pub struct CollectTensorOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    /// number of tensors to collect
    num_to_collect: i32,

    /// number of tensors visited
    num_visited: i32,
}

num_outputs!{CollectTensor, (1,INT_MAX)}

args!{CollectTensor, 
    0 => ("num_to_collect", "The max number of tensors to collect")
}

enforce_inplace!{CollectTensor,    
    |input: i32, output: i32| {
        input == output
    }
}

num_inputs!{CollectTensor,         
    |n: i32| {
        n > 0 && n % 2 == 0
    }
}

num_inputs_outputs!{CollectTensor, 
    |input: i32, output: i32| {
        input == output * 2
    }
}

impl<Context> CollectTensorOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            numToCollect_(
                OperatorStorage::GetSingleArgument<int>("num_to_collect", -1)),
            numVisited_(0) 

        CAFFE_ENFORCE(numToCollect_ > 0);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            int pos = -1;
        if (numVisited_ < numToCollect_) {
          // append
          pos = numVisited_;
        } else {
          // uniform between [0, numVisited_]
          at::uniform_int_from_to_distribution<int> uniformDist(numVisited_+1, 0);
          pos = uniformDist(context_.RandGenerator());
          if (pos >= numToCollect_) {
            // discard
            pos = -1;
          }
        }

        for (int i = 0; i < OutputSize(); ++i) {
          // TENSOR_VECTOR_IN is enforced inplace with TENSOR_VECTOR_OUT
          TensorVectorPtr& tensorVector = *OperatorStorage::Output<TensorVectorPtr>(i);

          if (numVisited_ >= numToCollect_) {
            CAFFE_ENFORCE(
                tensorVector->size() == numToCollect_,
                "TensorVecotor size = ",
                tensorVector->size(),
                " is different from numToCollect = ",
                numToCollect_);
          }

          const auto& tensor = Input(OutputSize() + i);

          if (pos < 0) {
            // discard
            CAFFE_ENFORCE(numVisited_ >= numToCollect_);
          } else if (pos >= tensorVector->size()) {
            // append
            tensorVector->emplace_back();
            ReinitializeAndCopyFrom(
                &tensorVector->back(),
                Context::GetDeviceType(),
                tensor); // sync copy
          } else {
            // replace
            tensorVector->at(pos).CopyFrom(tensor); // sync copy
          }
        }

        numVisited_++;
        return true;
        */
    }
}

