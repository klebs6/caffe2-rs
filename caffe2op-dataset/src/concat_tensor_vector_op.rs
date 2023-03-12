crate::ix!();

/**
  | Concat Tensors in the std::unique_ptr<std::vector<Tensor>>
  | along the first dimension.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ConcatTensorVectorOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{ConcatTensorVector, 1}

num_outputs!{ConcatTensorVector, 1}

inputs!{ConcatTensorVector, 
    0 => ("vector of Tensor", "std::unique_ptr<std::vector<Tensor> >")
}

outputs!{ConcatTensorVector, 
    0 => ("tensor", "tensor after concatenating")
}

impl<Context> ConcatTensorVectorOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const TensorVectorPtr& tensorVector =
            OperatorStorage::Input<TensorVectorPtr>(TENSOR_VECTOR);

        auto* tensor = Output(TENSOR);
        CAFFE_ENFORCE(!tensorVector->empty());

        vector<int64_t> outputDims(tensorVector->at(0).sizes().vec());
        CAFFE_ENFORCE(outputDims.size() > 0);
        for (int i = 1; i < tensorVector->size(); i++) {
          // the tensor shapes are the same except for the first dimension
          for (int j = 1; j < tensorVector->at(i).dim(); j++) {
            CAFFE_ENFORCE(outputDims[j] == tensorVector->at(i).sizes()[j]);
          }
          CAFFE_ENFORCE(tensorVector->at(0).dtype() == tensorVector->at(i).dtype());
          outputDims[0] += tensorVector->at(i).sizes()[0];
        }

        tensor->Resize(outputDims);
        int64_t offset = 0;
        auto* dst = (char*)tensor->raw_mutable_data(tensorVector->at(0).dtype());

        for (const auto& t : *tensorVector) {
          context_.CopyItemsSameDevice(
              t.dtype(), t.numel(), t.raw_data(), dst + offset);
          offset += t.nbytes();
        }

        return true;
        */
    }
}

input_tags!{
    ConcatTensorVectorOp {
        TensorVector
    }
}

output_tags!{
    ConcatTensorVectorOp {
        Tensor
    }
}
