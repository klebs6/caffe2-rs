crate::ix!();

/**
  | Given a dataset under a schema specified
  | by the `fields` argument, pack all the
  | input tensors into one, where each tensor
  | element represents a row of data (batch
  | of size 1). This format allows easier
  | use with the rest of Caffe2 operators.
  |
  */
pub struct PackRecordsOp {
    storage:                   OperatorStorage,
    context:                   CPUContext,
    fields:                    Vec<String>,
    pack_to_single_shared_ptr: bool,
}

num_inputs!{PackRecords, (1,INT_MAX)}

num_outputs!{PackRecords, 1}

outputs!{PackRecords, 
    0 => ("tensor", "One dimensional tensor having a complex type of SharedTensorVectorPtr. In order to reverse it back to the original input it has to be inserted into UnPackRecordsOp.")
}

args!{PackRecords, 
    0 => ("fields", "List of strings representing the string names in the format specified in the doc for CreateTreeCursor.")
}

impl PackRecordsOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            fields_(OperatorStorage::GetRepeatedArgument<std::string>("fields")),
            packToSingleSharedPtr_(OperatorStorage::GetSingleArgument<int>(
                "pack_to_single_shared_ptr",
                0))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // There should be one input per field
        CAFFE_ENFORCE_EQ(InputSize(), fields_.size());
        CAFFE_ENFORCE_EQ(OutputSize(), 1);

        TreeCursor cursor((TreeIterator(fields_)));

        TreeWalker walker(Inputs(), cursor);

        if (packToSingleSharedPtr_) {
          Output(0)->Resize(1);
          auto* dst = Output(0)->template mutable_data<Shared2DTensorVectorPtr>();
          dst[0] = std::make_shared<Tensor2DVector>();
          dst[0]->resize(walker.size());

          for (int batchId = 0; batchId < walker.size(); ++batchId) {
            std::vector<TensorCPU>& tensors = dst[0]->at(batchId);
            tensors.reserve(walker.fields().size());
            for (const auto& field : walker.fields()) {
              tensors.emplace_back(field.dim(), CPU);
              auto& tensor = tensors.back();
              context_.CopyItemsSameDevice(
                  field.meta(),
                  tensor.numel(),
                  field.ptr() /* src */,
                  tensor.raw_mutable_data(field.meta()) /* dst */);
            }
            walker.advance();
          }
        } else {
          Output(0)->Resize(walker.size());
          auto* dst = Output(0)->template mutable_data<SharedTensorVectorPtr>();

          for (int batchId = 0; batchId < walker.size(); ++batchId) {
            dst[batchId] = std::make_shared<std::vector<TensorCPU>>();
            dst[batchId]->reserve(walker.fields().size());
            for (const auto& field : walker.fields()) {
              dst[batchId]->emplace_back(field.dim(), CPU);
              auto& tensor = dst[batchId]->back();
              context_.CopyItemsSameDevice(
                  field.meta(),
                  tensor.numel(),
                  field.ptr() /* src */,
                  tensor.raw_mutable_data(field.meta()) /* dst */);
            }
            walker.advance();
          }
        }

        return true;
        */
    }
}

