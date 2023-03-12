crate::ix!();

/**
  | Trim the given dataset inplace, given
  | the dataset blobs and the field specs.
  | 
  | Trimming happens such that the dataset
  | will contain the largest possible number
  | of records that is a multiple of the 'multiple_of'
  | argument.
  |
  */
pub struct TrimDatasetOp {
    storage:     OperatorStorage,
    context:     CPUContext,
    iterator:    TreeIterator,
    multiple_of: i32,
}

num_inputs!{TrimDataset, (1,INT_MAX)}

num_outputs!{TrimDataset, (1,INT_MAX)}

args!{TrimDataset, 
    0 => ("fields", "List of strings representing the string names in the format specified in the doc for CreateTreeCursor.")
}

enforce_inplace!{TrimDataset, /*[](int input, int output) { return input == output; }*/}

impl TrimDatasetOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            iterator_(OperatorStorage::GetRepeatedArgument<std::string>("fields")),
            multiple_of_(OperatorStorage::GetSingleArgument<int>("multiple_of", 1)) 

        CAFFE_ENFORCE_GE(multiple_of_, 1);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            TreeCursor cursor(iterator_);
        TreeWalker walker(Inputs(), cursor);

        int trimmedSize = (walker.size() / multiple_of_) * multiple_of_;
        if (trimmedSize == walker.size()) {
          // we already satisfy the condition
          return true;
        }
        // advance desired number of records
        for (int i = 0; i < trimmedSize; ++i) {
          walker.advance();
        }
        // trim each column to the offset
        for (int col = 0; col < walker.fields().size(); ++col) {
          auto newOuterSize = walker.fields().at(col).offset();
          Output(col)->ShrinkTo(newOuterSize);
        }
        return true;
        */
    }
}
