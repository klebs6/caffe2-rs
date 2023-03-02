crate::ix!();

/**
  | The *FindDuplicateElements* op takes
  | a single 1-D tensor *data* as input and
  | returns a single 1-D output tensor *indices*.
  | The output tensor contains the indices
  | of the duplicate elements of the input,
  | excluding the first occurrences. If
  | all elements of *data* are unique, *indices*
  | will be empty.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/find_duplicate_elements_op.h
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/find_duplicate_elements_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_DISPATCH_HELPER]
pub struct FindDuplicateElementsOp<Context> {

    storage: OperatorStorage,
    context: Context,
}

num_inputs!{FindDuplicateElements, 1}

num_outputs!{FindDuplicateElements, 1}

inputs!{FindDuplicateElements, 
    0 => ("data", "a 1-D tensor.")
}

outputs!{FindDuplicateElements, 
    0 => ("indices", "Indices of duplicate elements in data, excluding first occurrences.")
}

impl<Context> FindDuplicateElementsOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double, int, long, std::string>>::
            call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& data = Input(0);
            CAFFE_ENFORCE(data.dim() == 1, "data should be 1-D.");

            const auto* data_ptr = data.template data<T>();
            std::unordered_map<T, int64_t> dict;
            std::vector<int64_t> dupIndices;
            // i is the index of unique elements, j is the index of all elements
            for (int64_t i = 0, j = 0; j < data.sizes()[0]; ++i, ++j) {
              bool retVal = dict.insert({data_ptr[j], i}).second;
              if (!retVal) {
                --i;
                dupIndices.push_back(j);
              }
            }

            const auto dupSize = dupIndices.size();

            auto* output =
                Output(0, {static_cast<int64_t>(dupSize)}, at::dtype<int64_t>());
            auto* out_ptr = output->template mutable_data<int64_t>();
            for (size_t i = 0; i < dupSize; ++i) {
              out_ptr[i] = dupIndices[i];
            }

            return true;
        */
    }
}

register_cpu_operator!{
    FindDuplicateElements,
    FindDuplicateElementsOp<CPUContext>
}

#[test] fn find_duplicate_elements_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "FindDuplicateElements",
        ["data"],
        ["indices"],
    )

    workspace.FeedBlob("data", np.array([8,2,1,1,7,8,1]).astype(np.float32))
    print("data:\n", workspace.FetchBlob("data"))

    workspace.RunOperatorOnce(op)
    print("indices: \n", workspace.FetchBlob("indices"))

    data:
     [8. 2. 1. 1. 7. 8. 1.]
    indices:
     [3 5 6]
    */
}

should_not_do_gradient!{FindDuplicateElements}
