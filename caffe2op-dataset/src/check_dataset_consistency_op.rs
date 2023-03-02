crate::ix!();

/**
  | Checks that the given data fields represents
  | a consistent dataset under the schema
  | specified by the `fields` argument.
  |
  | Operator fails if the fields are not
  | consistent. If data is consistent,
  | each field's data can be safely appended
  | to an existing dataset, keeping it consistent.
  |
  */
pub struct CheckDatasetConsistencyOp {
    storage: OperatorStorage,
    context: CPUContext,
    iterator: TreeIterator,
}

num_inputs!{CheckDatasetConsistency, (1,INT_MAX)}

num_outputs!{CheckDatasetConsistency, 0}

inputs!{CheckDatasetConsistency, 
    0 => ("field_0", "Data for field 0.")
}

args!{CheckDatasetConsistency, 
    0 => ("fields", "List of strings representing the string names in the format specified in the doc for CreateTreeCursor.")
}

impl CheckDatasetConsistencyOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            iterator_(OperatorStorage::GetRepeatedArgument<std::string>("fields"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            std::vector<const TLength*> lengths;
        std::vector<TOffset> limits;
        std::vector<TOffset> sizes;
        std::vector<TOffset> offsets;
        CAFFE_ENFORCE(
            InputSize() == iterator_.fields().size(),
            "Invalid number of fields. Expected ",
            iterator_.fields().size(),
            ", got ",
            InputSize());
        sizes.resize(iterator_.numOffsetFields());
        // gather length data
        lengths.resize(iterator_.numLengthFields());
        for (size_t i = 0; i < lengths.size(); ++i) {
          lengths[i] = Input(iterator_.lengthField(i).id).data<TLength>();
        }
        // gather size limits
        limits.assign(sizes.size(), TOffset::max);
        for (size_t i = 0; i < iterator_.fields().size(); ++i) {
          int lengthIdx = iterator_.fields()[i].lengthFieldId + 1;
          CAFFE_ENFORCE_GT(Input(i).dim(), 0);
          TOffset size = (TOffset)Input(i).sizes()[0];
          if (limits[lengthIdx] == TOffset::max) {
            limits[lengthIdx] = size;
          } else {
            CAFFE_ENFORCE(
                limits[lengthIdx] == size,
                "Inconsistent sizes for fields belonging to same domain.",
                " Field: ",
                i,
                " (",
                iterator_.fields()[i].name,
                "); Length field index: ",
                lengthIdx,
                "); Previous size: ",
                limits[lengthIdx],
                "; New size: ",
                size);
          }
        }
        // advance to the end
        offsets.assign(sizes.size(), 0);
        iterator_.advance(lengths, offsets, sizes, limits, limits[0]);
        for (size_t i = 0; i < limits.size(); ++i) {
          CAFFE_ENFORCE(limits[i] == offsets[i]);
        }
        return true;
        */
    }
}
