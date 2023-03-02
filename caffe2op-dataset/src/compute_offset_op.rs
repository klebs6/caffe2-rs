crate::ix!();

/**
  | Compute the offsets matrix given cursor
  | and data blobs. Need to be ran at beginning
  | or after reseting cursor
  | 
  | Input(0) is a blob pointing to a TreeCursor,
  | and [Input(1),... Input(num_fields)]
  | a list of tensors containing the data
  | for each field of the dataset.
  | 
  | ComputeOffset is thread safe.
  |
  */
pub struct ComputeOffsetOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{ComputeOffset, (1,INT_MAX)}

num_outputs!{ComputeOffset, 1}

inputs!{ComputeOffset, 
    0 => ("cursor", "A blob containing a pointer to the cursor."),
    1 => ("dataset_field_0", "First dataset field")
}

outputs!{ComputeOffset, 
    0 => ("field_0", "Tensor containing offset info for this chunk.")
}

impl ComputeOffsetOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& cursor = OperatorStorage::Input<std::unique_ptr<TreeCursor>>(0);
        CAFFE_ENFORCE(InputSize() == cursor->it.fields().size() + 1);
        auto* out = Output(0);
        std::vector<const TLength*> lengths;
        std::vector<TOffset> limits;
        std::vector<TOffset> sizes;
        std::vector<TOffset> offsets;
        TLength lenZero = 0;
        sizes.resize(cursor->it.numOffsetFields());
        // gather length data
        lengths.resize(cursor->it.numLengthFields());
        for (int i = 0; i < lengths.size(); ++i) {
          auto& a = Input(cursor->it.lengthField(i).id + 1);
          if (a.numel() > 0) {
            lengths[i] = a.data<int>();
          } else {
            lengths[i] = &lenZero;
          }
        }
        // gather size limits
        limits.assign(sizes.size(), TOffset::max);
        for (int i = 0; i < cursor->it.fields().size(); ++i) {
          int lengthFieldIdx = cursor->it.fields()[i].lengthFieldId + 1;
          limits[lengthFieldIdx] =
              std::min(limits[lengthFieldIdx], (TOffset)Input(i + 1).sizes()[0]);
        }
        out->Resize(limits.at(0) + 1, sizes.size());
        auto* out_data = out->template mutable_data<int64_t>();
        for (int k = 0; k <= limits.at(0); k++) {
          // advance cursor
          if (cursor->offsets.empty()) {
            cursor->offsets.assign(sizes.size(), 0);
          }
          // write output
          std::copy(cursor->offsets.begin(), cursor->offsets.end(), out_data);
          out_data += sizes.size();
          cursor->it.advance(lengths, cursor->offsets, sizes, limits, 1);
        }
        cursor->offsets.assign(sizes.size(), 0); // reSet after getting meta info
        return true;
        */
    }
}
