crate::ix!();

/**
  | Transfers a vector of segment ids to
  | a vector of segment lengths.
  | 
  | This operation supports non-consecutive
  | segment ids.
  | 
  | Segments not appearing in the input
  | vector will have length 0.
  | 
  | If the second input is provided, the
  | number of segments = the size of its first
  | dimension.
  | 
  | Otherwise, the number of segments =
  | the last index in the first input vector
  | + 1.
  | 
  | In general, for consecutive, zero-based
  | segment IDs, this is the inverse operation
  | of LengthsToSegmentIds, except that
  | a vector of segment IDs cannot represent
  | empty segments at the end (if the second
  | input is absent).
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SegmentIdsToLengthsOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{SegmentIdsToLengths, (1,2)}

num_outputs!{SegmentIdsToLengths, 1}

//todo, enable the filler
disallow_input_fillers!{SegmentIdsToLengths}

inputs!{SegmentIdsToLengths, 
    0 => ("segment_ids", "1-D int32_t or int64_t tensor of segment ids"),
    1 => ("data (optional)", "if provided, number of segments = the size of its first dimension")
}

outputs!{SegmentIdsToLengths, 
    0 => ("lengths", "1-D int64_t tensor of segment lengths")
}

impl<Context> SegmentIdsToLengthsOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
        todo!();
        /*
            auto& input = Input(0);
        if (input.dim() == 2) {
          CAFFE_ENFORCE(
              input.dim32(0) == 1 || input.dim32(1) == 1,
              "Input must be a vector.");
        } else {
          CAFFE_ENFORCE_EQ(input.dim(), 1, "Input must be a vector.");
        }
        auto* input_data = input.template data<Index>();
        auto input_size = input.numel();
        auto* output = Output(0);
        // segment id starts from 0
        auto num_segments = input_size ? input_data[input_size - 1] + 1 : 0;
        if (InputSize() > 1) {
          CAFFE_ENFORCE_GE(Input(1).dim(), 1);
          CAFFE_ENFORCE_LE(
              num_segments,
              Input(1).size(0),
              "The number of segments inferred should *NOT* be larger "
              "than the size of Input(1)'s first dimension");
          num_segments = Input(1).size(0);
        }
        CAFFE_ENFORCE(0 <= num_segments, "Indices must be in 0..K-1 range");
        output->Resize(num_segments);
        auto* output_data = output->template mutable_data<int32_t>();
        if (num_segments == 0) {
          return true;
        }
        std::fill(output_data, output_data + num_segments, 0);
        Index prev = 0; // Assume that segment_id >= 0.
        for (int64_t i = 0; i < input_size; i++) {
          CAFFE_ENFORCE(
              prev <= input_data[i],
              "Segment ids must be sorted: ",
              prev,
              " vs ",
              input_data[i]);
          prev = input_data[i];
          output_data[input_data[i]] += 1;
        }

        return true;
        */
    }
}
