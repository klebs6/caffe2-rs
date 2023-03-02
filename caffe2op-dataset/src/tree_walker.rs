crate::ix!();

/**
  | Simple wrapper class allowing an easy
  | traversal of the tensors representing
  | the hirerarchical structure.
  |
  */
pub struct TreeWalker<'a> {
    inputs:       &'a Vec<*const Blob>,
    cursor:       &'a mut TreeCursor,
    fields:       Vec<TreeWalkerField<'a>>,
    lengths:      Vec<*const TLength>,
    limits:       Vec<TOffset>,
    sizes:        Vec<TOffset>,
    offsets:      Vec<TOffset>,
    prev_offsets: Vec<TOffset>,
}

impl<'a> TreeWalker<'a> {

    /// Returns the number of records in a dataset
    #[inline] pub fn size(&self) -> TOffset {
        
        todo!();
        /*
            return limits_.at(0);
        */
    }
    
    #[inline] pub fn input(&self, idx: i32) -> &TensorCPU {
        
        todo!();
        /*
            return inputs_[idx]->Get<TensorCPU>();
        */
    }

    // TODO: Change to fieldDesc
    #[inline] pub fn field(&self, idx: i32) -> &TreeIteratorFieldDesc {
        
        todo!();
        /*
            return cursor_.it.fields().at(idx);
        */
    }
    
    #[inline] pub fn length_idx(&self, field_id: i32) -> i32 {
        
        todo!();
        /*
            return field(fieldId).lengthFieldId + 1;
        */
    }
    
    #[inline] pub fn offset(&self, field_id: i32) -> TOffset {
        
        todo!();
        /*
            return prevOffsets_[lengthIdx(fieldId)];
        */
    }

    /**
      | Notice that a reference is returned.
      | If advance() is called the fields will
      | be updated to represent the new state.
      |
      */
    #[inline] pub fn fields(&self) -> &Vec<TreeWalkerField> {
        
        todo!();
        /*
            return fields_;
        */
    }
    
    pub fn new(inputs: &Vec<*const Blob>, cursor: &mut TreeCursor) -> Self {
        todo!();
        /*
            : inputs_(inputs), cursor_(cursor), sizes_(cursor.it.numOffsetFields()) 

      CAFFE_ENFORCE_EQ(inputs.size(), cursor.it.fields().size());
      if (cursor.offsets.empty()) {
        cursor.offsets.assign(cursor.it.numOffsetFields(), 0);
      }

      for (int fieldId = 0; fieldId < cursor_.it.fields().size(); ++fieldId) {
        fields_.emplace_back(*this, fieldId);
      }

      gatherLengthData();

      gatherSizeLimits();

      // The invariant we hold is that we are always one step ahead
      advance();
        */
    }
    
    #[inline] pub fn advance(&mut self)  {
        
        todo!();
        /*
            prevOffsets_ = cursor_.offsets;
      cursor_.it.advance(lengths_, cursor_.offsets, sizes_, limits_, 1);
        */
    }
    
    #[inline] pub fn field_dim(&self, field_id: i32) -> Vec<i64> {
        
        todo!();
        /*
            auto tensorDim = input(fieldId).sizes().vec();
      tensorDim[0] = sizes_[lengthIdx(fieldId)];
      return tensorDim;
        */
    }
    
    #[inline] pub fn field_ptr(&self, field_id: i32) -> *mut c_void {
        
        todo!();
        /*
            auto& in = input(fieldId);
      return (char*)in.raw_data() +
          offset(fieldId) * in.size_from_dim(1) * in.dtype().itemsize();
        */
    }
    
    #[inline] pub fn gather_length_data(&mut self)  {
        
        todo!();
        /*
            static const TLength lenZero = 0;
      lengths_.resize(cursor_.it.numLengthFields());
      for (int i = 0; i < lengths_.size(); ++i) {
        auto& in = input(cursor_.it.lengthField(i).id);
        if (in.numel() > 0) {
          lengths_[i] = in.data<int>();
        } else {
          lengths_[i] = &lenZero;
        }
      }
        */
    }
    
    #[inline] pub fn gather_size_limits(&mut self)  {
        
        todo!();
        /*
            limits_.assign(sizes_.size(), TOffset::max);
      for (auto fieldId = 0; fieldId < cursor_.it.fields().size(); ++fieldId) {
        auto lengthFieldIdx = lengthIdx(fieldId);
        limits_[lengthFieldIdx] =
            std::min(limits_[lengthFieldIdx], (TOffset)input(fieldId).sizes()[0]);
      }
        */
    }
}

