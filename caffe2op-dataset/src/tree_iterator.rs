crate::ix!();

/**
  | Provides functionality to iterate
  | across a list of tensors where some of
  | those tensors represent lengths in
  | a hierarchical structure.
  |
  */
pub struct TreeIterator {

    /// Description of each field
    fields: Vec<TreeIteratorFieldDesc>,

    /**
      | Index into fields_ above for the fields
      | that are lengths.
      |
      */
    length_field_ids: Vec<i32>,
}

impl TreeIterator {

    /**
      | Corresponds to the number of fields
      | that have "length" as its last name
      |
      */
    #[inline] pub fn num_length_fields(&self) -> i32 {
        
        todo!();
        /*
            return lengthFieldIds_.size();
        */
    }

    /**
      | Corresponds to the number of length
      | fields + 1 (for the top-level domain)
      |
      */
    #[inline] pub fn num_offset_fields(&self) -> i32 {
        
        todo!();
        /*
            return numLengthFields() + 1;
        */
    }

    /**
      | Get lengthField description for the
      | given field
      |
      */
    #[inline] pub fn length_field_for(&mut self, 
        desc: &TreeIteratorFieldDesc) -> *const TreeIteratorFieldDesc {
        
        todo!();
        /*
            return (desc.lengthFieldId == -1)
            ? nullptr
            : &fields_.at(lengthFieldIds_.at(desc.lengthFieldId));
        */
    }

    /**
      | Get lengthField description for the
      | given lengthFieldId, where 0 <= lengthFieldId
      | < numLengthFields()
      |
      */
    #[inline] pub fn length_field(&mut self, 
        length_field_id: i32) -> &TreeIteratorFieldDesc {
        
        todo!();
        /*
            return fields_.at(lengthFieldIds_.at(lengthFieldId));
        */
    }

    /**
      | Returns the index into the 'offset'
      | vector for the given field.
      |
      */
    #[inline] pub fn offset_field_id_for(&mut self, 
        field_desc: &TreeIteratorFieldDesc) -> i32 {
        
        todo!();
        /*
            return fieldDesc.lengthFieldId + 1;
        */
    }

    /**
      | Returns the field description for all
      | fields.
      |
      */
    #[inline] pub fn fields(&mut self) -> &Vec<TreeIteratorFieldDesc> {
        
        todo!();
        /*
            return fields_;
        */
    }
    
    #[inline] pub fn length_field_ids(&self) -> &Vec<i32> {
        
        todo!();
        /*
            return lengthFieldIds_;
        */
    }
    
    pub fn new(fields: &Vec<String>) -> Self {
        todo!();
        /*
            // populate field vector and split field names
      fields_.resize(fields.size());
      std::vector<std::vector<std::string>> nameParts(fields_.size());
      for (size_t i = 0; i < fields.size(); ++i) {
        auto& field = fields_.at(i);
        field.name = fields[i];
        field.id = i;
        field.lengthFieldId = -1;
        nameParts.at(i) = split(kDatasetFieldSeparator, field.name);
      }

      // populate lengthFields
      for (const auto& field : fields_) {
        const auto& parts = nameParts.at(field.id);
        if (!parts.empty() && parts.back() == kDatasetLengthField) {
          lengthFieldIds_.push_back(field.id);
        }
      }

      // find length-field with maximum prefix matching for each field
      for (auto& field : fields_) {
        // by default, we are matching against the root domain
        size_t maxMatchLevel = 1;
        int maxMatchLengthFieldId = -1;
        for (int j = 0; j < numLengthFields(); ++j) {
          const auto& lenField = lengthField(j);
          // a length field can't have itself as its length field
          if (field.id == lenField.id) {
            continue;
          }
          auto lf = nameParts.at(lenField.id);
          auto lfEnd = lf.end() - 1;
          // check whether this lengthField is a prefix for this field name
          if (std::mismatch(lf.begin(), lfEnd, nameParts.at(field.id).begin())
                  .first != lfEnd) {
            continue;
          }
          if (lf.size() > maxMatchLevel) {
            maxMatchLevel = lf.size();
            maxMatchLengthFieldId = j;
          }
        }
        field.lengthFieldId = maxMatchLengthFieldId;
      }

      // check that fields are topologically sorted
      // (no length field depends on a length defined afterwards)
      for (const auto& field : fields_) {
        const auto* lengthField = lengthFieldFor(field);
        CAFFE_ENFORCE(
            (lengthField == nullptr) || (lengthField->id < field.id),
            "Error: Field ",
            field.id,
            " (",
            field.name,
            ") ",
            "depends on a field defined afterwards: ",
            lengthField->id,
            " (",
            lengthField->name,
            ").");
      }
        */
    }
    
    #[inline] pub fn advance(
        &mut self, 
        lengths: &Vec<*const TLength>,
        offsets: &mut Vec<TOffset>,
        sizes:   &mut Vec<TOffset>,
        limits:  &mut Vec<TOffset>,
        num:     TOffset)  
    {
        todo!();
        /*
            std::vector<TOffset> newOffsets;
      CAFFE_ENFORCE_EQ(lengths.size(), numLengthFields());
      CAFFE_ENFORCE_EQ(offsets.size(), numOffsetFields());
      sizes.resize(offsets.size());
      newOffsets.resize(offsets.size());
      // first index, top level
      {
        auto limit = limits[0];
        auto offset = offsets[0];
        CAFFE_ENFORCE(limit >= offset, "Tried to advance past end of cursor.");
        TOffset total = std::min(limit - offset, num);
        sizes[0] = total;
        newOffsets[0] = offset + total;
      }
      // child indices
      for (int j = 1; j < numOffsetFields(); ++j) {
        TOffset total = 0;
        int parentOffsetId = offsetFieldIdFor(lengthField(j - 1));
        const TLength* length = lengths[j - 1] + offsets[parentOffsetId];
        for (int k = 0; k < sizes[parentOffsetId]; ++k) {
          total += *(length++);
        }
        auto offset = offsets[j];
        CAFFE_ENFORCE(
            offset + total <= limits[j],
            "Inconsistent field length: ",
            "tried to advance past the end of field ",
            j);
        sizes[j] = total;
        newOffsets[j] = offset + total;
      }
      offsets = newOffsets;
        */
    }
}
