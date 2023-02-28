crate::ix!();

/// used for lengths tensors in the dataset
pub type TLength = i32;

/**
  | used for all internal dataset operations
  | (offsets, sizes to read, etc.)
  |
  */
pub type TOffset = i64;

///--------------------------------------------------
pub struct TreeIteratorFieldDesc {
    id:              i32,
    length_field_id: i32, // = -1;
    name:            String,
}

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

///---------------------------------------
pub struct TreeCursor {
    offsets: Vec<TOffset>,
    mutex:   parking_lot::RawMutex,
    it:      TreeIterator,
}

impl TreeCursor {
    
    pub fn new(iterator: &TreeIterator) -> Self {
        todo!();
        /*
            : it(iterator)
        */
    }
}

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
}

/**
  | Simple Proxy class to expose nicer API
  | for field access
  |
  */
pub struct TreeWalkerField<'a> {
    walker:   &'a TreeWalker<'a>,
    field_id: i32,
}

impl<'a> TreeWalkerField<'a> {
    
    pub fn new(walker: &mut TreeWalker, field_id: i32) -> Self {
        todo!();
        /*
            : walker_(walker), fieldId_(fieldId)
        */
    }
    
    #[inline] pub fn dim(&self) -> Vec<i64> {
        
        todo!();
        /*
            return walker_.fieldDim(fieldId_);
        */
    }
    
    #[inline] pub fn size(&self) -> i64 {
        
        todo!();
        /*
            int64_t size = 1;
          for (const auto d : dim()) {
            size *= d;
          }
          return size;
        */
    }
    
    #[inline] pub fn meta(&self) -> TypeMeta {
        
        todo!();
        /*
            return walker_.input(fieldId_).dtype();
        */
    }
    
    #[inline] pub fn ptr(&self)  {
        
        todo!();
        /*
            return walker_.fieldPtr(fieldId_);
        */
    }
    
    #[inline] pub fn field_id(&self) -> i32 {
        
        todo!();
        /*
            return fieldId_;
        */
    }
    
    #[inline] pub fn offset(&self) -> TOffset {
        
        todo!();
        /*
            return walker_.offset(fieldId_);
        */
    }
}

impl<'a> TreeWalker<'a> {
    
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

pub type SharedTensorVectorPtr   = Arc<Vec<TensorCPU>>;
pub type Shared2DTensorVectorPtr = Arc<Vec<Vec<TensorCPU>>>;
pub type Tensor2DVector          = Vec<Vec<TensorCPU>>;
pub type TensorVectorPtr         = Box<Vec<Tensor>>;

///---------------------------------------
pub struct SharedTensorVectorPtrSerializer {
    base: dyn BlobSerializerBase,
}

impl SharedTensorVectorPtrSerializer {
    
    #[inline] pub fn serialize(
        &mut self, 
        pointer:   *const c_void,
        type_meta: TypeMeta,
        name:      &String,
        acceptor:  SerializationAcceptor)  
    {
        todo!();
        /*
          /* This is dummy serialize that doesn't save anything. If saving the content
          is desired in future use case, you can change this serializer. Note: special
          care need to be taken for the parameter initialization of
          LastNWindowCollectorOp and ReservoirSamplingOp if this serializer actually
          saves the content.
          */
          CAFFE_ENFORCE(typeMeta.Match<std::shared_ptr<std::vector<TensorCPU>>>());
          BlobProto blob_proto;
          blob_proto.set_name(name);
          blob_proto.set_type("std::shared_ptr<std::vector<TensorCPU>>");
          blob_proto.set_content("");
          acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
        */
    }
}

///---------------------------------------
pub struct SharedTensorVectorPtrDeserializer {
    base: dyn BlobDeserializerBase,
}

impl SharedTensorVectorPtrDeserializer {
    
    #[inline] pub fn deserialize(
        &mut self, 
        unused: &BlobProto,
        blob: *mut Blob)  
    {
        todo!();
        /*
            /* This is dummy deserialize which creates a nullptr
       */
      blob->GetMutable<std::shared_ptr<std::vector<TensorCPU>>>();
        */
    }
}

caffe_known_type!{Box<TreeCursor>}
caffe_known_type!{TensorVectorPtr}
caffe_known_type!{SharedTensorVectorPtr}
caffe_known_type!{Shared2DTensorVectorPtr}

pub const kDatasetFieldSeparator: &'static str = ":";
pub const kDatasetLengthField:    &'static str = "lengths";

/// how much percent to grow the dataset when needed
pub const kDatasetGrowthPct: i32 = 40;

/**
 | Creates a cursor to iterate through a list of
 | tensors, where some of those tensors contain the
 | lengths in a nested schema. The schema is
 | determined by the `fields` arguments.
 |
 | For example, to represent the following schema:
 |
 |   Struct(
 |       a=Int(),
 |       b=List(List(Int)),
 |       c=List(
 |           Struct(
 |              c1=String,
 |              c2=List(Int),
 |           ),
 |       ),
 |   )
 |
 | the field list will be:
 |   [
 |       "a",
 |       "b:lengths",
 |       "b:values:lengths",
 |       "b:values:values",
 |       "c:lengths",
 |       "c:c1",
 |       "c:c2:lengths",
 |       "c:c2:values",
 |   ]
 |
 | And for the following instance of the struct:
 |
 |   Struct(
 |       a=3,
 |       b=[[4, 5], [6, 7, 8], [], [9]],
 |       c=[
 |           Struct(c1='alex', c2=[10, 11]),
 |           Struct(c1='bob', c2=[12]),
 |       ],
 |   )
 |
 | The values of the fields will be:
 |   {
 |       "a": [3],
 |       "b:lengths": [4],
 |       "b:values:lengths": [2, 3, 0, 1],
 |       "b:values:values": [4, 5, 6, 7, 8, 9],
 |       "c:lengths": [2],
 |       "c:c1": ["alex", "bob"],
 |       "c:c2:lengths": [2, 1],
 |       "c:c2:values", [10, 11, 12],
 |   }
 |
 | In general, every field name in the format
 | "{prefix}:lengths" defines a domain "{prefix}",
 | and every subsequent field in the format
 | "{prefix}:{field}" will be in that domain, and the
 | length of the domain is provided for each entry of
 | the parent domain. In the example, "b:lengths"
 | defines a domain of length 4, so every field under
 | domain "b" will have 4 entries. The "lengths"
 | field for a given domain must appear before any
 | reference to that domain.
 |
 | Returns a pointer to an instance of the Cursor,
 | which keeps the current offset on each of the
 | domains defined by `fields`. Cursor also ensures
 | thread-safety such that ReadNextBatch and
 | ResetCursor can be used safely in parallel.
 |
 | A cursor does not contain data per se, so calls to
 | ReadNextBatch actually need to pass a list of
 | blobs containing the data to read for each one of
 | the fields.
 |
 */
pub struct CreateTreeCursorOp {
    storage: OperatorStorage,
    context: CPUContext,
    fields:  Vec<String>,
}

num_inputs!{CreateTreeCursor, 0}

num_outputs!{CreateTreeCursor, 1}

outputs!{CreateTreeCursor, 
    0 => ("cursor", "A blob pointing to an instance of a new TreeCursor.")
}

args!{CreateTreeCursor, 
    0 => ("fields", "A list of strings each one representing a field of the dataset.")
}

impl CreateTreeCursorOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            fields_(OperatorStorage::GetRepeatedArgument<std::string>("fields"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            *OperatorStorage::Output<std::unique_ptr<TreeCursor>>(0) =
            std::unique_ptr<TreeCursor>(new TreeCursor(TreeIterator(fields_)));
        return true;
        */
    }
}

/**
  | Get the current offset in the cursor.
  |
  */
pub struct GetCursorOffsetOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{GetCursorOffset, 1}

num_outputs!{GetCursorOffset, 1}

inputs!{GetCursorOffset, 
    0 => ("cursor", "A blob containing a pointer to the cursor.")
}

outputs!{GetCursorOffset, 
    0 => ("offsets", "Tensor containing the offsets for the cursor.")
}

impl GetCursorOffsetOp {
    
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
        Output(0)->Resize(cursor->offsets.size());
        auto* output = Output(0)->template mutable_data<int>();
        for (size_t i = 0; i < cursor->offsets.size(); ++i) {
          output[i] = cursor->offsets[i];
        }
        return true;
        */
    }
}

/**
  | Resets the offsets for the given TreeCursor.
  | This operation is thread safe.
  |
  */
pub struct ResetCursorOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{ResetCursor, 1}

num_outputs!{ResetCursor, 0}

inputs!{ResetCursor, 
    0 => ("cursor", "A blob containing a pointer to the cursor.")
}

impl ResetCursorOp {
    
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
        std::lock_guard<std::mutex> lock(cursor->mutex_);
        cursor->offsets.clear();
        return true;
        */
    }
}

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

/**
  | Given a packed dataset (packed by the
  | PackRecordsOp) and the `fields` argument
  | describing the datasets schema, return
  | the original dataset format. Number
  | of returned tensors is equal to the number
  | of fields in the `fields` argument.
  | 
  | The first input is the packed tensor
  | to be unpacked. Optionally, you can
  | provide prototype tensors to give the
  | expected shapes of the output tensors.
  | This is helpful when you expected to
  | unpack empty tensor, e.g., output of
  | a sampling process.
  |
  */
pub struct UnPackRecordsOp {
    storage: OperatorStorage,
    context: CPUContext,
    fields:  Vec<String>,
}

num_inputs!{UnPackRecords, (1,INT_MAX)}

num_outputs!{UnPackRecords, (1,INT_MAX)}

inputs!{UnPackRecords, 
    0 => ("packed_tensor", "The tensor to be unpacked")
}

args!{UnPackRecords, 
    0 => ("fields", "List of strings representing the string names in the format specified in the doc for CreateTreeCursor.")
}

impl UnPackRecordsOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            fields_(OperatorStorage::GetRepeatedArgument<std::string>("fields"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            size_t numRows = 0;
        Shared2DTensorVectorPtr data_ptr = nullptr;
        if (Input(0).IsType<SharedTensorVectorPtr>()) {
          numRows = Input(0).numel();
          CAFFE_ENFORCE_GE(numRows, 0);
          data_ptr = std::make_shared<Tensor2DVector>();
          data_ptr->reserve(numRows);

          const auto* inputs = Input(0).template data<SharedTensorVectorPtr>();
          for (int i = 0; i < numRows; i++) {
            data_ptr->emplace_back(*inputs[i]);
          }
        } else if (Input(0).IsType<Shared2DTensorVectorPtr>()) {
          CAFFE_ENFORCE_EQ(Input(0).numel(), 1);
          const auto* inputs = Input(0).template data<Shared2DTensorVectorPtr>();
          CAFFE_ENFORCE(inputs[0] != nullptr);
          data_ptr = inputs[0];
          numRows = inputs[0]->size();
          CAFFE_ENFORCE_GE(numRows, 0);
        } else {
          // input contains a single tensor
          CAFFE_ENFORCE_EQ(InputSize(), 1);
          CAFFE_ENFORCE_EQ(OutputSize(), 1);
          Output(0)->CopyFrom(Input(0));
          return true;
        }

        auto numTensors = OutputSize();

        // Precomputer the output sizes to avoid resizing
        std::vector<std::vector<int64_t>> outputDims(numTensors);
        std::vector<TypeMeta> metas(numTensors);

        CAFFE_ENFORCE(
            numRows > 0 || InputSize() > 1,
            "Unpacking empty record without shape will leave output blobs in "
            "undefined state.");

        if (InputSize() == 1) {
          getShapeAndMetaFromInput(data_ptr, outputDims, metas);
        } else {
          getShapeAndMetaFromPrototypeBlobs(outputDims, metas);
        }

        // inputs contains a single shared_ptr of vector<vector<caffe2::TensorCPU>>
        auto& tensors = *data_ptr;
        for (int i = 0; i < numRows; ++i) {
          for (int j = 0; j < tensors[i].size(); ++j) {
            const auto& input = tensors[i][j];

            // Checks to ensure that dimensions/sizes match
            CAFFE_ENFORCE_EQ(outputDims[j].size(), input.dim());
            CAFFE_ENFORCE(metas[j] == input.dtype());
            // We look from first dimension, because we concat on the first.
            for (int k = 1; k < input.dim(); ++k) {
              CAFFE_ENFORCE_EQ(input.sizes()[k], outputDims[j][k]);
            }

            outputDims[j][0] += input.size(0);
          }
        }

        // Resize to the final output size
        std::vector<void*> destinations(numTensors);
        for (int i = 0; i < numTensors; ++i) {
          Output(i)->Resize(outputDims[i]);
          destinations[i] = Output(i)->raw_mutable_data(metas[i]);
        }

        for (int i = 0; i < numRows; ++i) {
          for (int j = 0; j < numTensors; ++j) {
            const auto& input = tensors[i][j];

            context_.CopyItemsSameDevice(
                metas[j],
                input.numel(),
                input.raw_data() /* src */,
                destinations[j] /* dst */
            );

            destinations[j] =
                (char*)destinations[j] + input.numel() * input.itemsize();
          }
        }

        return true;
        */
    }
    
    #[inline] pub fn get_shape_and_meta_from_input(
        &mut self, 
        inputs:      &Shared2DTensorVectorPtr,
        output_dims: &mut Vec<Vec<i64>>,
        metas:       &mut Vec<TypeMeta>)
    {
        todo!();
        /*
            const auto& inputZero = inputs->at(0);

        const auto numTensors = inputZero.size();

        CAFFE_ENFORCE_EQ(numTensors, fields_.size());
        CAFFE_ENFORCE_EQ(numTensors, OutputSize());

        for (int i = 0; i < numTensors; ++i) {
          outputDims[i] = inputZero[i].sizes().vec();
          outputDims[i][0] = 0;
          metas[i] = inputZero[i].dtype();
        }
        */
    }
    
    #[inline] pub fn get_shape_and_meta_from_prototype_blobs(
        &mut self, 
        output_dims: &mut Vec<Vec<i64>>,
        metas:       &mut Vec<TypeMeta>)  
    {
        
        todo!();
        /*
            const auto numTensors = fields_.size();
        CAFFE_ENFORCE_EQ(numTensors, InputSize() - 1);
        CAFFE_ENFORCE_EQ(numTensors, OutputSize());
        for (int i = 0; i < numTensors; ++i) {
          const auto& input = Input(i + 1);
          outputDims[i] = input.sizes().vec();
          outputDims[i][0] = 0;
          metas[i] = input.dtype();
        }
        */
    }
}

/**
  | Read the next batch of examples out of
  | the given cursor and data blobs.
  | 
  | Input(0) is a blob pointing to a TreeCursor,
  | and [Input(1),... Input(num_fields)]
  | a list of tensors containing the data
  | for each field of the dataset.
  | 
  | ReadNextBatch is thread safe.
  |
  */
pub struct ReadNextBatchOp {
    storage:            OperatorStorage,
    context:            CPUContext,
    batch_size:         i32,
    enforce_batch_size: bool,
}

num_inputs!{ReadNextBatch, (1,INT_MAX)}

num_outputs!{ReadNextBatch, (1,INT_MAX)}

inputs!{ReadNextBatch, 
    0 => ("cursor", "A blob containing a pointer to the cursor."),
    1 => ("dataset_field_0", "First dataset field")
}

outputs!{ReadNextBatch, 
    0 => ("field_0", "Tensor containing the next batch for field 0.")
}

args!{ReadNextBatch, 
    0 => ("batch_size", "Number of top-level entries to read.")
}

impl ReadNextBatchOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            batchSize_(OperatorStorage::GetSingleArgument<int>("batch_size", 1)),
            enforceBatchSize_(OperatorStorage::GetSingleArgument<bool>(
                "enforce_batch_size",
                false))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& cursor = OperatorStorage::Input<std::unique_ptr<TreeCursor>>(0);
        CAFFE_ENFORCE(InputSize() == cursor->it.fields().size() + 1);
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
        // advance cursor
        {
          std::lock_guard<std::mutex> lock(cursor->mutex_);
          if (cursor->offsets.empty()) {
            cursor->offsets.assign(sizes.size(), 0);
          }
          offsets = cursor->offsets;
          cursor->it.advance(lengths, cursor->offsets, sizes, limits, batchSize_);
          if (enforceBatchSize_ && sizes[0] < batchSize_) {
            // if we enforce batch_size but don't have enough rows left to
            // complete a full batch, return empty for all columns.
            // This signals end of dataset to the caller.
            sizes.assign(sizes.size(), 0);
          }
        }
        // gather data
        std::vector<int64_t> outDim;
        for (int i = 0; i < cursor->it.fields().size(); ++i) {
          auto lengthIdx = cursor->it.fields()[i].lengthFieldId + 1;
          auto size = sizes[lengthIdx];
          auto offset = offsets[lengthIdx];
          auto& in = Input(i + 1);
          auto innerSize = in.size_from_dim(1);
          outDim = in.sizes().vec();
          outDim[0] = size;
          auto* out = Output(i);
          out->Resize(outDim);
          void* src =
              (char*)in.raw_data() + offset * innerSize * in.dtype().itemsize();
          void* dst = out->raw_mutable_data(in.dtype()); // create the tensor
          if (out->numel() == 0) {
            continue;
          }
          context_.CopyItemsSameDevice(in.dtype(), out->numel(), src, dst);
        }
        return true;
        */
    }
}

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

/**
 | Compute the sorted indices given a field index to
 | sort by and break the sorted indices into chunks
 | of shuffle_size * batch_size and shuffle each
 | chunk, finally we shuffle between batches. If
 | sort_by_field_idx is -1 we skip sort.
 |
 | For example, we have data sorted as
 | 1,2,3,4,5,6,7,8,9,10,11,12
 |
 | and batchSize = 2 and shuffleSize = 3, when we
 | shuffle we get:
 | [3,1,4,6,5,2] [12,10,11,8,9,7]
 |
 | After this we will shuffle among different batches
 | with size 2
 | [3,1],[4,6],[5,2],[12,10],[11,8],[9,7]
 |
 | We may end up with something like
 | [9,7],[5,2],[12,10],[4,6],[3,1],[11,8]
 |
 | Input(0) is a blob pointing to a TreeCursor, and
 | [Input(1),... Input(num_fields)] a list of tensors
 | containing the data for each field of the dataset.
 |
 | SortAndShuffle is thread safe.
 */
pub struct SortAndShuffleOp {
    storage: OperatorStorage,
    context: CPUContext,
    sort_by_field_idx: i32,
    batch_size:        i32,
    shuffle_size:      i32,
}

num_inputs!{SortAndShuffle, (1,INT_MAX)}

num_outputs!{SortAndShuffle, 1}

inputs!{SortAndShuffle, 
    0 => ("cursor", "A blob containing a pointer to the cursor."),
    1 => ("dataset_field_0", "First dataset field")
}

outputs!{SortAndShuffle, 
    0 => ("indices", "Tensor containing sorted indices.")
}

impl SortAndShuffleOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            sort_by_field_idx_(
                OperatorStorage::GetSingleArgument<int>("sort_by_field_idx", 1)),
            batch_size_(OperatorStorage::GetSingleArgument<int>("batch_size", 1)),
            shuffle_size_(OperatorStorage::GetSingleArgument<int>("shuffle_size", 1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& cursor = OperatorStorage::Input<std::unique_ptr<TreeCursor>>(0);
        CAFFE_ENFORCE(InputSize() == cursor->it.fields().size() + 1);
        CAFFE_ENFORCE(-1 <= sort_by_field_idx_);
        CAFFE_ENFORCE(cursor->it.fields().size() - sort_by_field_idx_ > 0);
        int size;
        if (sort_by_field_idx_ != -1) {
          size = Input(sort_by_field_idx_ + 1).sizes()[0];
        } else {
          size = Input(1).sizes()[0];
        }

        CAFFE_ENFORCE(
            batch_size_ > 0 && shuffle_size_ > 0 &&
            0 < batch_size_ * shuffle_size_);
        // adjust shuffle_size_ if it is too large
        if (batch_size_ * shuffle_size_ > size) {
          shuffle_size_ = size / batch_size_;
        }

        int num_batch = size / batch_size_;
        auto* out = Output(0);
        out->Resize(size);
        auto* out_data = out->template mutable_data<int64_t>();

        vector<int> shuffle_idx(size);
        iota(shuffle_idx.begin(), shuffle_idx.end(), 0);

        if (sort_by_field_idx_ != -1) {
          auto& sortblob = Input(sort_by_field_idx_ + 1);
          auto* sortdata = sortblob.data<int>();
          // must sort by a field at the root level
          CAFFE_ENFORCE(
              cursor->it.fields()[sort_by_field_idx_].lengthFieldId == -1);
          sort(shuffle_idx.begin(), shuffle_idx.end(), [&sortdata](int i1, int i2) {
            return sortdata[i1] < sortdata[i2];
          });
        }

        if (batch_size_ * shuffle_size_ > 1) {
          int offset = 0;
          while (offset + batch_size_ * shuffle_size_ < size) {
            std::shuffle(
                shuffle_idx.begin() + offset,
                shuffle_idx.begin() + offset + batch_size_ * shuffle_size_,
                std::default_random_engine());
            offset += batch_size_ * shuffle_size_;
          }
        }

        vector<int> batch_idx(num_batch);
        iota(batch_idx.begin(), batch_idx.end(), 0);
        std::shuffle(
            batch_idx.begin(), batch_idx.end(), std::default_random_engine());

        for (int i = 0; i < num_batch; i++) {
          std::copy(
              shuffle_idx.begin() + batch_idx[i] * batch_size_,
              shuffle_idx.begin() + (batch_idx[i] + 1) * batch_size_,
              out_data);
          out_data += batch_size_;
        }
        std::copy(
            shuffle_idx.begin() + num_batch * batch_size_,
            shuffle_idx.end(),
            out_data);

        return true;
        */
    }
}

/**
 | Read the next batch of examples out of the given cursor,
 | idx blob, offset matrix and data blobs.
 |
 | Input(0) is a blob pointing to a TreeCursor,
 | Input(1) is a blob pointing to the shuffled idx
 | Input(2) is a blob pointing to the offset matrix and
 | [Input(3),... Input(num_fields)] a list of tensors containing the data for
 | each field of the dataset.
 |
 | ReadRandomBatch is thread safe.
 */
pub struct ReadRandomBatchOp {
    storage: OperatorStorage,
    context: CPUContext,
    batch_size:         i32,
    enforce_batch_size: bool,
    loop_over:          bool,
}

num_inputs!{ReadRandomBatch, (1,INT_MAX)}

num_outputs!{ReadRandomBatch, (1,INT_MAX)}

inputs!{ReadRandomBatch, 
    0 => ("cursor", "A blob containing a pointer to the cursor."),
    1 => ("idx", "idx with a shuffled order."),
    2 => ("offsetsmat", "offset matrix containing length offset info."),
    3 => ("dataset_field_0", "First dataset field")
}

outputs!{ReadRandomBatch, 
    0 => ("field_0", "Tensor containing the next batch for field 0.")
}

args!{ReadRandomBatch, 
    0 => ("batch_size", "Number of top-level entries to read."),
    1 => ("loop_over", "(bool) Repeat the dataset indefinitely")
}

impl ReadRandomBatchOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            batchSize_(OperatorStorage::GetSingleArgument<int>("batch_size", 1)),
            enforceBatchSize_(
                OperatorStorage::GetSingleArgument<bool>("enforce_batch_size", false)),
            loopOver_(OperatorStorage::GetSingleArgument<bool>("loop_over", false))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& cursor = OperatorStorage::Input<std::unique_ptr<TreeCursor>>(0);
        auto& idxblob = Input(1);
        auto& offsetsmat = Input(2);
        CAFFE_ENFORCE(InputSize() == cursor->it.fields().size() + 3);
        auto idxvec = idxblob.template data<int64_t>();
        auto offsetdim = offsetsmat.sizes();
        // gather data
        std::vector<int64_t> outDim;
        int64_t idx;
        {
          std::lock_guard<std::mutex> lock(cursor->mutex_);
          cursor->offsets.resize(1);
          idx = cursor->offsets.at(0);
          // if we want to enforce batch size but we dont have a complete
          // batch, skip the last rows.
          if (enforceBatchSize_ && idx + batchSize_ > idxblob.numel()) {
            idx = idxblob.numel();
          }
          if (loopOver_ && idx >= idxblob.numel()) {
            cursor->offsets.at(0) = 0;
            idx = 0;
          }
          cursor->offsets.at(0) += batchSize_;
        }

        for (int i = 0; i < cursor->it.fields().size(); ++i) {
          auto lengthIdx = cursor->it.fields()[i].lengthFieldId + 1;
          auto& in = Input(i + 3);
          outDim = in.sizes().vec();
          outDim.at(0) = 0;
          auto idxbegin = idx;
          for (int j = 0; j < batchSize_; ++j) {
            if (idx >= idxblob.numel()) {
              break;
            }
            CAFFE_ENFORCE(
                (idxvec[idx] + 1) * offsetdim[1] + lengthIdx < offsetsmat.numel(),
                "Out of bound when trying to get elem from offsetsmat");
            auto offsetptr = offsetsmat.template data<TOffset>() +
                idxvec[idx] * offsetdim[1] + lengthIdx;
            auto offset = *offsetptr;
            auto size = *(offsetptr + offsetdim[1]) - offset;
            outDim.at(0) += size; // accumulate over the batch
            idx++;
          }
          idx = idxbegin; // reSet
          auto* out = Output(i);
          out->Resize(outDim);
          if (out->numel() == 0) {
            continue;
          }
          auto dst = static_cast<char*>(out->raw_mutable_data(in.dtype()));
          int block_size = in.numel() / in.size(0);
          auto block_bytesize = in.size_from_dim(1) * in.dtype().itemsize();
          CAFFE_ENFORCE(
              block_bytesize == in.nbytes() / in.size(0),
              "block_bytesize should be consistent with data dim");
          auto src_base = static_cast<const char*>(in.raw_data());
          int start = 0;
          for (int j = 0; j < batchSize_; ++j) {
            if (idx >= idxblob.numel()) {
              break;
            }
            auto offsetptr = offsetsmat.template data<TOffset>() +
                idxvec[idx] * offsetdim[1] + lengthIdx;
            auto offset = *offsetptr;
            auto size = *(offsetptr + offsetdim[1]) - offset;
            // copy data
            auto src = src_base + offset * block_bytesize;
            context_.CopyItemsSameDevice(
                in.dtype(), size * block_size, src, dst + start * block_bytesize);
            start += size;
            idx++;
          }
          idx = idxbegin; // reSet
        }
        return true;
        */
    }
}

#[test] fn append_op_example() {
    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Append",
        ["A", "B"],
        ["A"],
    )

    workspace.FeedBlob("A", np.random.randint(10, size=(1,3,3)))
    workspace.FeedBlob("B", np.random.randint(10, size=(2,3,3)))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("A:", workspace.FetchBlob("A"))

    A:
    [[[3 8 7]
      [1 6 6]
      [5 0 6]]]
    B:
    [[[4 3 1]
      [7 9 6]
      [9 4 5]]

     [[7 7 4]
      [9 8 7]
      [1 6 6]]]
    A:
    [[[3 8 7]
      [1 6 6]
      [5 0 6]]

     [[4 3 1]
      [7 9 6]
      [9 4 5]]

     [[7 7 4]
      [9 8 7]
      [1 6 6]]]

    */
}

/**
  | Append input `B` to the end of input `A`.
  | 
  | - It is required that this operation
  | run in-place, meaning that the input
  | `A` blob must match the output blob.
  | 
  | - All except the outer-most dimension
  | must be the same between `A` and `B`.
  | 
  | - Input `A` may have to be re-allocated
  | in order for accommodate to the new size.
  | Currently, an exponential growth ratio
  | is used in order to ensure amortized
  | constant time complexity.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc
  |
  */
pub struct AppendOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{Append, 2}

num_outputs!{Append, 1}

inputs!{Append, 
    0 => ("A", "(*Tensor*): base input tensor of shape $(N, d_1, d_2, ..., d_n)$"),
    1 => ("B", "(*Tensor*): second input tensor of shape $(M, d_1, d_2, ..., d_n)$ to be appended to the base")
}

outputs!{Append, 
    0 => ("A", "(*Tensor*): output tensor of shape $(N+M, d_1, d_2, ..., d_n)$")
}

enforce_inplace!{Append, vec![(0, 0)]}

impl<Context> AppendOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& a = Input(0);
        auto& b = Input(1);
        auto* c = Output(0);
        CAFFE_ENFORCE(b.dim() >= 1);
        if (a.numel() == 0 && a.size(0) == 0) {
          c->CopyFrom(b);
          return true;
        }
        CAFFE_ENFORCE(&a == c, "First argument must be in-place.");
        CAFFE_ENFORCE(c->dim() == b.dim());
        CAFFE_ENFORCE(b.dim() == c->dim());
        CAFFE_ENFORCE(a.dtype() == b.dtype());
        for (int i = 1; i < a.dim(); ++i) {
          CAFFE_ENFORCE(a.sizes()[i] == b.sizes()[i]);
        }
        auto oldSize = c->numel();
        c->Extend(b.sizes()[0], kDatasetGrowthPct);
        auto* dst = (char*)c->raw_mutable_data() + oldSize * b.dtype().itemsize();
        context_.CopyItemsSameDevice(b.dtype(), b.numel(), b.raw_data(), dst);
        return true;
        */
    }
}

///----------------------------------------
pub struct AtomicAppendOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{AtomicAppend, (3,INT_MAX)}

num_outputs!{AtomicAppend, (1,INT_MAX)}

allow_inplace!{AtomicAppend, 
    |input: i32, output: i32| {
        input == out + 1
    }
}

impl<Context> AtomicAppendOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& mutex = OperatorStorage::Input<std::unique_ptr<std::mutex>>(0);
        const auto numFields = (InputSize() - 1) / 2;
        CAFFE_ENFORCE(OutputSize() == numFields);

        std::lock_guard<std::mutex> guard(*mutex);

        // 1: checks
        for (int i = 0; i < numFields; ++i) {
          auto& a = Input(1 + i);
          auto& b = Input(1 + i + numFields);
          auto* c = Output(i);
          CAFFE_ENFORCE(b.dim() >= 1);
          if (a.numel() == 0) {
            continue;
          }
          CAFFE_ENFORCE(
              (void*)&a == (void*)c, "Appended-to arguments must be in-place.");
          CAFFE_ENFORCE(c->dim() == b.dim());
          CAFFE_ENFORCE(b.dim() == c->dim());
          CAFFE_ENFORCE(a.dtype() == b.dtype());
          for (int j = 1; j < a.dim(); ++j) {
            CAFFE_ENFORCE(a.sizes()[j] == b.sizes()[j]);
          }
        }

        // 2: copies
        for (int i = 0; i < numFields; ++i) {
          auto& a = Input(1 + i);
          auto& b = Input(1 + i + numFields);
          auto* c = Output(i);
          if (a.numel() == 0 && a.size(0) == 0) {
            c->CopyFrom(b);
            continue;
          }
          auto oldSize = c->numel();
          c->Extend(b.sizes()[0], kDatasetGrowthPct);
          auto* dst = (char*)c->raw_mutable_data() + oldSize * b.dtype().itemsize();
          context_.CopyItemsSameDevice(b.dtype(), b.numel(), b.raw_data(), dst);
        }
        return true;
        */
    }
}

///------------------------------------------
///Create a std::unique_ptr<std::vector<Tensor> >
pub struct CreateTensorVectorOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{CreateTensorVector, 0}

num_outputs!{CreateTensorVector, 1}

impl<Context> CreateTensorVectorOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto ptr = make_unique<std::vector<Tensor>>();
        *OperatorStorage::Output<TensorVectorPtr>(TENSOR_VECTOR) = std::move(ptr);
        return true;
        */
    }
}

output_tags!{
    CreateTensorVectorOp {
        TensorVector
    }
}

///-----------------------------------------------
///Get the size of the input vector
pub struct TensorVectorSizeOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{TensorVectorSize, 1}

num_outputs!{TensorVectorSize, 1}

inputs!{TensorVectorSize, 
    0 => ("tensor vector", "std::unique_ptr<std::vector<Tensor> >")
}

outputs!{TensorVectorSize, 
    0 => ("size", "int32_t size")
}

impl<Context> TensorVectorSizeOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& vector_ptr = OperatorStorage::Input<TensorVectorPtr>(TENSOR_VECTOR);
        auto* size = Output(SIZE);
        size->Resize();
        // 32-bit should be enough here
        *size->template mutable_data<int32_t>() = vector_ptr->size();
        return true;
        */
    }
}

input_tags!{
    TensorVectorSizeOp {
        TensorVector
    }
}

output_tags!{
    TensorVectorSizeOp {
        Size
    }
}

/**
  | Concat Tensors in the std::unique_ptr<std::vector<Tensor>>
  | along the first dimension.
  |
  */
pub struct ConcatTensorVectorOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
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
    storage: OperatorStorage,
    context: CPUContext,
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

register_cpu_operator!{CreateTreeCursor,          CreateTreeCursorOp}
register_cpu_operator!{ResetCursor,               ResetCursorOp}
register_cpu_operator!{ReadNextBatch,             ReadNextBatchOp}
register_cpu_operator!{GetCursorOffset,           GetCursorOffsetOp}
register_cpu_operator!{ComputeOffset,             ComputeOffsetOp}
register_cpu_operator!{SortAndShuffle,            SortAndShuffleOp}
register_cpu_operator!{ReadRandomBatch,           ReadRandomBatchOp}
register_cpu_operator!{CheckDatasetConsistency,   CheckDatasetConsistencyOp}
register_cpu_operator!{Append,                    AppendOp<CPUContext>}
register_cpu_operator!{AtomicAppend,              AtomicAppendOp<CPUContext>}
register_cpu_operator!{CreateTensorVector,        CreateTensorVectorOp<CPUContext>}
register_cpu_operator!{TensorVectorSize,          TensorVectorSizeOp<CPUContext>}
register_cpu_operator!{ConcatTensorVector,        ConcatTensorVectorOp<CPUContext>}
register_cpu_operator!{CollectTensor,             CollectTensorOp<CPUContext>}
register_cpu_operator!{PackRecords,               PackRecordsOp}
register_cpu_operator!{UnPackRecords,             UnPackRecordsOp}
register_cpu_operator!{TrimDataset,               TrimDatasetOp}

should_not_do_gradient!{CreateTreeCursor}
should_not_do_gradient!{ResetCursor}
should_not_do_gradient!{ReadNextBatch}
should_not_do_gradient!{ComputeOffset}
should_not_do_gradient!{ReadRandomBatch}
should_not_do_gradient!{CheckDatasetConsistency}
should_not_do_gradient!{Append}
should_not_do_gradient!{AtomicAppend}
should_not_do_gradient!{CreateTensorVector}
should_not_do_gradient!{TensorVectorSize}
should_not_do_gradient!{ConcatTensorVector}
should_not_do_gradient!{CollectTensor}
should_not_do_gradient!{UnPackRecords}
should_not_do_gradient!{PackRecords}

pub struct TreeCursorSerializer {
    base: dyn BlobSerializerBase,
}

impl TreeCursorSerializer {
    
    #[inline] pub fn serialize(
        &mut self, 
        pointer:   *const c_void,
        type_meta: TypeMeta,
        name:      &String,
        acceptor:  SerializationAcceptor)  
    {

        todo!();
        /*
            CAFFE_ENFORCE(typeMeta.Match<std::unique_ptr<TreeCursor>>());
        const auto& cursor =
            *static_cast<const std::unique_ptr<TreeCursor>*>(pointer);
        BlobProto blob_proto;

        // serialize offsets as a tensor
        if (cursor->offsets.size() > 0) {
          Blob offsets_blob;
          auto* offsets = BlobGetMutableTensor(&offsets_blob, CPU);
          offsets->Resize(cursor->offsets.size());
          std::copy(
              cursor->offsets.begin(),
              cursor->offsets.end(),
              offsets->template mutable_data<TOffset>());
          TensorSerializer ser;
          ser.Serialize(
              *offsets, name, blob_proto.mutable_tensor(), 0, offsets->numel());
        }
        blob_proto.set_name(name);
        blob_proto.set_type("std::unique_ptr<TreeCursor>");

        // serialize field names in the content
        std::ostringstream os;
        for (const auto& field : cursor->it.fields()) {
          os << field.name << " ";
        }
        blob_proto.set_content(os.str());

        acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
        */
    }
}

pub struct TreeCursorDeserializer {
    base: dyn BlobDeserializerBase,
}

impl TreeCursorDeserializer {
    
    #[inline] pub fn deserialize(&mut self, proto: &BlobProto, blob: *mut Blob)  {
        
        todo!();
        /*
            // Deserialize the field names
        std::vector<std::string> fieldNames;
        std::istringstream is(proto.content());
        std::string field;
        while (true) {
          is >> field;
          if (is.eof()) {
            break;
          }
          fieldNames.push_back(field);
        }
        TreeIterator it(fieldNames);

        auto* base = blob->template GetMutable<std::unique_ptr<TreeCursor>>();
        CAFFE_ENFORCE(base != nullptr, "TreeCursor doesn't exist.");
        (*base).reset(new TreeCursor(it));

        // Deserialize the offset vector when it is not empty. The proto.tensor()
        // function will return a TensorProto associated with offset vector. The
        // offset vector contains fields of type int64_t, and we verify it is not
        // empty before calling the deserializer.
        if (proto.tensor().int64_data().size() > 0) {
          TensorDeserializer deser;
          Blob offset_blob;
          deser.Deserialize(proto, &offset_blob);
          auto& offsets = offset_blob.template Get<Tensor>();
          auto* offsets_ptr = offsets.data<TOffset>();
          (*base)->offsets.assign(offsets_ptr, offsets_ptr + offsets.numel());
        }
        */
    }
}

register_blob_serializer!{
    /*
    (TypeMeta::Id<Box<TreeCursor>>()),
    TreeCursorSerializer
    */
}

register_blob_deserializer!{
    /*
    Box<TreeCursor>, 
    TreeCursorDeserializer
    */
}

register_blob_serializer!{
    /*
    (TypeMeta::Id<Arc<Vec<TensorCPU>>>()),
    SharedTensorVectorPtrSerializer
    */
}

register_blob_deserializer!{
    /*
    Arc<Vec<TensorCPU>>,
    SharedTensorVectorPtrDeserializer
    */
}
