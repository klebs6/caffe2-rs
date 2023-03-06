crate::ix!();

#[inline] pub fn build_blob_name_from_db_key(
    db_key:       &String,
    strip_prefix: &String,
    add_prefix:   &String) -> String 
{
    todo!();
    /*
        std::string key = dbKey.substr(0, dbKey.find(kChunkIdSeparator));
      if (!strip_prefix.empty()) {
        auto match_pos = key.find(strip_prefix);
        if (match_pos != std::string::npos) {
          key = key.substr(match_pos + strip_prefix.size());
        }
      }
      key = add_prefix + key;
      return key;
    */
}

/**
  | We are tracking sizes of already read tensor
  | parts while reading data chunks. This way we
  | can make sure that all chunks were loaded in
  | the end.
  */
#[inline] pub fn process_blob(
    blob:            *mut Blob,
    proto:           &BlobProto,
    blob_states_ptr: *mut HashMap<String,BlobState>,
    key:             &String,
    loaded_blobs:    *mut i32)  
{
    todo!();
    /*
        prepareBlob(blob, blob_states_ptr, key);
      DeserializeBlob(proto, blob);
      updateBlobStates(proto, blob_states_ptr, key, loaded_blobs);
    */
}

#[inline] pub fn prepare_blob(
    blob:        *mut Blob,
    blob_states: *mut HashMap<String,BlobState>,
    key:         &String)  
{
    todo!();
    /*
        if (blob_states->count(key) == 0) {
        // We reset the blob so that any existing content is destroyed. This
        // is to guarantee correct device placement: if we are deserializing
        // into a TensorCUDA, without explicit Reset we might be loading data
        // into an existing TensorCUDA that has pre-allocated memory on a
        // different GPU.
        blob->Reset();
      }
    */
}

#[inline] pub fn update_blob_states(
    proto:              &BlobProto,
    blob_states_ptr:    *mut HashMap<String,BlobState>,
    key:                &String,
    loaded_blobs:       *mut i32)  
{
    todo!();
    /*
        auto& blob_states = *blob_states_ptr;

      if (proto.has_content_num_chunks()) {
        if (!blob_states.count(key)) {
          blob_states[key] = BlobState(proto.content_num_chunks());
        }
        CAFFE_ENFORCE(
            blob_states[key]
                .seen_chunks_ids.insert(proto.content_chunk_id())
                .second,
            "Chunk with the same id has occurred twice for: ",
            key);
        CAFFE_ENFORCE(
            proto.content_chunk_id() >= 0 &&
                proto.content_chunk_id() < blob_states[key].total_size,
            "Chunk id has to be not less than 0 and "
            "less than content_num_chunks for key: ",
            key);
        blob_states[key].current_size++;
        CAFFE_ENFORCE(
            !blob_states[key].is_tensor,
            "Proto with content_chunks can not store tensor: ",
            key);
        CAFFE_ENFORCE(
            blob_states[key].current_size <= blob_states[key].total_size,
            "Found an extra part for an already filled blob: ",
            key);
        if (blob_states[key].current_size == blob_states[key].total_size) {
          (*loaded_blobs)++;
        }
        return;
      }
      if (!proto.has_tensor()) {
        // If blob is divided into chunks the field content_chunks has to be set,
        // otherwise only tensors can be seen multiple times as chunks.
        CAFFE_ENFORCE(blob_states.count(key) == 0, "Blob duplicated: ", key);
        blob_states[key] = BlobState();
        (*loaded_blobs)++;
        return;
      }
      CAFFE_ENFORCE(proto.has_tensor());
      if (blob_states.count(key)) {
        CAFFE_ENFORCE(blob_states[key].is_tensor, "Must be tensor ", key);
        CAFFE_ENFORCE(
            blob_states[key].current_size < blob_states[key].total_size,
            "Found an extra part for an already filled tensor: ",
            key);
        CAFFE_ENFORCE(
            proto.tensor().has_segment(),
            "Partial tensor must have a segment: ",
            key);
        blob_states[key].current_size +=
            proto.tensor().segment().end() - proto.tensor().segment().begin();
        CAFFE_ENFORCE(
            blob_states[key].current_size <= blob_states[key].total_size,
            "Tensor parts are bigger than target size for tensor: ",
            key);
      } else {
        const auto& dims = proto.tensor().dims();
        int64_t total_size = 1;
        for (const auto& dim : dims) {
          total_size *= dim;
        }
        auto current_size = total_size;
        if (proto.tensor().has_segment()) {
          current_size =
              proto.tensor().segment().end() - proto.tensor().segment().begin();
        }
        blob_states[key] =
            BlobState(total_size, current_size, true /* is_tensor */);
      }

      if (blob_states[key].current_size == blob_states[key].total_size) {
        (*loaded_blobs)++;
      }
    */
}

#[inline] pub fn validate_blob_states(blob_states: &HashMap<String,BlobState>)  {
    
    todo!();
    /*
        for (const auto& iter : blob_states) {
        const BlobState& blob_state = iter.second;
        CAFFE_ENFORCE(
            blob_state.current_size == blob_state.total_size,
            "Data size mismatch for blob ",
            iter.first,
            ". Expected: ",
            blob_state.total_size,
            " Read: ",
            blob_state.current_size);
      }
    */
}
