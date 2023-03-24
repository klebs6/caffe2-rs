crate::ix!();

#[inline] pub fn infer_blob_shapes_and_types_from_workspace(
    ws:   *mut Workspace,
    nets: &Vec<*mut NetDef>)  
{
    
    todo!();
    /*
        CaffeMap<string, TensorShape> blob_desc;
      // Populate shapes from workplace
      const std::vector<string>& ws_blobs = ws->Blobs();
      for (const auto& s : ws_blobs) {
        Blob* b = ws->GetBlob(s);
        TensorShape tp = GetTensorShapeOfBlob(b);
        blob_desc[s] = tp;
      }
      return InferBlobShapesAndTypes(blob_desc, nets);
    */
}

#[inline] pub fn infer_blob_shapes_and_types_from_map(
    blob_dimensions: &HashMap<String,Vec<i64>>,
    nets: &Vec<*mut NetDef>)  
{
    todo!();
    /*
        CaffeMap<string, TensorShape> blob_desc;
      // Populate shapes from known blobs
      for (const auto& blob : blob_dimensions) {
        TensorShape tp;
        for (auto d : blob.second) {
          CAFFE_ENFORCE_GE(d, 0, blob.first);
          tp.add_dims(d);
        }
        blob_desc[blob.first] = tp;
      }
      return InferBlobShapesAndTypes(blob_desc, nets);
    */
}


#[inline] pub fn infer_blob_shapes_and_types_from_map_with_blob_types(
    blob_dimensions: &HashMap<String,Vec<i64>>,
    blob_types: &HashMap<String,TensorProto_DataType>,
    nets: &Vec<*mut NetDef>)  
{
    todo!();
    /*
        CaffeMap<string, TensorShape> blob_desc;
      // Populate shapes from known blobs
      for (const auto& blob : blob_dimensions) {
        TensorShape tp;
        for (auto d : blob.second) {
          CAFFE_ENFORCE_GE(d, 0, blob.first);
          tp.add_dims(d);
        }
        auto blob_type = blob_types.find(blob.first);
        if (blob_type == blob_types.end()) {
          LOG(WARNING) << "Missing type of " << blob.first
                       << "; assuming to be UNDEFINED";
          tp.set_data_type(TensorProto_DataType_UNDEFINED);
        } else {
          tp.set_data_type(blob_type->second);
        }
        blob_desc[blob.first] = tp;
      }
      return InferBlobShapesAndTypes(blob_desc, nets);
    */
}


