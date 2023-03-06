crate::ix!();

#[inline] pub fn add_scalar_input<DataT>(
    value:    &DataT,
    name:     &String,
    ws:       *mut Workspace,
    is_empty: bool) 
{
    todo!();
    /*
        Blob* blob = ws->CreateBlob(name);
      auto* tensor = BlobGetMutableTensor(blob, CPU);
      if (!isEmpty) {
        tensor->Resize(vector<int64_t>{1});
        *(tensor->template mutable_data<DataT>()) = value;
      } else {
        tensor->Resize(vector<int64_t>{0});
        tensor->template mutable_data<DataT>();
      }
      return;
    */
}
