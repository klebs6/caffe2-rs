crate::ix!();

#[inline] pub fn blob_to_tensor_descriptor(
    name:        &String,
    ws:          *mut Workspace,
    desc:        *mut OnnxTensorDescriptorV1,
    shapes:      *mut Vec<Vec<u64>>,
    all_scales:  *mut Vec<Vec<f32>>,
    all_offsets: *mut Vec<Vec<i32>>)  
{
    
    todo!();
    /*
        const Blob* blob = ws->GetBlob(name);
      CAFFE_ENFORCE(blob, "Blob ", name, " doesn't exist");
      const bool is_int8tensor =
          blob->meta().id() == TypeMeta::Id<int8::Int8TensorCPU>();
      bool is_external_tensor;
    #ifndef C10_MOBILE
      auto function_ptr =
          ExternalTensorFunctionsBaseRegistry()->Create(blob->meta().id());
      is_external_tensor = function_ptr != nullptr;
    #else
      is_external_tensor = false;
    #endif
      // Memory type
      // We only allow weights to be CPU tensor or int8tensor for now
      CAFFE_ENFORCE(
          (BlobIsTensorType(*blob, CPU) || BlobIsInt8TensorCPUType(*blob) ||
           is_external_tensor),
          "Initialization blob ",
          name,
          " needs to be TensorCPU or Int8TensorCPU or Int8FCDNNLowPPackedWeightBlob Based class: ",
          blob->TypeName());
      desc->tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
      desc->memoryType = ONNXIFI_MEMORY_TYPE_CPU;
      desc->isOffline = false;

      if (is_int8tensor) {
        // Data type
        const auto& cpu_int8tensor = blob->template Get<int8::Int8TensorCPU>();
        const auto& cpu_tensor = cpu_int8tensor.t;
        setInputTensorDescriptorTypeAndBuffer(cpu_int8tensor, desc);
        // Set dims
        const auto shape = cpu_tensor.sizes();
        desc->dimensions = shape.size();
        shapes->emplace_back(shape.cbegin(), shape.cend());
        desc->shape = shapes->back().data();
      } else if (is_external_tensor) {
    #ifndef C10_MOBILE
        ExternalTensorDescriptor ext_desc;
        function_ptr->SetupExternalTensorDescriptor(
            blob, shapes, all_scales, all_offsets, &ext_desc);
        copyDescriptor(&ext_desc, desc);
    #endif
      } else {
        // Data type
        const auto& cpu_tensor = blob->template Get<TensorCPU>();
        setInputTensorDescriptorTypeAndBuffer(cpu_tensor, desc);
        // Set dims
        const auto shape = cpu_tensor.sizes();
        desc->dimensions = shape.size();
        shapes->emplace_back(shape.cbegin(), shape.cend());
        desc->shape = shapes->back().data();
        desc->quantizationParams = 0;
      }
    */
}

