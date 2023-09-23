crate::ix!();

pub fn from_file(
    filename:   &str,
    shared:     Option<bool>,
    size:       Option<i64>,
    dtype:      Option<ScalarType>,
    layout:     Option<Layout>,
    device:     Option<Device>,
    pin_memory: Option<bool>

) -> Tensor {

    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

        TORCH_CHECK(!options.pinned_memory(), "tensors constructed from a file cannot be pinned");
        i64 my_size = size.value_or(0);
        int flags = shared.value_or(false) ? TH_ALLOCATOR_MAPPED_SHARED : 0;
        auto my_dtype = options.dtype();
        usize size_bytes = my_size * my_dtype.itemsize();
        auto storage_impl = make_intrusive<StorageImpl>(
            StorageImpl::use_byte_Size(),
            size_bytes,
            THMapAllocator::makeDataPtr(
                string(filename), flags, size_bytes, nullptr),
            /*allocator=*/nullptr,
            /*resizable=*/false);
        auto tensor = make_tensor<TensorImpl>(
            storage_impl, DispatchKey::CPU, my_dtype);
        tensor.unsafeGetTensorImpl()->set_sizes_contiguous({my_size});
        return tensor;
        */
}
