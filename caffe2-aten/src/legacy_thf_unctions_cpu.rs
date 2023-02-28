crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/LegacyTHFunctionsCPU.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/LegacyTHFunctionsCPU.cpp]

pub fn infer_scalar_type(t: &Tensor) -> ScalarType {
    
    todo!();
        /*
            return t.scalar_type();
        */
}

pub fn infer_scalar_type_with_list(tl: &TensorList) -> ScalarType {
    
    todo!();
        /*
            TORCH_CHECK(tl.size() > 0, "expected a non-empty list of Tensors");
        return tl[0].scalar_type();
        */
}

pub fn options(s: ScalarType) -> TensorOptions {
    
    todo!();
        /*
            return TensorOptions().dtype(s)
                              .device(DeviceType_CPU)
                              .layout(kStrided);
        */
}

pub fn allocator() -> *mut Allocator {
    
    todo!();
        /*
            return getCPUAllocator();
        */
}

pub fn th_histc_out(
        self_:  &Tensor,
        bins:   i64,
        min:    &Scalar,
        max:    &Scalar,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            // DeviceGuard omitted
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_histc_out", false, DeviceType_CPU, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_histc_out", false, DeviceType_CPU, dispatch_scalar_type);
                auto min_ = min.toDouble();
                auto max_ = max.toDouble();
                THDoubleTensor_histc(result_, self_, bins, min_, max_);
                break;
            }
            case ScalarType::Float: {
                auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_histc_out", false, DeviceType_CPU, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_histc_out", false, DeviceType_CPU, dispatch_scalar_type);
                auto min_ = min.toFloat();
                auto max_ = max.toFloat();
                THFloatTensor_histc(result_, self_, bins, min_, max_);
                break;
            }
            default:
                AT_ERROR("_th_histc_out not supported on CPUType for ", dispatch_scalar_type);
        }
        return result;
        */
}

pub fn th_histc(
        self_: &Tensor,
        bins:  i64,
        min:   &Scalar,
        max:   &Scalar) -> Tensor {
    
    todo!();
        /*
            // DeviceGuard omitted
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto result_ = make_intrusive<TensorImpl, UndefinedTensorImpl>(Storage(Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto result = Tensor(intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_histc", false, DeviceType_CPU, dispatch_scalar_type);
                auto min_ = min.toDouble();
                auto max_ = max.toDouble();
                THDoubleTensor_histc(result_, self_, bins, min_, max_);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_histc", false, DeviceType_CPU, dispatch_scalar_type);
                auto min_ = min.toFloat();
                auto max_ = max.toFloat();
                THFloatTensor_histc(result_, self_, bins, min_, max_);
                break;
            }
            default:
                AT_ERROR("_th_histc not supported on CPUType for ", dispatch_scalar_type);
        }
        return result;
        */
}
