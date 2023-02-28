crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/LegacyTHFunctionsCUDA.cpp]

pub fn infer_scalar_type(t: &Tensor) -> ScalarType {
    
    todo!();
        /*
            return t.scalar_type();
        */
}

pub fn infer_scalar_type_with_tensor_list(tl: &TensorList) -> ScalarType {
    
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
                              .device(DeviceType::CUDA)
                              .layout(kStrided);
        */
}

pub fn allocator() -> *mut Allocator {
    
    todo!();
        /*
            return at::cuda::getCUDADeviceAllocator();
        */
}


pub fn th_cross_kernel_out(
        result: &mut Tensor,
        self_:  &Tensor,
        other:  &Tensor,
        dim:    i64) -> &mut Tensor {
    
    todo!();
        /*
            // DeviceGuard omitted
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Byte: {
                auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaByteTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
                break;
            }
            case ScalarType::Char: {
                auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaCharTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
                break;
            }
            case ScalarType::Double: {
                auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaDoubleTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
                break;
            }
            case ScalarType::Float: {
                auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
                break;
            }
            case ScalarType::Int: {
                auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaIntTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
                break;
            }
            case ScalarType::Long: {
                auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaLongTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
                break;
            }
            case ScalarType::Short: {
                auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaShortTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
                break;
            }
            case ScalarType::Half: {
                auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaHalfTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
                break;
            }
            default:
                AT_ERROR("_th_cross_kernel_out not supported on CUDAType for ", dispatch_scalar_type);
        }
        return result;
        */
}


pub fn th_cross_kernel(
        self_: &Tensor,
        other: &Tensor,
        dim:   i64) -> Tensor {
    
    todo!();
        /*
            // DeviceGuard omitted
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
        switch (dispatch_scalar_type) {
            case ScalarType::Byte: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
                auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaByteTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
                break;
            }
            case ScalarType::Char: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
                auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaCharTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
                break;
            }
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
                auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaDoubleTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
                auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
                break;
            }
            case ScalarType::Int: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
                auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaIntTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
                break;
            }
            case ScalarType::Long: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
                auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaLongTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
                break;
            }
            case ScalarType::Short: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
                auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaShortTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
                break;
            }
            case ScalarType::Half: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
                auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaHalfTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
                break;
            }
            default:
                AT_ERROR("_th_cross_kernel not supported on CUDAType for ", dispatch_scalar_type);
        }
        return result;
        */
}


pub fn th_gels_out(
    self_: &Tensor,
    A:     &Tensor,
    res1:  &mut Tensor,
    res2:  &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            TORCH_WARN_ONCE(
          "torch.lstsq is deprecated in favor of torch.linalg.lstsq and will be removed in a future PyTorch release.\n",
          "torch.linalg.lstsq has reversed arguments and does not return the QR decomposition in "
          "the returned tuple (although it returns other information about the problem).\n",
          "To get the qr decomposition consider using torch.linalg.qr.\n",
          "The returned solution in torch.lstsq stored the residuals of the solution in the ",
          "last m - n columns of the returned value whenever m > n. In torch.linalg.lstsq, the ",
          "residuals in the field 'residuals' of the returned named tuple.\n",
          "The unpacking of the solution, as in\n",
          "X, _ = torch.lstsq(B, A).solution[:A.size(1)]\n",
          "should be replaced with\n",
          "X = torch.linalg.lstsq(A, B).solution"
        );
        // DeviceGuard omitted
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto res1_ = checked_dense_tensor_unwrap(res1, "res1", 0, "_th_gels_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto res2_ = checked_dense_tensor_unwrap(res2, "res2", 0, "_th_gels_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_gels_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto A_ = checked_dense_tensor_unwrap(A, "A", 2, "_th_gels_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaDoubleTensor_gels(globalContext().getTHCState(), res1_, res2_, self_, A_);
                break;
            }
            case ScalarType::Float: {
                auto res1_ = checked_dense_tensor_unwrap(res1, "res1", 0, "_th_gels_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto res2_ = checked_dense_tensor_unwrap(res2, "res2", 0, "_th_gels_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_gels_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto A_ = checked_dense_tensor_unwrap(A, "A", 2, "_th_gels_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaTensor_gels(globalContext().getTHCState(), res1_, res2_, self_, A_);
                break;
            }
            default:
                AT_ERROR("_th_gels_out not supported on CUDAType for ", dispatch_scalar_type);
        }
        return std::tuple<Tensor &, Tensor &>(res1, res2);
        */
}

pub fn th_gels(
    self_: &Tensor,
    A:     &Tensor) -> (Tensor,Tensor) {
    
    todo!();
        /*
            TORCH_WARN_ONCE(
          "torch.lstsq is deprecated in favor of torch.linalg.lstsq and will be removed in a future PyTorch release.\n",
          "torch.linalg.lstsq has reversed arguments and does not return the QR decomposition in "
          "the returned tuple (although it returns other information about the problem).\n",
          "To get the qr decomposition consider using torch.linalg.qr.\n",
          "The returned solution in torch.lstsq stored the residuals of the solution in the ",
          "last m - n columns of the returned value whenever m > n. In torch.linalg.lstsq, the ",
          "residuals in the field 'residuals' of the returned named tuple.\n",
          "The unpacking of the solution, as in\n",
          "X, _ = torch.lstsq(B, A).solution[:A.size(1)]\n",
          "should be replaced with\n",
          "X = torch.linalg.lstsq(A, B).solution"
        );
        // DeviceGuard omitted
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto res1_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto res1 = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(res1_));
        auto res2_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto res2 = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(res2_));
        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_gels", false, DeviceType::CUDA, dispatch_scalar_type);
                auto A_ = checked_dense_tensor_unwrap(A, "A", 2, "_th_gels", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaDoubleTensor_gels(globalContext().getTHCState(), res1_, res2_, self_, A_);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_gels", false, DeviceType::CUDA, dispatch_scalar_type);
                auto A_ = checked_dense_tensor_unwrap(A, "A", 2, "_th_gels", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaTensor_gels(globalContext().getTHCState(), res1_, res2_, self_, A_);
                break;
            }
            default:
                AT_ERROR("_th_gels not supported on CUDAType for ", dispatch_scalar_type);
        }
        return std::tuple<Tensor, Tensor>(res1, res2);
        */
}


pub fn th_copy_ignoring_overlaps(
        self_: &mut Tensor,
        src:   &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            // DeviceGuard omitted
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Byte: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
                auto src_ = checked_dense_tensor_unwrap(src, "src", 2, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaByteTensor_copyIgnoringOverlaps(globalContext().getTHCState(), self_, src_);
                break;
            }
            case ScalarType::Char: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
                auto src_ = checked_dense_tensor_unwrap(src, "src", 2, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaCharTensor_copyIgnoringOverlaps(globalContext().getTHCState(), self_, src_);
                break;
            }
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
                auto src_ = checked_dense_tensor_unwrap(src, "src", 2, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaDoubleTensor_copyIgnoringOverlaps(globalContext().getTHCState(), self_, src_);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
                auto src_ = checked_dense_tensor_unwrap(src, "src", 2, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaTensor_copyIgnoringOverlaps(globalContext().getTHCState(), self_, src_);
                break;
            }
            case ScalarType::Int: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
                auto src_ = checked_dense_tensor_unwrap(src, "src", 2, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaIntTensor_copyIgnoringOverlaps(globalContext().getTHCState(), self_, src_);
                break;
            }
            case ScalarType::Long: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
                auto src_ = checked_dense_tensor_unwrap(src, "src", 2, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaLongTensor_copyIgnoringOverlaps(globalContext().getTHCState(), self_, src_);
                break;
            }
            case ScalarType::Short: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
                auto src_ = checked_dense_tensor_unwrap(src, "src", 2, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaShortTensor_copyIgnoringOverlaps(globalContext().getTHCState(), self_, src_);
                break;
            }
            case ScalarType::Half: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
                auto src_ = checked_dense_tensor_unwrap(src, "src", 2, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
                THCudaHalfTensor_copyIgnoringOverlaps(globalContext().getTHCState(), self_, src_);
                break;
            }
            default:
                AT_ERROR("_th_copy_ignoring_overlaps_ not supported on CUDAType for ", dispatch_scalar_type);
        }
        return self;
        */
}


pub fn thnn_multi_margin_loss_forward_out(
        self_:      &Tensor,
        target:     &Tensor,
        p:          &Scalar,
        margin:     &Scalar,
        weight_opt: &Option<Tensor>,
        reduction:  i64,
        output:     &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

        const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multi_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multi_margin_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto p_ = p.toDouble();
                auto margin_ = margin.toDouble();
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 5, "_thnn_multi_margin_loss_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 6, "_thnn_multi_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleMultiMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, p_, weight_ ? weight_ : NULL, margin_);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multi_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multi_margin_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto p_ = p.toDouble();
                auto margin_ = margin.toDouble();
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 5, "_thnn_multi_margin_loss_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 6, "_thnn_multi_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaMultiMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, p_, weight_ ? weight_ : NULL, margin_);
                break;
            }
            case ScalarType::Half: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multi_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multi_margin_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto p_ = p.toDouble();
                auto margin_ = margin.toDouble();
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 5, "_thnn_multi_margin_loss_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 6, "_thnn_multi_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfMultiMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, p_, weight_ ? weight_ : NULL, margin_);
                break;
            }
            default:
                AT_ERROR("_thnn_multi_margin_loss_forward_out not supported on CUDAType for ", dispatch_scalar_type);
        }
        return output;
        */
}


pub fn thnn_multi_margin_loss_forward(
        self_:      &Tensor,
        target:     &Tensor,
        p:          &Scalar,
        margin:     &Scalar,
        weight_opt: &Option<Tensor>,
        reduction:  i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

        const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto output_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto output = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(output_));
        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multi_margin_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multi_margin_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
                auto p_ = p.toDouble();
                auto margin_ = margin.toDouble();
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 5, "_thnn_multi_margin_loss_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleMultiMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, p_, weight_ ? weight_ : NULL, margin_);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multi_margin_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multi_margin_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
                auto p_ = p.toDouble();
                auto margin_ = margin.toDouble();
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 5, "_thnn_multi_margin_loss_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaMultiMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, p_, weight_ ? weight_ : NULL, margin_);
                break;
            }
            case ScalarType::Half: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multi_margin_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multi_margin_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
                auto p_ = p.toDouble();
                auto margin_ = margin.toDouble();
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 5, "_thnn_multi_margin_loss_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfMultiMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, p_, weight_ ? weight_ : NULL, margin_);
                break;
            }
            default:
                AT_ERROR("_thnn_multi_margin_loss_forward not supported on CUDAType for ", dispatch_scalar_type);
        }
        return output;
        */
}


pub fn thnn_multi_margin_loss_backward_out(
        grad_output: &Tensor,
        self_:       &Tensor,
        target:      &Tensor,
        p:           &Scalar,
        margin:      &Scalar,
        weight_opt:  &Option<Tensor>,
        reduction:   i64,
        grad_input:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

        const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto p_ = p.toDouble();
                auto margin_ = margin.toDouble();
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 6, "_thnn_multi_margin_loss_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleMultiMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, p_, weight_ ? weight_ : NULL, margin_);
                break;
            }
            case ScalarType::Float: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto p_ = p.toDouble();
                auto margin_ = margin.toDouble();
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 6, "_thnn_multi_margin_loss_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaMultiMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, p_, weight_ ? weight_ : NULL, margin_);
                break;
            }
            case ScalarType::Half: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto p_ = p.toDouble();
                auto margin_ = margin.toDouble();
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 6, "_thnn_multi_margin_loss_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfMultiMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, p_, weight_ ? weight_ : NULL, margin_);
                break;
            }
            default:
                AT_ERROR("_thnn_multi_margin_loss_backward_out not supported on CUDAType for ", dispatch_scalar_type);
        }
        return grad_input;
        */
}


pub fn thnn_multi_margin_loss_backward(
        grad_output: &Tensor,
        self_:       &Tensor,
        target:      &Tensor,
        p:           &Scalar,
        margin:      &Scalar,
        weight_opt:  &Option<Tensor>,
        reduction:   i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

        const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto grad_input_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto grad_input = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_input_));
        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multi_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multi_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multi_margin_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
                auto p_ = p.toDouble();
                auto margin_ = margin.toDouble();
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 6, "_thnn_multi_margin_loss_backward", true, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleMultiMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, p_, weight_ ? weight_ : NULL, margin_);
                break;
            }
            case ScalarType::Float: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multi_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multi_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multi_margin_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
                auto p_ = p.toDouble();
                auto margin_ = margin.toDouble();
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 6, "_thnn_multi_margin_loss_backward", true, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaMultiMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, p_, weight_ ? weight_ : NULL, margin_);
                break;
            }
            case ScalarType::Half: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multi_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multi_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multi_margin_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
                auto p_ = p.toDouble();
                auto margin_ = margin.toDouble();
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 6, "_thnn_multi_margin_loss_backward", true, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfMultiMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, p_, weight_ ? weight_ : NULL, margin_);
                break;
            }
            default:
                AT_ERROR("_thnn_multi_margin_loss_backward not supported on CUDAType for ", dispatch_scalar_type);
        }
        return grad_input;
        */
}

pub fn thnn_multilabel_margin_loss_forward_out(
    self_:     &Tensor,
    target:    &Tensor,
    reduction: i64,
    output:    &mut Tensor,
    is_target: &mut Tensor) -> (&mut Tensor,&mut Tensor) 
{
    todo!();
        /*
            const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 3, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 3, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleMultiLabelMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, is_target_, reduction);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 3, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 3, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaMultiLabelMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, is_target_, reduction);
                break;
            }
            case ScalarType::Half: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 3, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 3, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfMultiLabelMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, is_target_, reduction);
                break;
            }
            case ScalarType::BFloat16: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 3, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 3, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaBFloat16MultiLabelMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, is_target_, reduction);
                break;
            }
            default:
                AT_ERROR("_thnn_multilabel_margin_loss_forward_out not supported on CUDAType for ", dispatch_scalar_type);
        }
        return std::tuple<Tensor &, Tensor &>(output, is_target);
        */
}


pub fn thnn_multilabel_margin_loss_forward(
    self_:     &Tensor,
    target:    &Tensor,
    reduction: i64) -> (Tensor,Tensor) {
    
    todo!();
        /*
            const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto output_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto output = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(output_));
        auto is_target_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto is_target = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(is_target_));
        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multilabel_margin_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multilabel_margin_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
                THNN_CudaDoubleMultiLabelMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, is_target_, reduction);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multilabel_margin_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multilabel_margin_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
                THNN_CudaMultiLabelMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, is_target_, reduction);
                break;
            }
            case ScalarType::Half: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multilabel_margin_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multilabel_margin_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
                THNN_CudaHalfMultiLabelMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, is_target_, reduction);
                break;
            }
            case ScalarType::BFloat16: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multilabel_margin_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multilabel_margin_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
                THNN_CudaBFloat16MultiLabelMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, is_target_, reduction);
                break;
            }
            default:
                AT_ERROR("_thnn_multilabel_margin_loss_forward not supported on CUDAType for ", dispatch_scalar_type);
        }
        return std::tuple<Tensor, Tensor>(output, is_target);
        */
}


pub fn thnn_multilabel_margin_loss_backward_out(
    grad_output: &Tensor,
    self_:       &Tensor,
    target:      &Tensor,
    reduction:   i64,
    is_target:   &Tensor,
    grad_input:  &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 5, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 5, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleMultiLabelMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, is_target_, reduction);
                break;
            }
            case ScalarType::Float: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 5, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 5, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaMultiLabelMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, is_target_, reduction);
                break;
            }
            case ScalarType::Half: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 5, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 5, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfMultiLabelMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, is_target_, reduction);
                break;
            }
            case ScalarType::BFloat16: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 5, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 5, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaBFloat16MultiLabelMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, is_target_, reduction);
                break;
            }
            default:
                AT_ERROR("_thnn_multilabel_margin_loss_backward_out not supported on CUDAType for ", dispatch_scalar_type);
        }
        return grad_input;
        */
}

pub fn thnn_multilabel_margin_loss_backward(
    grad_output: &Tensor,
    self_:       &Tensor,
    target:      &Tensor,
    reduction:   i64,
    is_target:   &Tensor) -> Tensor {
    
    todo!();
        /*
            const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto grad_input_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto grad_input = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_input_));
        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
                auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 5, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleMultiLabelMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, is_target_, reduction);
                break;
            }
            case ScalarType::Float: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
                auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 5, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaMultiLabelMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, is_target_, reduction);
                break;
            }
            case ScalarType::Half: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
                auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 5, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfMultiLabelMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, is_target_, reduction);
                break;
            }
            case ScalarType::BFloat16: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
                auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 5, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaBFloat16MultiLabelMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, is_target_, reduction);
                break;
            }
            default:
                AT_ERROR("_thnn_multilabel_margin_loss_backward not supported on CUDAType for ", dispatch_scalar_type);
        }
        return grad_input;
        */
}


pub fn thnn_nll_loss_forward_out(
    self_:        &Tensor,
    target:       &Tensor,
    weight_opt:   &Option<Tensor>,
    reduction:    i64,
    ignore_index: i64,
    output:       &mut Tensor,
    total_weight: &mut Tensor) -> (&mut Tensor,&mut Tensor) {

    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

        const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 5, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 5, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 5, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 5, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::Half: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 5, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 5, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::BFloat16: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 5, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 5, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaBFloat16ClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            default:
                AT_ERROR("_thnn_nll_loss_forward_out not supported on CUDAType for ", dispatch_scalar_type);
        }
        return std::tuple<Tensor &, Tensor &>(output, total_weight);
        */
}

pub fn thnn_nll_loss_forward(
    self_:        &Tensor,
    target:       &Tensor,
    weight_opt:   &Option<Tensor>,
    reduction:    i64,
    ignore_index: i64) -> (Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

        const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto output_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto output = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(output_));
        auto total_weight_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto total_weight = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(total_weight_));
        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::Half: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::BFloat16: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaBFloat16ClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            default:
                AT_ERROR("_thnn_nll_loss_forward not supported on CUDAType for ", dispatch_scalar_type);
        }
        return std::tuple<Tensor, Tensor>(output, total_weight);
        */
}

pub fn thnn_nll_loss_backward_out(
    grad_output:  &Tensor,
    self_:        &Tensor,
    target:       &Tensor,
    weight_opt:   &Option<Tensor>,
    reduction:    i64,
    ignore_index: i64,
    total_weight: &Tensor,
    grad_input:   &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

        const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::Float: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::Half: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::BFloat16: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaBFloat16ClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            default:
                AT_ERROR("_thnn_nll_loss_backward_out not supported on CUDAType for ", dispatch_scalar_type);
        }
        return grad_input;
        */
}

pub fn thnn_nll_loss_backward(
    grad_output:  &Tensor,
    self_:        &Tensor,
    target:       &Tensor,
    weight_opt:   &Option<Tensor>,
    reduction:    i64,
    ignore_index: i64,
    total_weight: &Tensor) -> Tensor {

    todo!();
    /*
    // See [Note: hacky wrapper removal for optional tensor]
      c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

        const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto grad_input_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto grad_input = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_input_));
        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss_backward", true, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::Float: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss_backward", true, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::Half: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss_backward", true, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::BFloat16: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss_backward", true, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaBFloat16ClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            default:
                AT_ERROR("_thnn_nll_loss_backward not supported on CUDAType for ", dispatch_scalar_type);
        }
        return grad_input;
        */
}


pub fn thnn_nll_loss2d_forward_out(
    self_:        &Tensor,
    target:       &Tensor,
    weight_opt:   &Option<Tensor>,
    reduction:    i64,
    ignore_index: i64,
    output:       &mut Tensor,
    total_weight: &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

        const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 5, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 5, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleSpatialClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 5, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 5, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaSpatialClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::Half: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 5, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 5, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfSpatialClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::BFloat16: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 5, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 5, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaBFloat16SpatialClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            default:
                AT_ERROR("_thnn_nll_loss2d_forward_out not supported on CUDAType for ", dispatch_scalar_type);
        }
        return std::tuple<Tensor &, Tensor &>(output, total_weight);
        */
}

pub fn thnn_nll_loss2d_forward(
    self_:        &Tensor,
    target:       &Tensor,
    weight_opt:   &Option<Tensor>,
    reduction:    i64,
    ignore_index: i64) -> (Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

        const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto output_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto output = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(output_));
        auto total_weight_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto total_weight = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(total_weight_));
        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss2d_forward", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleSpatialClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss2d_forward", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaSpatialClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::Half: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss2d_forward", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfSpatialClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::BFloat16: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss2d_forward", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaBFloat16SpatialClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            default:
                AT_ERROR("_thnn_nll_loss2d_forward not supported on CUDAType for ", dispatch_scalar_type);
        }
        return std::tuple<Tensor, Tensor>(output, total_weight);
        */
}

pub fn thnn_nll_loss2d_backward_out(
    grad_output:  &Tensor,
    self_:        &Tensor,
    target:       &Tensor,
    weight_opt:   &Option<Tensor>,
    reduction:    i64,
    ignore_index: i64,
    total_weight: &Tensor,
    grad_input:   &mut Tensor) -> &mut Tensor {

    todo!();
    /*
            // See [Note: hacky wrapper removal for optional tensor]
      c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

        const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleSpatialClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::Float: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaSpatialClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::Half: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfSpatialClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::BFloat16: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaBFloat16SpatialClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            default:
                AT_ERROR("_thnn_nll_loss2d_backward_out not supported on CUDAType for ", dispatch_scalar_type);
        }
        return grad_input;
        */
}


pub fn thnn_nll_loss2d_backward(
    grad_output:  &Tensor,
    self_:        &Tensor,
    target:       &Tensor,
    weight_opt:   &Option<Tensor>,
    reduction:    i64,
    ignore_index: i64,
    total_weight: &Tensor) -> Tensor {

    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

        const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto grad_input_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto grad_input = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_input_));
        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss2d_backward", true, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleSpatialClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::Float: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss2d_backward", true, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaSpatialClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::Half: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss2d_backward", true, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfSpatialClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            case ScalarType::BFloat16: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, ScalarType::Long);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss2d_backward", true, DeviceType::CUDA, dispatch_scalar_type);
                auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaBFloat16SpatialClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
                break;
            }
            default:
                AT_ERROR("_thnn_nll_loss2d_backward not supported on CUDAType for ", dispatch_scalar_type);
        }
        return grad_input;
        */
}

pub fn thnn_glu_forward_out(
    self_:  &Tensor,
    dim:    i64,
    output: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_glu_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 2, "_thnn_glu_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleGatedLinear_updateOutput(globalContext().getTHCState(), self_, output_, dim);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_glu_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 2, "_thnn_glu_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaGatedLinear_updateOutput(globalContext().getTHCState(), self_, output_, dim);
                break;
            }
            case ScalarType::Half: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_glu_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 2, "_thnn_glu_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfGatedLinear_updateOutput(globalContext().getTHCState(), self_, output_, dim);
                break;
            }
            default:
                AT_ERROR("_thnn_glu_forward_out not supported on CUDAType for ", dispatch_scalar_type);
        }
        return output;
        */
}

pub fn thnn_glu_forward(
    self_: &Tensor,
    dim:   i64) -> Tensor {
    
    todo!();
        /*
            const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto output_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto output = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(output_));
        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_glu_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleGatedLinear_updateOutput(globalContext().getTHCState(), self_, output_, dim);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_glu_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaGatedLinear_updateOutput(globalContext().getTHCState(), self_, output_, dim);
                break;
            }
            case ScalarType::Half: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_glu_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfGatedLinear_updateOutput(globalContext().getTHCState(), self_, output_, dim);
                break;
            }
            default:
                AT_ERROR("_thnn_glu_forward not supported on CUDAType for ", dispatch_scalar_type);
        }
        return output;
        */
}

pub fn thnn_glu_backward_out(
    grad_output: &Tensor,
    self_:       &Tensor,
    dim:         i64,
    grad_input:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_glu_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_glu_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 3, "_thnn_glu_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleGatedLinear_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, dim);
                break;
            }
            case ScalarType::Float: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_glu_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_glu_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 3, "_thnn_glu_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaGatedLinear_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, dim);
                break;
            }
            case ScalarType::Half: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_glu_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_glu_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 3, "_thnn_glu_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfGatedLinear_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, dim);
                break;
            }
            default:
                AT_ERROR("_thnn_glu_backward_out not supported on CUDAType for ", dispatch_scalar_type);
        }
        return grad_input;
        */
}


pub fn thnn_glu_backward(
    grad_output: &Tensor,
    self_:       &Tensor,
    dim:         i64) -> Tensor {
    
    todo!();
        /*
            const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto grad_input_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto grad_input = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_input_));
        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_glu_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_glu_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleGatedLinear_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, dim);
                break;
            }
            case ScalarType::Float: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_glu_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_glu_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaGatedLinear_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, dim);
                break;
            }
            case ScalarType::Half: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_glu_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_glu_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfGatedLinear_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, dim);
                break;
            }
            default:
                AT_ERROR("_thnn_glu_backward not supported on CUDAType for ", dispatch_scalar_type);
        }
        return grad_input;
        */
}

pub fn thnn_log_sigmoid_forward_out(
    self_:  &Tensor,
    output: &mut Tensor,
    buffer: &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_log_sigmoid_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 1, "_thnn_log_sigmoid_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto buffer_ = checked_dense_tensor_unwrap(buffer, "buffer", 1, "_thnn_log_sigmoid_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleLogSigmoid_updateOutput(globalContext().getTHCState(), self_, output_, buffer_);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_log_sigmoid_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 1, "_thnn_log_sigmoid_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto buffer_ = checked_dense_tensor_unwrap(buffer, "buffer", 1, "_thnn_log_sigmoid_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaLogSigmoid_updateOutput(globalContext().getTHCState(), self_, output_, buffer_);
                break;
            }
            case ScalarType::Half: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_log_sigmoid_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 1, "_thnn_log_sigmoid_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto buffer_ = checked_dense_tensor_unwrap(buffer, "buffer", 1, "_thnn_log_sigmoid_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfLogSigmoid_updateOutput(globalContext().getTHCState(), self_, output_, buffer_);
                break;
            }
            default:
                AT_ERROR("_thnn_log_sigmoid_forward_out not supported on CUDAType for ", dispatch_scalar_type);
        }
        return std::tuple<Tensor &, Tensor &>(output, buffer);
        */
}

pub fn thnn_log_sigmoid_forward(self_: &Tensor) -> (Tensor,Tensor) {
    
    todo!();
        /*
            const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto output_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto output = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(output_));
        auto buffer_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto buffer = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(buffer_));
        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_log_sigmoid_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleLogSigmoid_updateOutput(globalContext().getTHCState(), self_, output_, buffer_);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_log_sigmoid_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaLogSigmoid_updateOutput(globalContext().getTHCState(), self_, output_, buffer_);
                break;
            }
            case ScalarType::Half: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_log_sigmoid_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfLogSigmoid_updateOutput(globalContext().getTHCState(), self_, output_, buffer_);
                break;
            }
            default:
                AT_ERROR("_thnn_log_sigmoid_forward not supported on CUDAType for ", dispatch_scalar_type);
        }
        return std::tuple<Tensor, Tensor>(output, buffer);
        */
}

pub fn thnn_log_sigmoid_backward_out(
    grad_output: &Tensor,
    self_:       &Tensor,
    buffer:      &Tensor,
    grad_input:  &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto buffer_ = checked_dense_tensor_unwrap(buffer, "buffer", 3, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 3, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleLogSigmoid_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, buffer_);
                break;
            }
            case ScalarType::Float: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto buffer_ = checked_dense_tensor_unwrap(buffer, "buffer", 3, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 3, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaLogSigmoid_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, buffer_);
                break;
            }
            case ScalarType::Half: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto buffer_ = checked_dense_tensor_unwrap(buffer, "buffer", 3, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 3, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfLogSigmoid_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, buffer_);
                break;
            }
            default:
                AT_ERROR("_thnn_log_sigmoid_backward_out not supported on CUDAType for ", dispatch_scalar_type);
        }
        return grad_input;
        */
}

pub fn thnn_log_sigmoid_backward(
    grad_output: &Tensor,
    self_:       &Tensor,
    buffer:      &Tensor) -> Tensor {
    
    todo!();
        /*
            const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto grad_input_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto grad_input = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_input_));
        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_log_sigmoid_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_log_sigmoid_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto buffer_ = checked_dense_tensor_unwrap(buffer, "buffer", 3, "_thnn_log_sigmoid_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleLogSigmoid_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, buffer_);
                break;
            }
            case ScalarType::Float: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_log_sigmoid_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_log_sigmoid_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto buffer_ = checked_dense_tensor_unwrap(buffer, "buffer", 3, "_thnn_log_sigmoid_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaLogSigmoid_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, buffer_);
                break;
            }
            case ScalarType::Half: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_log_sigmoid_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_log_sigmoid_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto buffer_ = checked_dense_tensor_unwrap(buffer, "buffer", 3, "_thnn_log_sigmoid_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfLogSigmoid_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, buffer_);
                break;
            }
            default:
                AT_ERROR("_thnn_log_sigmoid_backward not supported on CUDAType for ", dispatch_scalar_type);
        }
        return grad_input;
        */
}

pub fn thnn_conv2d_forward_out(
    self_:       &Tensor,
    weight:      &Tensor,
    kernel_size: &[i32],
    bias_opt:    &Option<Tensor>,
    stride:      &[i32],
    padding:     &[i32],
    output:      &mut Tensor,
    columns:     &mut Tensor,
    ones:        &mut Tensor) -> (&mut Tensor,&mut Tensor,&mut Tensor) {

    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

        const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
                auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleSpatialConvolutionMM_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
                auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaSpatialConvolutionMM_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
                break;
            }
            case ScalarType::Half: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
                auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfSpatialConvolutionMM_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
                break;
            }
            case ScalarType::BFloat16: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
                auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaBFloat16SpatialConvolutionMM_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
                break;
            }
            default:
                AT_ERROR("_thnn_conv2d_forward_out not supported on CUDAType for ", dispatch_scalar_type);
        }
        return std::tuple<Tensor &, Tensor &, Tensor &>(output, columns, ones);
        */
}

pub fn thnn_conv2d_forward(
    self_:       &Tensor,
    weight:      &Tensor,
    kernel_size: &[i32],
    bias_opt:    &Option<Tensor>,
    stride:      &[i32],
    padding:     &[i32]) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

        const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto output_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto output = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(output_));
        auto columns_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto columns = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(columns_));
        auto ones_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto ones = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(ones_));
        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
                auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                THNN_CudaDoubleSpatialConvolutionMM_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
                auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                THNN_CudaSpatialConvolutionMM_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
                break;
            }
            case ScalarType::Half: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
                auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                THNN_CudaHalfSpatialConvolutionMM_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
                break;
            }
            case ScalarType::BFloat16: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
                auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                THNN_CudaBFloat16SpatialConvolutionMM_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
                break;
            }
            default:
                AT_ERROR("_thnn_conv2d_forward not supported on CUDAType for ", dispatch_scalar_type);
        }
        return std::tuple<Tensor, Tensor, Tensor>(output, columns, ones);
        */
}


pub fn thnn_conv2d_backward_out(
        grad_input:  &mut Tensor,
        grad_weight: &mut Tensor,
        grad_bias:   &mut Tensor,
        grad_output: &Tensor,
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        columns:     &Tensor,
        ones:        &Tensor) -> (&mut Tensor,&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 7, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 8, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_weight_ = checked_dense_tensor_unwrap(grad_weight, "grad_weight", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_bias_ = checked_dense_tensor_unwrap(grad_bias, "grad_bias", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                if (grad_input_) THNN_CudaDoubleSpatialConvolutionMM_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
                if (grad_weight_ || grad_bias_) THNN_CudaDoubleSpatialConvolutionMM_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, grad_bias_ ? grad_bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
                break;
            }
            case ScalarType::Float: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 7, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 8, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_weight_ = checked_dense_tensor_unwrap(grad_weight, "grad_weight", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_bias_ = checked_dense_tensor_unwrap(grad_bias, "grad_bias", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                if (grad_input_) THNN_CudaSpatialConvolutionMM_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
                if (grad_weight_ || grad_bias_) THNN_CudaSpatialConvolutionMM_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, grad_bias_ ? grad_bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
                break;
            }
            case ScalarType::Half: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 7, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 8, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_weight_ = checked_dense_tensor_unwrap(grad_weight, "grad_weight", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_bias_ = checked_dense_tensor_unwrap(grad_bias, "grad_bias", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                if (grad_input_) THNN_CudaHalfSpatialConvolutionMM_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
                if (grad_weight_ || grad_bias_) THNN_CudaHalfSpatialConvolutionMM_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, grad_bias_ ? grad_bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
                break;
            }
            case ScalarType::BFloat16: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 7, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 8, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_weight_ = checked_dense_tensor_unwrap(grad_weight, "grad_weight", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_bias_ = checked_dense_tensor_unwrap(grad_bias, "grad_bias", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                if (grad_input_) THNN_CudaBFloat16SpatialConvolutionMM_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
                if (grad_weight_ || grad_bias_) THNN_CudaBFloat16SpatialConvolutionMM_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, grad_bias_ ? grad_bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
                break;
            }
            default:
                AT_ERROR("_thnn_conv2d_backward_out not supported on CUDAType for ", dispatch_scalar_type);
        }
        return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
        */
}



pub fn thnn_conv2d_backward(
        grad_output: &Tensor,
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        columns:     &Tensor,
        ones:        &Tensor,
        output_mask: Array3<bool>) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto grad_input_ = output_mask[0] ? c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release() : nullptr;
        auto grad_input = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensorImpl::singleton() : (TensorImpl*)grad_input_));
        auto grad_weight_ = output_mask[1] ? c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release() : nullptr;
        auto grad_weight = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensorImpl::singleton() : (TensorImpl*)grad_weight_));
        auto grad_bias_ = output_mask[2] ? c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release() : nullptr;
        auto grad_bias = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensorImpl::singleton() : (TensorImpl*)grad_bias_));
        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 7, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 8, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                if (grad_input_) THNN_CudaDoubleSpatialConvolutionMM_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
                if (grad_weight_ || grad_bias_) THNN_CudaDoubleSpatialConvolutionMM_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, grad_bias_ ? grad_bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
                break;
            }
            case ScalarType::Float: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 7, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 8, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                if (grad_input_) THNN_CudaSpatialConvolutionMM_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
                if (grad_weight_ || grad_bias_) THNN_CudaSpatialConvolutionMM_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, grad_bias_ ? grad_bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
                break;
            }
            case ScalarType::Half: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 7, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 8, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                if (grad_input_) THNN_CudaHalfSpatialConvolutionMM_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
                if (grad_weight_ || grad_bias_) THNN_CudaHalfSpatialConvolutionMM_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, grad_bias_ ? grad_bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
                break;
            }
            case ScalarType::BFloat16: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 7, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 8, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                if (grad_input_) THNN_CudaBFloat16SpatialConvolutionMM_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
                if (grad_weight_ || grad_bias_) THNN_CudaBFloat16SpatialConvolutionMM_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, grad_bias_ ? grad_bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
                break;
            }
            default:
                AT_ERROR("_thnn_conv2d_backward not supported on CUDAType for ", dispatch_scalar_type);
        }
        return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
        */
}


pub fn thnn_conv_depthwise2d_forward_out(
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        bias_opt:    &Option<Tensor>,
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32],
        output:      &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

        const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
                auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv_depthwise2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 7, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaDoubleSpatialDepthwiseConvolution_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
                auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv_depthwise2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 7, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaSpatialDepthwiseConvolution_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                break;
            }
            case ScalarType::Half: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
                auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv_depthwise2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 7, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaHalfSpatialDepthwiseConvolution_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                break;
            }
            case ScalarType::BFloat16: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
                auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv_depthwise2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
                auto output_ = checked_dense_tensor_unwrap(output, "output", 7, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                THNN_CudaBFloat16SpatialDepthwiseConvolution_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                break;
            }
            default:
                AT_ERROR("_thnn_conv_depthwise2d_forward_out not supported on CUDAType for ", dispatch_scalar_type);
        }
        return output;
        */
}


pub fn thnn_conv_depthwise2d_forward(
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        bias_opt:    &Option<Tensor>,
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32]) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

        const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto output_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
        auto output = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(output_));
        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv_depthwise2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv_depthwise2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
                auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv_depthwise2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
                THNN_CudaDoubleSpatialDepthwiseConvolution_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                break;
            }
            case ScalarType::Float: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv_depthwise2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv_depthwise2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
                auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv_depthwise2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
                THNN_CudaSpatialDepthwiseConvolution_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                break;
            }
            case ScalarType::Half: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv_depthwise2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv_depthwise2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
                auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv_depthwise2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
                THNN_CudaHalfSpatialDepthwiseConvolution_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                break;
            }
            case ScalarType::BFloat16: {
                auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv_depthwise2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv_depthwise2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
                auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv_depthwise2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
                THNN_CudaBFloat16SpatialDepthwiseConvolution_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                break;
            }
            default:
                AT_ERROR("_thnn_conv_depthwise2d_forward not supported on CUDAType for ", dispatch_scalar_type);
        }
        return output;
        */
}

pub fn thnn_conv_depthwise2d_backward_out(
    grad_input:  &mut Tensor,
    grad_weight: &mut Tensor,
    grad_output: &Tensor,
    self_:       &Tensor,
    weight:      &Tensor,
    kernel_size: &[i32],
    stride:      &[i32],
    padding:     &[i32],
    dilation:    &[i32]) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);

        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_conv_depthwise2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_weight_ = checked_dense_tensor_unwrap(grad_weight, "grad_weight", 7, "_thnn_conv_depthwise2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                if (grad_input_) THNN_CudaDoubleSpatialDepthwiseConvolution_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                if (grad_weight_) THNN_CudaDoubleSpatialDepthwiseConvolution_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                break;
            }
            case ScalarType::Float: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_conv_depthwise2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_weight_ = checked_dense_tensor_unwrap(grad_weight, "grad_weight", 7, "_thnn_conv_depthwise2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                if (grad_input_) THNN_CudaSpatialDepthwiseConvolution_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                if (grad_weight_) THNN_CudaSpatialDepthwiseConvolution_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                break;
            }
            case ScalarType::Half: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_conv_depthwise2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_weight_ = checked_dense_tensor_unwrap(grad_weight, "grad_weight", 7, "_thnn_conv_depthwise2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                if (grad_input_) THNN_CudaHalfSpatialDepthwiseConvolution_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                if (grad_weight_) THNN_CudaHalfSpatialDepthwiseConvolution_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                break;
            }
            case ScalarType::BFloat16: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
                auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_conv_depthwise2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                auto grad_weight_ = checked_dense_tensor_unwrap(grad_weight, "grad_weight", 7, "_thnn_conv_depthwise2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
                if (grad_input_) THNN_CudaBFloat16SpatialDepthwiseConvolution_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                if (grad_weight_) THNN_CudaBFloat16SpatialDepthwiseConvolution_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                break;
            }
            default:
                AT_ERROR("_thnn_conv_depthwise2d_backward_out not supported on CUDAType for ", dispatch_scalar_type);
        }
        return std::tuple<Tensor &, Tensor &>(grad_input, grad_weight);
        */
}

pub fn thnn_conv_depthwise2d_backward(
    grad_output: &Tensor,
    self_:       &Tensor,
    weight:      &Tensor,
    kernel_size: &[i32],
    stride:      &[i32],
    padding:     &[i32],
    dilation:    &[i32],
    output_mask: Array2<bool>) -> (Tensor,Tensor) {

    todo!();
    /*
            const OptionalDeviceGuard device_guard(device_of(self));
        auto dispatch_scalar_type = infer_scalar_type(self);
        auto grad_input_ = output_mask[0] ? c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release() : nullptr;
        auto grad_input = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensorImpl::singleton() : (TensorImpl*)grad_input_));
        auto grad_weight_ = output_mask[1] ? c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release() : nullptr;
        auto grad_weight = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensorImpl::singleton() : (TensorImpl*)grad_weight_));
        switch (dispatch_scalar_type) {
            case ScalarType::Double: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
                if (grad_input_) THNN_CudaDoubleSpatialDepthwiseConvolution_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                if (grad_weight_) THNN_CudaDoubleSpatialDepthwiseConvolution_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                break;
            }
            case ScalarType::Float: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
                if (grad_input_) THNN_CudaSpatialDepthwiseConvolution_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                if (grad_weight_) THNN_CudaSpatialDepthwiseConvolution_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                break;
            }
            case ScalarType::Half: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
                if (grad_input_) THNN_CudaHalfSpatialDepthwiseConvolution_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                if (grad_weight_) THNN_CudaHalfSpatialDepthwiseConvolution_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                break;
            }
            case ScalarType::BFloat16: {
                auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
                auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
                auto stride_ = check_intlist<2>(stride, "stride", 5);
                auto padding_ = check_intlist<2>(padding, "padding", 6);
                auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
                if (grad_input_) THNN_CudaBFloat16SpatialDepthwiseConvolution_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                if (grad_weight_) THNN_CudaBFloat16SpatialDepthwiseConvolution_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
                break;
            }
            default:
                AT_ERROR("_thnn_conv_depthwise2d_backward not supported on CUDAType for ", dispatch_scalar_type);
        }
        return std::tuple<Tensor, Tensor>(grad_input, grad_weight);
        */
}

