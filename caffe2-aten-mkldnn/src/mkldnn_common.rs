crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkldnn/MKLDNNCommon.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkldnn/MKLDNNCommon.cpp]

#[cfg(feature = "mkldnn")]
pub mod mkldnn_enabled {

    use super::*;

    /**
      | `IntrusivePtrTargetWrapper` wraps
      | a custom storage handle of a tensor (as
      | template param) and inherits `c10::intrusive_ptr_target`
      | so that it can be used with `c10::intrusive_ptr`.
      | 
      | It currently only supports wrapping
      | the custom handle by:
      | 
      | - Constructing with an existing custom
      | handle by copy/move constructor.
      | 
      | See `OpaqueTensorImpl::opaque_handle_`.
      | 
      | -----------
      | @note
      | 
      | if this is generally useful we may want
      | to move this to its own header.
      |
      */
    pub struct IntrusivePtrTargetWrapper<T> {
        base:   IntrusivePtrTarget,
        target: T,
    }

    impl IntrusivePtrTargetWrapper<T> {

        pub fn new(target: &T) -> Self {
        
            todo!();
            /*
            : target(target),

            
            */
        }
        
        pub fn new(target: T) -> Self {
        
            todo!();
            /*
            : target(std::move(target)),

            
            */
        }
        
        pub fn get_target(&mut self) -> &mut T {
            
            todo!();
            /*
                return target_;
            */
        }
    }

    pub type IDeepTensorWrapper    = IntrusivePtrTargetWrapper<IDEEP::Tensor>;
    pub type IDeepTensorWrapperPtr = IntrusivePtr<IDeepTensorWrapper>;
    pub type MKLDNNTensorImpl      = OpaqueTensorImpl<IDeepTensorWrapperPtr>;
    pub type MKLDNNTensor          = Tensor;

    /**
      | Mapping ScalarType to ideep tensor
      | data_type
      |
      */
    pub fn get_mkldnn_dtype(ty: ScalarType) -> DataType {
        
        todo!();
            /*
                switch (type) {
            case ScalarType::Float:
              return ideep::tensor::data_type::f32;
            case ScalarType::QInt32:
              return ideep::tensor::data_type::s32;
            case ScalarType::QInt8:
              return ideep::tensor::data_type::s8;
            case ScalarType::QUInt8:
            case ScalarType::Byte:
              return ideep::tensor::data_type::u8;
            case ScalarType::BFloat16:
              return ideep::tensor::data_type::bf16;
            default:
              TORCH_CHECK(false, "get_mkldnn_dtype: unsupported data type");
          }
            */
    }

    /**
      | Construct aten MKL-DNN tensor given
      | an ideep tensor
      |
      */
    pub fn new_with_itensor_mkldnn(
            it:     IDEEP::Tensor,
            dtype:  Option<ScalarType>,
            device: Option<Device>) -> Tensor {
        
        todo!();
            /*
                // NOTE: i32 dims from ideep::tensor but sizes needs i64
          // TODO: support i64 dims in ideep::tensor to avoid extra conversion
          auto dims = it.get_dims();
          IDeepTensorWrapperPtr handle = make_intrusive<IDeepTensorWrapper>(std::move(it));
          caffe2::TypeMeta dtype_ = scalarTypeToTypeMeta(dtype_or_default(dtype));
          Device device_ = device_or_default(device);
          return detail::make_tensor<MKLDNNTensorImpl>(
            DispatchKeySet(DispatchKey::MkldnnCPU),
            dtype_, device_, handle,
            std::vector<i64>(dims.begin(), dims.end()));
            */
    }

    // Retrieve `ideep::tensor` from MKL-DNN tensor
    pub fn itensor_from_mkldnn(mkldnn_tensor: &MKLDNNTensor) -> &mut IDEEP::Tensor {
        
        todo!();
            /*
                TORCH_CHECK(mkldnn_tensor.is_mkldnn(),
                     "itensor_from_mkldnn expects MKL-DNN tensor input");
          TORCH_INTERNAL_ASSERT(at::impl::variable_excluded_from_dispatch());
          MKLDNNTensorImpl *mklimpl = static_cast<MKLDNNTensorImpl *>(mkldnn_tensor.unsafeGetTensorImpl());
          return mklimpl->unsafe_opaque_handle()->get_target();
            */
    }

    /**
      | Construct an `ideep::tensor` "view"
      | from dense tensor, note the ideep::tensor
      | will share the underlying buffer
      |
      */
    pub fn itensor_view_from_dense(tensor: &Tensor) -> IDEEP::Tensor {
        
        todo!();
            /*
                TORCH_CHECK(
              tensor.device().is_cpu(),
              "itensor_view_from_dense expects CPU tensor input");
          TORCH_CHECK(
              tensor.layout() == Layout::Strided,
              "itensor_view_from_dense expects dense tensor input");
          TORCH_CHECK(tensor.scalar_type() == ScalarType::Float,
                     "itensor_view_from_dense expects float tensor input");
          TORCH_INTERNAL_ASSERT(at::impl::variable_excluded_from_dispatch());
          return {{{tensor.sizes().cbegin(), tensor.sizes().cend()},
                   ideep::tensor::data_type::f32},
                  tensor.template data_ptr<float>()};
            */
    }

    /**
      | Helper function for getting an ideep
      | tensor out of an aten Tensor or MKL-DNN
      | tensor.
      |
      | Note in case the aten Tensor is a dense tensor,
      | the returned ideep tensor is just a view of the
      | storage of the aten dense tensor, so caller
      | needs to make sure the aten dense tensor's
      | lifetime is longer than the ideep tensor.
      */
    pub fn itensor_from_tensor(tensor: &Tensor) -> IDEEP::Tensor {
        
        todo!();
            /*
                if (tensor.is_mkldnn()) {
            return itensor_from_mkldnn(tensor);
          } else {
            return itensor_view_from_dense(tensor);
          }
            */
    }
}
