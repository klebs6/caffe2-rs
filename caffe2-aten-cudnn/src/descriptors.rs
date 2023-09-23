crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cudnn/Descriptors.h]

// TODO: Add constructors for all of the
// descriptors
//
#[inline] pub fn data_size(data_type: cudnn::DataType) -> i32 {
    
    todo!();
        /*
            switch (dataType) {
        case CUDNN_DATA_HALF: return 2;
        case CUDNN_DATA_FLOAT: return 4;
        default: return 8;
      }
        */
}

/**
  | The stride for a size-1 dimensions is not
  | uniquely determined; in fact, it can be
  | anything you want, because the fact that the
  | tensor is size 1 at this dimension means that
  | you will never actually try advancing your
  | pointer by this stride.
  |
  | However, CuDNN has a much more stringent
  | requirement on strides: if you are passing
  | a contiguous input, it better be the case that
  | the stride for dim i is the product of the
  | sizes of dims i+1 to the end.  This stride is
  | indeed uniquely determined.  This function
  | modifies 'stride' in place so this invariant
  | holds.
  */
#[inline] pub fn fix_size_one_dim_stride(
        dim:    i32,
        size:   *const i32,
        stride: *mut i32,
        nhwc:   bool)  {
    
    todo!();
        /*
            i64 z = 1;
      int index = 0;
      vector<int> permutation(dim);

      if (nhwc) {
        permutation[index++] = 1;
      }
      for (int d = dim-1; d > 1; d--) {
        permutation[index++] = d;
      }
      if (!nhwc) {
        permutation[index++] = 1;
      }
      permutation[index++] = 0;
      for (int d : permutation) {
        if (size[d] == 1) {
          stride[d] = z;
        } else {
          z *= size[d];
        }
      }
        */
}

pub struct DescriptorDeleter {

}

impl DescriptorDeleter {
    
    //template <typename T, cudnnStatus_t (*dtor)(T*)>
    pub fn invoke(&mut self, x: *mut T)  {
        
        todo!();
        /*
            if (x != nullptr) {
          AT_CUDNN_CHECK(dtor(x));
        }
        */
    }
}

/**
  | A generic class for wrapping cuDNN descriptor
  | types.
  |
  | All you need is to give the underlying type the
  | Descriptor_t points to (usually, if it's
  | cudnnTensorDescriptor_t it points to
  | cudnnTensorStruct), the constructor and the
  | destructor.
  |
  | Subclasses are responsible for defining a set()
  | function to actually set the descriptor.
  |
  | Descriptors default construct to a nullptr, and
  | have a descriptor initialized the first time
  | you call set() or any other initializing
  | function.
  */
//template <typename T, cudnnStatus_t (*ctor)(T**), cudnnStatus_t (*dtor)(T*)>
pub struct Descriptor {
    desc: Box<T,DescriptorDeleter<T,Dtor>>,
}

impl Descriptor {
    
    /**
      | TODO: Figure out why const-correctness
      | doesn't work here
      | 
      | Use desc() to access the underlying
      | descriptor pointer in a read-only fashion.
      | 
      | Most client code should use this.
      | 
      | If the descriptor was never initialized,
      | this will return nullptr.
      |
      */
    pub fn desc(&self) -> *mut T {
        
        todo!();
        /*
            return desc_.get();
        */
    }
    
    pub fn desc(&mut self) -> *mut T {
        
        todo!();
        /*
            return desc_.get();
        */
    }
    
    /**
      | Use mut_desc() to access the underlying
      | descriptor pointer if you intend to
      | modify what it points to (e.g., using
      | cudnnSetFooDescriptor).
      | 
      | This will ensure that the descriptor
      | is initialized. Code in this file will
      | use this function.
      |
      */
    pub fn mut_desc(&mut self) -> *mut T {
        
        todo!();
        /*
            init(); return desc_.get();
        */
    }
    
    pub fn init(&mut self)  {
        
        todo!();
        /*
            if (desc_ == nullptr) {
          T* raw_desc;
          AT_CUDNN_CHECK(ctor(&raw_desc));
          desc_.reset(raw_desc);
        }
        */
    }
}

pub struct TensorDescriptor {
    base: Descriptor<CuDnnTensorStruct,CuDnnCreateTensorDescriptor,CuDnnDestroyTensorDescriptor>,
}

impl TensorDescriptor {
    
    pub fn new(
        t:   &Tensor,
        pad: usize) -> Self {
        let pad: usize = pad.unwrap_or(0);
        todo!();
        /*


            set(t, pad);
        */
    }

    /**
      | Note [CuDNN broadcast padding]
      | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |
      | pad specifies the minimum dimensionality of
      | the tensor descriptor we produce (it doesn't
      | have anything to do with, e.g., convolution
      | padding).  If 't' is lower-dimensional than
      | 'pad', the remaining dimensions (on the
      | right) are padded with ones.
      |
      | This doesn't affect the underlying data
      | layout.  This is particularly useful for
      | dealing with a pecularity of the CuDNN API,
      | which is that broadcasting in CuDNN is done
      | in two steps: first, the client code is
      | expected to pad out (the dimensions) input
      | tensors to be the same dimension as the
      | target broadcast, and then second, CuDNN
      | takes of actually broadcasting size
      | 1 dimensions.
      */
    pub fn set(&mut self, 
        t:   &Tensor,
        pad: usize)  {
        let pad: usize = pad.unwrap_or(0);

        todo!();
        /*
        
        */
    }
    
    pub fn set(&mut self, 
        data_type: cudnn::DataType,
        sizes:     &[i32],
        strides:   &[i32],
        pad:       usize)  {
        let pad: usize = pad.unwrap_or(0);

        todo!();
        /*
        
        */
    }
    
    pub fn print(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set(&mut self, 
        data_type: cudnn::DataType,
        sizes:     &[i32],
        strides:   &[i32],
        pad:       usize,
        nhwc:      bool)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set(&mut self, 
        data_type: cudnn::DataType,
        dim:       i32,
        size:      *mut i32,
        stride:    *mut i32,
        nhwc:      bool)  {
        
        todo!();
        /*
            fixSizeOneDimStride(dim, size, stride, nhwc);
        AT_CUDNN_CHECK(cudnnSetTensorNdDescriptor(mut_desc(), dataType, dim, size, stride));
        */
    }
}

//------------------------------------------
pub struct FilterDescriptor {
    base: Descriptor<CuDnnFilterStruct,CuDnnCreateFilterDescriptor,CuDnnDestroyFilterDescriptor>,
}

impl FilterDescriptor {
    
    pub fn set(&mut self, 
        t:   &Tensor,
        pad: i64)  {
        let pad: i64 = pad.unwrap_or(0);

        todo!();
        /*
            set(t, MemoryFormat::Contiguous, pad);
        */
    }
    
    pub fn set(&mut self, 
        t:             &Tensor,
        memory_format: MemoryFormat,
        pad:           i64)  {
        let pad: i64 = pad.unwrap_or(0);

        todo!();
        /*
        
        */
    }
    
    pub fn print(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set(&mut self, 
        data_type:     cudnn::DataType,
        dim:           i32,
        size:          *mut i32,
        filter_format: cudnn::TensorFormat)  {
        
        todo!();
        /*
            AT_CUDNN_CHECK(cudnnSetFilterNdDescriptor(mut_desc(), dataType, filter_format, dim, size));
        */
    }
}

pub struct ConvolutionDescriptor {
    base: Descriptor<CuDnnConvolutionStruct,CuDnnCreateConvolutionDescriptor,CuDnnDestroyConvolutionDescriptor>,
}

impl ConvolutionDescriptor {
    
    pub fn set(&mut self, 
        data_type:            cudnn::DataType,
        dim:                  i32,
        pad:                  *mut i32,
        stride:               *mut i32,
        upscale_aka_dilation: *mut i32,
        groups:               i32,
        allow_tf32:           bool)  {
        
        todo!();
        /*
            cudnnDataType_t mathType = dataType;
        if (dataType == CUDNN_DATA_HALF) mathType = CUDNN_DATA_FLOAT;
        AT_CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(mut_desc(), dim, pad, stride, upscale,
                                              CUDNN_CROSS_CORRELATION, mathType));
        AT_CUDNN_CHECK(cudnnSetConvolutionGroupCount(mut_desc(), groups));
        // See Note [behavior of cudnnFind and cudnnGet]
        AT_CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(), CUDNN_DEFAULT_MATH));
        if(dataType == CUDNN_DATA_HALF) {
          AT_CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(), CUDNN_TENSOR_OP_MATH));
        } else if (dataType == CUDNN_DATA_FLOAT && !allow_tf32) {
    #if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8000
          AT_CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(), CUDNN_FMA_MATH));
    #endif
        }
        */
    }
}

//-----------------------------------------
pub struct SpatialTransformerDescriptor {
    base: Descriptor<CuDnnSpatialTransformerStruct,CuDnnCreateSpatialTransformerDescriptor,CuDnnDestroySpatialTransformerDescriptor>,
}

impl SpatialTransformerDescriptor {
    
    pub fn set(&mut self, 
        data_type: cudnn::DataType,
        dim:       i32,
        size:      *mut i32)  {
        
        todo!();
        /*
            AT_CUDNN_CHECK(cudnnSetSpatialTransformerNdDescriptor(mut_desc(), CUDNN_SAMPLER_BILINEAR, dataType, dim, size));
        */
    }
}

pub struct DropoutDescriptor {
    base:  Descriptor<CuDnnDropoutStruct,CuDnnCreateDropoutDescriptor,CuDnnDestroyDropoutDescriptor>,
    state: Tensor,
}

impl DropoutDescriptor {

    /**
      | Initialize a dropout descriptor's RNG state.
      |
      | WARNING: This function is very expensive,
      | avoid calling this function!
      |
      */
    pub fn initialize_rng(&mut self, 
        handle:  CuDnnHandle,
        dropout: f32,
        seed:    i64,
        options: &TensorOptions)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(dropout > 0, "dropout must be nonzero; otherwise call set_no_dropout");
        usize state_size;
        AT_CUDNN_CHECK(cudnnDropoutGetStatesSize(handle, &state_size));
        AT_ASSERT(options.device().type() == kCUDA);
        AT_ASSERT(options.dtype() == kByte);
        state = empty({static_cast<i64>(state_size)}, options);
        AT_CUDNN_CHECK(cudnnSetDropoutDescriptor(mut_desc(), handle, dropout, state.data_ptr(), state_size, seed));
        */
    }
    
    /**
      | Restore a dropout descriptor given
      | a dropout probability and existing
      | RNG state.
      |
      */
    pub fn set(&mut self, 
        handle:  CuDnnHandle,
        dropout: f32,
        state:   Tensor)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(dropout > 0, "dropout must be nonzero; otherwise call set_no_dropout");
        state = state_;
        void *state_ptr = state.data_ptr();
        usize state_size = state.size(0);
        // NB: The seed doesn't actually matter, so we give a dummy value
        AT_CUDNN_CHECK(cudnnRestoreDropoutDescriptor(mut_desc(), handle, dropout, state_ptr, state_size, 0 /* seed */));
        */
    }

    /// Restore a dropout descriptor corresponding
    /// to no dropout
    ///
    pub fn set_no_dropout(&mut self, handle: CuDnnHandle)  {
        
        todo!();
        /*
            // NB: seed doesn't matter when dropout = 0, because no random number
        // initialization actually takes place when there is no dropout.
        // NB: Empirically, cudnnSetDropoutDescriptor is cheap when
        // dropoot == 0
        AT_CUDNN_CHECK(cudnnSetDropoutDescriptor(mut_desc(), handle, 0 /* dropout */, nullptr, 0 /* state_size */, 0 /* seed */));
        */
    }
}

//-----------------------------------
pub struct RNNDescriptor {
    base:         Descriptor<CuDnnRNNStruct,CuDnnCreateRNNDescriptor,CuDnnDestroyRNNDescriptor>,
    dropout_desc: DropoutDescriptor,
}

impl RNNDescriptor {
    
    pub fn set(&mut self, 
        handle:        CuDnnHandle,
        hidden_size:   i32,
        proj_size:     i32,
        num_layers:    i32,
        dropout_desc:  DropoutDescriptor,
        input_mode:    CudnnRNNInputMode,
        bidirectional: CudnnDirectionMode,
        mode:          CudnnRNNMode,
        datatype:      CudnnDataType,
        input_type:    CudnnDataType,
        algo:          CudnnRNNAlgo,
        allow_tf32:    bool)  {
        
        todo!();
        /*
            dropout_desc_ = move(dropout_desc);

        AT_CUDNN_CHECK(cudnnSetRNNDescriptor_v6(
              handle,
              mut_desc(),
              hidden_size,
              num_layers,
              dropout_desc_.desc(),
              input_mode,
              bidirectional,
              mode,
              algo,
              datatype));
        if (proj_size != 0) {
          AT_CUDNN_CHECK(cudnnSetRNNProjectionLayers(
                handle,
                /*rnnDesc=*/mut_desc(),
                /*recProjSize=*/proj_size,
                /*outProjSize=*/0));
        }
        cudaDeviceProp* prop = getCurrentDeviceProperties();
        if (prop->major >= 7) {
          if (input_type == CUDNN_DATA_HALF) {
            cudnnSetRNNMatrixMathType(mut_desc(), CUDNN_TENSOR_OP_MATH);
          }
    #if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8000
          else if (input_type == CUDNN_DATA_FLOAT && !allow_tf32) {
            cudnnSetRNNMatrixMathType(mut_desc(), CUDNN_FMA_MATH);
          }
    #endif
          else {
            // Technically, as the default it's not necessary to explicitly
            // set this.
            cudnnSetRNNMatrixMathType(mut_desc(), CUDNN_DEFAULT_MATH);
          }
        }
        */
    }
}

pub struct CTCLossDescriptor {
    base: Descriptor<CudnnCTCLossStruct,CudnnCreateCTCLossDescriptor,CudnnDestroyCTCLossDescriptor>,
}

impl CTCLossDescriptor {

    pub fn set(&mut self, datatype: CudnnDataType)  {
        
        todo!();
        /*
            AT_CUDNN_CHECK(cudnnSetCTCLossDescriptor(mut_desc(), datatype));
        */
    }

    #[cfg(CUDNN_VERSION_gte_7600)]
    pub fn set_ex(&mut self, 
        datatype:  CudnnDataType,
        norm_mode: CudnnLossNormalizationMode,
        grad_mode: CudnnNanPropagation)  {
        
        todo!();
        /*
            AT_CUDNN_CHECK(
            cudnnSetCTCLossDescriptorEx(mut_desc(), datatype, normMode, gradMode));
        */
    }
}

//----------------------------------------
pub struct ActivationDescriptor {
    base: Descriptor<CudnnActivationStruct,CudnnCreateActivationDescriptor,CudnnDestroyActivationDescriptor>,
}

impl ActivationDescriptor {
    
    pub fn set(&mut self, mode: CudnnActivationMode)  {
        
        todo!();
        /*
            AT_ASSERT(
            mode == CUDNN_ACTIVATION_RELU,
            "TODO: support more cuDNN activation modes");
        AT_CUDNN_CHECK(cudnnSetActivationDescriptor(
            mut_desc(),
            mode,
            cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
            double::max));
        */
    }
}

//-----------------------------
pub union Constant
{
    f: f32,
    d: f64,
}

impl Constant {

    pub fn new(
        data_type: CudnnDataType,
        value:     f64) -> Self {
    
        todo!();
        /*


            if (dataType == CUDNN_DATA_HALF || dataType == CUDNN_DATA_FLOAT) {
          f = static_cast<float>(value);
        } else {
          d = value;
        }
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cudnn/Descriptors.cpp]

#[inline] pub fn get_data_type(t: &Tensor) -> CudnnDataType {
    
    todo!();
        /*
            auto scalar_type = t.scalar_type();
      if (scalar_type == kFloat) {
        return CUDNN_DATA_FLOAT;
      } else if (scalar_type == kHalf) {
        return CUDNN_DATA_HALF;
      } else if (scalar_type == kDouble) {
        return CUDNN_DATA_DOUBLE;
      }
      throw runtime_error("TensorDescriptor only supports double, float and half tensors");
        */
}

impl TensorDescriptor {
    
    pub fn set(&mut self, 
        t:   &Tensor,
        pad: usize)  {
        
        todo!();
        /*
            auto memory_format = t.suggest_memory_format();
      set(getDataType(t), t.sizes(), t.strides(), pad,
        memory_format == MemoryFormat::ChannelsLast ||
        memory_format == MemoryFormat::ChannelsLast3d);
        */
    }
    
    pub fn set(&mut self, 
        datatype:  CudnnDataType,
        t_sizes:   &[i32],
        t_strides: &[i32],
        pad:       usize)  {
        
        todo!();
        /*
            set(datatype, t_sizes, t_strides, pad,
        is_channels_last_strides_2d(t_sizes, t_strides) ||
        is_channels_last_strides_3d(t_sizes, t_strides));
        */
    }
    
    pub fn set(&mut self, 
        datatype:  CudnnDataType,
        t_sizes:   &[i32],
        t_strides: &[i32],
        pad:       usize,
        nhwc:      bool)  {
        
        todo!();
        /*
            usize dim = t_sizes.size();
      if (dim > CUDNN_DIM_MAX || pad > CUDNN_DIM_MAX)
    #define _STR(X) #X
    #define STR(X) _STR(X)
        throw runtime_error("cuDNN supports only up to " STR(CUDNN_DIM_MAX) " dimensions");
    #undef _STR
    #undef STR
      int size[CUDNN_DIM_MAX];
      int stride[CUDNN_DIM_MAX];
      for (usize i = 0; i < dim; ++i) {
        size[i] = static_cast<int>(t_sizes[i]);
        stride[i] = static_cast<int>(t_strides[i]);
      }
      for (usize i = dim; i < pad; ++i) {
        size[i] = 1;
        stride[i] = 1;
      }
      set(datatype, static_cast<int>(max(dim, pad)), size, stride, nhwc);
        */
    }
}

pub fn cudnn_type_to_string(dtype: CudnnDataType) -> String {
    
    todo!();
        /*
            switch (dtype) {
        case CUDNN_DATA_FLOAT:
          return "CUDNN_DATA_FLOAT";
        case CUDNN_DATA_DOUBLE:
          return "CUDNN_DATA_DOUBLE";
        case CUDNN_DATA_HALF:
          return "CUDNN_DATA_HALF";
        case CUDNN_DATA_INT8:
          return "CUDNN_DATA_INT8";
        case CUDNN_DATA_INT32:
          return "CUDNN_DATA_INT32";
        case CUDNN_DATA_INT8x4:
          return "CUDNN_DATA_INT8x4";
    #if CUDNN_VERSION >= 7100
        case CUDNN_DATA_UINT8:
          return "CUDNN_DATA_UINT8";
        case CUDNN_DATA_UINT8x4:
          return "CUDNN_DATA_UINT8x4";
    #endif
        default:
          ostringstream oss;
          oss << "(unknown data-type " << static_cast<int>(dtype) << ")";
          return oss.str();
      }
        */
}

impl fmt::Display for TensorDescriptor {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            out << "TensorDescriptor " << static_cast<void*>(d.desc()) << "\n";
      int nbDims;
      int dimA[CUDNN_DIM_MAX];
      int strideA[CUDNN_DIM_MAX];
      cudnnDataType_t dtype;
      cudnnGetTensorNdDescriptor(d.desc(), CUDNN_DIM_MAX, &dtype, &nbDims, dimA, strideA);
      out << "    type = " << cudnnTypeToString(dtype) << "\n";
      out << "    nbDims = " << nbDims << "\n";
      // Read out only nbDims of the arrays!
      out << "    dimA = ";
      for (auto i : ArrayRef<int>{dimA, static_cast<usize>(nbDims)}) {
        out << i << ", ";
      }
      out << "\n";
      out << "    strideA = ";
      for (auto i : ArrayRef<int>{strideA, static_cast<usize>(nbDims)}) {
        out << i << ", ";
      }
      out << "\n";
      return out;
        */
    }
}

impl TensorDescriptor {
    
    pub fn print(&mut self)  {
        
        todo!();
        /*
            cout << *this;
        */
    }
}

impl FilterDescriptor {
    
    pub fn set(&mut self, 
        t:             &Tensor,
        memory_format: MemoryFormat,
        pad:           i64)  {
        
        todo!();
        /*
            auto dim = t.ndimension();
      if (dim > CUDNN_DIM_MAX || pad > CUDNN_DIM_MAX)
    #define _STR(X) #X
    #define STR(X) _STR(X)
        throw runtime_error("cuDNN supports only up to " STR(CUDNN_DIM_MAX) " dimensions");
    #undef _STR
    #undef STR
      // NB: It is possible for this test to be insufficient, because the
      // Tensor passed in to set the filter descriptor may not be the actual
      // Tensor whose data pointer is passed to cuDNN.  Nevertheless,
      // that is the common case, so we can catch most client errors with this test.
      TORCH_CHECK(t.is_contiguous(memory_format),
          "cuDNN filters (a.k.a. weights) must be contiguous in desired memory_format");

      int size[CUDNN_DIM_MAX];
      for (int i = 0; i < dim; ++i) {
        size[i] = (int) t.size(i);
      }
      for (int i = dim; i < pad; ++i) {
        size[i] = (int) 1;
      }
      dim = max(dim, pad);
      cudnnTensorFormat_t filter_format;
      switch(memory_format) {
        case MemoryFormat::Contiguous:
          filter_format = CUDNN_TENSOR_NCHW;
          break;
        case MemoryFormat::ChannelsLast:
        case MemoryFormat::ChannelsLast3d:
          filter_format = CUDNN_TENSOR_NHWC;
          break;
        default:
          TORCH_INTERNAL_ASSERT(false, "unsurpported memory_format for cuDNN filters");
      }
      set(getDataType(t), (int) dim, size, filter_format);
        */
    }
}

pub fn cudnn_memory_format_to_string(tformat: cudnn::TensorFormat) -> String {
    
    todo!();
        /*
            switch (tformat) {
        case CUDNN_TENSOR_NCHW:
          return "CUDNN_TENSOR_NCHW";
        case CUDNN_TENSOR_NHWC:
          return "CUDNN_TENSOR_NHWC";
        default:
          ostringstream oss;
          oss << "(unknown cudnn tensor format " << static_cast<int>(tformat) << ")";
          return oss.str();
      }
        */
}

impl fmt::Display for FilterDescriptor {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            out << "FilterDescriptor " << static_cast<void*>(d.desc()) << "\n";
      int nbDims;
      int dimA[CUDNN_DIM_MAX];
      cudnnDataType_t dtype;
      cudnnTensorFormat_t tformat;
      cudnnGetFilterNdDescriptor(d.desc(), CUDNN_DIM_MAX, &dtype, &tformat, &nbDims, dimA);
      out << "    type = " << cudnnTypeToString(dtype) << "\n";
      out << "    tensor_format = " << cudnnMemoryFormatToString(tformat) << "\n";
      out << "    nbDims = " << nbDims << "\n";
      // Read out only nbDims of the arrays!
      out << "    dimA = ";
      for (auto i : ArrayRef<int>{dimA, static_cast<usize>(nbDims)}) {
        out << i << ", ";
      }
      out << "\n";
      return out;
        */
    }
}

impl FilterDescriptor {
    
    pub fn print(&mut self)  {
        
        todo!();
        /*
            cout << *this;
        */
    }
}
