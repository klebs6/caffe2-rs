crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/miopen/Descriptors.h]

#[inline] pub fn data_size(data_type: miopen::DataType) -> i32 {
    
    todo!();
        /*
            switch (dataType) {
        case miopenHalf: return 2;
        case miopenFloat: return 4;
        case miopenBFloat16: return 2;
        default: return 8;
      }
        */
}

/**
  | This function modifies 'stride' in place so
  | that the stride for dim i is the product of the
  | sizes of dims i+1 to the end.
  |
  */
#[inline] pub fn fix_size_one_dim_stride(
        dim:    i32,
        size:   *const i32,
        stride: *mut i32)  {
    
    todo!();
        /*
            i64 z = 1;
      for(int d = dim-1; d >= 0; d--)
      {
        if (size[d] == 1) {
          stride[d] = z;
        } else {
          z *= size[d];
        }
      }
        */
}

//template <typename T, miopenStatus_t (*dtor)(T*)>
pub struct DescriptorDeleter {

}

impl DescriptorDeleter {

    pub fn invoke(&mut self, x: *mut T)  {
        
        todo!();
        /*
            if (x != nullptr) {
          MIOPEN_CHECK(dtor(x));
        }
        */
    }
}

/**
  | A generic class for wrapping MIOpen descriptor
  | types.  All you need is to give the underlying
  | type the Descriptor_t points to (usually, if
  | it's miopenTensorDescriptor_t it points to
  | miopenTensorStruct), the constructor and the
  | destructor.  Subclasses are responsible for
  | defining a set() function to actually set the
  | descriptor.
  |
  | Descriptors default construct to a nullptr, and
  | have a descriptor initialized the first time
  | you call set() or any other initializing
  | function.
  */
//template <typename T, miopenStatus_t (*ctor)(T**), miopenStatus_t (*dtor)(T*)>
pub struct Descriptor {
    desc: Box<T,DescriptorDeleter<T,Dtor>>,
}

impl Descriptor {

    /**
      | Use desc() to access the underlying
      | descriptor pointer in a read-only fashion.
      | Most client code should use this.
      |
      | If the descriptor was never initialized, this
      | will return nullptr.
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
      | descriptor pointer if you intend to modify
      | what it points to (e.g., using
      | miopenSetFooDescriptor).  This will ensure
      | that the descriptor is initialized.  Code in
      | this file will use this function.
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
          MIOPEN_CHECK(ctor(&raw_desc));
          desc_.reset(raw_desc);
        }
        */
    }
}

pub struct TensorDescriptor {
    base: Descriptor<MiOpenTensorDescriptor,MiOpenCreateTensorDescriptor,MiOpenDestroyTensorDescriptor>,
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
    
    pub fn set(&mut self, 
        t:   &Tensor,
        pad: usize)  {
        let pad: usize = pad.unwrap_or(0);

        todo!();
        /*
        
        */
    }
    
    pub fn set(&mut self, 
        data_type: miopen::DataType,
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
        data_type: miopen::DataType,
        dim:       i32,
        size:      *mut i32,
        stride:    *mut i32)  {
        
        todo!();
        /*
            fixSizeOneDimStride(dim, size, stride);
        MIOPEN_CHECK(miopenSetTensorDescriptor(mut_desc(), dataType, dim, size, stride));
        */
    }
}

impl fmt::Display for TensorDescriptor {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
        
        */
    }
}

pub struct FilterDescriptor {
    base: Descriptor<MiOpenTensorDescriptor,MiOpenCreateTensorDescriptor,MiOpenDestroyTensorDescriptor>,
}

impl FilterDescriptor {
    
    pub fn set(&mut self, 
        t:   &Tensor,
        pad: i64)  {
        let pad: i64 = pad.unwrap_or(0);

        todo!();
        /*
        
        */
    }
    
    pub fn set(&mut self, 
        data_type: miopen::DataType,
        dim:       i32,
        size:      *mut i32,
        stride:    *mut i32)  {
        
        todo!();
        /*
            fixSizeOneDimStride(dim, size, stride);
        MIOPEN_CHECK(miopenSetTensorDescriptor(mut_desc(), dataType, dim, size, stride));
        */
    }
}

pub struct ConvolutionDescriptor {
    base: Descriptor<MiOpenConvolutionDescriptor,MiOpenCreateConvolutionDescriptor,MiOpenDestroyConvolutionDescriptor>,
}

impl ConvolutionDescriptor {

    pub fn set(&mut self, 
        data_type:            miopen::DataType,
        c_mode:               MiOpenConvolutionMode,
        dim:                  i32,
        pad:                  *mut i32,
        stride:               *mut i32,
        upscale_aka_dilation: *mut i32,
        groups:               i32)  {
        
        todo!();
        /*
            MIOPEN_CHECK(miopenInitConvolutionNdDescriptor(mut_desc(), dim, pad, stride, upscale_aka_dilation, c_mode));
        MIOPEN_CHECK(miopenSetConvolutionGroupCount(mut_desc(), groups));
        */
    }
}

pub struct RNNDescriptor {
    base: Descriptor<MiOpenRNNDescriptor,MiOpenCreateRNNDescriptor,MiOpenDestroyRNNDescriptor>,
}

impl RNNDescriptor {
    
    pub fn set(&mut self, 
        hidden_size: i64,
        num_layers:  i64,
        input_mode:  miopen::RNNInputMode,
        direction:   miopen::RNNDirectionMode,
        rnn_mode:    miopen::RNNMode,
        bias_mode:   miopen::RNNBiasMode,
        algorithm:   miopen::RNNAlgo,
        datatype:    miopen::DataType)  {
        
        todo!();
        /*
            MIOPEN_CHECK(miopenSetRNNDescriptor(mut_desc(), hidden_size, num_layers, input_mode, direction, rnn_mode, bias_mode, algorithm, datatype));
        */
    }
}

pub union Constant
{
    f: f32,
    d: f64,
}

impl Constant {
    
    pub fn new(
        data_type: miopen::DataType,
        value:     f64) -> Self {
    
        todo!();
        /*


            if (dataType == miopenHalf || dataType == miopenFloat || dataType == miopenBFloat16) {
          f = static_cast<float>(value);
        } else {
          d = value;
        }
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/miopen/Descriptors.cpp]

#[inline] pub fn get_data_type(t: &Tensor) -> miopen::DataType {
    
    todo!();
        /*
            auto scalar_type = t.scalar_type();
      if (scalar_type == kFloat) {
        return miopenFloat;
      } else if (scalar_type == kHalf) {
        return miopenHalf;
      } else if (scalar_type == kBFloat16) {
        return miopenBFloat16;
      } else {
      throw runtime_error("TensorDescriptor only supports float, half and bfloat16 tensors");
      }
        */
}

lazy_static!{
    /*
    static int MIOPEN_DIM_MAX = 5;
    */
}

impl TensorDescriptor {
    
    pub fn set(&mut self, 
        t:   &Tensor,
        pad: usize)  {
        
        todo!();
        /*
            set(getDataType(t), t.sizes(), t.strides(), pad);
        */
    }
    
    pub fn set(&mut self, 
        datatype:  miopen::DataType,
        t_sizes:   &[i32],
        t_strides: &[i32],
        pad:       usize)  {
        
        todo!();
        /*
            usize dim = t_sizes.size();
      if (dim > MIOPEN_DIM_MAX || pad > MIOPEN_DIM_MAX)
    #define _STR(X) #X
    #define STR(X) _STR(X)
        throw runtime_error("MIOpen supports only up to " STR(MIOPEN_DIM_MAX) " dimensions");
    #undef _STR
    #undef STR
      int size[MIOPEN_DIM_MAX];
      int stride[MIOPEN_DIM_MAX];
      for (usize i = 0; i < dim; ++i) {
        size[i] = static_cast<int>(t_sizes[i]);
        stride[i] = static_cast<int>(t_strides[i]);
      }
      for (usize i = dim; i < pad; ++i) {
        size[i] = 1;
        stride[i] = 1;
      }
      set(datatype, static_cast<int>(max(dim, pad)), size, stride);
        */
    }
    
    pub fn print(&mut self)  {
        
        todo!();
        /*
            cout << *this;
        */
    }
}

pub fn miopen_type_to_string(dtype: miopen::DataType) -> String {
    
    todo!();
        /*
            switch (dtype) {
        case miopenFloat:
          return "miopenFloat";
        case miopenHalf:
          return "miopenHalf";
        case miopenBFloat16:
          return "miopenBFloat16";
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
      int nbDims = 4;
      int dimA[MIOPEN_DIM_MAX];
      int strideA[MIOPEN_DIM_MAX];
      miopenDataType_t dtype;
      miopenGetTensorDescriptor(d.desc(), &dtype, dimA, strideA);
      out << "    type = " << miopenTypeToString(dtype) << "\n";
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

impl FilterDescriptor {
    
    pub fn set(&mut self, 
        t:   &Tensor,
        pad: i64)  {
        
        todo!();
        /*
            auto dim = t.ndimension();
      if (dim > MIOPEN_DIM_MAX || pad > MIOPEN_DIM_MAX)
    #define _STR(X) #X
    #define STR(X) _STR(X)
        throw runtime_error("MIOpen supports only up to " STR(MIOPEN_DIM_MAX) " dimensions");
    #undef _STR
    #undef STR
      if (!t.is_contiguous()) {
        throw runtime_error("MIOpen filters (a.k.a. weights) must be contiguous");
      }
      int size[MIOPEN_DIM_MAX];
      int stride[MIOPEN_DIM_MAX];
      for (int i = 0; i < dim; ++i) {
        size[i] = (int) t.size(i);
      }
      for (int i = dim; i < pad; ++i) {
        size[i] = (int) 1;
      }
      for (int i = dim - 1; i >=0; --i) {
        stride[i] = (i == dim - 1) ? 1 : stride[i+1] * size[i+1];
      }
      dim = max(dim, pad);
      set(getDataType(t), (int) dim, size, stride);
        */
    }
}
