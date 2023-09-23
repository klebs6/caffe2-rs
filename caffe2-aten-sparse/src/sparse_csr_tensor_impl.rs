crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/SparseCsrTensorImpl.h]

/**
  | Struct implementing a sparse CSR tensor. It
  | uses three 1-D tensors for denoting the data:
  | `crow_indices_`, `col_indices_` and `values_`.
  |
  | The `crow_indices_` tensor is a integer tensor
  | of shape `(size(0) + 1)` that represents the
  | compressed row indices of the CSR tensor. The
  | `col_indices_` tensor is an integer tensor of
  | shape `(nnz())` that explicitly stores the
  | column indices of each value of the sparse
  | tensor.
  |
  | The `values_` tensor can be of any
  | pytorch-supported data type and has shape
  | `(nnz())`.
  |
  | Since the main advantage of the CSR format over
  | the COO format is speed of computation, care
  | must be taken to facilitate smooth interfacing
  | of these data structures with optimized
  | libraries such as MKL and MAGMA.
  |
  | Since the MKL interface for pytorch currently
  | uses indexing with int32 type, it is important
  | to make sure that the `crow_indices` and
  | `col_indices` are of type int32 when calling
  | MKL routines such as SPMM or SPMV.
  |
  | If not calling MKL, it should be alright to use
  | 64 bit integer tensors for indexing.
  |
  */
pub struct SparseCsrTensorImpl {
    base:         TensorImpl,
    crow_indices: Tensor,
    col_indices:  Tensor,
    values:       Tensor,
}

impl SparseCsrTensorImpl {

    pub fn crow_indices(&self) -> &Tensor {
        
        todo!();
        /*
            return crow_indices_;
        */
    }
    
    pub fn col_indices(&self) -> &Tensor {
        
        todo!();
        /*
            return col_indices_;
        */
    }
    
    pub fn values(&self) -> &Tensor {
        
        todo!();
        /*
            return values_;
        */
    }
    
    pub fn nnz(&mut self) -> i32 {
        
        todo!();
        /*
            return values_.size(0);
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/SparseCsrTensorImpl.cpp]

pub fn sparse_csr_tensor_set_to_device_type(key_set: DispatchKeySet) -> DeviceType {
    
    todo!();
        /*
            if (key_set.has(DispatchKey::SparseCsrCPU)) {
        return kCPU;
      } else if (key_set.has(DispatchKey::SparseCsrCUDA)) {
        return kCUDA;
      } else {
        TORCH_CHECK(false,
            "Cannot construct SparseCsrTensor with non-sparse tensor type ID ",
            key_set);
      }
        */
}

impl SparseCsrTensorImpl {
    
    pub fn new(
        key_set:   DispatchKeySet,
        data_type: TypeMeta) -> Self {
    
        todo!();
        /*


            : SparseCsrTensorImpl(
              key_set,
              data_type,
              empty(
                  {0},
                  initialTensorOptions()
                      .device(SparseCsrTensorSetToDeviceType(key_set))
                      .dtype(ScalarType::Int)) // crow_indices
              ,
              empty(
                  {0},
                  initialTensorOptions()
                      .device(SparseCsrTensorSetToDeviceType(key_set))
                      .dtype(ScalarType::Int)) // col_indices
              ,
              empty(
                  {0},
                  initialTensorOptions()
                      .device(SparseCsrTensorSetToDeviceType(key_set))
                      .dtype(data_type)) // values
          )
        */
    }
    
    pub fn new(
        key_set:      DispatchKeySet,
        data_type:    TypeMeta,
        crow_indices: Tensor,
        col_indices:  Tensor,
        values:       Tensor) -> Self {
    
        todo!();
        /*
        : tensor_impl(key_set, data_type, values.device()),
        : crow_indices(move(crow_indices)),
        : col_indices(move(col_indices)),
        : values(move(values)),

        
        */
    }
    
    pub fn resize_as_sparse_csr_tensor(&mut self, src: &Tensor)  {
        
        todo!();
        /*
            crow_indices_ = empty_like(
          src.crow_indices(),
          src.crow_indices().options(),
          src.crow_indices().suggest_memory_format());
      col_indices_ = empty_like(
          src.col_indices(),
          src.col_indices().options(),
          src.col_indices().suggest_memory_format());
      values_ = empty_like(
          src.values(),
          src.values().options(),
          src.values().suggest_memory_format());
      sizes_and_strides_.set_sizes(src.sizes());
      refresh_numel();
        */
    }
    
    pub fn set_member_tensors(&mut self, 
        crow_indices: &Tensor,
        col_indices:  &Tensor,
        values:       &Tensor,
        size:         &[i32])  {
        
        todo!();
        /*
            // CSR Type Invariants
      TORCH_CHECK(
          values.scalar_type() == typeMetaToScalarType(dtype()),
          "dtype of values (",
          values.scalar_type(),
          ") must match dtype of sparse tensor (",
          typeMetaToScalarType(dtype()),
          ")");

      crow_indices_ = crow_indices;
      col_indices_ = col_indices;
      values_ = values;

      sizes_and_strides_.set_sizes(size);
      refresh_numel();
        */
    }
}
