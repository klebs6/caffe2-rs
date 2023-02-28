crate::ix!();

pub enum Unsafe { IDoWantAliasing, IDontWantAliasing }

/**
  | @brief
  | 
  | Tensor class holds a shared pointer
  | to the implementation TensorImpl,
  | redirects API calls to TensorImpl;
  | 
  | Copying of Tensor results in sharing
  | the same underlying implementation
  | object
  | 
  | NB: See TensorImpl for documentation
  | on these methods.
  |
  */
pub struct Tensor {
    impl_:  TensorImplPtr,
}

pub type TensorImplPtr = IntrusivePtr<TensorImpl>;

//was called copy_ctor in the c++
impl Clone for Tensor {
    fn clone(&self) -> Self {
        todo!();
        /*
           return X.UnsafeSharedInstance();
           */
    }
}

impl Default for Tensor {
    
    fn default() -> Self {
        todo!();
        /*
            : impl_(
        */
    }
}

impl Into<bool> for Tensor {
    fn into(self) -> bool {
        todo!();
        /*
        self.impl_.defined()
        */
    }

}

impl From<DeviceType> for Tensor {

    /**
      | -----------
      | @brief
      | 
      | Creates a tensor of the given device
      | type.
      | 
      | -----------
      | @note
      | 
      | the actual data allocation is not going
      | to be carried out until you resize the
      | tensor and then call mutable_data().
      |
      */
    fn from(device: DeviceType) -> Self {
        todo!();
        /*
            : impl_(c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
                Storage::create_legacy(device),
                c10::computeDispatchKey(c10::nullopt, at::kStrided, device),
                TypeMeta()))
        */
    }
}

impl Tensor {
    
    #[inline] pub fn get_intrusive_ptr(&self) -> &IntrusivePtr<TensorImpl> {
        
        todo!();
        /*
            return impl_;
        */
    }

    pub unsafe fn new(other: &Tensor) -> Self {
    
        todo!();
        /*
            : impl_(other.getIntrusivePtr())
        */
    }
    
    #[inline] pub fn resize<Ts>(&self, dim_source: Ts)  {
        todo!();
        /*
            impl_.get()->Resize(dim_source...);
        */
    }
    
    #[inline] pub fn resize_with_dim_vec<T>(&self, dim_source: &Vec<T>)  {
        todo!();
        /*
            impl_.get()->Resize(&[T](dim_source));
        */
    }
    
    #[inline] pub fn unsafe_get_tensor_impl(&self) -> *mut TensorImpl {
        
        todo!();
        /*
            return impl_.get();
        */
    }
    
    #[inline] pub fn unsafe_shared_instance(&self) -> Tensor {
        
        todo!();
        /*
            return Tensor(*this, IDoWantAliasing);
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Creates a tensor of the given dimension.
      | 
      | -----------
      | @note
      | 
      | the actual data allocation is not going
      | to be carried out until the first time
      | mutable_data() is called.
      |
      */
    pub fn new_with_dimension_and_type(dims: &[i32], ty: DeviceType) -> Self {
        todo!();
        /*
            : Tensor(type) 

        // TODO: here, we create a Storage
        // and immediately discard it in Resize() since
        // reset_tensor will be true and FreeMemory will be called,
        // we might want to avoid creating Storage twice?
        Resize(dims);
        */
    }

    /// we want to preserve index information
    pub fn new_with_dimension_and_device(dims: &[i32], device: Device) -> Self {
        todo!();
        /*
            : Tensor(device) 
        Resize(dims);
        */
    }

    /// TODO: remove?
    pub fn new_with_dimension_vec_and_type(dims: &Vec<i32>, ty: DeviceType) -> Self {
        todo!();
        /*
            : Tensor(type) 
        Resize(dims);
        */
    }

    /**
      | @brief
      | 
      | : Create a Tensor of at::DeviceType
      | `type` and initialize it with src Tensor
      |
      */
    pub fn new_with_type_from_src(src: &Tensor, ty: DeviceType) -> Self {
        todo!();
        /*
            : Tensor(type) 
        CopyFrom(src);
        */
    }
}

pub struct TensorInfo {
    min:        f32,
    max:        f32,
    total_min:  f32,
    total_max:  f32,
    name:       String,
}

impl TensorInfo {
    
    pub fn new(name: &String) -> Self {
    
        todo!();
        /*
            : min(float::max),
            max(std::numeric_limits<float>::lowest()),
            total_min(float::max),
            total_max(std::numeric_limits<float>::lowest()),
            name(name)
        */
    }
    
    #[inline] pub fn update(&mut self, cur_min: f32, cur_max: f32)  {
        
        todo!();
        /*
            min = std::min(min, cur_min);
            max = std::max(max, cur_max);
            total_min = std::min(total_min, cur_min);
            total_max = std::max(total_max, cur_max);
        */
    }
}

#[inline] pub fn reinitialize_and_copy_from(
    t:       *mut Tensor,
    options: TensorOptions,
    src:     &Tensor,
    async_:  Option<bool>)  
{
    let async_: bool = async_.unwrap_or(false);

    todo!();
    /*
    
    */
}

pub type TensorCPU  = Tensor;
pub type TensorCUDA = Tensor;

pub const k_limit_default: i32 = 1000;

/**
  | TODO: the following logic can be merged
  | into regular Tensor class methods after
  | MKLMemory starts to implement Tensor
  | interface
  |
  */

/// Type call registry
pub type TypeCall = fn() -> TypeMeta;

/// Shape call registry
pub type TensorInfoCall = fn(*const c_void, capacity: *mut usize,device: *mut DeviceOption);

/**
  | -----------
  | @brief
  | 
  | Creates a CPU tensor, and fills its contents
  | with the given values. Values are copied
  | in
  |
  */

/**
  | TODO: can be unified with at::from_blob
  | when Tensor is merged and string types are
  | supported
  |
  */
#[inline] pub fn tensor_cpu_from_values<T>(dims: &[i32], values: &[T]) -> Tensor {

    todo!();
    /*
        Tensor r = empty(dims, at::device(CPU).dtype<T>());
      CAFFE_ENFORCE_EQ(values.size(), r.numel());
      CPUContext context;
      context.CopyItemsFromCPU(
          r.dtype(), values.size(), values.data(), r.mutable_data<T>());
      return r;
    */
}

///-----------------
pub struct TensorPrinter<W: Write> {
    to_file:      bool,
    limit:        i32,
    log_file:     Box<std::io::BufWriter<W>>,
    tensor_name:  String,
}

impl<W: Write> Default for TensorPrinter<W> {
    fn default() -> Self {
        todo!();
    }
}

caffe_known_type!{Tensor}

impl<W: Write> TensorPrinter<W> {
    
    pub fn new_from_file(
        tensor_name: &mut String,
        file_name: &mut String,
        limit: i32) -> Self {
        todo!();
        /*
            : to_file_(!file_name.empty()),
          limit_(limit ? limit : k_limit_default_),
          tensor_name_(tensor_name) 

      if (to_file_) {
        // We will output to file instead of printing on screen.
        // We will write each individual tensor to its individual file.
        log_file_.reset(new std::ofstream(
            file_name, std::ofstream::out | std::ofstream::trunc));
        CAFFE_ENFORCE(
            log_file_->good(),
            "Failed to open TensorPrinter file ",
            file_name,
            ". rdstate() = ",
            log_file_->rdstate());
      }
        */
    }

    pub fn print<T>(&mut self, tensor: &Tensor) {
        todo!();
        /*
            std::stringstream values_stream;
          // One most likely doesn't want to print int64-number of items for visual
          // inspection, so we cast down to int here.
          int total_count = static_cast<int>(std::min(tensor.numel(), int64_t(limit_)));

          const T* tensor_data = tensor.template data<T>();
          for (int i = 0; i < total_count - 1; ++i) {
            values_stream << tensor_data[i] << ",";
          }
          if (total_count) {
            // We do not add a comma after the last item.
            values_stream << tensor_data[total_count - 1];
          }

          if (to_file_) {
            (*log_file_) << MetaStr(tensor) << values_stream.str() << std::endl;
          } else {
            // Log to console.
            LOG(INFO) << MetaStr(tensor) << values_stream.str();
          }
        */
    }
    
    pub fn print_meta(&mut self, tensor: &Tensor)  {
        
        todo!();
        /*
            if (to_file_) {
        (*log_file_) << MetaStr(tensor) << std::endl;
      } else {
        LOG(INFO) << MetaStr(tensor);
      }
        */
    }
    
    pub fn meta_str(&mut self, tensor: &Tensor) -> String {
        
        todo!();
        /*
            std::stringstream meta_stream;
      meta_stream << "Tensor " << tensor_name_ << " of type "
                  << tensor.dtype().name() << ". Dims: (";
      for (const auto dim : tensor.sizes()) {
        meta_stream << dim << ",";
      }
      meta_stream << "): ";
      return meta_stream.str();
        */
    }
}

impl<W: Write> Drop for TensorPrinter<W> {
    fn drop(&mut self) {
        todo!();
        /*
          if (log_file_.get()) {
            log_file_->close();
          }
        */
    }
}

#[inline] pub fn get_gpuid_for_pointer(ptr: *const c_void) -> i32 {
    
    todo!();
    /*
    
    */
}

lazy_static!{

    /// TODO(jerryzh): Remove
    static ref type_call_registry: HashMap<TypeIdentifier, TypeCall> = {
        todo!();
        /*
        let m = HashMap::new();
        m.insert(Tensor::type_id(), get_tensor_type);
        m.insert(Int8TensorCPU::type_id(), get_int_8tensor_type);
        m
        */
    };

    /**
      | since we only have one tensor, probably
      | need to remove this at some point?
      |
      */
    static ref tensor_info_call_registry: HashMap<TypeIdentifier, TensorInfoCall> = {
        todo!();
        /*
        let m = HashMap::new();
        m.insert(Tensor::type_id(), get_tensor_info);
        m.insert(Int8TensorCPU::type_id(), get_int_8tensor_info);
        m
        */
    };
}

impl Tensor {

    pub fn is_same(&self, other: &Tensor) -> bool {
        
        todo!();
        /*
            return impl_ == other.impl_;
        */
    }

    pub fn clone(&self) -> Tensor {
        
        todo!();
        /*
            Tensor x(GetDevice());
        x.CopyFrom(*this);
        return x;
        */
    }
    
    /**
      | Clone self as a Tensor that share the
      | same
      | 
      | Storage, that is, both Tensors are views
      | on the same Storage.
      | 
      | If we change the sizes or strides of one
      | 
      | Tensor, it does not affect the other
      | Tensor that it shares Storage with.
      | 
      | A similar yet different usage is `Tensor
      | x = y;`, this will make x and y pointing
      | to the same Tensor and resizing one of
      | them will resize the other as well.
      | 
      | TODO: Deduplicate this with
      | 
      | THTensor_(newWithTensor) (exposed
      | in ATen as at::alias but not otherwise
      | available)
      |
      */
    pub fn alias(&self) -> Tensor {
        
        todo!();
        /*
            Tensor x(sizes(), GetDevice());
        if (!dtype_initialized()) {
          C10_LOG_EVERY_MS(WARNING, 1000)
              << "Cloning a tensor that don't have a data type (did you call mutable_data<T> on the tensor?)";
        }
        AT_ASSERTM(
            storage_initialized(),
            "Cloning a tensor that has no content and has size > 0");
        // set_storage already sets data_type_ of TensorImpl
        x.impl_->set_storage_and_dtype(storage(), impl_->dtype());
        x.impl_->set_storage_offset(impl_->storage_offset());
        x.impl_->set_sizes_and_strides(sizes(), strides());
        return x;
        */
    }
    
    pub fn get_device_type(&self) -> DeviceType {
        
        todo!();
        /*
            return impl_->device_type();
        */
    }
    
    pub fn get_device(&self) -> Device {
        
        todo!();
        /*
            return impl_.get()->device();
        */
    }

    /**
      | -----------
      | @brief
      | 
      | Extend the outer-most dimension of
      | this tensor to dimension of `num`.
      |
      */
    pub fn extend_to(&self, num: i64,
        growth_pct: f32)  {

        todo!();
        /*
            CAFFE_ENFORCE_GE_WITH_CALLER(impl_->dim(), 1);
        CAFFE_ENFORCE_GE_WITH_CALLER(growthPct, 0);
        Extend(num - impl_->size(0), growthPct);
        */
    }
    
    pub fn extend(&self, num: i64, growth_pct: f32)  {
        
        todo!();
        /*
            impl_.get()->Extend(num, growthPct);
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Shrinks the outer-most dimension to
      | given size, keeping the data.
      | 
      | This method guarantees that no re-allocations
      | are carried out, which means that the
      | extra capacity after the end of the shrunk
      | tensor is maintained.
      | 
      | Notably, this function does NOT respect
      | caffe2_keep_on_shrink.
      |
      */
    pub fn shrink_to(&self, outer_dim: i64)  {
        
        todo!();
        /*
            CAFFE_ENFORCE_WITH_CALLER(
            impl_->is_contiguous(),
            "Right now ShrinkTo is only supported on contiguous Tensor.");
        CAFFE_ENFORCE_WITH_CALLER(impl_->dim() >= 1, "Tensor must be at least 1D");
        CAFFE_ENFORCE_WITH_CALLER(
            outer_dim <= impl_->size(0),
            "New outer dimension must be smaller than current.");
        CAFFE_ENFORCE(
            impl_->storage().unique(),
            "Can't call ShrinkTo on shared storage, please call Resize instead.");
        impl_.get()->set_size(0, outer_dim);
        */
    }

    pub fn reserve_space<T>(&self, outer_dim: &T)  {
        todo!();
        /*
           impl_.get()->ReserveSpace(outer_dim);
           */
    }

    /**
      | Resize the tensor like the source tensor.
      | Note that this is just a sugar wrapper
      | that essentially calls
      | 
      | Resize(src_tensor.dims()).
      | 
      | This method respects caffe2_keep_on_shrink.
      |
      */
    #[inline] pub fn resize_like(&self, src_tensor: &Tensor)  {
        
        todo!();
        /*
            CAFFE_ENFORCE_WITH_CALLER(
            src_tensor.is_contiguous(),
            "Right now ResizeLike is only supported for contiguous Tensor.");
        if (impl_ != src_tensor.impl_) {
          impl_.get()->Resize(src_tensor.sizes());
        }
        */
    }
    
    #[inline] pub fn reshape<T: PrimInt>(&self, dims: &Vec<T>)  {
        todo!();
        /*
            impl_.get()->Reshape(ToVectorint64_t(dims));
        */
    }
    
    #[inline] pub fn free_memory(&self)  {
        
        todo!();
        /*
            impl_.get()->FreeMemory();
        */
    }
    
    /**
      | A utility function to print the debug
      | string for the tensor. Note that this
      | is very slow since it involves quite
      | some string operations, so do not use
      | it in your performance-critical code.
      |
      */
    pub fn debug_string(&self) -> String {
        
        todo!();
        /*
            std::stringstream ss;
        ss << "A Tensor of item size " << impl_->dtype().itemsize() << " and type "
           << impl_->dtype().name() << " and dimension (";
        for (int d : impl_->sizes()) {
          ss << d << ",";
        }
        ss << ").";
        return ss.str();
        */
    }

    /// To be deprecated
    pub fn share_data(&self, src: &Tensor)  {
        
        todo!();
        /*
            impl_.get()->ShareData(*src.impl_.get());
        */
    }

    /**
      | @brief
      | 
      | Shares the data with an externally managed
      | pointer.
      | 
      | This is similar to ShareData() but the
      | source is a pointer with an advanced
      | deleter option.
      | 
      | In default, no deletion takes place,
      | and one needs to make sure that the external
      | memory is deallocated only after the
      | tensor finishes using it.
      | 
      | If a Deleter object is passed in, when
      | this tensor is reallocated or freed,
      | the deleter function is going to be called.
      |
      */
    pub fn share_external_pointer_with_externally_managed_pointer<T>(
        &self, 
        src:    *mut T,
        nbytes: usize,
        d:      MemoryDeleter) {
        todo!();
        /*
            ShareExternalPointer((void*)src, caffe2::TypeMeta::Make<T>(), nbytes, d);
        */
    }
    
    pub fn share_external_pointer_with_deleter(
        &self, 
        src:        *mut c_void,
        data_type:  TypeMeta,
        nbytes:     Option<usize>,
        d:          Option<MemoryDeleter>)  
    {
        let nbytes: usize = nbytes.unwrap_or(0);

        todo!();
        /*
            CAFFE_ENFORCE_WITH_CALLER(
            impl_->is_contiguous(),
            "Right now ShareExternalPointer is only supported for contiguous Tensor.");
        CAFFE_ENFORCE_WITH_CALLER(
            data_type != ScalarType::Undefined,
            "To share with a raw external pointer you need to pass in an "
            "initialized data_type(TypeMeta).");
        impl_.get()->ShareExternalPointer(
            at::DataPtr(src, src, d, impl_->device_type()), data_type, nbytes);
        */
    }


    pub fn share_external_pointer<T>(
        &self, 
        data_ptr: DataPtr, 
        nbytes:   usize) 
    {
        todo!();
        /*
           ShareExternalPointer( std::move(data_ptr), caffe2::TypeMeta::Make<T>(), nbytes);
           */
    }
    
    pub fn share_external_pointer_with_data_type(
        &mut self, 
        data_ptr:  DataPtr,
        data_type: TypeMeta,
        nbytes:    usize)  {
        
        todo!();
        /*
            impl_.get()->ShareExternalPointer(std::move(data_ptr), data_type, nbytes);
        */
    }
    
    pub fn defined(&self) -> bool {
        
        todo!();
        /*
            return impl_;
        */
    }
    
    /**
      | Returns a raw void* pointer of the underlying
      | storage. mutable_data() or raw_mutable_data()
      | must have been called prior to this function
      | call.
      |
      */
    #[inline] pub fn raw_data(&self)  {
        
        todo!();
        /*
            return impl_->data();
        */
    }

    #[inline] pub fn data<T>(&self) -> *mut T {
        todo!();
        /*
            return impl_.get()->data<T>();
        */
    }
    
    #[inline] pub fn raw_mutable_data_with_type_meta(&self, meta: TypeMeta)  {
        
        todo!();
        /*
            return impl_.get()->raw_mutable_data(meta);
        */
    }

    /**
      | Returns a mutable raw pointer of the
      | underlying storage. This can only be
      | used when you know for sure that the underlying
      | storage of the tensor is already created
      | via an earlier raw_mutable_data(meta)
      | call or a mutable_data<T>() call.
      | 
      | If the existing data does not match the
      | desired type, it will be deleted and
      | a new storage will be created.
      |
      */
    #[inline] pub fn raw_mutable_data(&self)  {
        
        todo!();
        /*
            const auto& data_type = impl_->dtype();
        CAFFE_ENFORCE_WITH_CALLER(
            data_type != ScalarType::Undefined,
            "Calling raw_mutable_data() without meta, but the current meta is "
            "of unknown type.");
        return raw_mutable_data(data_type);
        */
    }

    #[inline] pub fn mutable_data<T>(&self) -> *mut T {
        todo!();
        /*
            return impl_.get()->mutable_data<T>();
        */
    }
    
    /**
      | Returns the number of dimensions of
      | the data.
      |
      */
    #[inline] pub fn dim(&self) -> i32 {
        
        todo!();
        /*
            return impl_->dim();
        */
    }
    
    /**
      | (To be deprecated) Returns the number
      | of dimensions of the data.
      |
      */
    #[inline] pub fn ndim(&self) -> i32 {
        
        todo!();
        /*
            return impl_->dim();
        */
    }
    
    /**
      | (To be deprecated) Returns the size
      | (i.e. the number of items) of the tensor.
      |
      */
    #[inline] pub fn size(&self) -> i64 {
        
        todo!();
        /*
            return impl_->numel();
        */
    }
    
    /**
      | Returns the number of items of the tensor.
      |
      */
    #[inline] pub fn numel(&self) -> i64 {
        
        todo!();
        /*
            return impl_->numel();
        */
    }
    
    /**
      | Return the number of bytes each item
      | takes in the tensor.
      |
      */
    #[inline] pub fn itemsize(&self) -> usize {
        
        todo!();
        /*
            return impl_->dtype().itemsize();
        */
    }
    
    /**
      | Returns the total number of bytes of
      | the storage.
      | 
      | This is equivalent to calling size()
      | * itemsize().
      |
      */
    #[inline] pub fn nbytes(&self) -> usize {
        
        todo!();
        /*
            return impl_->numel() * itemsize();
        */
    }
    
    #[inline] pub fn sizes(&self) -> &[i32] {
        
        todo!();
        /*
            return impl_.get()->sizes();
        */
    }
    
    #[inline] pub fn size_from_dim(&self, k: i32) -> i64 {
        
        todo!();
        /*
            return size_from_dim_(k, impl_->sizes());
        */
    }
    
    #[inline] pub fn size_to_dim(&self, k: i32) -> i64 {
        
        todo!();
        /*
            return size_to_dim_(k, impl_->sizes());
        */
    }
    
    #[inline] pub fn size_between_dim(&self, k: i32, l: i32) -> i64 {
        
        todo!();
        /*
            return size_between_dim_(k, l, impl_->sizes());
        */
    }
    
    /**
     | Returns the 'canonical' version of
     | a (usually)  user-specified axis, allowing
     | for negative indexing (e.g., -1 for the last
     | axis).
     |
     | @param axis_index the axis index.
     |        If 0 <= index < dim(), return index.
     |        If -ndim <= index <= -1, return (dim() - (-index)),
     |        e.g., the last axis index (dim() - 1) if index == -1,
     |        the second to last if index == -2, etc.
     |        Dies on out of range index.
     */
    #[inline] pub fn canonical_axis_index(&self, axis_index: i32) -> i32 {
        
        todo!();
        /*
            return canonical_axis_index_(axis_index, impl_->dim());
        */
    }
    
    #[inline] pub fn stride(&self, dim: i64) -> i64 {
        
        todo!();
        /*
            return impl_.get()->stride(dim);
        */
    }
    
    #[inline] pub fn strides(&self) -> &[i32] {
        
        todo!();
        /*
            return impl_.get()->strides();
        */
    }
    
    #[inline] pub fn is_contiguous(&self, memory_format: Option<MemoryFormat>) -> bool {

        let memory_format: MemoryFormat = 
            todo!(); // memory_format.unwrap_or(MemoryFormat::Contiguous);

        todo!();
        /*
            return impl_.get()->is_contiguous(memory_format);
        */
    }

    /**
      | Checks if the tensor content is of the
      | given data type.
      |
      */
    #[inline] pub fn is_type<T>(&self) -> bool {
        todo!();
        /*
            return impl_->dtype().Match<T>();
        */
    }

    /**
      | Returns the TypeMeta object associated
      | with the current data type.
      |
      */
    #[inline] pub fn dtype(&self) -> TypeMeta {
        
        todo!();
        /*
            return impl_->dtype();
        */
    }
    
    /**
      | (To be deprecated) Returns the TypeMeta
      | object associated with the current
      | data type.
      |
      */
    #[inline] pub fn meta(&self) -> TypeMeta {
        
        todo!();
        /*
            return impl_->dtype();
        */
    }
    
    /**
      | Returns the i-th dimension of the tensor
      | in int.
      | 
      | This function returns an int value instead
      | of int64_t, which depending on the typedef
      | could be int64. If you want int64 dim
      | values, make sure you call dim() instead.
      |
      */
    #[inline] pub fn dim32(&self, i: i32) -> i32 {
        
        todo!();
        /*
            #ifndef NDEBUG
        CAFFE_ENFORCE_LT_WITH_CALLER(
            i, static_cast<int>(impl_->dim()), "Exceeding ndim limit");
        CAFFE_ENFORCE_GE_WITH_CALLER(i, 0, "Cannot have negative dimension index");
    #endif
        // Avoid TensorImpl::size() because it is a virtual call that
        // supports out-of-range indexing like Python.
        auto s = impl_->sizes()[i];
        CAFFE_ENFORCE_LT_WITH_CALLER(s, int::max);
        return static_cast<int>(s);
        */
    }
    
    #[inline] pub fn size_from_index(&self, i: i32) -> i64 {
        
        todo!();
        /*
            return impl_->size(i);
        */
    }

    // To be deprecated
    #[inline] pub fn dim_from_index(&self, i: i32) -> i64 {
        
        todo!();
        /*
            return impl_->size(i);
        */
    }
    
    pub fn mut_storage<'a>(&'a mut self) -> &'a mut Storage {
        
        todo!();
        /*
            return impl_->storage();
        */
    }
    
    pub fn storage<'a>(&'a self) -> &'a Storage {
        
        todo!();
        /*
            return impl_->storage();
        */
    }
    
    pub fn storage_initialized(&self) -> bool {
        
        todo!();
        /*
            return impl_->storage_initialized();
        */
    }
    
    pub fn dtype_initialized(&self) -> bool {
        
        todo!();
        /*
            return impl_->dtype_initialized();
        */
    }
    
    /**
      | Reinitialize a Tensor to given dims
      | and options if necessary, note that
      | this will not do anything if the Tensor
      | already has correct size and data type
      |
      */
    pub fn reinitialize_tensor(&mut self, 
        tensor: *mut Tensor,
        dims: &[i32],
        options: TensorOptions)  {

        todo!();
        /*
            CAFFE_ENFORCE(options.device_opt() != c10::nullopt);
      if (*tensor) {
        // Note: we don't compare device_id here because of the purpose of
        // ReinitializeTensor: https://github.com/pytorch/pytorch/pull/13147
        // In the original code, we don't have device_id defined, therefore, we
        // should not include device_id in the comparison
        if (tensor->GetDeviceType() == options.device().type()) {
          if (tensor->sizes() != dims) {
            // Resize when the dims doesn't match
            tensor->Resize(dims);
          }
          if (tensor->dtype() == options.dtype()) {
            tensor->raw_mutable_data();
          } else {
            // This C10 logging API is not thread-safe, and should not be called here
            // This can lead to a memory corruption in glog.
            // C10_LOG_FIRST_N(WARNING, 1)
            //     << "Changing the data type of Tensor is discouraged."
            //     << " Attempt to change data type from: " << tensor->dtype()
            //     << " to: " << options.dtype();
            // create a new Tensor when the data_type doesn't match
            *tensor = caffe2::empty(dims, options);
          }
          return;
        }
        // create a new Tensor when device doesn't match
      }

      VLOG(1) << "Create new mutable object " << TypeMeta::TypeName<Tensor>()
              << " dims: " << dims;
      *tensor = caffe2::empty(dims, options);
        */
    }
    
    pub fn reinitialize_and_copy_from(&mut self, 
        t:       *mut Tensor,
        options: TensorOptions,
        src:     &Tensor,
        async_:  bool)  {

        todo!();
        /*
            auto device_type = options.device().type();
      CAFFE_ENFORCE(t != nullptr, "Target tensor ptr is null.");
      if (!*t || device_type != t->GetDeviceType()) {
        *t = Tensor(device_type);
      }
      CAFFE_ENFORCE(
          !t->dtype_initialized() || t->dtype() == src.dtype(),
          "We don't allow a change of data type in ReinitializeAndCopyFrom. Attempt to "
          " change from: ",
          t->dtype(),
          " to: ",
          src.dtype());
      t->CopyFrom(src, async);
        */
    }
    
    pub fn enforce_invariants(&mut self)  {
        
        todo!();
        /*
            if (impl_.get() == nullptr) {
        throw std::runtime_error("TensorImpl with nullptr is not supported");
      }
      // TODO: only check `!impl_->requires_grad()` after Variable and Tensor are
      // merged
    #if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
      CAFFE_ENFORCE(
          !(impl_->requires_grad() && at::GradMode::is_enabled()),
          "Caffe2 tensor wrapper doesn't support autograd variables that require grad");
    #endif
      CAFFE_ENFORCE_EQ(
          impl_->layout(),
          at::kStrided,
          "Caffe2 tensor wrapper supports only regular non-sparse tensors");
      CAFFE_ENFORCE(
          impl_->is_contiguous(),
          "Caffe2 tensor wrapper supports only contiguous tensors");
        */
    }
    
    /**
      | @brief
      | 
      | Copies the data from a source tensor,
      | with a context provided to carry out
      | the underlying memcpy operation. This
      | method respects caffe2_keep_on_shrink.
      | 
      | After CopyFrom, this function guarantees
      | that the destination tensor will have
      | the same initialization state and dtype
      | as src.
      | 
      | This function preserves the DeviceType
      | of the source tensor (so, e.g., if you
      | allocate a tensor on CPU and then CopyFrom
      | a CUDA tensor, that will to a CUDA-to-CPU
      | transfer). 'async' parameter triggers
      | async copy for
      | 
      | CUDA tensors
      |
      */
    pub fn copy_from(&mut self, src: &Tensor, async_: Option<bool>)  {

        let async_ = async_.unwrap_or(false);
        
        todo!();
        /*
            // TODO: only check `!impl_->requires_grad()` after Variable and Tensor are
      // merged
    #if defined(EXPOSE_C2_OPS) || \
        !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
      AT_ASSERT(!(impl_->requires_grad() && at::GradMode::is_enabled()));
    #endif
      AT_ASSERTM(
          src.impl_->is_contiguous(),
          "Right now only copy of contiguous source Tensor is supported.");
      AT_ASSERTM(
          src.impl_->storage_initialized(),
          "Cannot copy from an uninitialized Tensor");

      if (src.impl_.get() == impl_.get()) {
        return;
      }

      // Test if we need to allocate a new storage
      // Uninitialized storages are guaranteed to be uniquely owned,
      // so we don't need to swap in dst case.
      // If the dtype changed, we need to reallocate storage.
      if (impl_->dtype() != src.impl_->dtype()) {
        // NB: copy preserves device_type
        // This storage will get initialized by the mutable_data call below.
        impl_->set_storage_and_dtype(
            at::Storage::create_legacy(impl_->device_type()), src.impl_->dtype());
      }
      impl_->Resize(src.impl_->sizes());

      if (impl_->numel() > 0) {
        if (impl_->dtype().copy()) {
          AT_ASSERTM(
              impl_->device_type() == ::at::DeviceType::CPU,
              "In CopyFrom source and dest tensors must both be CPU for "
              "non-POD copy, but dest tensor was ",
              impl_->device_type());
          AT_ASSERTM(
              src.impl_->device_type() == ::at::DeviceType::CPU,
              "In CopyFrom source and dest tensors must both be CPU for "
              "non-POD copy, but src tensor was ",
              src.impl_->device_type());
          impl_->dtype().copy()(
              src.impl_->data(),
              impl_->raw_mutable_data(impl_->dtype()),
              impl_->numel());
        } else {
          // The following copy uses the current (thread local) stream for copying
          // and also takes the GPU id from the device() field passed in.
          //
          // TODO: Potentially more enforcements are necessary to avoid accidental
          // switch to sync copy if the currently set device is wrong.
          //
          // Specifically, we might need to switch to a different context device
          // here explicitly to avoid relying on user synchronizing things
          // properly.
          //
          // note: raw_mutable_data initializes device here
          void* new_data = impl_->raw_mutable_data(impl_->dtype());
          at::CopyBytes(
              impl_->numel() * impl_->itemsize(),
              src.impl_->data(),
              src.impl_->device(),
              new_data,
              impl_->device(),
              async);
        }
      }
        */
    }
}

/**
  | TODO: Remove this code in a separate
  | diff, since we only have one GetTensorInfo
  | function now
  |
  */
pub fn get_tensor_info_function(id: TypeIdentifier) -> TensorInfoCall {
    
    todo!();
    /*
        auto f = tensor_info_call_registry_.find(id);
      if (f == tensor_info_call_registry_.end()) {
        return nullptr;
      }
      return f->second;
    */
}

pub fn register_tensor_info_function(id: TypeIdentifier, c: TensorInfoCall)  {
    
    todo!();
    /*
        tensor_info_call_registry_[id] = c;
    */
}

/// resize helper function
pub fn tensor_vector_resize(
    tensors: &mut Vec<Tensor>,
    size:    i32,
    ty:      DeviceType)  
{
    todo!();
    /*
        tensors.reserve(size);
      for (auto i = 0; i < size; ++i) {
        tensors.emplace_back(type);
      }
    */
}

/// Tensor factory function
pub fn empty(dims: &[i32], options: TensorOptions) -> Tensor {
    
    todo!();
    /*
        // TODO: merge this with at::empty after Tensor is merged
      auto tensor = Tensor(dims, options.device());
      tensor.raw_mutable_data(options.dtype());
      return tensor;
    */
}

pub fn get_tensor_info(
    c:        *const c_void,
    capacity: *mut usize,
    device:   *mut DeviceOption) -> Vec<i64> {
    
    todo!();
    /*
        CHECK(capacity);
      const Tensor* tc = static_cast<const Tensor*>(c);
      CHECK(tc);
      CHECK(tc->unsafeGetTensorImpl());
      CHECK(tc->unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
      *capacity = tc->storage().nbytes();
      ExtractDeviceOption(device, tc->GetDevice());
      return tc->sizes().vec();
    */
}

pub fn get_int_8tensor_info(
    c:        *const c_void,
    capacity: *mut usize,
    device:   *mut DeviceOption) -> Vec<i64> {
    
    todo!();
    /*
        const int8::Int8TensorCPU* int8_tensor =
          static_cast<const int8::Int8TensorCPU*>(c);
      return GetTensorInfo(&(int8_tensor->t), capacity, device);
    */
}

pub fn get_type_call_function(id: TypeIdentifier) -> TypeCall {
    
    todo!();
    /*
        auto f = type_call_registry_.find(id);
      if (f == type_call_registry_.end()) {
        return nullptr;
      }
      return f->second;
    */
}

pub fn register_type_call_function(id: TypeIdentifier, c: TypeCall)  {
    
    todo!();
    /*
        type_call_registry_[id] = c;
    */
}

pub fn get_tensor_type(c: *const c_void) -> TypeMeta {
    
    todo!();
    /*
        const Tensor* tc = static_cast<const Tensor*>(c);
      return tc->dtype();
    */
}

pub fn get_int_8tensor_type(c: *const c_void) -> TypeMeta {
    
    todo!();
    /*
        const int8::Int8TensorCPU* int8_tensor =
          static_cast<const int8::Int8TensorCPU*>(c);
      return (int8_tensor->t).dtype();
    */
}
