crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/TensorImpl.h]

pub struct C10Tensor {
    dims: Vec<i64>,
    data: *mut u8,
}

/**
  | A utility function to convert vector<int>
  | to vector<int64_t>.
  |
  */
#[inline] pub fn to_vectorint64_t(src: &[i32]) -> Vec<i64> {
    
    todo!();
        /*
            return vector<int64_t>(src.begin(), src.end());
        */
}

/**
  | Return product of all dimensions starting
  | from k
  |
  */
#[inline] pub fn size_from_dim(
        k:    i32,
        dims: &[i32]) -> i64 {
    
    todo!();
        /*
            int64_t r = 1;
      for (size_t i = k; i < dims.size(); ++i) {
        r *= dims[i];
      }
      return r;
        */
}

/**
  | Product of all dims up to k (not including
  | dims[k])
  |
  */
#[inline] pub fn size_to_dim(
        k:    i32,
        dims: &[i32]) -> i64 {
    
    todo!();
        /*
            TORCH_CHECK((unsigned)k <= dims.size());
      int64_t r = 1;
      for (int i = 0; i < k; ++i) {
        r *= dims[i];
      }
      return r;
        */
}

/**
  | Product of all dims between k and l (not
  | including dims[k] and dims[l])
  |
  */
#[inline] pub fn size_between_dim(
    k:    i32,
    l:    i32,
    dims: &[i32]) -> i64 {
    
    todo!();
        /*
            TORCH_CHECK((unsigned)l < dims.size());
      int64_t r = 1;
      if (k < l) {
        for (int i = k + 1; i < l; ++i) {
          r *= dims[i];
        }
      } else {
        for (int i = l + 1; i < k; ++i) {
          r *= dims[i];
        }
      }
      return r;
        */
}

/**
  | Wrap around axis_index if it is negative,
  | s.t., -1 is the last dim
  |
  */
#[inline] pub fn canonical_axis_index(
    axis_index: i32,
    ndims:      i32) -> i32 {
    
    todo!();
        /*
            TORCH_CHECK(axis_index >= -ndims);
      TORCH_CHECK(axis_index < ndims);
      if (axis_index < 0) {
        return axis_index + ndims;
      }
      return axis_index;
        */
}

pub type PlacementDtor = fn(_0: *mut c_void, _1: usize) -> c_void;

/**
  | A Context that will call extra placement
  | deleter during deconstruction.
  | 
  | Accept a already constructed DataPtr
  | and store it as member during destruction,
  | we'll call extra deleter on the underlying
  | data pointer before the DataPtr is destructed.
  | `data_ptr_` owns the memory.
  |
  */
pub struct PlacementDeleteContext {
    data_ptr:       DataPtr,
    placement_dtor: PlacementDtor,
    size:           usize,
}

impl Drop for PlacementDeleteContext {

    fn drop(&mut self) {
        todo!();
        /*
            placement_dtor_(data_ptr_.get(), size_);
        // original memory will be freed when data_ptr_ is destructed
        */
    }
}

impl PlacementDeleteContext {
    
    pub fn new(
        data_ptr:       DataPtr,
        placement_dtor: PlacementDtor,
        size:           usize) -> Self {
    
        todo!();
        /*
        : data_ptr(move(data_ptr)),
        : placement_dtor(placement_dtor),
        : size(size),

        
        */
    }
}

pub trait AutogradMetaInterface:
SetRequiresGrad
+ RequiresGrad
+ MutableGrad
+ Grad
+ FwGrad
+ SetFwGrad {}

pub trait SetRequiresGrad {

    fn set_requires_grad(&mut self, 
        requires_grad: bool,
        self_impl:     *mut TensorImpl);
}

pub trait RequiresGrad {

    fn requires_grad(&self) -> bool;
}

pub trait MutableGrad {

    fn mutable_grad(&mut self) -> &mut C10Tensor;
}

pub trait Grad {

    fn grad(&self) -> &C10Tensor;
}

pub trait FwGrad {

    fn fw_grad(&self, 
        level: u64,
        self_: &C10Tensor) -> &C10Tensor;
}

pub trait SetFwGrad {

    fn set_fw_grad(&mut self, 
        new_grad:      &C10Tensor,
        self_:         &C10Tensor,
        level:         u64,
        is_inplace_op: bool);
}

/**
  | Unfortunately, the definition of AutogradMeta
  | lives in a separate compilation unit than
  | TensorImpl (libtorch.so versus libc10.so) which
  | means that we cannot construct an AutogradMeta
  | from TensorImpl, not even from the cpp file.
  |
  | So we have to indirect it through a factory
  | function which will be initialized when we load
  | libtorch.so.
  */
pub trait AutogradMetaFactoryInterface:
Make
+ UndefinedTensor {}

pub trait Make {
    
    fn make(&self) -> Box<dyn AutogradMetaInterface>;
}

pub trait UndefinedTensor {

    /**
     | This method is the dumbest method. But
     | I don't have access to C10Tensor (not TensorImpl)
     | which is undefined in this header.
     |
     */
    fn undefined_tensor(&self) -> &C10Tensor;
}

pub struct AutogradMetaFactoryRegisterer {

}

impl AutogradMetaFactoryRegisterer {
    
    pub fn new(factory: *mut dyn AutogradMetaFactoryInterface) -> Self {
    
        todo!();
        /*
            SetAutogradMetaFactory(factory);
        */
    }
}

pub type PyInterpreterNameSig   = fn(_0: *const PyInterpreter) -> String;
pub type PyInterpreterDecrefSig = fn(_0: *const PyInterpreter, _1: *mut PyObject) -> ();

/**
  | Note [Python interpreter tag]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  | We store a PyObject on TensorImpl so that we
  | can efficiently translate tensors into the
  | Python representations.  However, in some
  | situations (torchdeploy) there may be multiple
  | Python interpreters in a single process and we
  | must take care not to accidentally mix up
  | PyObjects with the wrong interpreters.  Thus,
  | we also tag every TensorImpl with the Python
  | interpreter it corresponds to.
  |
  | With torchdeploy, we have these invariants:
  |
  |  - Any given TensorImpl can be associated with
  |  AT MOST one Python interpreter.
  |
  |    We represent the interpreter tag as a memory
  |    address to an instance of a virtual class
  |    that is allocated once per interpreter (this
  |    is so that we can request the interpreter to
  |    perform operations for us, if necessary).
  |
  |  - A given TensorImpl's interpreter tag can
  |    only go from uninitialized to tagged; once
  |    tagged, this is a quiescent state (once
  |    tagged to an interpreter, ALWAYS tagged to
  |    that interpreter)
  |
  |  - A thread may mutate the PyObject field of
  |    a TensorImpl if and only if it holds the GIL
  |    for the interpreter tagged on the
  |    TensorImpl.  (If the TensorImpl is not
  |    tagged, it must first atomically claim its
  |    tag before it can validly write)
  |
  | The PyInterpreter object itself is a class that
  | contains some function pointers for interacting
  | with the interpreter.  For now this is just for
  | debugging, but if a C10Tensor can own a PyObject,
  | the interpreter can be used to free it.
  |
  | WARNING: This class has to be written very
  | carefully, because it may be possible for
  | a C10Tensor to have a reference an interpreter
  | corresponding to a shared library that has
  | ALREADY BEEN UNLOADED.  This makes blindly
  | calling virtual methods very dangerous, because
  | the vtable may be garbage at that point (on
  | a good day, you might get "pure virtual method
  | called").
  |
  | The idea to solve this problem is we always
  | leak PyInterpreters (so they always stay live
  | even after dlclose), and disarm the "virtual
  | methods" by replacing them with function
  | pointers that just no-op.  This can't be done
  | with a traditional C++ vtable, so we have to
  | roll our own.
  |
  | NB: The downside with representing
  | PyInterpreter tags as full objects is that it
  | takes an extra word on TensorImpl.  If tags
  | were instead just integer indices, on 64-bit
  | architectures we could pack the tag and
  | PyObject together into a single atomic word.
  | On 32-bit architectures we could simply say
  | that only one Python interpreter is supported
  | (erroring if a nontrivial interpreter tag is
  | attempted to be set).
  |
  | The difficulty with this scheme is we need to
  | maintain an out-of-line table to get at the
  | PyInterpreters so that we can do virtual method
  | calls on them, and registration/deregistration
  | to this table must be done in a thread safe
  | manner.  This can be easily done if the number
  | of possible PyInterpreters is small enough
  | (e.g., 8-bit integer) by simply preallocating
  | an array of sufficient size to hold all
  | possible interpreters.  Surely 128 threads is
  | more than enough for anyone!
  |
  | I didn't decide to do this technique at the
  | moment, because the extra word added by the
  | PyInterpreter tag takes us to 24 words, which
  | means that we still fit inside three eight word
  | cache lines.  If you need to penny pinch
  | another word consider doing this!
  */
pub struct PyInterpreter {

    /**
     | For debugging purposes only
     |
     */
    name_fn:   *mut PyInterpreterNameSig,
    decref_fn: *mut PyInterpreterDecrefSig,
}

impl PyInterpreter {

    pub fn new(
        name_fn:   *mut PyInterpreterNameSig,
        decref_fn: *mut PyInterpreterDecrefSig) -> Self {
    
        todo!();
        /*
        : name_fn(name_fn),
        : decref_fn(decref_fn),

        
        */
    }

    /**
      | UBSAN suppression fixes: "call to function
      | (anonymous namespace)::concrete_decref_fn(PyInterpreter const*,
      | _object*) through pointer to incorrect function type 'void (*)(const
      | PyInterpreter *, _object *)'" See
      | https://github.com/google/sanitizers/issues/911
      */
    pub fn name(&self) -> String {
        
        todo!();
        /*
            return (*name_fn_)(this);
        */
    }

    /**
      | Run Py_DECREF on a PyObject. We DO NOT
      | assume the GIL is held on call
      |
      */
    pub fn decref(&self, pyobj: *mut PyObject)  {
        
        todo!();
        /*
            return (*decref_fn_)(this, pyobj);
        */
    }
}

/**
  | PyInterpreterStatus describes what the state of
  | its interpreter tag is, relative to the thread
  | currently holding the GIL.
  |
  */
pub enum PyInterpreterStatus {

    /**
     | We just allocated the C10Tensor, it hasn't
     | escaped to other threads, we know that it
     | definitely hasn't been tagged to be
     | associated with an interpreter.
     */
    DEFINITELY_UNINITIALIZED,

    /**
     | We queried the interpreter field and it
     | looked uninitialized.  But another thread
     | may have raced with us to tag it with some
     | other interpreter id.  So we will have to
     | do a CEX to make sure we can actually nab
     | it.
     */
    MAYBE_UNINITIALIZED,

    /**
     | We queried the interpreter field and it was
     |  tagged to belong to us.
     |
     | This means we have sole write access (as we
     | hold the GIL for this interpreter)
     */
    TAGGED_BY_US,

    /**
     | Someone else tagged this. We can't use
     | this TensorImpl from Python.
     |
     */
    TAGGED_BY_OTHER,
}

pub trait NamedTensorMetaInterface {
    
    fn clone(&self) -> Box<dyn NamedTensorMetaInterface> {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(
            false, "Not implemented: NamedTensorMetaInterface::clone");
      }{
        */
    }
    
    fn slow_dim(&self) -> i64 {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(
            false, "Not implemented: NamedTensorMetaInterface::slow_dim");
      }{
        */
    }
}

pub struct VersionCounter {
    link:    LinkedListLink,
    version: Atomic<u32>,
}

intrusive_adapter!(pub VersionCounterAdapter = Box<VersionCounter>: VersionCounter { link: LinkedListLink });

impl VersionCounter {
    
    pub fn new(version: u32) -> Self {
    
        todo!();
        /*
        : version(version),

        
        */
    }
}

/**
  | Note [Disabled VariableVersion]
  |
  | VariableVersion struct has an intrusive_ptr
  |   pointing VersionCounter struct with an atomic
  |   variable. Thus
  |   `VariableVersion(/*version=*/0)` is not as
  |   cheap as we expected. In some cases
  |   constructing a VariableVersion with version
  |   0 is not necessary so we add a cheap
  |   constructor which doesn't allocate the
  |   intrusive_ptr.
  |
  | Example use cases are:
  |
  |  - Inference tensors don't track version
  |      counter, so they'll just always have
  |      disbaled VariableVersion.
  |
  |  - In SavedVariable class we override
  |    version_counter_ inside its construtor so
  |    that we can use the cheap constructor there.
  |
  */
pub enum VariableVersionDisabled { DISABLED }

/**
  | NOTE [ Version Counter Sharing ]
  |
  | Every C10Tensor has a version counter. Version
  | counters are incremented whenever the data or
  | size of a tensor changes through in-place
  | Variable operations.
  |
  | Version counters are used to detect
  | modifications to saved variables which would
  | result in incorrect gradient
  | calculations. Version counters may be shared
  | between Variables:
  |
  | 1. A view shares the version counter of the
  |    base Variable,
  |
  | 2. `x.detach()` shares the version counter of
  |    `x`,
  |
  | 3. Unpacked saved variables share the version
  |    counter of the source.
  |
  | Version counters are not shared in these
  | scenarios:
  |
  | 1. When we replace a `Variable`'s underlying
  | `C10Tensor` by calling `set_data(...)`,
  |
  | 2. `x.data` does not share the version counter
  | of `x`. (See discussion at
  | https://github.com/pytorch/pytorch/issues/5396)
  |
  | Question: Why do we put the version counter in
  | TensorImpl instead of AutogradMeta?
  |
  | Answer: After the Variable/C10Tensor merge,
  | a tensor will not have AutogradMeta when its
  | `requires_grad_` is false, but when we use this
  | tensor in the forward pass of a function that
  | requires saving this tensor for backward, we
  | need to keep track of this tensor's version to
  | make sure it's always valid in the autograd
  | graph.
  |
  | To achieve this goal, we put the version
  | counter in TensorImpl instead of AutogradMeta,
  | and have it always be available. This allows us
  | to have the optimization of not carrying
  | AutogradMeta when a tensor doesn't require
  | gradient.
  |
  | A hypothetical alternative way to achieve this
  | goal is to initialize AutogradMeta and create
  | the version counter for the non-requires-grad
  | tensor only when it's saved for
  | backward. However, since saving a tensor for
  | backward happens in the forward pass, and our
  | invariant is that forward pass needs to be
  | thread-safe, lazy-initializing AutogradMeta
  | when saving a tensor can introduce race
  | conditions when we are running the forward pass
  | in multi-thread scenarios, thus making the
  | forward pass not thread-safe anymore, which
  | breaks the invariant.
  */
pub struct VariableVersion {
    version_counter: VersionCounterAdapter,
}

impl VariableVersion {

    /**
      | It's okay to return true even for inference
      | tensor which doesn't have version counter
      | enabled.
      |
      | We want to be permissive here since in many
      | cases (e.g. make_variable) we can move
      | a TensorImpl if there's no other uses which
      | saves us an additional TensorImpl
      | allocation.
      */
    pub fn unique(&self) -> bool {
        
        todo!();
        /*
            return version_counter_ ? 1 == version_counter_.use_count() : true;
        */
    }

    /**
      | NOTE: As of C++11 and 14,
      | default-constructing a atomic variable
      | leaves it in a persistently undefined
      | state. See
      | https://cplusplus.github.io/LWG/issue2334.
      */
    pub fn new_with_version(version: u32) -> Self {
    
        todo!();
        /*


            : version_counter_(make_intrusive<VersionCounter>(version))
        */
    }
    
    pub fn enabled(&self) -> bool {
        
        todo!();
        /*
            return version_counter_;
        */
    }

    /**
      | Note [Inplace update inference tensor]
      |
      | 1. Inplace update to inference tensor is
      | forbidden in normal mode. For example:
      | inference_tensor.copy_(normal_tensor_requires_grad)
      |
      |   This inplace makes inference_tensor have
      |   requires_grad=True and have a grad_fn.
      |   This is bad because views of
      |   `inference_tensor` created in
      |   InferenceMode won't be able to know the
      |   grad_fn since their ViewMeta were not
      |   recorded. To match NoGradMode behavior
      |   that "inplace update to a view created in
      |   NoGradMode raise an error", we just ban
      |   inplace update to inference tensor since
      |   we can't tell if an inference tensor is
      |   a view created in InferenceMode.
      |
      |   Note that views of normal tensor created
      |   in InferenceMode has proper ViewMeta so
      |   that they're aware of the grad_fn
      |   correctly.
      |
      | 2. Inplace update to inference tensor in
      |  inference tensor doesn't bump version
      |  counter.
      |
      |    * It either doesn't call bump() by
      |    skipping ADInplaceOrView kernel,
      |
      |      - e.g. inference_tensor.add_(1)
      |
      |    * or bump() is a no-op for inference
      |    tensor.
      |
      |      - e.g. inference_tensor.add_(normal_tensor)
      */
    pub fn bump(&mut self)  {
        
        todo!();
        /*
            // TODO: Replace the link to the documentation once it's available.
        TORCH_CHECK(
            version_counter_ || InferenceMode::is_enabled(),
            "Inplace update to inference tensor outside InferenceMode is not allowed."
            "You can make a clone to get a normal tensor before doing inplace update."
            "See https://github.com/pytorch/rfcs/pull/17 for more details.");
        if (version_counter_) {
          ++version_counter_->version_;
        }
        */
    }

    /**
      | Inference tensor doesn't have version
      | counter so it shouldn't be accessed.
      |
      */
    pub fn current_version(&self) -> u32 {
        
        todo!();
        /*
            TORCH_CHECK(
            version_counter_, "Inference tensors do not track version counter.");
        return version_counter_->version_;
        */
    }
}

/**
  | -----------
  | @note
  | 
  | Some TensorImpl methods are small and
  | not overridden in the
  | PyTorch codebase itself, but may theoretically
  | need to be overridden by third-party
  | TensorImpl subclasses. This macro
  | allows users that need maximum performance
  | and don't need these extension points
  | to disable them with a build-time flag.
  | (In particular,
  | XLA's XLATensorImpl currently overrides
  | these methods, so we can't enable this
  | flag by default.)
  |
  */
lazy_static!{
    /*
    #ifdef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
    #define TENSORIMPL_MAYBE_VIRTUAL
    #else
    #define TENSORIMPL_MAYBE_VIRTUAL virtual
    #endif
    */
}

/**
  | Note [Enum ImplType]
  |
  | This enum is temporary.
  |
  | In the followup refactor we should think about
  | how to specialize TensorImpl creation for
  | view tensors.
  |
  | Currently we only special case its key_set_ but
  | there's also potential to share
  | version_counter_ directly without creating
  | first and then override in as_view.
  */
pub enum TensorImplType { VIEW }

pub trait TensorImplInterface:
ReleaseResources
+ IsContiguousCustom
+ Strides {

    /**
      | Returns the human-readable name of
      | the actual type of this object (e.g.,
      | TensorImpl, BatchedTensorImpl, etc.).
      | Used for error messages.
      |
      */
    fn tensorimpl_type_name(&self) -> *const u8 {
        
        todo!();
        /*
            return "TensorImpl";
        */
    }

    /**
      | Change the size at some dimension. This
      | DOES NOT update strides; thus, most
      | changes to size will not preserve contiguity.
      | You probably also want to call set_stride()
      | when you call this.
      | 
      | TODO: This should be jettisoned in favor
      | of `set_sizes_and_strides`, which
      | is harder to misuse.
      |
      */
    fn set_size(&mut self, 
        dim:      i64,
        new_size: i64)  {
        
        todo!();
        /*
            TORCH_CHECK(
            allow_tensor_metadata_change(),
            "set_size ",
            err_msg_tensor_metadata_change_not_allowed);
        sizes_and_strides_.size_at(dim) = new_size;
        refresh_numel();
        refresh_contiguous();
        */
    }

    /**
      | Change the stride at some dimension.
      | 
      | TODO: This should be jettisoned in favor
      | of `set_sizes_and_strides`, which
      | is harder to misuse.
      |
      */
    fn set_stride(&mut self, 
        dim:        i64,
        new_stride: i64)  {
        
        todo!();
        /*
            TORCH_CHECK(
            allow_tensor_metadata_change(),
            "set_stride ",
            err_msg_tensor_metadata_change_not_allowed);
        sizes_and_strides_.stride_at_unchecked(dim) = new_stride;
        refresh_contiguous();
        */
    }

    /**
      | Set the offset into the storage of this
      | tensor.
      | 
      | WARNING: This does NOT check if the tensor
      | is in bounds for the new location at the
      | storage; the caller is responsible
      | for checking this (and resizing if necessary.)
      |
      */
    fn set_storage_offset(&mut self, storage_offset: i64)  {
        
        todo!();
        /*
            TORCH_CHECK(
            allow_tensor_metadata_change(),
            "set_storage_offset ",
            err_msg_tensor_metadata_change_not_allowed);
        storage_offset_ = storage_offset;
        */
    }

    /**
      | Shallow-copies data from another TensorImpl
      | into this TensorImpl.
      | 
      | For why this function doesn't check
      | this TensorImpl's `allow_tensor_metadata_change_`,
      | see NOTE [ TensorImpl Shallow-Copying
      | ].
      |
      */
    fn shallow_copy_from(&mut self, impl_: TensorImplAdapter)  {
        
        todo!();
        /*
            copy_tensor_metadata(
            /*src_impl=*/impl.get(),
            /*dest_impl=*/this,
            /*version_counter=*/version_counter(),
            /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
        refresh_numel();
        refresh_contiguous();
        */
    }
}

pub trait ReleaseResources {

    /**
     | Release (decref) storage, and any other
     | external allocations. This override
     | is for `intrusive_ptr_target` and
     | is used to implement weak tensors.
     |
     */
    fn release_resources(&mut self);
}

pub trait Strides {

    /**
     | Return a reference to the strides of
     | this tensor. This reference remains
     | valid as long as the tensor is live and
     | not restrided.
     |
     */
    fn strides(&self) -> &[i32];
}

pub trait IsContiguousCustom {

    /**
      | Customization point for is_contiguous;
      | must also set_has_contiguity_policy(HasContiguityPolicy::Custom)
      | for this to be called.
      |
      */
    fn is_contiguous_custom(&self, memory_format: MemoryFormat) -> bool;
}

/**
 | Policy for adjusting the behavior of
 | is_contiguous(). Allows subclass customization
 |   while still being able to inline
 |   is_contiguous() in the common case.
 |
 */
#[repr(u8)]
pub enum HasContiguityPolicy {

    /**
      | Default behavior: check is_contiguous_
      | and similar bitflags.
      |
      */
    Default,

    /**
      | Throw a generic error message that this
      | tensor type does not support is_contiguous.
      |
      */
    ContiguityNotSupported,

    /**
      | Call virtual is_contiguous_custom
      | method to implement custom is_contiguous
      | behavior.
      |
      */
    CustomBehavior,
}

/**
 | The low-level representation of a tensor, which
 | contains a pointer to a storage (which contains
 | the actual data) and metadata (e.g., sizes and
 | strides) describing this particular view of the
 | data as a tensor.
 |
 | Some basic characteristics about our in-memory
 |  representation of tensors:
 |
 |  - It contains a pointer to a storage struct
 |     (Storage/StorageImpl) which contains the
 |     pointer to the actual data and records the
 |     data type and device of the view.  This
 |     allows multiple tensors to alias the same
 |     underlying data, which allows to efficiently
 |    implement differing *views* on a tensor.
 |
 |  - The tensor struct itself records
 |     view-specific metadata about the tensor,
 |     e.g., sizes, strides and offset into
 |     storage. Each view of a storage can have
 |     a different size or offset.
 |
 |  - This class is intrusively refcounted.  It is
 |     refcounted so that we can support prompt
 |     deallocation of large tensors; it is
 |     intrusively refcounted so that we can still
 |     perform reference counted operations on raw
 |     pointers, which is often more convenient
 |    when passing tensors across language
 |    boundaries.
 |
 |  - For backwards-compatibility reasons, a tensor
 |     may be in an uninitialized state.  A tensor
 |     may be uninitialized in the following two
 |     ways:
 |
 |      - A tensor may be DTYPE UNINITIALIZED.
 |         A tensor of this form has an
 |         uninitialized dtype.  This situation
 |         most frequently arises when a user
 |         writes C10Tensor x(CPU).  The dtype and is
 |         subsequently initialized when
 |         mutable_data<T>() is
 |        invoked for the first time.
 |
 |      - A tensor may be STORAGE UNINITIALIZED.
 |         A tensor of this form has non-zero size,
 |         but has a storage with a null data
 |         pointer. This situation most frequently
 |         arises when a user calls Resize() or
 |         FreeMemory().  This is because Caffe2
 |         historically does lazy allocation: allocation of data
 |         doesn't occur until mutable_data<T>() is
 |         invoked.  A tensor with zero size is
 |         always storage initialized, because no
 |         allocation is necessary in this case.
 |
 |    All combinations of these two uninitialized
 |    states are possible.
 |
 |    Consider the following transcript in
 |    idiomatic Caffe2 API:
 |
 |      // x is storage-initialized, dtype-UNINITIALIZED
 |      C10Tensor x(CPU); 
 |
 |      // x is storage-UNINITIALIZED, dtype-UNINITIALIZED
 |      x.Resize(4); 
 |
 |      // x is storage-initialized, dtype-initialized
 |      x.mutable_data<float>(); 
 |
 |      // x is storage-UNINITIALIZED, dtype-initialized.
 |      x.FreeMemory(); 
 |
 |    All other fields on tensor are always
 |    initialized.  In particular, size is always
 |    valid. (Historically, a tensor declared as
 |    C10Tensor x(CPU) also had uninitialized size,
 |    encoded as numel == -1, but we have now
 |    decided to default to zero size, resulting
 |    in numel == 0).
 |
 |    Uninitialized storages MUST be uniquely
 |    owned, to keep our model simple.  Thus, we
 |    will reject operations which could cause an
 |    uninitialized storage to become shared (or
 |    a shared storage to become uninitialized,
 |    e.g., from FreeMemory).
 |
 |    In practice, tensors which are
 |    storage-UNINITIALIZED and
 |    dtype-UNINITIALIZED are *extremely*
 |    ephemeral: essentially, after you do
 |    a Resize(), you basically always call
 |    mutable_data() immediately afterwards.  Most
 |    functions are not designed to work if given
 |    a storage-UNINITIALIZED, dtype-UNINITIALIZED
 |    tensor.
 |
 |    We intend to eliminate all uninitialized
 |    states, so that every tensor is fully
 |    initialized in all fields.  Please do not
 |    write new code that depends on these
 |    uninitialized states.
 */
pub struct TensorImpl {
    link:    LinkedListLink,
    storage: Storage,

    /**
      | This pointer points to an AutogradMeta struct that stores autograd-specific
      | fields (such as grad_ / grad_fn_ / grad_accumulator_). This pointer always
      | has unique ownership (meaning only one TensorImpl can own it at a time).
      |
      | autograd_meta_ can be nullptr, as an optimization.
      |
      | When this occurs, it is
      | equivalent to having an autograd_meta_ pointing to a default constructed
      | AutogradMeta; intuitively, tensors which don't require grad will have this
      | field set to null.
      |
      | This means accessors on autograd_meta_ have to be careful to test if they
      | got a nullptr, and handle default behavior appropriately in that case.
      |
      | Note that we don't enforce the invariant that if the AutogradMeta is
      | default constructed, it is nullptr (to do this, we'd have to continuously
      | check if an AutogradMeta became, by mutation, equal to the default
      | constructed form.
      |
      | (This might be useful, but it seems rare enough that
      | a requires_grad=True variable will turn back into the requires_grad=False
      | version.)
      |
      | So there are three representable states:
      |
      |    - 1. autograd_meta_ == nullptr
      |    - 2. autograd_meta_ is default constructed (semantically, same as (1))
      |    - 3. autograd_meta_ has nontrivial information content
      |
      */
    autograd_meta:     Box<dyn AutogradMetaInterface>,    // default = nullptr
    named_tensor_meta: Box<dyn NamedTensorMetaInterface>, // default = nullptr
    version_counter:   VariableVersion,

    /**
      | This field contains the interpreter tag for
      | this object.
      |
      | See Note [Python interpreter tag] for
      | general context
      |
      | Note [Memory ordering on Python interpreter
      | tag]
      | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |
      | What memory_order do we need when accessing
      | this atomic?
      |
      | We don't need a single total
      | modification order (as provided by
      | memory_order_seq_cst) as pyobj_interpreter_
      | is monotonic:
      |
      | it can only transition from
      | -1 to some positive integer and never
      | changes afterwards.
      |
      | Because there is only one modification, it
      | trivially already has a total modification
      | order (e.g., we don't need fences or locked
      | instructions on x86)
      |
      | In fact, one could make a reasonable
      | argument that relaxed reads are OK, due to
      | the presence of external locking (GIL) to
      | ensure that interactions with other data
      | structures are still correctly
      | synchronized, so that we fall in the
      | "Single-Location Data Structures" case as
      | described in
      | http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p2055r0.pdf
      |
      | However, on x86, it doesn't matter if I use
      | acquire or relaxed on the load as I get the
      | same assembly in both cases.  So I just use
      | the more conservative acquire (which will
      | impede compiler optimizations but I don't
      | care)
      */
    pyobj_interpreter: Atomic<*mut PyInterpreter>,

    /**
      | This field contains a weak reference to
      | a PyObject representing this C10Tensor.
      |
      | It MUST NOT be a strong reference, as that
      | would create a reference cycle between
      | C10Tensor and the PyObject.  If pyobj is
      | nullptr, when we transfer C10Tensor to Python,
      | we allocate a new PyObject for it and set
      | this field.
      |
      | This field does not have to be protected by
      | an atomic as it is only allowed to be
      | accessed when you hold the GIL.
      |
      | When a PyObject dies, you are obligated to
      | clear this field (otherwise, you will try
      | to use-after-free the pyobj);
      |
      | this currently occurs in THPVariable_clear
      | in torch/csrc/autograd/python_variable.cpp
      */
    pyobj:             *mut PyObject,
    sizes_and_strides: SizesAndStrides,
    storage_offset:    i64, // default = 0

    /**
     | If sizes and strides are empty, the numel
     | is 1!!
     | 
     | However, most of the time, we will immediately
     | set sizes to {0} and reset numel to 0.
     | (Can't do that in the default initializers,
     | because there's no way to spell "allocate
     | a one-element array" for strides_).
     |
     */
    numel:      i64, // default = 1

    /**
      | INVARIANT: When storage is non-null,
      | this type meta must agree with the type
      | meta in storage
      |
      */
    data_type:  TypeMeta,

    /**
      | NOTE [optional operator usage in Cuda]
      | 
      | Our optional definition doesn't compile
      | in .cu file if `value()` or `operator->`
      | are used. Instead, we always use `operator*`.
      | 
      | See https://github.com/pytorch/pytorch/issues/18496
      | for more info.
      | 
      | If this is too burdensome to maintain,
      | we can just manually implement this
      | with an additional bool.
      | 
      | INVARIANT: When storage is non-null,
      | this Device must agree with the type
      | meta in storage.
      | 
      | INVARIANT: device_opt_ is only nullopt
      | for undefined tensors (which do not
      | have a device.)
      |
      */
    device_opt: Option<Device>,

    /**
     | C10Tensor is contiguous
     |
     */
    is_contiguous:                  bool, // : 1;

    /**
      | HasContiguityPolicy : 2;
      |
      */
    has_contiguity:                 u8,

    /**
      | C10Tensor is a subclass that does not permit
      | storage access. : 1;
      |
      */
    storage_access_should_throw:    bool,

    /**
      | C10Tensor is stored in the channels last
      | 2d memory format, when dimensions order
      | is (N)CHW and C-strides < W-strides
      | < H-strides (< N-strides) (If size of
      | any dimension is equal to 1, this dimension
      | strides value is not taken into account).
      | : 1;
      |
      */
    is_channels_last:               bool,

    /**
      | Channels last contiguous tensor is
      | channel last tensor which occupies
      | contiguous memory block. : 1;
      |
      */
    is_channels_last_contiguous:    bool,

    /**
      | C10Tensor is stored in the channels last
      | 3d memory format, when dimensions order
      | is (N)CDHW and C-strides < W-strides
      | < H-strides < D - strides (<
      | 
      | N-strides) (If size of any dimension
      | is equal to 1, this dimension strides
      | value is not taken into account). : 1;
      |
      */
    is_channels_last_3d:            bool,

    /**
      | Channels last 3d contiguous tensor
      | is channel last 3d tensor which occupies
      | contiguous memory block. : 1;
      |
      */
    is_channels_last_3d_contiguous: bool,

    /**
      | Dense tensor is the tensor that store
      | values in a contiguous block of memory.
      | Non-overlapping tensor is the tensor
      | in which elements occupy individual
      | non-repetitive memory. : 1;
      |
      */
    is_non_overlapping_and_dense:   bool,

    /**
      | : 1;
      |
      */
    is_wrapped_number:              bool,

    /**
      | NOTE [ Metadata Change for a Detached
      | C10Tensor ]
      | 
      | Normally, a user is allowed to change
      | the tensor metadata (e.g. sizes / strides
      | / storage / storage_offset) of a tensor.
      | 
      | However, if the tensor is created by
      | `t1_detached = t1.data` in Python or
      | `t1_detached = t1.detach()` in Python/C++,
      | those changes to the tensor metadata
      | of `t1_detached` will not be propagated
      | back to the original tensor `t1`. In
      | order to make such changes explicitly
      | illegal, we created the `allow_tensor_metadata_change_`
      | flag, to prevent users from changing
      | metadata of the detached tensor and
      | expecting the original tensor to also
      | be updated.
      | 
      | -----------
      | @note
      | 
      | For a full list of tensor metadata fields,
      | please see `copy_tensor_metadata()`
      | in TensorImpl and its subclasses to
      | find which fields are copied by value.
      | : 1;
      |
      */
    allow_tensor_metadata_change:   bool,

    /**
      | we decide to keep reserved_ and it will
      | live in C10Tensor after the split
      | 
      | The logic is that if Extend() or ReserveSpace()
      | were ever called, then subsequent Resize()s
      | will not free up Storage. : 1;
      |
      */
    reserved:                       bool,

    /**
      | If pyobj_ is nullptr, this is always
      | false.
      | 
      | Otherwise, this indicates whether
      | or not TensorImpl owns the pyobj_ or
      | vice versa. Ordinarily, pyobj_ owns
      | TensorImpl, but if the
      | 
      | Python object's refcount goes to zero,
      | we flip the ownership direction (to
      | make sure the pyobj stays live). : 1;
      |
      */
    owns_pyobj:                     bool,

    /**
      | The set of DispatchKeys which describe
      | this tensor. NB: this does NOT include
      | Autograd (historically, it did, but
      | not anymore!)
      | 
      | INVARIANT: named_tensor_meta_ !=
      | nullptr <==> key_set_.has(DispatchKey::Named)
      |
      */
    key_set:                        DispatchKeySet,
}

intrusive_adapter!(pub TensorImplAdapter = Box<TensorImpl>: TensorImpl { link: LinkedListLink });

/**
  | Error message to show when the user tries to
  | change tensor metadata on C10Tensor created from
  |   .data or .detach().
  |
  | See NOTE [ Metadata Change for a Detached
  | C10Tensor ] for details.
  */
lazy_static!{
    /*

    static String err_msg_tensor_metadata_change_not_allowed;

    const char* const TensorImpl::err_msg_tensor_metadata_change_not_allowed =
        "is not allowed on a C10Tensor created from .data or .detach().\n"
        "If your intent is to change the metadata of a C10Tensor (such as sizes / strides / storage / storage_offset)\n"
        "without autograd tracking the change, remove the .data / .detach() call and wrap the change in a `with torch.no_grad():` block.\n"
        "For example, change:\n"
        "    x.data.set_(y)\n"
        "to:\n"
        "    with torch.no_grad():\n"
        "        x.set_(y)";

    */
}

impl TensorImpl {

    /**
      | Construct a 1-dim 0-size tensor backed
      | by the given storage.
      |
      */
    pub fn new_with_storage(
        storage:   Storage,
        _1:        DispatchKeySet,
        data_type: TypeMeta) -> Self {
    
        todo!();
        /*
        
        */
    }

    /**
      | Construct a 1-dim 0 size tensor that
      | doesn't have a storage.
      |
      */
    pub fn new_no_storage(
        _0:         DispatchKeySet,
        data_type:  TypeMeta,
        device_opt: Option<Device>) -> Self {
    
        todo!();
        /*
        
        */
    }

    /**
      | Return the DispatchKeySet corresponding
      | to this C10Tensor, specifying all of the
      | DispatchKeys that this C10Tensor identifies
      | as. This is the information used to dispatch
      | operations on this tensor.
      |
      */
    pub fn key_set(&self) -> DispatchKeySet {
        
        todo!();
        /*
            return key_set_;
        */
    }

    /**
      | Return a reference to the sizes of this
      | tensor. This reference remains valid
      | as long as the tensor is live and not resized.
      |
      */
    pub fn sizes_maybe_virtual(&self) -> &[i32] {
        
        todo!();
        /*
            #ifdef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
            return sizes_and_strides_.sizes_arrayref();
        #else
              ;
        #endif
        */
    }

    /**
      | Return the number of dimensions of this
      | tensor. Note that 0-dimension represents
      | a C10Tensor that is a Scalar, e.g., one that
      | has a single element.
      |
      */
    pub fn dim_maybe_virtual(&self) -> i64 {
        
        todo!();
        /*
            #ifdef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
        return sizes_and_strides_.size();
    #else
          ;
    #endif
        */
    }

    /**
      | True if this tensor has storage. See
      | storage() for details.
      |
      | Allow subclasses to check that their storage_
      | is never getting set in debug builds.
      |
      | NOTE: we devirtualize this because it
      | arguably shouldn't be an error just to ask
      | subclasses if they have storage. This used
      | to throw for most subclasses, but
      | OpaqueTensorImpl wanted it to successfully
      | return false, so we went ahead and made it
      | a non-error.
      */
    pub fn has_storage_maybe_virtual(&self) -> bool {
        
        todo!();
        /*
            #ifdef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
        return storage_;
    #else
          ;
    #endif
        */
    }

    /**
      | Return the underlying storage of a C10Tensor.
      | Multiple tensors may share a single
      | storage. A Storage is an impoverished,
      | C10Tensor-like class which supports far
      | less operations than C10Tensor.
      | 
      | Avoid using this method if possible;
      | try to use only C10Tensor APIs to perform
      | operations.
      |
      */
    //#[TENSORIMPL_MAYBE_VIRTUAL] 
    pub fn storage(&self) -> &Storage {
        
        todo!();
        /*
            if (C10_UNLIKELY(storage_access_should_throw_)) {
          throw_storage_access_error();
        }
        return storage_;
        */
    }

    /**
      | Return the underlying storage, unsafely
      | assuming this is a basic strided tensor.
      | In cases where `storage` access would
      | throw, this returns a default-constructed
      | Storage.
      |
      */
    #[inline] pub fn unsafe_storage(&self) -> &Storage {
        
        todo!();
        /*
            return storage_;
        */
    }

    /**
      | The number of elements in a tensor.
      | 
      | WARNING: Previously, if you were using
      | the Caffe2 API, you could test numel()
      | == -1 to see if a tensor was uninitialized.
      | This is no longer true; numel always
      | accurately reports the product of sizes
      | of a tensor.
      |
      */
    //#[TENSORIMPL_MAYBE_VIRTUAL] 
    pub fn numel(&self) -> i64 {
        
        todo!();
        /*
            #ifdef DEBUG
        TORCH_INTERNAL_ASSERT(compute_numel() == numel_);
    #endif
        return numel_;
        */
    }
    
    pub fn unique_version(&self) -> bool {
        
        todo!();
        /*
            return version_counter_.unique();
        */
    }

    /**
      | Whether or not a tensor is laid out in
      | contiguous memory.
      | 
      | Tensors with non-trivial strides are
      | not contiguous. See compute_contiguous()
      | for the exact definition of whether
      | or not a tensor is contiguous or not.
      | 
      | -----------
      | @note
      | 
      | is_contiguous is only `TENSORIMPL_MAYBE_VIRTUAL`
      | for backward compatibility. See `set_has_contiguity_policy`
      | and `is_contiguous_custom` for the
      | encouraged customization point.
      |
      */
    //#[TENSORIMPL_MAYBE_VIRTUAL] 
    pub fn is_contiguous(&self, memory_format: Option<MemoryFormat>) -> bool {

        let memory_format: MemoryFormat = memory_format.unwrap_or(MemoryFormat::Contiguous);

        todo!();
        /*
            if (C10_UNLIKELY(
                has_contiguity_ !=
                static_cast<uint8_t>(HasContiguityPolicy::Default))) {
          return is_contiguous_nondefault_policy_impl(memory_format);
        }
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(compute_contiguous() == is_contiguous_);
        if (memory_format == MemoryFormat::ChannelsLast) {
          return is_channels_last_contiguous_;
        } else if (memory_format == MemoryFormat::ChannelsLast3d) {
          return is_channels_last_3d_contiguous_;
        }
        return is_contiguous_;
        */
    }
    
    pub fn is_sparse(&self) -> bool {
        
        todo!();
        /*
            // NB: This method is not virtual and avoid dispatches for performance
        // reasons.
        return key_set_.has(DispatchKey::SparseCPU) ||
            key_set_.has(DispatchKey::SparseCUDA) ||
            key_set_.has(DispatchKey::SparseHIP) ||
            key_set_.has(DispatchKey::SparseXPU);
        */
    }

    /**
      | Whether a tensor is sparse COO or not.
      | Use is_sparse_csr for checking CSR
      | format.
      |
      */
    pub fn is_sparse_csr(&self) -> bool {
        
        todo!();
        /*
            return key_set_.has(DispatchKey::SparseCsrCPU) ||
            key_set_.has(DispatchKey::SparseCsrCUDA);
        */
    }
    
    pub fn is_quantized(&self) -> bool {
        
        todo!();
        /*
            // NB: This method is not virtual and avoid dispatches for performance
        // reasons.
        return key_set_.has(DispatchKey::QuantizedCPU) ||
            key_set_.has(DispatchKey::QuantizedCUDA) ||
            key_set_.has(DispatchKey::QuantizedXPU);
        */
    }
    
    pub fn is_meta(&self) -> bool {
        
        todo!();
        /*
            // NB: This method is not virtual and avoid dispatches for performance
        // reasons.
        return key_set_.has(DispatchKey::Meta);
        */
    }
    
    pub fn is_cpu(&self) -> bool {
        
        todo!();
        /*
            // NB: This method is not virtual and avoid dispatches for performance
        // reasons.
        return key_set_.has(DispatchKey::CPU) ||
            key_set_.has(DispatchKey::SparseCPU) ||
            key_set_.has(DispatchKey::SparseCsrCPU) ||
            key_set_.has(DispatchKey::QuantizedCPU) ||
            key_set_.has(DispatchKey::MkldnnCPU);
        */
    }
    
    pub fn is_cuda(&self) -> bool {
        
        todo!();
        /*
            // NB: This method is not virtual and avoid dispatches for performance
        // reasons.
        return key_set_.has(DispatchKey::Cuda) ||
            key_set_.has(DispatchKey::SparseCUDA) ||
            key_set_.has(DispatchKey::SparseCsrCUDA) ||
            key_set_.has(DispatchKey::QuantizedCUDA);
        */
    }
    
    pub fn is_xpu(&self) -> bool {
        
        todo!();
        /*
            // NB: This method is not virtual and avoid dispatches for performance
        // reasons.
        return key_set_.has(DispatchKey::XPU) ||
            key_set_.has(DispatchKey::SparseXPU) ||
            key_set_.has(DispatchKey::QuantizedXPU);
        */
    }
    
    pub fn is_xla(&self) -> bool {
        
        todo!();
        /*
            return key_set_.has(DispatchKey::XLA);
        */
    }
    
    pub fn is_hip(&self) -> bool {
        
        todo!();
        /*
            // NB: This method is not virtual and avoid dispatches for performance
        // reasons.
        return key_set_.has(DispatchKey::HIP) ||
            key_set_.has(DispatchKey::SparseHIP);
        */
    }
    
    pub fn is_mkldnn(&self) -> bool {
        
        todo!();
        /*
            return key_set_.has(DispatchKey::MkldnnCPU);
        */
    }
    
    pub fn is_vulkan(&self) -> bool {
        
        todo!();
        /*
            return key_set_.has(DispatchKey::Vulkan);
        */
    }
    
    pub fn is_metal(&self) -> bool {
        
        todo!();
        /*
            return key_set_.has(DispatchKey::Metal);
        */
    }
    
    pub fn is_mlc(&self) -> bool {
        
        todo!();
        /*
            return key_set_.has(DispatchKey::MLC);
        */
    }

    /**
      | TODO: remove this once we don't automatically
      | enabled Autograd dispatch keys in
      | TensorImpl constructor.
      |
      | DON'T USE THIS API!! It's only created for
      | testing purpose in file
      | aten/src/ATen/core/boxing/impl/test_helpers.h
      */
    pub fn remove_autograd_key(&mut self)  {
        
        todo!();
        /*
            key_set_ = key_set_ - autograd_dispatch_keyset;
        */
    }

    /**
      | Inference tensor doesn't have autograd or
      | ADInplaceOrView key.
      |
      | Invariant:
      |
      |   Inference tensor has
      |   version_counter_.enabled() == false
      |
      */
    pub fn is_inference(&mut self) -> bool {
        
        todo!();
        /*
            bool no_ADInplaceOrView = !key_set_.has(DispatchKey::ADInplaceOrView);
        bool no_Autograd = (key_set_ & autograd_dispatch_keyset).empty();
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            no_ADInplaceOrView == no_Autograd,
            "ADInplaceOrView and Autograd keys must be on/off at the same time.");
        return no_ADInplaceOrView && no_Autograd;
        */
    }
    
    pub fn get_device(&self) -> i64 {
        
        todo!();
        /*
            TORCH_CHECK(device_opt_.has_value(), "tensor does not have a device");
        // See NOTE [optional operator usage in Cuda]
        return (*device_opt_).index();
        */
    }
    
    pub fn device(&self) -> Device {
        
        todo!();
        /*
            TORCH_CHECK(device_opt_.has_value(), "tensor does not have a device");
        // See NOTE [optional operator usage in Cuda]
        return *device_opt_;
        */
    }
    
    pub fn layout(&self) -> Layout {
        
        todo!();
        /*
            // NB: This method is not virtual and avoid dispatches for perf.
        if (is_sparse()) {
          return kSparse;
        } else if (is_sparse_csr()) {
          return kSparseCsr;
        } else if (is_mkldnn()) {
          return kMkldnn;
        } else {
          return kStrided;
        }
        */
    }

    /**
      | True if a tensor was auto-wrapped from
      | a C++ or Python number.
      | 
      | For example, when you write 't + 2', 2
      | is auto-wrapped into a C10Tensor with `is_wrapped_number_`
      | set to true.
      | 
      | Wrapped numbers do not participate
      | in the result type computation for mixed-type
      | operations if there are any Tensors
      | that are not wrapped numbers.
      | 
      | This is useful, because we want 't + 2'
      | to work with any type of tensor, not just
      | LongTensor (which is what integers
      | in Python represent).
      | 
      | Otherwise, they behave like their non-wrapped
      | equivalents.
      | 
      | See [Result type computation] in TensorIterator.h.
      | 
      | Why did we opt for wrapped numbers, as
      | opposed to just having an extra function
      | add(C10Tensor, Scalar)?
      | 
      | This helps greatly reduce the amount
      | of code we have to write for add, when
      | actually a C10Tensor-Scalar addition
      | is really just a C10Tensor-C10Tensor addition
      | when the RHS is 0-dim (except for promotion
      | behavior.)
      |
      */
    pub fn is_wrapped_number(&self) -> bool {
        
        todo!();
        /*
            return is_wrapped_number_;
        */
    }

    /**
      | Set whether or not a tensor was auto-wrapped
      | from a C++ or Python number. You probably
      | don't want to call this, unless you are
      | writing binding code.
      |
      */
    pub fn set_wrapped_number(&mut self, value: bool)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(dim() == 0);
        is_wrapped_number_ = value;
        */
    }

    /**
      | Returns true if C10Tensor supports as_strided
      | and as_strided_backward.
      | 
      | This is used in autograd to perform inplace
      | update on view Tensors.
      | 
      | See Note [View + Inplace update for base
      | tensor] and [View + Inplace update for
      | view tensor] for details.
      | 
      | Note this method only returns true for
      | XLA backend, where it simulates strided
      | C10Tensor to support most view ops, but
      | it cannot fully support general `as_strided`
      | case.
      | 
      | It can be expanded as needed in the future,
      | e.g sparse C10Tensor.
      |
      */
    #[inline] pub fn support_as_strided(&self) -> bool {
        
        todo!();
        /*
            return device().supports_as_strided();
        */
    }

    /*
      | ~~~~~ Autograd API ~~~~~
      | Some methods below are defined in
      | TensorImpl.cpp because C10Tensor is an incomplete
      | type.
      */

    /**
      | Whether or not the imaginary part of
      | the tensor should be negated
      |
      */
    #[inline] pub fn is_conj(&self) -> bool {
        
        todo!();
        /*
            return key_set_.has(DispatchKey::Conjugate);
        */
    }

    /**
      | Set whether or not to take the conjugate
      | of the tensor (flip the imaginary bit).
      |
      */
    pub fn set_conj(&mut self, value: bool)  {
        
        todo!();
        /*
            if (value) {
          key_set_ = key_set_.add(DispatchKey::Conjugate);
          TORCH_INTERNAL_ASSERT(isComplexType(typeMetaToScalarType(dtype())));
        } else {
          key_set_ = key_set_.remove(DispatchKey::Conjugate);
        }
        */
    }

    /**
      | Return a typed data pointer to the actual
      | data which this tensor refers to.
      | 
      | This checks that the requested type
      | (from the template parameter) matches
      | the internal type of the tensor.
      | 
      | It is invalid to call data() on a dtype-uninitialized
      | tensor, even if the size is 0.
      | 
      | WARNING: If a tensor is not contiguous,
      | you MUST use strides when performing
      | index calculations to determine the
      | location of elements in the tensor.
      | We recommend using 'TensorAccessor'
      | to handle this computation for you;
      | this class is available from 'C10Tensor'.
      |
      */
    #[inline] pub fn data_generic<T>(&self) -> *mut T {
    
        todo!();
        /*
            TORCH_CHECK(
            data_type_.Match<T>(),
            "C10Tensor type mismatch, caller expects elements to be ",
            TypeMeta::TypeName<T>(),
            ", while tensor contains ",
            data_type_.name(),
            ". ");
        return data_ptr_impl<T>();
        */
    }

    /**
      | More efficient helper for C10Tensor::data_ptr().
      | Like data<T>(), but does not do a type
      | check. Unlike the untemplated data(),
      | does check has_storage() and storage_initialized().
      |
      */
    #[inline] pub fn data_ptr_impl<T>(&self) -> *mut T {
    
        todo!();
        /*
            TORCH_CHECK(
            has_storage(),
            "Cannot access data pointer of C10Tensor that doesn't have storage");
        TORCH_CHECK(
            storage_initialized(),
            "The tensor has a non-zero number of elements, but its data is not allocated yet. "
            "Caffe2 uses a lazy allocation, so you will need to call "
            "mutable_data() or raw_mutable_data() to actually allocate memory.");
        // Caller does the type check.
        return storage_.unsafe_data<T>() + storage_offset_;
        */
    }

    /**
      | Return a void* data pointer to the actual
      | data which this tensor refers to.
      | 
      | It is invalid to call data() on a dtype-uninitialized
      | tensor, even if the size is 0.
      | 
      | WARNING: The data pointed to by this
      | tensor may not contiguous; do NOT assume
      | that itemsize() * numel() is sufficient
      | to compute the bytes that can be validly
      | read from this tensor.
      |
      */
    #[inline] pub fn data(&self)  {
        
        todo!();
        /*
            TORCH_CHECK(
            has_storage(),
            "Cannot access data pointer of C10Tensor that doesn't have storage");
        TORCH_CHECK(
            dtype_initialized(),
            "Cannot access data pointer of C10Tensor that doesn't have initialized dtype "
            "(e.g., C10Tensor x(CPU), prior to calling mutable_data<T>() on x)");
        return static_cast<void*>(
            static_cast<char*>(storage_.data()) +
            data_type_.itemsize() * storage_offset_);
        */
    }

    /**
      | Like data<T>(), but performs no checks.
      | You are responsible for ensuring that
      | all invariants required by data() are
      | upheld here.
      |
      */
    #[inline] pub fn unsafe_data<T>(&self) -> *mut T {
    
        todo!();
        /*
            return storage_.unsafe_data<T>() + storage_offset_;
        */
    }

    /**
      | Returns the TypeMeta of a tensor, which
      | describes what data type it is (e.g.,
      | int, float, ...)
      |
      */
    pub fn dtype(&self) -> TypeMeta {
        
        todo!();
        /*
            return data_type_;
        */
    }

    /**
      | Return the size of a single element of
      | this tensor in bytes.
      |
      */
    pub fn itemsize(&self) -> usize {
        
        todo!();
        /*
            TORCH_CHECK(
            dtype_initialized(),
            "Cannot report itemsize of C10Tensor that doesn't have initialized dtype "
            "(e.g., C10Tensor x(CPU), prior to calling mutable_data<T>() on x)");
        return data_type_.itemsize();
        */
    }

    /**
      | Return the offset in number of elements
      | into the storage that this tensor points
      | to. Most tensors have storage_offset()
      | == 0, but, for example, an index into
      | a tensor will have a non-zero storage_offset().
      | 
      | WARNING: This is NOT computed in bytes.
      |
      */
    //#[TENSORIMPL_MAYBE_VIRTUAL] 
    pub fn storage_offset(&self) -> i64 {
        
        todo!();
        /*
            return storage_offset_;
        */
    }

    /**
      | True if a tensor has no elements (e.g.,
      | numel() == 0).
      |
      */
    #[inline] pub fn is_empty(&self) -> bool {
        
        todo!();
        /*
            return numel() == 0;
        */
    }

    /**
      | Like set_sizes_and_strides but assumes
      | contiguous strides.
      | 
      | WARNING: This function does not check
      | if the requested sizes/strides are
      | in bounds for the storage that is allocated;
      | this is the responsibility of the caller
      |
      */
    pub fn set_sizes_contiguous(&mut self, new_size: &[i32])  {
        
        todo!();
        /*
            TORCH_CHECK(
            allow_tensor_metadata_change(),
            "set_sizes_contiguous ",
            err_msg_tensor_metadata_change_not_allowed);

        sizes_and_strides_.set_sizes(new_size);

        refresh_numel();
        empty_tensor_restride(MemoryFormat::Contiguous);
        */
    }

    /**
      | Set the sizes and strides of a tensor.
      | 
      | WARNING: This function does not check
      | if the requested sizes/strides are
      | in bounds for the storage that is allocated;
      | this is the responsibility of the caller
      |
      */
    pub fn set_sizes_and_strides(&mut self, 
        new_size:   &[i32],
        new_stride: &[i32])  {
        
        todo!();
        /*
            TORCH_CHECK(
            allow_tensor_metadata_change(),
            "set_sizes_and_strides ",
            err_msg_tensor_metadata_change_not_allowed);
        TORCH_CHECK(
            new_size.size() == new_stride.size(),
            "dimensionality of sizes (",
            new_size.size(),
            ") must match dimensionality of strides (",
            new_stride.size(),
            ")");
        const auto new_dim = new_size.size();

        sizes_and_strides_.set_sizes(new_size);

        if (new_dim > 0) {
          for (size_t dim = new_dim - 1;; dim--) {
            if (new_stride[dim] >= 0) {
              sizes_and_strides_.stride_at_unchecked(dim) = new_stride[dim];
            } else {
              // XXX: This behavior is surprising and may need to be removed to
              // support negative strides. Some pytorch functions rely on it:
              // for example, torch.cat (run TestTorch.test_cat_empty).
              if (dim == new_dim - 1) {
                sizes_and_strides_.stride_at_unchecked(dim) = 1;
              } else {
                // Keep stride monotonically increasing to match NumPy.
                sizes_and_strides_.stride_at_unchecked(dim) =
                    max<int64_t>(
                        sizes_and_strides_.size_at_unchecked(dim + 1), 1) *
                    sizes_and_strides_.stride_at_unchecked(dim + 1);
              }
            }
            if (dim == 0)
              break;
          }
        }

        refresh_numel();
        refresh_contiguous();
        */
    }

    /**
      | Set whether a tensor allows changes
      | to its metadata (e.g. sizes / strides
      | / storage / storage_offset). See NOTE
      | [ Metadata Change for a Detached C10Tensor ] for details.
      |
      */
    pub fn set_allow_tensor_metadata_change(&mut self, value: bool)  {
        
        todo!();
        /*
            allow_tensor_metadata_change_ = value;
        */
    }

    /**
      | True if a tensor allows changes to its
      | metadata (e.g. sizes / strides / storage
      | / storage_offset). See NOTE [ Metadata
      | Change for a Detached C10Tensor ] for details.
      |
      */
    pub fn allow_tensor_metadata_change(&self) -> bool {
        
        todo!();
        /*
            return allow_tensor_metadata_change_;
        */
    }

    /**
      | Set the pointer to named tensor metadata.
      |
      */
    pub fn set_named_tensor_meta(&mut self, named_tensor_meta: Box<dyn NamedTensorMetaInterface>)  {
        
        todo!();
        /*
            TORCH_WARN_ONCE(
            "Named tensors and all their associated APIs are an experimental feature ",
            "and subject to change. Please do not use them for anything important ",
            "until they are released as stable.");
    #ifdef DEBUG
        if (named_tensor_meta) {
          TORCH_INTERNAL_ASSERT(named_tensor_meta->slow_dim() == dim());
        }
    #endif
        named_tensor_meta_ = move(named_tensor_meta);
        if (named_tensor_meta_ == nullptr) {
          key_set_ = key_set_.remove(DispatchKey::Named);
        } else {
          key_set_ = key_set_.add(DispatchKey::Named);
        }
        */
    }

    /**
      | Return the pointer to named tensor metadata.
      |
      */
    pub fn named_tensor_meta(&self) -> *const dyn NamedTensorMetaInterface {
        
        todo!();
        /*
            return named_tensor_meta_.get();
        */
    }
    
    pub fn named_tensor_meta_mut(&mut self) -> *mut dyn NamedTensorMetaInterface {
        
        todo!();
        /*
            return named_tensor_meta_.get();
        */
    }
    
    pub fn has_named_tensor_meta(&self) -> bool {
        
        todo!();
        /*
            return named_tensor_meta_ != nullptr;
        */
    }

    /*
      | NOTE [ TensorImpl Shallow-Copying ]
      |
      | TensorImpl shallow-copying is used when we want
      | to have two Variables share the same tensor
      | metadata (e.g. sizes / strides / storage
      | pointer / storage_offset), but each with
      | a different autograd history. Example call
      | sites:
      |
      | 1. `var_detached = var.detach()` uses
      | `shallow_copy_and_detach()` to create
      | `var_detached` that shares the same tensor
      | metadata with `var`, but with a completely
      | new autograd history.
      |
      | 2. `var.set_data(tensor)` uses
      | `shallow_copy_from()` to copy tensor metadata
      | from `tensor` into `var`, while keeping
      | `var`'s original AutogradMeta.
      |
      | Functions that shallow-copy a TensorImpl (such
      | as `shallow_copy_and_detach()`
      | / `shallow_copy_from()`
      | / `copy_tensor_metadata()`) copy the tensor
      | metadata fields (e.g. sizes / strides
      | / storage pointer / storage_offset) by
      | value. However, the following fields are not
      | copied:
      |
      | 1. the AutogradMeta pointer, because it is
      | unique for each Variable.
      |
      | 2. the version counter, because the destination
      | TensorImpl's version counter is either set to
      | the passed-in `version_counter` (in
      | `shallow_copy_and_detach()` and
      | `copy_tensor_metadata()`), or it is kept
      | intact (in `shallow_copy_from()`). See NOTE
      | [ Version Counter Sharing ] for details.
      |
      | In `shallow_copy_and_detach()` and
      | `copy_tensor_metadata()`, the passed-in
      | `allow_tensor_metadata_change` determines
      | whether the TensorImpl shallow-copy allows
      | changes to its metadata (e.g. sizes / strides
      | / storage / storage_offset). See NOTE
      | [ Metadata Change for a Detached C10Tensor ] for
      | details.
      |
      | In `shallow_copy_from()`, we don't check the
      | destination TensorImpl's
      | `allow_tensor_metadata_change_`, because
      | `shallow_copy_from()` is used for
      | implementing functions such as
      | `var.set_data(tensor)`, which changes `var`'s
      | tensor metadata and expects its
      | `allow_tensor_metadata_change_` to be
      | ignored.
      */

    /**
      | One TensorImpl can be copied to another
      | TensorImpl if they have the same
      | 
      | DispatchKeySet. The only two special
      | cases (for legacy reason) are:
      | 
      | CPU is compatible with Cuda and SparseCPU
      | is compatible with SparseCUDA.
      |
      */
    #[inline] pub fn has_compatible_shallow_copy_type(&mut self, from: DispatchKeySet) -> bool {
        
        todo!();
        /*
            auto is_dense = [](DispatchKeySet ts) {
          return ts.has(DispatchKey::CPU) || ts.has(DispatchKey::Cuda) ||
              ts.has(DispatchKey::HIP) || ts.has(DispatchKey::XPU);
        };
        auto is_sparse = [](DispatchKeySet ts) {
          return ts.has(DispatchKey::SparseCPU) ||
              ts.has(DispatchKey::SparseCUDA) || ts.has(DispatchKey::SparseHIP) ||
              ts.has(DispatchKey::SparseXPU);
        };
        return (key_set_ == from) || (is_dense(key_set_) && is_dense(from)) ||
            (is_sparse(key_set_) && is_sparse(from));
        */
    }

    /**
      | Inference tensor doesn't have version
      | counter, set_version_counter is no-op
      | for them.
      |
      */
    pub fn set_version_counter_ref(&mut self, version_counter: &VariableVersion)  {
        
        todo!();
        /*
            TORCH_CHECK(
            !(is_inference() && version_counter.enabled()),
            "Cannot set version_counter for inference tensor");
        version_counter_ = version_counter;
        */
    }
    
    pub fn set_version_counter(&mut self, version_counter: VariableVersion)  {
        
        todo!();
        /*
            TORCH_CHECK(
            !(is_inference() && version_counter.enabled()),
            "Cannot set version_counter for inference tensor");
        version_counter_ = move(version_counter);
        */
    }
    
    pub fn version_counter(&self) -> &VariableVersion {
        
        todo!();
        /*
            return version_counter_;
        */
    }
    
    pub fn bump_version(&mut self)  {
        
        todo!();
        /*
            version_counter_.bump();
        */
    }

    /**
      | Associate the TensorImpl with the specified
      | PyObject, and, if necessary, also tag the
      | interpreter.
      |
      | NB: This lives in a header so that we can
      | inline away the switch on status
      |
      | NB: THIS FUNCTION CAN RAISE AN EXCEPTION.
      | Make sure to clean up after PyObject if
      | necessary!
      */
    pub fn init_pyobj(&mut self, 
        self_interpreter: *mut PyInterpreter,
        pyobj:            *mut PyObject,
        status:           PyInterpreterStatus)  {
        
        todo!();
        /*
            PyInterpreter* expected = nullptr;
        switch (status) {
          case PyInterpreterStatus::DEFINITELY_UNINITIALIZED:
            // caller guarantees there is no multithreaded access; if there is
            // no data race OK to do a relaxed store
            pyobj_interpreter_.store(self_interpreter, memory_order_relaxed);
            break;
          case PyInterpreterStatus::TAGGED_BY_US:
            // no tagging is necessary, the tag is already correct
            break;
          case PyInterpreterStatus::MAYBE_UNINITIALIZED:
            // attempt to claim this TensorImpl with the specified interpreter
            // tag
            if (pyobj_interpreter_.compare_exchange_strong(
                    expected, self_interpreter, memory_order_acq_rel)) {
              break;
            }
            // test if, actually, it was already tagged by us!  this situation can't
            // be caused by a race, but it could be caused by a situation
            // where someone conservatively tagged the tensor as MAYBE_UNINITIALIZED
            // (because they didn't pre-check the tag) when actually it was
            // owned by the interpreter
            if (expected == self_interpreter) {
              break;
            }
            // fallthrough, we lost the race.  We are guaranteed not to lose the
            // race with ourself, as calls to init_pyobj with the same interpreter
            // ID must be sequentialized by the GIL
            C10_FALLTHROUGH;
          case PyInterpreterStatus::TAGGED_BY_OTHER:
            TORCH_CHECK(
                false,
                "cannot allocate PyObject for C10Tensor on interpreter ",
                self_interpreter,
                " that has already been used by another torch deploy interpreter ",
                pyobj_interpreter_.load());
        }

        // we are the ONLY thread that can have gotten to this point.  It is not
        // possible to conflict with another zero interpreter as access is protected
        // by GIL
        pyobj_ = pyobj;
        */
    }

    /**
      | Test the interpreter tag.  If tagged for the
      | current interpreter, return a non-nullopt (but
      | possibly null) PyObject.  If (possibly)
      | untagged, returns a nullopt.  If it is
      | definitely invalid, raises an error.
      |
      | NB: this lives in header so that we can avoid
      | actually creating the optional
      */
    pub fn check_pyobj(&mut self, self_interpreter: *mut PyInterpreter) -> Option<*mut PyObject> {
        
        todo!();
        /*
            // Note [Memory ordering on Python interpreter tag]
        PyInterpreter* interpreter =
            pyobj_interpreter_.load(memory_order_acquire);
        if (interpreter == nullptr) {
          // NB: This never returns DEFINITELY_UNINITIALIZED because there is
          // always the possibility that another thread races to initialize
          // after we query here.  The only time when we can conclude a tensor
          // is definitely uninitialized is when we have just allocated it and
          // it cannot have escaped to other threads yet
          return nullopt;
        } else if (interpreter == self_interpreter) {
          // NB: pyobj_ could still be null!
          return make_optional(pyobj_);
        } else {
          TORCH_CHECK(
              false,
              "cannot access PyObject for C10Tensor on interpreter ",
              self_interpreter->name(),
              " that has already been used by another torch deploy interpreter ",
              pyobj_interpreter_.load()->name());
        }
        */
    }

    /**
      | Clear the PyObject field for an interpreter,
      | in situations where we statically know the
      | tensor is tagged with our interpreter.
      |
      */
    pub fn unchecked_clear_pyobj(&mut self, interpreter: *mut PyInterpreter)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(interpreter == pyobj_interpreter_.load());
        pyobj_ = nullptr;
        */
    }
 
    /**
      | See NOTE [optional operator usage in Cuda]
      |
      | We probably don't want to expose this publicly
      |   until the note is addressed.
      |
      */
    pub fn device_opt(&self) -> Option<Device> {
        
        todo!();
        /*
            return device_opt_;
        */
    }

    /**
      | The device type of a C10Tensor, e.g., DeviceType::CPU
      | or DeviceType::CUDA.
      |
      */
    pub fn device_type(&self) -> DeviceType {
        
        todo!();
        /*
            // TODO: A useful internal assert would be to show that device_opt_ is null
        // only if you are an undefined tensor
        TORCH_CHECK(
            device_opt_.has_value(),
            "device_type cannot be run on undefined C10Tensor");
        // See NOTE [optional operator usage in Cuda]
        return (*device_opt_).type();
        */
    }

    /**
      | -----------
      | @brief
      | 
      | Extends the outer-most dimension of
      | this tensor by num elements, preserving
      | the existing data.
      | 
      | The underlying data may be reallocated
      | in order to accommodate the new elements,
      | in which case this tensors' capacity
      | is grown at a factor of growthPct. This
      | ensures that Extend runs on an amortized
      | O(1) time complexity.
      | 
      | This op is auto-asynchronous if the
      | underlying device (Cuda) supports
      | it.
      |
      */
    pub fn extend(&mut self, 
        num:        i64,
        growth_pct: f32)  {
        
        todo!();
        /*
            TORCH_CHECK(sizes_and_strides_.size() >= 1u);
        TORCH_CHECK(num >= 0, "`num` must be non-negative for Extend");
        TORCH_CHECK(
            is_contiguous_,
            "Right now Extend is only supported for contiguous C10Tensor.");
        using SizesVector = SmallVector<int64_t, 5>;
        SizesVector newDims(
            sizes_and_strides_.sizes_begin(), sizes_and_strides_.sizes_end());
        newDims[0] += num;
        if (!storage_.data()) {
          Resize(newDims);
          return;
        }
        const auto newNumel =
            multiply_integers(newDims.begin(), newDims.end());
        if (newNumel * data_type_.itemsize() <= storage_.nbytes()) {
          sizes_and_strides_.set_sizes(newDims);
          numel_ = newNumel;
          return;
        }
        SizesVector newCapacity(
            sizes_and_strides_.sizes_begin(), sizes_and_strides_.sizes_end());
        newCapacity[0] = max(
            newDims[0],
            static_cast<int64_t>(ceil(
                sizes_and_strides_.size_at_unchecked(0) * (1 + growthPct / 100))));
        auto oldData = move(storage_.data_ptr());
        auto oldSize = numel_;
        Resize(newCapacity);
        auto* newData = raw_mutable_data(data_type_);
        if (data_type_.copy()) {
          TORCH_CHECK(
              device_type() == DeviceType::CPU, "non-POD types work only on CPU");
          data_type_.copy()(oldData.get(), newData, oldSize);
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
          CopyBytes(
              oldSize * itemsize(),
              oldData.get(),
              device(),
              newData,
              device(),
              true); // non-blocking
        }
        reserved_ = true;
        sizes_and_strides_.set_sizes(newDims);
        numel_ = newNumel;
        */
    }

    /**
      | -----------
      | @brief
      | 
      | Reserve space for the underlying tensor.
      | 
      | This must be called after Resize(),
      | since we only specify the first dimension
      | This does not copy over the old data to
      | the newly allocated space
      |
      */
    pub fn reserve_space<T>(&mut self, outer_dim: &T)  {
    
        todo!();
        /*
            TORCH_CHECK(
            is_contiguous_,
            "Right now ReserveSpace is only supported for contiguous C10Tensor.");
        TORCH_CHECK(
            storage_.unique(), "Can't call ReserveSpace on shared storage.");
        // TODO: eliminate newCapacity.
        SmallVector<int64_t, 5> newCapacity(
            sizes_and_strides_.sizes_begin(), sizes_and_strides_.sizes_end());
        newCapacity[0] = outer_dim;
        auto newNumel = multiply_integers(newCapacity);
        if (newNumel * data_type_.itemsize() <= storage_.nbytes()) {
          return;
        }
        // Old data is discarded
        storage_.data_ptr().clear();
        auto oldSize = numel_;
        SmallVector<int64_t, 5> oldDims(
            sizes_and_strides_.sizes_begin(), sizes_and_strides_.sizes_end());
        Resize(newCapacity);
        // Allocate new memory but don't copy over the data
        raw_mutable_data(data_type_);
        sizes_and_strides_.set_sizes(oldDims);
        numel_ = oldSize;
        reserved_ = true;
        */
    }

    /**
      | -----------
      | @brief
      | 
      | Resizes a tensor.
      | 
      | Resize takes in a vector of ints specifying
      | the dimensions of the tensor.
      | 
      | You can pass in an empty vector to specify
      | that it is a scalar (i.e. containing
      | one single item).
      | 
      | The underlying storage may be deleted
      | after calling Resize: if the new shape
      | leads to a different number of items
      | in the tensor, the old memory is deleted
      | and new memory will be allocated next
      | time you call mutable_data(). However,
      | if the shape is different but the total
      | number of items is the same, the underlying
      | storage is kept.
      | 
      | This method respects caffe2_keep_on_shrink.
      | Consult the internal logic of this method
      | to see exactly under what circumstances
      | this flag matters.
      |
      */
    pub fn resize<Ts>(&mut self, dim_source: Ts)  {
    
        todo!();
        /*
            bool size_changed = SetDims(dim_source...);
        if (size_changed) {
          HandleResize();
        }
        */
    }
    
    pub fn resize_from_vec<T>(&mut self, dim_source: &Vec<T>)  {
    
        todo!();
        /*
            Resize(&[T](dim_source));
        */
    }

    /**
      | Resizes the tensor without touching
      | underlying storage.
      | 
      | This requires the total size of the tensor
      | to remains constant.
      |
      */
    #[inline] pub fn reshape(&mut self, dims: &Vec<i64>)  {
        
        todo!();
        /*
            TORCH_CHECK(
            is_contiguous_,
            "Right now Reshape is only supported for contiguous C10Tensor.");
        int64_t new_size = 1;
        for (auto d : dims) {
          TORCH_CHECK(d >= 0);
          new_size *= d;
        }
        TORCH_CHECK(
            new_size == numel_,
            "New size and old size are not equal. You cannot use Reshape, "
            "but should use Resize."
            // TODO(jiayq): remove the following warning after pending diffs
            // stabilize.
            " The old caffe2 mixes Reshape and Resize but this behavior has "
            "been changed. If you find this error, most likely you will need "
            "to change corresponding code from Reshape to Resize.");
        sizes_and_strides_.set_sizes(dims);
        empty_tensor_restride(MemoryFormat::Contiguous);
        */
    }

    /**
      | Release whatever memory the tensor
      | was holding but keep size and type information.
      | Subsequent call to mutable_data will
      | trigger new memory allocation.
      |
      */
    #[inline] pub fn free_memory(&mut self)  {
        
        todo!();
        /*
            // We'll detach from the old Storage and create a new one
        storage_ = Storage::create_legacy(storage_.device());
        storage_offset_ = 0;
        */
    }

    /**
      | -----------
      | @brief
      | 
      | Shares the data with another tensor.
      | 
      | To share data between two tensors, the
      | sizes of the two tensors must be equal
      | already. The reason we do not implicitly
      | do a Resize to make the two tensors have
      | the same shape is that we want to allow
      | tensors of different shapes but the
      | same number of items to still be able
      | to share data. This allows one to e.g.
      | have a n-dimensional C10Tensor and a flattened
      | version sharing the same underlying
      | storage.
      | 
      | The source tensor should already have
      | its data allocated.
      |
      */
    // To be deprecated
    pub fn share_data(&mut self, src: &TensorImpl)  {
        
        todo!();
        /*
            // Right now, we are assuming the device_type are the same, since it is
        // inherently the same in the non-templatized code. We should probably add
        // an assert here which might affect perf a little bit.
        TORCH_CHECK(
            src.numel_ == numel_,
            "Size mismatch - did you call reshape before sharing the data?");
        // It is possible that the source tensor hasn't called mutable_data() yet,
        // in which case ShareData() doesn't make much sense since we don't really
        // know what to share yet.
        // TODO: Add the assert after all uninitialized states are eliminated
        // TORCH_CHECK(src.dtype_initialized(),
        //            "Source tensor don't have a data type (did you call
        //            mutable_data<T> on the tensor?)");
        if (!src.dtype_initialized()) {
          C10_LOG_EVERY_MS(WARNING, 1000)
              << "Source tensor don't have a data type (did you call mutable_data<T> on the tensor?)";
        }
        TORCH_CHECK(
            src.storage_initialized(),
            "Source tensor has no content and has size > 0");
        // Finally, do sharing.
        /* Since we create new Storage whenever we need to change data_type/nbytes
         * this still keeps the original semantics
         */
        storage_ = src.storage();
        data_type_ = src.dtype();
        device_opt_ = src.device_opt();
        storage_offset_ = src.storage_offset();
        */
    }
    
    pub fn share_external_pointer<A: std::alloc::Allocator>(&mut self, 
        data_ptr:   DataPtr,
        data_type:  TypeMeta,
        size_bytes: usize)  {
        
        todo!();
        /*
            TORCH_CHECK(
            data_type != ScalarType::Undefined,
            "To share with a raw external pointer you need to pass in an "
            "initialized data_type(TypeMeta).");
        if (!size_bytes) {
          size_bytes = numel_ * data_type.itemsize();
        }
        if (storage_.unique()) {
          storage_.UniqueStorageShareExternalPointer(
              move(data_ptr), size_bytes);
          data_type_ = data_type;
          device_opt_ = storage_.device();
          storage_offset_ = 0;
        } else {
          // Create a new Storage
          storage_ = Storage(
              Storage::use_byte_size_t(),
              size_bytes,
              move(data_ptr),
              /*allocator=*/nullptr,
              /*resizable=*/false);
          data_type_ = data_type;
          device_opt_ = storage_.device();
          storage_offset_ = 0;
        }
        */
    }

    /**
      | Returns a mutable raw pointer of the
      | underlying storage. Since we will need
      | to know the type of the data for allocation,
      | a TypeMeta object is passed in to specify
      | the necessary information. This is
      | conceptually equivalent of calling
      | mutable_data<T>() where the TypeMeta
      | parameter meta is derived from the type
      | T. This function differs from mutable_data<T>()
      | in the sense that the type T can be specified
      | during runtime via the TypeMeta object.
      | 
      | If the existing data does not match the
      | desired type, it will be deleted and
      | a new storage will be created.
      |
      */
    #[inline] pub fn raw_mutable_data(&mut self, meta: TypeMeta)  {
        
        todo!();
        /*
            // For 0-size tensors it's fine to return any pointer (including nullptr)
        if (data_type_ == meta && storage_initialized()) {
          return static_cast<void*>(
              static_cast<char*>(storage_.data()) +
              storage_offset_ * meta.itemsize());
        } else {
          bool had_special_dtor = data_type_.placementDelete() != nullptr;
          storage_offset_ = 0;
          data_type_ = meta;
          // NB: device is not changed

          // We can reuse the existing buffer if the current data does not have
          // a special destructor and the new data doesn't have a special
          // constructor.
          if (numel_ == 0 ||
              (meta.placementNew() == nullptr && !had_special_dtor &&
               (storage_.nbytes() >= (numel_ * data_type_.itemsize())))) {
            TORCH_INTERNAL_ASSERT(
                storage_offset_ == 0); // because we just reallocated
            return storage_.data();
          }
          const Allocator* allocator = storage_.allocator();
          // Storage might have nullptr allocator in rare cases, for example, if
          // an external memory segment has been wrapped with C10Tensor and we don't
          // know how to reallocate it. However, in order to preserve legacy C2
          // behavior, we allow reallocating the memory using default allocator.
          if (allocator == nullptr) {
            allocator = GetAllocator(storage_.device_type());
          }
          if (meta.placementNew()) {
            // For types that need placement new, we will call it, as well as
            // making sure that when the data is freed, it calls the right
            // destruction procedure.
            auto size = numel_;
            auto dtor = data_type_.placementDelete();
            auto data_ptr = allocator->allocate(numel_ * data_type_.itemsize());
            storage_.set_data_ptr_noswap(PlacementDeleteContext::makeDataPtr(
                move(data_ptr), dtor, size, storage_.device()));
            data_type_.placementNew()(storage_.data(), numel_);
          } else {
            // For fundamental type, new and delete is easier.
            storage_.set_data_ptr_noswap(
                allocator->allocate(numel_ * data_type_.itemsize()));
          }
          storage_.set_nbytes(numel_ * data_type_.itemsize());
          TORCH_INTERNAL_ASSERT(
              storage_offset_ == 0); // because we just reallocated
          device_opt_ = storage_.device();
          return storage_.data();
        }
        */
    }

    /**
      | Returns a typed pointer of the underlying
      | storage.
      | 
      | For fundamental types, we reuse possible
      | existing storage if there is sufficient
      | capacity.
      |
      */
    #[inline] pub fn mutable_data<T>(&mut self) -> *mut T {
    
        todo!();
        /*
            if (storage_initialized() && data_type_.Match<T>()) {
          return static_cast<T*>(storage_.data()) + storage_offset_;
        }
        // Check it here statically - otherwise TypeMeta would throw the runtime
        // error in attempt to invoke TypeMeta::ctor()
        static_assert(
            is_default_constructible<T>::value,
            "C10Tensor can't hold non-default-constructable types");
        return static_cast<T*>(raw_mutable_data(TypeMeta::Make<T>()));
        */
    }

    /**
      | True if a tensor is storage initialized.
      | A tensor may become storage UNINITIALIZED
      | after a Resize() or FreeMemory()
      |
      */
    pub fn storage_initialized(&self) -> bool {
        
        todo!();
        /*
            TORCH_CHECK(
            has_storage(),
            "cannot call storage_initialized on tensor that does not have storage");
        return storage_.data() || numel_ == 0;
        */
    }

    /**
      | True if a tensor is dtype initialized.
      | A tensor allocated with
      | 
      | Caffe2-style constructors is dtype
      | uninitialized until the first time
      | mutable_data<T>() is called.
      |
      */
    pub fn dtype_initialized(&self) -> bool {
        
        todo!();
        /*
            return data_type_ != TypeMeta();
        */
    }
    
    pub fn set_storage_keep_dtype(&mut self, storage: Storage)  {
        
        todo!();
        /*
            TORCH_CHECK(
            allow_tensor_metadata_change(),
            "set_storage ",
            err_msg_tensor_metadata_change_not_allowed);
        storage_ = move(storage);
        device_opt_ = storage_.device();
        */
    }
    
    pub fn set_storage_and_dtype(&mut self, 
        storage:   Storage,
        data_type: TypeMeta)  {
        
        todo!();
        /*
            set_storage_keep_dtype(storage);
        data_type_ = data_type;
        */
    }

    /**
      | Set the strides of the tensor to match
      | memory_format
      | 
      | WARNING: This function doesn't rearrange
      | data and assumes tensor is a memory contiguous
      |
      */
    pub fn empty_tensor_restride(&mut self, memory_format: MemoryFormat)  {
        
        todo!();
        /*
            #ifdef DEBUG
        TORCH_INTERNAL_ASSERT(
            compute_numel() == numel_,
            "If you are seeing this error, that means empty_tensor_restride was "
            "called before setting correct numel");
    #endif
        switch (memory_format) {
          case MemoryFormat::Contiguous: {
            // dim_ is a virtual call, don't repeat it
            const auto dim_ = dim();
            sizes_and_strides_.resize(dim_);
            if (dim_ > 0) {
              const auto last_idx = dim_ - 1;
              sizes_and_strides_.stride_at_unchecked(last_idx) = 1;
              for (auto i = last_idx - 1; i >= 0; --i) {
                sizes_and_strides_.stride_at_unchecked(i) =
                    sizes_and_strides_.stride_at_unchecked(i + 1) *
                    max<int64_t>(
                        sizes_and_strides_.size_at_unchecked(i + 1), 1);
              }
            }
            break;
          }
          case MemoryFormat::ChannelsLast: {
            TORCH_CHECK(
                dim() == 4, "required rank 4 tensor to use channels_last format");
            set_sizes_and_strides(sizes(), get_channels_last_strides_2d(sizes()));
            break;
          }
          case MemoryFormat::ChannelsLast3d: {
            TORCH_CHECK(
                dim() == 5,
                "required rank 5 tensor to use channels_last_3d format");
            set_sizes_and_strides(sizes(), get_channels_last_strides_3d(sizes()));
            break;
          }
          case MemoryFormat::Preserve:
            TORCH_CHECK(false, "unsupported memory format ", memory_format);
            // Cleaning warning messages, no need to break as TORCH_CHECK(false)
            // terminates flow.
            // break;
        }
        // recompute contiguous flag, as currently NHWC/NCHW flags are not mutually
        // exclusive see #24090
        refresh_contiguous();
        */
    }
    
    pub fn is_strides_like_channels_last(&self) -> bool {
        
        todo!();
        /*
            return is_channels_last_;
        */
    }
    
    pub fn is_strides_like_channels_last_3d(&self) -> bool {
        
        todo!();
        /*
            return is_channels_last_3d_;
        */
    }
    
    pub fn is_non_overlapping_and_dense(&self) -> bool {
        
        todo!();
        /*
            return is_non_overlapping_and_dense_;
        */
    }
    
    /**
      | The Caffe2 Resize() method supports being
      | called both as Resize({2,2}) as well as
      | variadic with Resize(2, 2).  These overloads
      | provide all of the supported calling
      | configurations, while being overloads (and
      | not templates) so that implicit conversions
      | still work.
      |
      | SetDims on ArrayRef is internally implemented
      | as a template, so we can handle both
      | ArrayRefs of different types (there are some
      | uses of Resize in Caffe2 which pass in int,
      | not int64_t.)
      */
    // template < typename T, typename = typename enable_if<is_integral<T>::value>::type>
    pub fn set_dims_template<T: PrimInt>(&mut self, src: &[T]) -> bool {
        
        todo!();
        /*
            auto old_numel = numel_;
        sizes_and_strides_.resize(src.size());
        int64_t new_numel = 1;
        for (size_t i = 0; i < src.size(); ++i) {
          new_numel *= src[i];
          sizes_and_strides_.size_at_unchecked(i) = src[i];
        }
        numel_ = new_numel;
        empty_tensor_restride(MemoryFormat::Contiguous);
        return numel_ != old_numel;
        */
    }
    
    
    pub fn set_dims_a(&mut self, s: &[i64]) -> bool {
        
        todo!();
        /*
            return SetDimsTemplate(s);
        */
    }
    
    
    pub fn set_dims_b(&mut self, s: &[i32]) -> bool {
        
        todo!();
        /*
            return SetDimsTemplate(s);
        */
    }
    
    
    pub fn set_dims_c(&mut self, s: &[usize]) -> bool {
        
        todo!();
        /*
            return SetDimsTemplate(s);
        */
    }
    
    
    pub fn set_dims_d(&mut self) -> bool {
        
        todo!();
        /*
            return SetDims(IntArrayRef{});
        */
    }
    
    
    pub fn set_dims_e(&mut self, d0: i64) -> bool {
        
        todo!();
        /*
            return SetDims(IntArrayRef{d0});
        */
    }
    
    
    pub fn set_dims_f(&mut self, d0: i64, d1: i64) -> bool {
        
        todo!();
        /*
            return SetDims(IntArrayRef{d0, d1});
        */
    }
    
    pub fn set_dims_g(&mut self, 
        d0: i64,
        d1: i64,
        d2: i64) -> bool {
        
        todo!();
        /*
            return SetDims(IntArrayRef{d0, d1, d2});
        */
    }
    
    pub fn set_dims_h(&mut self, 
        d0: i64,
        d1: i64,
        d2: i64,
        d3: i64) -> bool {
        
        todo!();
        /*
            return SetDims(IntArrayRef{d0, d1, d2, d3});
        */
    }

    /**
      | Compute the number of elements based
      | on the sizes of a tensor.
      |
      */
    pub fn compute_numel(&self) -> i64 {
        
        todo!();
        /*
            int64_t n = 1;
        for (auto s : sizes()) {
          n *= s;
        }
        return n;
        */
    }

    /**
      | Compute the number of elements based
      | on the sizes of a tensor. Catches integer
      | overflow that may occur when a tensor
      | using a sparse layout has multiple dimensions
      | with large sizes.
      |
      */
    pub fn safe_compute_numel(&self) -> i64 {
        
        todo!();
        /*
            int64_t n = 1;
        for (auto s : sizes()) {
          TORCH_CHECK(
              s == 0 || n <= int64_t::max / s,
              "numel: integer multiplication overflow");
          n *= s;
        }
        return n;
        */
    }

    /**
      | Recompute the cached numel of a tensor.
      | Call this if you modify sizes.
      | 
      | For tensors with sparse layouts, use
      | safe_refresh_numel() instead because
      | it will catch integer overflow that
      | may occur for tensors with sparse layouts
      | and large dimensions.
      |
      */
    pub fn refresh_numel(&mut self)  {
        
        todo!();
        /*
            numel_ = compute_numel();
        */
    }

    /**
      | Recompute the cached numel of a tensor.
      | Call this if you modify sizes. Use only
      | for tensors with sparse layouts because
      | only sparse tensor are likely to have
      | sizes that may lead to integer overflow
      | when computing numel.
      |
      */
    pub fn safe_refresh_numel(&mut self)  {
        
        todo!();
        /*
            numel_ = safe_compute_numel();
        */
    }

    /**
      | Recompute the cached contiguity of
      | a tensor. Call this if you modify sizes
      | or strides.
      |
      */
    pub fn refresh_contiguous(&mut self)  {
        
        todo!();
        /*
            is_contiguous_ = compute_contiguous();
        // Note:
        // Dim 0, 1, 2 will never be a channels last 2d/3d format
        // Dim 3+ is possibly be a channels last 2d format (Dim 4 only at this
        // point) Dim 4+ is possibly be a channels last 3d format (Dim 5 only at
        // this point)
        switch (dim()) {
          case 4:
            is_channels_last_contiguous_ = compute_channels_last_contiguous_2d();
            is_channels_last_3d_contiguous_ = false;
            is_channels_last_ = compute_strides_like_channels_last_2d();
            is_channels_last_3d_ = false;
            is_non_overlapping_and_dense_ = is_contiguous_ ||
                is_channels_last_contiguous_ || compute_non_overlapping_and_dense();
            break;
          case 5:
            is_channels_last_contiguous_ = compute_channels_last_contiguous_2d();
            is_channels_last_3d_contiguous_ = !is_channels_last_contiguous_ &&
                compute_channels_last_contiguous_3d();
            is_channels_last_ = !is_channels_last_3d_contiguous_ &&
                compute_strides_like_channels_last_2d();
            is_channels_last_3d_ =
                !is_channels_last_ && compute_strides_like_channels_last_3d();
            is_non_overlapping_and_dense_ = is_contiguous_ ||
                is_channels_last_contiguous_ || is_channels_last_3d_contiguous_ ||
                compute_non_overlapping_and_dense();
            break;
          default:
            is_channels_last_contiguous_ = false;
            is_channels_last_3d_contiguous_ = false;
            // is_channels_last_ and is_channels_last_3d_ are suggested
            // memory_format. Being channels_last_contiguous doesn't necessarily
            // mean the tensor is strided like channels_last: for strides on channel
            // dimension could suggest desired memory_layout, but it doesn't affect
            // memory storage
            is_channels_last_ = false;
            is_channels_last_3d_ = false;
            is_non_overlapping_and_dense_ =
                is_contiguous_ || compute_non_overlapping_and_dense();
        }
        */
    }

    /**
      | Copy the tensor metadata fields (e.g.
      | sizes / strides / storage pointer / storage_offset)
      | from one TensorImpl to another TensorImpl.
      | 
      | For usage of `version_counter` and
      | `allow_tensor_metadata_change`,
      | see NOTE [ TensorImpl Shallow-Copying
      | ].
      |
      */
    pub fn copy_tensor_metadata_a(
        src_impl:                     *const TensorImpl,
        dest_impl:                    *mut TensorImpl,
        version_counter:              &VariableVersion,
        allow_tensor_metadata_change: bool)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Copy the tensor metadata fields (e.g.
      | sizes / strides / storage pointer / storage_offset)
      | from one TensorImpl to another TensorImpl.
      | 
      | For usage of `version_counter` and
      | `allow_tensor_metadata_change`,
      | see NOTE [ TensorImpl Shallow-Copying
      | ].
      |
      */
    pub fn copy_tensor_metadata_b(
        src_impl:                     *const TensorImpl,
        dest_impl:                    *mut TensorImpl,
        version_counter:              VariableVersion,
        allow_tensor_metadata_change: bool)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_storage_access_should_throw(&mut self)  {
        
        todo!();
        /*
            storage_access_should_throw_ = true;
        */
    }
    
    
    pub fn owns_pyobj(&mut self) -> bool {
        
        todo!();
        /*
            return owns_pyobj_;
        */
    }
    
    
    pub fn set_owns_pyobj(&mut self, b: bool)  {
        
        todo!();
        /*
            owns_pyobj_ = b;
        */
    }
    
    pub fn set_has_contiguity_policy(&mut self, p: HasContiguityPolicy)  {
        
        todo!();
        /*
            has_contiguity_ = static_cast<uint8_t>(p);
        */
    }

    /**
      | default member initializers for bit-fields
      | only available with -std=c++2a or
      | -std=gnu++2a
      |
      */
    #[inline] pub fn init_bitfields(&mut self)  {
        
        todo!();
        /*
            is_contiguous_ = true;
        has_contiguity_ = static_cast<uint8_t>(HasContiguityPolicy::Default);

        is_channels_last_ = false;
        is_channels_last_contiguous_ = false;
        is_channels_last_3d_ = false;
        is_channels_last_3d_contiguous_ = false;
        is_non_overlapping_and_dense_ = true;
        is_wrapped_number_ = false;
        allow_tensor_metadata_change_ = true;
        reserved_ = false;
        owns_pyobj_ = false;
        storage_access_should_throw_ = false;
        */
    }
}

/**
  | Note [TensorImpl size constraints]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  | Changed the size of TensorImpl?  If the size
  | went down, good for you!  Adjust the
  | documentation below and the expected size. Did
  | it go up?  Read on...
  |
  | Struct size matters.  In some production
  | systems at Facebook, we have 400M live tensors
  | during a training run.  Do the math: every
  | 64-bit word you add to C10Tensor is an extra 3.2
  | gigabytes in RAM.
  |
  | If you are a Facebook employee, you can check
  | if the run in question has tipped you over the
  | point using the command here:
  | https://fburl.com/q5enpv98
  |
  | For reference, we OOMed at 160 bytes (20 words)
  | per TensorImpl. This is not counting overhead
  | from strides out-of-line allocation and
  | StorageImpl space and this is from before we
  | inlined sizes and strides directly into
  | TensorImpl as SmallVectors.
  |
  | Our memory usage on 32-bit systems is
  | suboptimal, but we're not checking for it at
  | the moment (to help avoid rage inducing cycles
  | when the 32-bit number is wrong).
  |
  | Current breakdown:
  |
  |    vtable pointer
  |    strong refcount           TODO: pack these into one word
  |    weak refcount
  |    storage pointer
  |    autograd metadata pointer
  |    named tensor metadata pointer
  |    version counter pointer
  |    Python interpreter pointer
  |    PyObject pointer
  |    SizesAndStrides size/pointer
  |    SizesAndStrides sizes (pre-allocated 0)
  |    SizesAndStrides sizes (pre-allocated 1)
  |    SizesAndStrides sizes (pre-allocated 2)
  |    SizesAndStrides sizes (pre-allocated 3)
  |    SizesAndStrides sizes (pre-allocated 4)
  |    SizesAndStrides strides (pre-allocated 0)
  |    SizesAndStrides strides (pre-allocated 1)
  |    SizesAndStrides strides (pre-allocated 2)
  |    SizesAndStrides strides (pre-allocated 3)
  |    SizesAndStrides strides (pre-allocated 4)
  |    storage offset
  |    numel
  |    data type, device, is_contiguous, storage_access_should_throw_, bitfields
  |    DispatchKeySet
  |
  */
/// You changed the size of TensorImpl on 64-bit arch. See Note [TensorImpl size constraints] on how to proceed."
const_assert!{
    size_of::<*mut c_void>() != size_of::<i64>() || // if 64-bit...
    size_of::<TensorImpl>() == size_of::<i64>() * 24
}

//-------------------------------------------[.cpp/pytorch/c10/core/TensorImpl.cpp]

pub fn noop_name_fn(_0: *const PyInterpreter) -> String {
    
    todo!();
        /*
            return "<unloaded interpreter>";
        */
}

pub fn noop_decref_fn(
        _0: *const PyInterpreter,
        _1: *mut PyObject)  {
    
    todo!();
        /*
            // no-op
        */
}

impl PyInterpreter {
    
    /**
      | Disarm this PyInterpreter, making all of its
      | methods noops.
      |
      | Because the function pointers are raw pointers
      | (not atomics), a disarm() invocation that is
      | concurrent with active destructors is not
      | thread safe and will trigger TSAN.
      |
      | My hope is that this situations doesn't ever
      | actually happen;
      |
      | tensor destruction should quiesce when
      | a dlclose happens, and any long lived tensors
      | whose destructors would be disarmed here only
      | begin the destruction process on process
      | shutdown (long after the dlclose has
      | occurred).
      */
    pub fn disarm(&mut self)  {
        
        todo!();
        /*
            name_fn_ = &noop_name_fn;
      decref_fn_ = &noop_decref_fn;
        */
    }
}

impl TensorImpl {
    
    /**
      | Return a mutable reference to the gradient.
      | This is conventionally used as `t.grad()
      | = x` to set a gradient to a completely
      | new tensor.
      |
      */
    pub fn mutable_grad(&mut self) -> &mut C10Tensor {
        
        todo!();
        /*
            if (!autograd_meta_)
        autograd_meta_ = GetAutogradMetaFactory()->make();
      return autograd_meta_->mutable_grad();
        */
    }
    
    /**
      | Return the accumulated gradient of
      | a tensor. This gradient is written into
      | when performing backwards, when this
      | tensor is a leaf tensor.
      |
      */
    pub fn grad(&self) -> &C10Tensor {
        
        todo!();
        /*
            // Yes, I know this looks really weird.  But I don't really have a choice as
      // long as this function returns a const reference to C10Tensor.  I'm not
      // really sure how I would have designed this API differently, but it
      // is not so easy to fix right now because the mutable counterpart of
      // this function must keep working so that "x.grad() = ..." keeps working
      // (part of public API).
      if (!autograd_meta_)
        return GetAutogradMetaFactory()->undefined_tensor();
      return autograd_meta_->grad();
        */
    }
    
    /**
      | Return the accumulated gradient of
      | a tensor. This gradient is computed
      | using forward mode AD.
      | 
      | This is an internal API that should never
      | be used by end users.
      | 
      | The API is as follows:
      | 
      | - "level" allows to specify the level
      | of forward AD nesting for which the gradient
      | should be returned. Note that since
      | levels are not fully supported yet,
      | this argument should be 0. See documentation
      | for
      | 
      | Torchautograd::enter_dual_level
      | for more details about forward AD nesting.
      | 
      | - "self" should represent the C10Tensor
      | whose forward grad is accessed. It is
      | required when dealing with view.
      |
      */
    pub fn fw_grad(&self, 
        level: u64,
        self_: &C10Tensor) -> &C10Tensor {
        
        todo!();
        /*
            // See TensorImpl::grad() above for explanation about the line below
      if (!autograd_meta_)
        return GetAutogradMetaFactory()->undefined_tensor();
      return autograd_meta_->fw_grad(level, self);
        */
    }
    
    /**
      | Sets the forward gradient for this C10Tensor.
      | 
      | The given C10Tensor might not be used directly
      | and its content will be copied.
      | 
      | This is an internal API that should never
      | be used by end users.
      | 
      | The API is as follows:
      | 
      | - "new_grad" is a C10Tensor containing
      | the new value of the gradient that should
      | be set
      | 
      | - "self" should represent the C10Tensor
      | whose forward grad is accessed. It is
      | required when dealing with view.
      | 
      | - "level" allows to specify the level
      | of forward AD nesting for which the gradient
      | should be set. Note that since levels
      | are not fully supported yet, this argument
      | should be 0. See documentation for
      | 
      | Torchautograd::enter_dual_level
      | for more details about forward AD nesting.
      | 
      | - "is_inplace_op" is a boolean flag
      | that tells if this gradient was generated
      | by an inplace operation or an out of place
      | one. This allows better error checking.
      |
      */
    pub fn set_fw_grad(&mut self, 
        new_grad:      &C10Tensor,
        self_:         &C10Tensor,
        level:         u64,
        is_inplace_op: bool)  {
        
        todo!();
        /*
            if (!autograd_meta_)
        autograd_meta_ = GetAutogradMetaFactory()->make();
      autograd_meta_->set_fw_grad(new_grad, self, level, is_inplace_op);
        */
    }
    
    pub fn new(
        storage:   Storage,
        key_set:   DispatchKeySet,
        data_type: TypeMeta) -> Self {
    
        todo!();
        /*


            // Use forward to suppress static analyzer false positive.
        : TensorImpl(
              forward<Storage>(storage),
              key_set,
              data_type,
              storage.device())
        */
    }
    
    pub fn new_a(
        ty:        TensorImplType,
        storage:   Storage,
        key_set:   DispatchKeySet,
        data_type: TypeMeta) -> Self {
    
        todo!();
        /*


            : storage_(move(storage)),
          pyobj_interpreter_(nullptr),
          pyobj_(nullptr),
          storage_offset_(0),
          numel_(0),
          data_type_(data_type),
          device_opt_(storage_.device()),
          key_set_(key_set) 

      init_bitfields();
      // Inference tensor doesn't have version counter.
      if (!is_inference()) {
        version_counter_ = VariableVersion(/*version=*/0);
      }
        */
    }
    
    pub fn new_b(
        key_set:    DispatchKeySet,
        data_type:  TypeMeta,
        device_opt: Option<Device>) -> Self {
    
        todo!();
        /*
            : TensorImpl({}, key_set, data_type, move(device_opt))
        */
    }
    
    pub fn new_c(
        storage:    Storage,
        key_set:    DispatchKeySet,
        data_type:  TypeMeta,
        device_opt: Option<Device>) -> Self {
    
        todo!();
        /*


            : storage_(move(storage)),
          pyobj_interpreter_(nullptr),
          pyobj_(nullptr),
          storage_offset_(0),
          numel_(0),
          data_type_(data_type),
          device_opt_(device_opt) 
      init_bitfields();

      if (!key_set.empty()) {
        TORCH_INTERNAL_ASSERT(
            data_type == ScalarType::Undefined || device_opt_.has_value());
        // UndefinedTensorImpl is a singleton, so we skip logging it
        C10_LOG_API_USAGE_ONCE("tensor.create");
      }

      bool inference_mode = InferenceMode::is_enabled();

      // TODO: be more explicit about the full key set at call sites so we
      // don't have to keep recomputing it here
      DispatchKey k = key_set.highestPriorityBackendTypeId();

      key_set = key_set | getAutocastRelatedKeySetFromBackend(k);

      // Inference tensor doesn't have autograd related keys.
      if (inference_mode) {
        // See Note [Expected TLS state in InferenceMode] for why we exclude
        // Autograd & ADInplaceOrView keys. Normally key_set only contains backend
        // keys but we do the substraction here to make sure.
        key_set_ = key_set - autograd_dispatch_keyset_with_ADInplaceOrView;
      } else {
        // TODO: Ideally we only add AutogradBackend key when the tensor requires
        // grad.
        //       See Note [Dream: skip VariableType kernel when requires_grad=false]
        key_set_ = key_set | getAutogradRelatedKeySetFromBackend(k);
      }

      // Inference tensor doesn't have version counter.
      if (!is_inference()) {
        version_counter_ = VariableVersion(/*version=*/0);
      }

      // we would also like to check that non-cpu devices have an index, but some
      // Caffe2 operators create Storages with default devices.
        */
    }

    #[cfg(not(C10_DISABLE_TENSORIMPL_EXTENSIBILITY))]
    pub fn sizes(&self) -> &[i32] {
        
        todo!();
        /*
            return sizes_and_strides_.sizes_arrayref();
        */
    }
    
    pub fn strides(&self) -> &[i32] {
        
        todo!();
        /*
            return sizes_and_strides_.strides_arrayref();
        */
    }
    
    pub fn handle_resize(&mut self)  {
        
        todo!();
        /*
            // If needed, we will free the data. the next mutable_data() call
      // will create the data storage.
      bool reset_tensor = false;
      if (reserved_) {
        // If tensor is reserved then don't claim its memeory unless nbytes()
        // is smaller than new size
        reset_tensor =
            storage_.nbytes() < (storage_offset_ + numel_) * data_type_.itemsize();
      } else {
        reset_tensor = storage_.nbytes() <
                (storage_offset_ + numel_) * data_type_.itemsize() ||
            !FLAGS_caffe2_keep_on_shrink ||
            storage_.nbytes() - (storage_offset_ + numel_) * data_type_.itemsize() >
                static_cast<size_t>(FLAGS_caffe2_max_keep_on_shrink_memory);
      }

      if (reset_tensor && storage_initialized()) {
        FreeMemory();
      }
        */
    }
    
    /**
      | Compute whether or not a tensor is contiguous
      | based on the sizes and strides of a tensor.
      |
      */
    pub fn compute_contiguous(&self) -> bool {
        
        todo!();
        /*
            bool is_contiguous = true;
      if (is_empty())
        return is_contiguous;
      int64_t z = 1;
      for (int64_t d = dim() - 1; d >= 0; d--) {
        const auto size_d = sizes_and_strides_.size_at_unchecked(d);
        if (size_d != 1) {
          if (sizes_and_strides_.stride_at_unchecked(d) == z) {
            z *= size_d;
          } else {
            is_contiguous = false;
            break;
          }
        }
      }
      return is_contiguous;
        */
    }
    
    pub fn compute_channels_last_contiguous_2d(&self) -> bool {
        
        todo!();
        /*
            // Please don't combine these code, constant array is used here to let
      // compiler fully unroll the loop to get better performance
      switch (sizes_and_strides_.size()) {
        case 4: {
          int64_t expected = 1;
          for (auto& d : {1, 3, 2, 0}) {
            const auto size_d = sizes_and_strides_.size_at_unchecked(d);
            if (size_d != 1) {
              if (sizes_and_strides_.stride_at_unchecked(d) != expected) {
                return false;
              }
              expected *= size_d;
            }
          }
          return true;
        }
        // NOLINTNEXTLINE(bugprone-branch-clone)
        case 3:
          // TODO dim == 3 case will be enabled once it is fully tested
          return false;
        default:
          return false;
      }
        */
    }
    
    pub fn compute_channels_last_contiguous_3d(&self) -> bool {
        
        todo!();
        /*
            // Please don't combine these code, constant array is used here to let
      // compiler fully unroll the loop to get better performance
      switch (sizes_and_strides_.size()) {
        case 5: {
          int64_t expected = 1;
          for (auto& d : {1, 4, 3, 2, 0}) {
            const auto size_d = sizes_and_strides_.size_at_unchecked(d);
            if (size_d != 1) {
              if (sizes_and_strides_.stride_at_unchecked(d) != expected) {
                return false;
              }
              expected *= size_d;
            }
          }
          return true;
        }
        // NOLINTNEXTLINE(bugprone-branch-clone)
        case 4:
          // TODO dim == 4 case will be enabled once it is fully tested
          return false;
        default:
          return false;
      }
        */
    }
    
    pub fn compute_strides_like_channels_last_2d(&self) -> bool {
        
        todo!();
        /*
            return is_channels_last_strides_2d(
          TensorImpl::sizes(), TensorImpl::strides());
        */
    }
    
    pub fn compute_strides_like_channels_last_3d(&self) -> bool {
        
        todo!();
        /*
            return is_channels_last_strides_3d(
          TensorImpl::sizes(), TensorImpl::strides());
        */
    }
    
    pub fn compute_non_overlapping_and_dense(&self) -> bool {
        
        todo!();
        /*
            if (dim() == 1) {
        return sizes_and_strides_.size_at_unchecked(0) < 2 ||
            sizes_and_strides_.stride_at_unchecked(0) == 1;
      }
      SmallVector<int64_t, 5> perm;
      perm.resize(dim());
      for (int64_t i = 0; i < dim(); i++) {
        perm[i] = i;
      }
      // Sort by strides, leaving 0 and 1 sized dims at the end of the array
      sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
        if (sizes_and_strides_.size_at_unchecked(a) < 2) {
          return false;
        } else if (sizes_and_strides_.size_at_unchecked(b) < 2) {
          return true;
        }
        return sizes_and_strides_.stride_at_unchecked(a) <
            sizes_and_strides_.stride_at_unchecked(b);
      });
      auto require_stride = 1;
      for (int64_t i = 0; i < dim(); i++) {
        const auto size_perm_i = sizes_and_strides_.size_at_unchecked(perm[i]);
        if (size_perm_i < 2) {
          return true;
        }
        if (sizes_and_strides_.stride_at_unchecked(perm[i]) != require_stride) {
          return false;
        }
        require_stride *= size_perm_i;
      }
      return true;
        */
    }
    
    pub fn release_resources(&mut self)  {
        
        todo!();
        /*
            autograd_meta_.reset();
      if (storage_) {
        storage_ = {};
      }
      if (owns_pyobj_) {
        TORCH_INTERNAL_ASSERT(pyobj_interpreter_ != nullptr);
        TORCH_INTERNAL_ASSERT(pyobj_ != nullptr);
        pyobj_interpreter_.load(memory_order_acquire)->decref(pyobj_);
        // NB: this destructor can only be entered when there are no
        // references to this C++ object (obviously), NOR any references
        // to the PyObject (if there are references to the PyObject,
        // then the PyObject holds an owning reference to the tensor).
        // So it is OK to clear pyobj_ here as it is impossible for it to
        // be used again (modulo weak reference races)
        pyobj_ = nullptr; // for safety
      }
        */
    }

    #[cfg(not(C10_DISABLE_TENSORIMPL_EXTENSIBILITY))]
    pub fn dim(&self) -> i64 {
        
        todo!();
        /*
            return sizes_and_strides_.size();
        */
    }
    
    /**
      | Return the size of a tensor at some dimension.
      |
      */
    pub fn size(&self, d: i64) -> i64 {
        
        todo!();
        /*
            d = maybe_wrap_dim(d, dim(), false);
      return sizes_and_strides_.size_at_unchecked(d);
        */
    }
    
    /**
      | Return the stride of a tensor at some
      | dimension.
      |
      */
    pub fn stride(&self, d: i64) -> i64 {
        
        todo!();
        /*
            d = maybe_wrap_dim(d, dim(), false);
      return sizes_and_strides_.stride_at_unchecked(d);
        */
    }

    #[cfg(not(C10_DISABLE_TENSORIMPL_EXTENSIBILITY))]
    pub fn has_storage(&self) -> bool {
        
        todo!();
        /*
            return storage_;
        */
    }
    
    pub fn throw_storage_access_error(&self)  {
        
        todo!();
        /*
            TORCH_CHECK_NOT_IMPLEMENTED(
          false, "Cannot access storage of ", tensorimpl_type_name());
        */
    }
    
    pub fn is_contiguous_nondefault_policy_impl(&self, memory_format: MemoryFormat) -> bool {
        
        todo!();
        /*
            if (has_contiguity_ ==
          static_cast<uint8_t>(HasContiguityPolicy::ContiguityNotSupported)) {
        TORCH_CHECK_NOT_IMPLEMENTED(
            false,
            "Tensors of type ",
            tensorimpl_type_name(),
            " do not have is_contiguous");
      } else {
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            has_contiguity_ ==
            static_cast<uint8_t>(HasContiguityPolicy::CustomBehavior));
        return is_contiguous_custom(memory_format);
      }
        */
    }
    
    pub fn is_contiguous_custom(&self, memory_format: MemoryFormat) -> bool {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          false,
          "TensorImpl::is_contiguous_custom should never be called; did you "
          "set_has_contiguity_policy and forget to override is_contiguous_custom?");
        */
    }
}

pub fn delete_placement_delete_context(ptr: *mut c_void)  {
    
    todo!();
        /*
            delete static_cast<PlacementDeleteContext*>(ptr);
        */
}

impl PlacementDeleteContext {
    
    pub fn make_data_ptr(&mut self, 
        data_ptr:       DataPtr,
        placement_dtor: PlacementDtor,
        size:           usize,
        device:         Device) -> DataPtr {
        
        todo!();
        /*
            auto* ptr = data_ptr.get();
      return {
          ptr,
          new PlacementDeleteContext(move(data_ptr), placement_dtor, size),
          &deletePlacementDeleteContext,
          device};
        */
    }
}

impl TensorImpl {
    
    /**
      | Set whether or not a tensor requires
      | gradient.
      |
      | Setting requires_grad to true on inference
      | tensor outside InferenceMode is forbidden.
      | Ideally it would also be illegal inside
      | InferenceMode.
      |
      | But there's no way that we can directly
      | allocate a tensor to have requires_grad = true
      | in C++ constructor so set_requires_grad is
      | widely used in C++ frontend. Forbidding it
      | inside InferenceMode will force users to delete
      | these setter code in their code which is not
      | ideal.
      */
    pub fn set_requires_grad(&mut self, requires_grad: bool)  {
        
        todo!();
        /*
            TORCH_CHECK(
          !(requires_grad && is_inference() && !InferenceMode::is_enabled()),
          "Setting requires_grad=True on inference tensor outside InferenceMode is not allowed.");
      if (!requires_grad && !autograd_meta_)
        return;
      if (!autograd_meta_)
        autograd_meta_ = GetAutogradMetaFactory()->make();
      // NB: In principle, setting requires_grad to false could result in
      // the AutogradMeta becoming equal to a default constructed state,
      // in which case we could apply the nullptr AutogradMeta optimization
      // (see autograd_meta_ docs).  But we don't do this right now.  Note
      // that it is unsound to unconditionally set AutogradMeta to false
      // when you set requires_grad to False, as there may be nontrivial
      // information content in the other fields; for example, we may
      // have set the string name for a Variable, or there may be hooks
      // registered for it.
      autograd_meta_->set_requires_grad(requires_grad, this);
        */
    }
    
    /**
      | True if a tensor requires gradient.
      | 
      | Tensors which require gradient have
      | history tracked for any operations
      | performed on them, so that we can automatically
      | differentiate back to them.
      | 
      | A tensor that requires gradient and
      | has no history is a "leaf" tensor, which
      | we accumulate gradients into.
      |
      */
    pub fn requires_grad(&self) -> bool {
        
        todo!();
        /*
            if (!autograd_meta_)
        return false;
      return autograd_meta_->requires_grad();
        */
    }
    
    /**
      | Set the pointer to autograd metadata.
      |
      */
    pub fn set_autograd_meta(&mut self, autograd_meta: Box<dyn AutogradMetaInterface>)  {
        
        todo!();
        /*
            // NB: autograd_meta may be null!  That just means it's the default
      // constructor
      autograd_meta_ = move(autograd_meta);
        */
    }
    
    /**
      | Return the pointer to autograd metadata.
      | May return nullptr if the tensor does
      | not track gradients.
      |
      */
    pub fn autograd_meta(&self) -> *mut dyn AutogradMetaInterface {
        
        todo!();
        /*
            // NB: Might return null!
      return autograd_meta_.get();
        */
    }
    
    /**
      | Return a TensorImpl that is a shallow-copy
      | of this TensorImpl.
      | 
      | For usage of `version_counter` and
      | `allow_tensor_metadata_change`,
      | see NOTE [ TensorImpl Shallow-Copying ].
      |
      */
    pub fn shallow_copy_and_detach_a(&self, 
        version_counter:              &VariableVersion,
        allow_tensor_metadata_change: bool) -> TensorImplAdapter {
        
        todo!();
        /*
            auto impl = make_intrusive<TensorImpl>(
          // No need to populate Storage; copy_tensor_metadata will do it for us.
          key_set_,
          data_type_,
          device_opt_);
      copy_tensor_metadata(
          /*src_impl=*/this,
          /*dest_impl=*/impl.get(),
          /*version_counter=*/version_counter,
          /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
      impl->refresh_numel();
      impl->refresh_contiguous();
      return impl;
        */
    }
    
    /**
      | Return a TensorImpl that is a shallow-copy
      | of this TensorImpl.
      | 
      | For usage of `version_counter` and
      | `allow_tensor_metadata_change`,
      | see NOTE [ TensorImpl Shallow-Copying ].
      |
      */
    pub fn shallow_copy_and_detach_b(&self, 
        version_counter:              VariableVersion,
        allow_tensor_metadata_change: bool) -> TensorImplAdapter {
        
        todo!();
        /*
            auto impl = make_intrusive<TensorImpl>(
          // No need to populate Storage; copy_tensor_metadata will do it for us.
          key_set_,
          data_type_,
          device_opt_);
      copy_tensor_metadata(
          /*src_impl=*/this,
          /*dest_impl=*/impl.get(),
          /*version_counter=*/move(version_counter),
          /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
      impl->refresh_numel();
      impl->refresh_contiguous();
      return impl;
        */
    }
    
    pub fn copy_tensor_metadata_except_version_counter(&mut self, 
        src_impl:                     *const TensorImpl,
        dest_impl:                    *mut TensorImpl,
        allow_tensor_metadata_change: bool)  {
        
        todo!();
        /*
            dest_impl->storage_ = src_impl->storage_;
      dest_impl->sizes_and_strides_ = src_impl->sizes_and_strides_;
      dest_impl->storage_offset_ = src_impl->storage_offset_;
      dest_impl->data_type_ = src_impl->data_type_;
      dest_impl->device_opt_ = src_impl->device_opt_;
      dest_impl->key_set_ = src_impl->key_set_;
      dest_impl->is_contiguous_ = src_impl->is_contiguous_;
      dest_impl->has_contiguity_ = src_impl->has_contiguity_;
      dest_impl->is_channels_last_contiguous_ =
          src_impl->is_channels_last_contiguous_;
      dest_impl->is_channels_last_3d_contiguous_ =
          src_impl->is_channels_last_3d_contiguous_;
      dest_impl->is_channels_last_ = src_impl->is_channels_last_;
      dest_impl->is_channels_last_3d_ = src_impl->is_channels_last_3d_;
      dest_impl->is_non_overlapping_and_dense_ =
          src_impl->is_non_overlapping_and_dense_;
      dest_impl->is_wrapped_number_ = src_impl->is_wrapped_number_;
      dest_impl->reserved_ = src_impl->reserved_;
      dest_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
      dest_impl->storage_access_should_throw_ =
          src_impl->storage_access_should_throw_;
      if (src_impl->named_tensor_meta_ != nullptr) {
        dest_impl->named_tensor_meta_ = src_impl->named_tensor_meta_->clone();
      }
        */
    }
    
    pub fn copy_tensor_metadata_c(&mut self, 
        src_impl:                     *const TensorImpl,
        dest_impl:                    *mut TensorImpl,
        version_counter:              &VariableVersion,
        allow_tensor_metadata_change: bool)  {
        
        todo!();
        /*
            copy_tensor_metadata_except_version_counter(
          src_impl, dest_impl, allow_tensor_metadata_change);
      // TODO: In the ideal end state, it's okay to set disabled version_counter
      // on inference tensor since it's a no-op. This requires refactor on call
      // sites.
      if (!dest_impl->is_inference()) {
        dest_impl->set_version_counter(version_counter);
      }
        */
    }
    
    pub fn copy_tensor_metadata_d(&mut self, 
        src_impl:                     *const TensorImpl,
        dest_impl:                    *mut TensorImpl,
        version_counter:              VariableVersion,
        allow_tensor_metadata_change: bool)  {
        
        todo!();
        /*
            copy_tensor_metadata_except_version_counter(
          src_impl, dest_impl, allow_tensor_metadata_change);
      if (!dest_impl->is_inference()) {
        dest_impl->set_version_counter(move(version_counter));
      }
        */
    }
}

lazy_static!{
    /*
    AutogradMetaFactory* meta_factory = nullptr;
    */
}

pub fn set_autograd_meta_factory(factory: *mut dyn AutogradMetaFactoryInterface)  {
    
    todo!();
        /*
            meta_factory = factory;
        */
}

pub fn get_autograd_meta_factory() -> *mut dyn AutogradMetaFactoryInterface {
    
    todo!();
        /*
            TORCH_CHECK(
          meta_factory,
          "Support for autograd has not been loaded; have you linked against libtorch.so?")
      return meta_factory;
        */
}
