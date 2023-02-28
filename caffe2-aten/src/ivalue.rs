crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/ivalue.h]

pub struct CustomClassHolder {
    base: IntrusivePtrTarget,
}

pub type TypePtr = Arc<Type>;
pub type ClassTypePtr = Arc<ClassType>;

/**
  | A comparator that checks ordering of
  | two IValues of same type.
  |
  */
pub type IValueComparator = fn(a: &IValue, b: &IValue) -> bool;

/**
  | We need a ComplexHolder because currently the
  | payloads in the Union only take 64 bits.
  |
  | Since ComplexDouble takes up 128 bits, and is
  | too big to fit in the IValue directly, we
  | indirect complex numbers through an intrusive
  | pointer to ComplexHolder (which contains
  | a complex).
  */
#[derive(Default)]
pub struct ComplexHolder {
    base: IntrusivePtrTarget,
    val:  Complex<f64>,
}

impl ComplexHolder {
    
    pub fn new<T>(c: Complex<T>) -> Self {
    
        todo!();
        /*
            val = convert<decltype(val), complex<T>>(c);
        */
    }
}

/**
  | This is an owning wrapper for
  | a optional<vector<T>> that can be implicitly
  | converted to a (non-owning)
  | optional<ArrayRef<T>>.
  |
  | Its purpose is to be used in generated code to
  | keep the vector alive either until the end of
  | a statement (as a temporary), or as a saved arg
  | in autograd.
  |
  */
#[derive(Default)]
pub struct OptionalArray<T> {
    list: Option<Vec<T>>,
}

impl OptionalArray<T> {
    
    pub fn new(val: Vec<T>) -> Self {
    
        todo!();
        /*
        : list(move(val)),

        
        */
    }

    /**
      | Used when saving an argument for the
      | backwards pass.
      |
      */
    pub fn assign_from(&mut self, ref_: Option<&[T]>) -> &mut OptionalArray {
        
        todo!();
        /*
            if (ref) {
          list = vector<T>(ref->begin(), ref->end());
        } else {
          list = nullopt;
        }
        return *this;
        */
    }
    
    pub fn operator_optional_array_ref_t(&mut self) -> Option<&[T]> {
        
        todo!();
        /*
            if (!list) {
          return nullopt;
        }
        return *list;
        */
    }
}

/**
  | Capsule is an internal implementation detail of
  | custom C++ classes. We define it as an owning
  | wrapper for
  | intrusive_ptr<TorchCustomClassHolder>
  |
  | This wrapper is here to serve as an abstraction
  | of the type erased custom class object pointer.
  |
  | It also allow pybind11 to treat this as
  | a standalone class to register as a separate
  | type caster, instead of a custom pointer holder
  | which the pointer holder type caster try to
  | "unwrap" it automatically.
  |
  */
pub struct Capsule {
    obj_ptr: IntrusivePtr<TorchCustomClassHolder>,
}

impl Capsule {
    
    pub fn new(ptr: IntrusivePtr<TorchCustomClassHolder>) -> Self {
    
        todo!();
        /*
        : obj_ptr(move(ptr)),

        
        */
    }
}

/**
  | IValue is the generic tagged union used by the
  | interpreter to hold all value types.
  |
  | It is a 16-byte object with an 8-byte payload
  | and an 8-byte tag.
  |
  | The tag is currently 4 bytes to determine the
  | type, and 1 byte to mark whether that type is
  | a subtype of intrusive_ptr_target and needs
  | retain/release calls.
  */
#[macro_export] macro_rules! torch_forall_tags {
    ($_:ident) => {
        /*
        
          _(None)                    
          _(Tensor)                  
          _(Storage)                 
          _(Double)                  
          _(ComplexDouble)           
          _(Int)                     
          _(Bool)                    
          _(Tuple)                   
          _(String)                  
          _(Blob)                    
          _(GenericList)             
          _(GenericDict)             
          _(Future)                  
          _(Device)                  
          _(Stream)                  
          _(Object)                  
          _(PyObject)                
          _(Uninitialized)           
          _(Capsule)                 
          _(RRef)                    
          _(Quantizer)               
          _(Generator)               
          _(Enum)
        */
    }
}


/*
  | [doxygen private]
  |
  | These methods are not actually private but we
  | don't want to document them, so they are marked
  | `@private`, which hides them on the doxygen
  | documentation for this page.
  */

/**
  | IValue (Interpreter Value) is a tagged union
  | over the types supported by the TorchScript
  | interpreter. IValues contain their values as
  | an `IValue::Payload`, which holds primitive
  | types (`i64`, `bool`, `double`, `Device`)
  | and `Tensor` as values, and all other types as
  | a `intrusive_ptr`. In order to optimize
  | performance of the destructor and related
  | operations by making the `Tensor` and
  | `intrusive_ptr` paths generate the same code,
  | we represent a null `intrusive_ptr` as
  | `UndefinedTensorImpl::singleton()`, *not*
  | `nullptr`.
  |
  | IValues are used as inputs to and outputs from
  | the TorchScript interpreter.
  |
  | To retrieve the value contained within an
  | IValue, use the `.toX()` methods, where `X` is
  | the type you are trying to get. Note that
  | neither the `.toX()` methods nor the templated
  | `.to<T>` functions do any kind of casting,
  | they only unwrap the contained value.
  |
  | For example:
  |
  | \rst
  | .. code-block:: cpp
  |
  |   // Make the IValue
  |   TorchIValue my_ivalue(26);
  |   cout << my_ivalue << "\n";
  |
  |   // Unwrap the IValue
  |   i64 my_int = my_ivalue.toInt();
  |   cout << my_int << "\n";
  |
  |   // This will throw an error!
  |   // `my_ivalue` is tagged as an int and cannot be used as another type
  |   TorchTensor my_tensor = my_ivalue.toTensor();
  | \endrst
  */
pub struct IValue {
    payload:          Payload,
    tag:              Tag,
    is_intrusive_ptr: bool,
}

impl Drop for IValue {

    /// @private [doxygen private]
    fn drop(&mut self) {
        todo!();
        /*
            destroy();
        */
    }
}

lazy_static!{
    /*
    impl IValue {
        // Some template constructors of IValue calls another constructor recursively.
          // This SNIFAEs the called constructor exists.
          template <class T>
          using enable_if_ivalue_constructible =
              enable_if_t<is_constructible<IValue, T>::value, nullptr_t>;

          template <class T, enable_if_ivalue_constructible<T> = nullptr>
          IValue(List<T>&& v);
          template <class T, enable_if_ivalue_constructible<T> = nullptr>
          IValue(const List<T>& v);
          template <class T, enable_if_ivalue_constructible<T> = nullptr>
          IValue(ArrayRef<T> v);
          template <class T, enable_if_ivalue_constructible<T> = nullptr>
          IValue(const vector<T>& v);
          template <class T, usize N>
          IValue(array<T, N> v);
    }
    */
}

impl Default for IValue {
    
    fn default() -> Self {
        todo!();
        /*


            : tag(Tag::None), is_intrusive_ptr(false)
        */
    }
}

/// Detect aliased tensors.
pub struct HashAliasedIValue {

}

impl HashAliasedIValue {
    
    pub fn invoke(&self, val: &IValue) -> usize {
        
        todo!();
        /*
            if (val.isTensor()) {
                if (val.toTensor().is_mkldnn()) {
                    // MKLDNN tensors dont have storage and dont create views
                    // or aliasing so we can just use Tensor pointer, TODO: find way
                    // to use mkldnn storage
                    return reinterpret_cast<usize>(val.toTensor().unsafeGetTensorImpl());
                } else {
                    return reinterpret_cast<usize>(
                        val.toTensor().storage().unsafeGetStorageImpl());
                }
            }
            // If it is not a Tensor, then two mutable IValues alias each other only
            // if they are the same pointer.
            return val.payload.u.as_int;
        */
    }
}

pub struct CompAliasedIValues {

}

impl CompAliasedIValues {
    
    pub fn invoke(&self, 
        lhs: &IValue,
        rhs: &IValue) -> bool {
        
        todo!();
        /*
            return lhs.isAliasOf(rhs);
        */
    }
}

pub type HashAliasedIValues   = HashSet<IValue,HashAliasedIValue,CompAliasedIValues>;
pub type HashAliasedIValueMap = HashMap<IValue,IValue,HashAliasedIValue,CompAliasedIValues>;

/**
  | NOTE: IValue tags are intentionally private.
  |
  | In the future we may encode this value
  | different (e.g. using NaN boxing), and this
  | would make it more costly to determine the
  | tag for all types vs just determining if
  | something is a particular type.
  |
  | Instead we want clients to use the `isX`
  | methods when possible.
  |
  | If for perf. reasons you really, absolutely,
  | must have a jump table, then we can revisit
  | this.
  |
  */
#[repr(u32)]
pub enum IValueTag {
    None,
    Tensor,                 
    Storage,                
    Double,                 
    ComplexDouble,          
    Int,                    
    Bool,                   
    Tuple,                  
    String,                 
    Blob,                   
    GenericList,            
    GenericDict,            
    Future,                 
    Device,                 
    Stream,                 
    Object,                 
    PyObject,               
    Uninitialized,          
    Capsule,                
    RRef,                   
    Quantizer,              
    Generator,              
    Enum,
}

/**
  | We use a nested union here so that we can make
  | the copy easy and efficient in the non-tensor
  | (i.e., trivially copyable) case.
  |
  | Specifically, we do not have to do
  | a switch-on-tag to figure out which union
  | member to assign; we can just use
  | TriviallyCopyablePayload::operator=.
  */
pub union IValuePayloadTriviallyCopyablePayload {
    as_int:           i64,
    as_double:        f64,
    as_bool:          bool,

    /**
      | Invariant: never nullptr; null state
      | is represented as
      | 
      | UndefinedTensorImpl::singleton()
      | for consistency of representation
      | with Tensor.
      |
      */
    as_intrusive_ptr: *mut IntrusivePtrTarget,

    as_device:        DeviceDescriptor,
}

impl Default for IValuePayloadTriviallyCopyablePayload {
    
    fn default() -> Self {
        todo!();
        /*
        : as_int(0),

        
        */
    }
}

pub struct DeviceDescriptor {
    ty:    DeviceType,
    index: DeviceIndex,
}

pub union IValuePayload {
    u:         IValuePayloadTriviallyCopyablePayload,
    as_tensor: Tensor,
}

impl Default for Payload {
    
    fn default() -> Self {
        todo!();
        /*
        : u(),
        */
    }
}

impl IValue {

    pub fn new(rhs: &IValue) -> Self {
    
        todo!();
        /*
        : i_value(rhs.payload, rhs.tag, rhs.is_intrusive_ptr),

            if (is_intrusive_ptr && payload.u.as_intrusive_ptr != UndefinedTensorImpl::singleton()) {
          raw::intrusive_ptr::incref(payload.u.as_intrusive_ptr);
        }
        */
    }
    
    pub fn new(rhs: IValue) -> Self {
    
        todo!();
        /*


            : tag(rhs.tag), is_intrusive_ptr(rhs.is_intrusive_ptr) 

        moveFrom(move(rhs));
        */
    }

    #[inline(always)] 
    pub fn assign_from(&mut self, rhs: IValue) -> &mut IValue {
        
        todo!();
        /*
            if (&rhs == this) {
          return *this;
        }

        destroy();
        moveFrom(move(rhs));
        return *this;
        */
    }
    
    pub fn assign_from(&mut self, rhs: &IValue) -> &mut IValue {
        
        todo!();
        /*
            IValue(rhs).swap(*this);
        return *this;
        */
    }
    
    pub fn dump(&self)  {
        
        todo!();
        /*
        
        */
    }

    /** 
      | Equality comparison. The semantics are the
      | same as Python's `==`:
      |
      | 1. Numerical types are compared by value.
      |
      | 2. Tensors compute element-wise equality,
      | returning a BoolTensor (see: `torch.eq()`)
      |
      | 3. Strings are compared by value.
      |
      | 4. Sequence types (list, tuple) are compared
      |    lexicographically by comparing their
      |    elements. Different sequence types never
      |    compare equal.
      |
      | 5. Mappings (dict) must have equal (key,
      | value) pairs.
      |
      | 6. If not listed above, the default behavior
      | for is to test identity equality
      | (e.g. pointer equality).
      |
      | Why does this return an IValue instead of
      | a bool? Because in PyTorch, `tensor1 ==
      | tensor2` returns a `BoolTensor`, not a bool.
      |
      | NOTE: we (like Python) assume that identity
      | equality implies value equality for
      | efficiency.
      |
      | TODO: need to support customizing equality
      */
    pub fn equals(&self, rhs: &IValue) -> IValue {
        
        todo!();
        /*
        
        */
    }

    /**
      | Identity comparison. Checks if `this`
      | is the same object as `rhs`. The semantics
      | are the same as Python's `is` operator.
      | 
      | -----------
      | @note
      | 
      | Like in Python, this operation is poorly
      | defined for primitive types like numbers
      | and strings. Prefer to use `==` unless
      | you really want to check identity equality.
      |
      */
    pub fn is(&self, rhs: &IValue) -> bool {
        
        todo!();
        /*
        
        */
    }

    /**
      | Hashing for IValues. Returns an IValue-boxed
      | int.
      | 
      | Some notes:
      | 
      | - Like eager, Tensors are hashed by looking
      | at the pointer. This is not strictly
      | correct because two value-equal tensors
      | with different tensor pointers will
      | hash differently, but we choose to reproduce
      | the eager semantics.
      | 
      | - Hashing is not defined on all built-in
      | IValue types (e.g. list and dict), following
      | Python. Calling `hash()` on these types
      | will throw.
      |
      */
    pub fn hash(&self) -> IValue {
        
        todo!();
        /*
            return (i64)IValue::hash(*this);
        */
    }

    /**
      | This is defined because `hash` dispatches
      | to a function of this signature. See
      | the member function `hash()`.
      |
      */
    pub fn hash(iv: &IValue) -> usize {
        
        todo!();
        /*
        
        */
    }

    /**
      | @private [doxygen private] [container
      | equality]
      | 
      | This is an equality implementation
      | that assumes objects with the same identity
      | equal themselves, for efficiency reasons.
      | 
      | We primarily have this for consistency,
      | because Python does the same thing.
      | 
      | This actually provokes user-visible
      | changes in behavior due to quirks in
      | torch: [tensor1] == [tensor1] -> True
      | (because container equality will first
      | compare identity) [tensor1] == [tensor1_copy]
      | ->
      | 
      | RuntimeError: bool value of Tensor
      | is ambiguous
      |
      */
    pub fn fast_equals_for_container(&mut self, 
        lhs: &IValue,
        rhs: &IValue) -> bool {
        
        todo!();
        /*
        
        */
    }

    /// @private [doxygen private]
    pub fn is_alias_of(&self, rhs: &IValue) -> bool {
        
        todo!();
        /*
            if (this->tag != rhs.tag) {
          // Trivially don't alias if the type is different
          return false;
        }

        // Tensors should be compared based on internal storage
        if (this->isTensor()) {
          const auto& thisTensor = this->toTensor();
          const auto& rhsTensor = rhs.toTensor();
          // mkldnn tensors dont have views or storage, so we compare
          // based on tensor impl. //TODO: find a way to use mkldnn storage
          if (thisTensor.is_mkldnn() || rhsTensor.is_mkldnn()) {
            return thisTensor.unsafeGetTensorImpl() ==
                rhsTensor.unsafeGetTensorImpl();
          }

          return thisTensor.is_alias_of(rhsTensor);
        }

        if (!this->is_intrusive_ptr) {
          // Primitive types don't alias anything
          return false;
        }

        AT_ASSERT(rhs.is_intrusive_ptr);

        // Other types can be compared by their ptr value
        return this->payload.u.as_intrusive_ptr == rhs.payload.u.as_intrusive_ptr;
        */
    }

    /// @private [doxygen private]
    pub fn use_count(&self) -> usize {
        
        todo!();
        /*
            if (isTensor()) {
          return payload.as_tensor.use_count();
        }

        if (!is_intrusive_ptr) {
          return 1;
        }

        if (payload.u.as_intrusive_ptr == UndefinedTensorImpl::singleton()) {
          return 0;
        }
        return raw::intrusive_ptr::use_count(payload.u.as_intrusive_ptr);
        */
    }

    /// @private [doxygen private]
    pub fn swap(&mut self, rhs: &mut IValue)  {
        
        todo!();
        /*
            if (isTensor() && rhs.isTensor()) {
          swap(payload.as_tensor, rhs.payload.as_tensor);
        } else if (isTensor()) {
          Tensor t = move(payload.as_tensor);
          // As far as I can tell, omitting the usual explicit destructor call
          // is not UB in and of itself, and it's a slight perf win. The
          // destructor is a no-op, because the moved-from Tensor is
          // effectively an intrusive_ptr in the null state, so we don't need
          // the behavior for correctness reasons either. Leaving this
          // explanatory comment, including commented-out destructor call, to
          // make this abundantly clear.
          //
          // payload.as_tensor.~Tensor();
          payload.u = rhs.payload.u;
          new (&rhs.payload.as_tensor) Tensor(move(t));
        } else if (rhs.isTensor()) {
          rhs.swap(*this);
          return;
        } else {
          swap(payload.u, rhs.payload.u);
        }
        swap(is_intrusive_ptr, rhs.is_intrusive_ptr);
        swap(tag, rhs.tag);
        */
    }

    /**
      | Accessors for subtypes are arranged together
      | below
      |
      | While some of these accessors could be
      | generated through templates, we prefer to
      | write them manually for clarity
      */
    pub fn new(t: Tensor) -> Self {
    
        todo!();
        /*
        : tag(Tag::Tensor),
        : is_intrusive_ptr(false),

            new (&payload.as_tensor) Tensor(move(t));
        */
    }
    
    pub fn is_tensor(&self) -> bool {
        
        todo!();
        /*
            return Tag::Tensor == tag;
        */
    }

    /**
      | Outlined error path so that toTensor()
      | can be inlined.
      |
      */
    pub fn report_to_tensor_type_error(&self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_tensor(&mut self) -> Tensor {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_tensor(&mut self) -> &mut Tensor {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_tensor(&mut self) -> &Tensor {
        
        todo!();
        /*
        
        */
    }
    
    pub fn unsafe_to_tensor_impl(&self) -> *mut TensorImpl {
        
        todo!();
        /*
            return payload.as_tensor.unsafeGetTensorImpl();
        */
    }
    
    pub fn new(s: Storage) -> Self {
    
        todo!();
        /*


            : tag(Tag::Storage), is_intrusive_ptr(static_cast<bool>(s)) 

        // Note: the undefined tensor is not refcounted, so while it
        // is tagged as a tensor, is_intrusive_ptr is set to false.
        // This is not an optional optimization: our incref call
        // *will not* do the right thing when called on an
        // undefined tensor.
        payload.u.as_intrusive_ptr = null_to_undefined_tensor(s.unsafeReleaseStorageImpl());
        */
    }
    
    pub fn is_storage(&self) -> bool {
        
        todo!();
        /*
            return Tag::Storage == tag;
        */
    }
    
    pub fn to_storage(&mut self) -> Storage {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_ivalue(&self) -> &IValue {
        
        todo!();
        /*
            return *this;
        */
    }
    
    pub fn to_ivalue(&mut self) -> &mut IValue {
        
        todo!();
        /*
            return *this;
        */
    }

    /// @private [doxygen private]
    pub fn new(blob: IntrusivePtr<Blob>) -> Self {
    
        todo!();
        /*


            : tag(Tag::Blob), is_intrusive_ptr(true) 
        // TODO (after Tensor merge) If we pass in a Blob holding a Tensor, extract
        // and store it as a Tensor instead.
        payload.u.as_intrusive_ptr = null_to_undefined_tensor(blob.release());
        */
    }

    /// @private [doxygen private]
    pub fn is_blob(&self) -> bool {
        
        todo!();
        /*
            return Tag::Blob == tag;
        */
    }

    /// @private [doxygen private]
    pub fn to_blob(&mut self) -> IntrusivePtr<Blob> {
        
        todo!();
        /*
        
        */
    }

    /**
      | Capsule. No new callsites of these APIs
      | should be introduced.
      |
      */
    #[inline] pub fn make_capsule(blob: IntrusivePtr<TorchCustomClassHolder>) -> IValue {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_capsule(&self) -> bool {
        
        todo!();
        /*
            return Tag::Capsule == tag;
        */
    }
    
    pub fn to_capsule(&mut self) -> IntrusivePtr<TorchCustomClassHolder> {
        
        todo!();
        /*
        
        */
    }

    pub fn new(custom_class: IntrusivePtr<T>) -> Self {
    
        lazy_static!{
            /*
            // Custom C++ classes
              template <
                  typename T,
                  enable_if_t<
                      is_base_of<TorchCustomClassHolder, T>::value,
                      int> = 0>
              IValue(intrusive_ptr<T> custom_class);
            */
        }

        todo!();
        /*


        
        */
    }
    
    pub fn is_custom_class(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_custom_class<T>(&mut self) -> IntrusivePtr<T> {
    
        todo!();
        /*
        
        */
    }

    pub fn new(v: IntrusivePtr<Tuple>) -> Self {
    
        todo!();
        /*


        
        */
    }

    pub fn is_tuple(&self) -> bool {
        
        lazy_static!{
            /*
            template <
                  typename... Args,
                  enable_if_t<
                      !disjunction<
                          is_lvalue_reference<Args>...,
                          negation<is_constructible<IValue, Args>>...>::value,
                      nullptr_t> = nullptr>
              IValue(const tuple<Args...>& t);
              template <
                  typename... Args,
                  enable_if_t<
                      !disjunction<
                          is_lvalue_reference<Args>...,
                          negation<is_constructible<IValue, Args>>...>::value,
                      nullptr_t> = nullptr>
              IValue(tuple<Args...>&& t);
              bool isTuple() const {
                return Tag::Tuple == tag;
              }
            */
        }

        todo!();
        /*
            return Tag::Tuple == tag;
        */
    }
    
    pub fn to_tuple(&mut self) -> IntrusivePtr<Tuple> {
        
        todo!();
        /*
        
        */
    }

    pub fn new(d: f64) -> Self {
    
        todo!();
        /*
        : tag(Tag::Double),
        : is_intrusive_ptr(false),

            payload.u.as_double = d;
        */
    }
    
    pub fn is_double(&self) -> bool {
        
        todo!();
        /*
            return Tag::Double == tag;
        */
    }
    
    pub fn to_double(&self) -> f64 {
        
        todo!();
        /*
            AT_ASSERT(isDouble());
        return payload.u.as_double;
        */
    }

    pub fn new<T>(c: Complex<T>) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn is_complex_double(&self) -> bool {
        
        todo!();
        /*
            return Tag::ComplexDouble == tag;
        */
    }
    
    pub fn to_complex_double(&self) -> Complex<f64> {
        
        todo!();
        /*
        
        */
    }

    pub fn new(v: IntrusivePtr<Future>) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn is_future(&self) -> bool {
        
        todo!();
        /*
            return Tag::Future == tag;
        */
    }
    
    pub fn to_future(&mut self) -> IntrusivePtr<Future> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new(v: IntrusivePtr<RRefInterface>) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn is_rref(&self) -> bool {
        
        todo!();
        /*
            return Tag::RRef == tag;
        */
    }
    
    pub fn to_rref(&mut self) -> IntrusivePtr<RRefInterface> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new(v: IntrusivePtr<Quantizer>) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn is_quantizer(&self) -> bool {
        
        todo!();
        /*
            return Tag::Quantizer == tag;
        */
    }
    
    pub fn to_quantizer(&mut self) -> IntrusivePtr<Quantizer> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new(i: i64) -> Self {
    
        todo!();
        /*
        : tag(Tag::Int),
        : is_intrusive_ptr(false),

            payload.u.as_int = i;
        */
    }

    /// allow you to pass literals (3, 4) without
    /// ambiguity
    ///
    pub fn new(i: i32) -> Self {
    
        todo!();
        /*


            : IValue(static_cast<i64>(i))
        */
    }
    
    pub fn is_int(&self) -> bool {
        
        todo!();
        /*
            return Tag::Int == tag;
        */
    }
    
    pub fn to_int(&self) -> i64 {
        
        todo!();
        /*
            AT_ASSERT(isInt());
        return payload.u.as_int;
        */
    }

    pub fn new(b: bool) -> Self {
    
        todo!();
        /*


            : tag(Tag::Bool), is_intrusive_ptr(false) 

    #if defined(__clang__) && defined(__x86_64__)
        // Initializing entire payload stops valgrind's from reporting
        // "jump or move depends on uninitialised value" in IValue copy constructor
        // See https://github.com/pytorch/pytorch/issues/37117
        payload.u.as_int = b;
    #else
        payload.u.as_bool = b;
    #endif
        */
    }
    
    pub fn is_bool(&self) -> bool {
        
        todo!();
        /*
            return Tag::Bool == tag;
        */
    }
    
    pub fn to_bool(&self) -> bool {
        
        todo!();
        /*
            AT_ASSERT(isBool());
        return payload.u.as_bool;
        */
    }
    
    pub fn is_int_list(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_int_list(&mut self) -> List<i64> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_int_vector(&self) -> Vec<i64> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new(v: IntrusivePtr<ConstantString>) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn new(v: String) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn new(v: *const u8) -> Self {
    
        todo!();
        /*
        : i_value(string(v)),

        
        */
    }
    
    pub fn new(v: StringView) -> Self {
    
        todo!();
        /*
        : i_value(string(v)),

            }{
        */
    }
    
    pub fn is_string(&self) -> bool {
        
        todo!();
        /*
            return Tag::String == tag;
        */
    }
    
    pub fn to_string(&mut self) -> IntrusivePtr<ConstantString> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_string_ref(&self) -> &String {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_optional_string_ref(&self) -> Option<ReferenceWrapper<String>> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_string_view(&self) -> StringView {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_double_list(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_double_list(&mut self) -> List<f64> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_double_vector(&self) -> Vec<f64> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_complex_double_list(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_complex_double_list(&mut self) -> List<Complex<f64>> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_complex_double_vector(&self) -> Vec<Complex<f64>> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_bool_list(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_bool_list(&mut self) -> List<bool> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_tensor_list(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_tensor_list(&mut self) -> List<Tensor> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_tensor_vector(&self) -> Vec<Tensor> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new(v: List<IValue>) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn is_list(&self) -> bool {
        
        todo!();
        /*
            return Tag::GenericList == tag;
        */
    }
    
    pub fn to_list(&mut self) -> List<IValue> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_list_ref(&self) -> &[IValue] {
        
        todo!();
        /*
        
        */
    }

    pub fn new(v: Dict<IValue,IValue>) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn is_generic_dict(&self) -> bool {
        
        todo!();
        /*
            return Tag::GenericDict == tag;
        */
    }
    
    pub fn is_object(&self) -> bool {
        
        todo!();
        /*
            return tag == Tag::Object;
        */
    }
    
    pub fn is_py_object(&self) -> bool {
        
        todo!();
        /*
            return tag == Tag::PyObject;
        */
    }
    
    pub fn is_enum(&self) -> bool {
        
        todo!();

        /*
            return tag == Tag::Enum;
        */
    }
    
    pub fn is_none(&self) -> bool {
        
        todo!();
        /*
            return Tag::None == tag;
        */
    }
    
    pub fn to_none(&self) -> String {
        
        todo!();
        /*
            AT_ASSERT(isNone());
        return "None";
        */
    }
    
    pub fn uninitialized() -> IValue {
        
        todo!();
        /*
            auto i = IValue();
        i.tag = Tag::Uninitialized;
        return i;
        */
    }

    /// Scalar, which gets encoded as either an
    /// Int, a Double or a ComplexDouble
    ///
    pub fn new(s: &Scalar) -> Self {
    
        todo!();
        /*
        : i_value(),

            if (s.isFloatingPoint()) {
          *this = s.toDouble();
        } else if (s.isComplex()) {
          *this = s.toComplexDouble();
        } else if (s.isBoolean()) {
          *this = s.toBool();
        } else if (s.isIntegral(false)) {
          *this = s.toLong();
        } else {
          TORCH_CHECK(false, "Unknown type in Scalar");
        }
        */
    }
    
    pub fn is_scalar(&self) -> bool {
        
        todo!();
        /*
            return isDouble() || isInt() || isComplexDouble() || isBool();
        */
    }
    
    pub fn to_scalar(&self) -> Scalar {
        
        todo!();
        /*
            if (isDouble())
          return toDouble();
        else if (isInt())
          return toInt();
        else if (isComplexDouble())
          return toComplexDouble();
        else if (isBool())
          return toBool();
        throw runtime_error("IValue is not a Scalar");
        */
    }
    
    pub fn new(d: Device) -> Self {
    
        todo!();
        /*
        : tag(Tag::Device),
        : is_intrusive_ptr(false),

            payload.u.as_device.type = d.type();
        payload.u.as_device.index = d.index();
        */
    }
    
    pub fn is_device(&self) -> bool {
        
        todo!();
        /*
            return Tag::Device == tag;
        */
    }
    
    pub fn to_device(&self) -> Device {
        
        todo!();
        /*
            AT_ASSERT(isDevice());
        return Device(payload.u.as_device.type, payload.u.as_device.index);
        */
    }
    
    pub fn new(stream: Stream) -> Self {
    
        todo!();
        /*
        : tag(Tag::Stream),
        : is_intrusive_ptr(false),

            payload.u.as_int = stream.pack();
        */
    }
    
    pub fn is_stream(&self) -> bool {
        
        todo!();
        /*
            return Tag::Stream == tag;
        */
    }

    pub fn new(t: ScalarType) -> Self {
    
        todo!();
        /*


            : IValue(static_cast<underlying_type<ScalarType>::type>(t))
        */
    }
    
    pub fn to_scalar_type(&self) -> ScalarType {
        
        todo!();
        /*
            return static_cast<ScalarType>(toInt());
        */
    }

    pub fn new(l: Layout) -> Self {
    
        todo!();
        /*


            : IValue(static_cast<underlying_type<Layout>::type>(l))
        */
    }
    
    pub fn to_layout(&self) -> Layout {
        
        todo!();
        /*
            return static_cast<Layout>(toInt());
        */
    }
    
    pub fn new(m: MemoryFormat) -> Self {
    
        todo!();
        /*


            : IValue(static_cast<underlying_type<MemoryFormat>::type>(m))
        */
    }
    
    pub fn to_memory_format(&self) -> MemoryFormat {
        
        todo!();
        /*
            return static_cast<MemoryFormat>(toInt());
        */
    }

    pub fn new(qscheme: QScheme) -> Self {
    
        todo!();
        /*


            : tag(Tag::Int), is_intrusive_ptr(false) 
        payload.u.as_int = static_cast<i64>(qscheme);
        */
    }
    
    pub fn to_qscheme(&self) -> QScheme {
        
        todo!();
        /*
            return static_cast<QScheme>(toInt());
        */
    }
    
    pub fn new(dimname: Dimname) -> Self {
    
        todo!();
        /*


            : IValue(dimname.symbol().toQualString())
        */
    }
    
    pub fn to_dimname(&self) -> Dimname {
        
        todo!();
        /*
            return Dimname::fromSymbol(Symbol::fromQualString(toStringRef()));
        */
    }
    
    pub fn new(g: Generator) -> Self {
    
        todo!();
        /*
        : tag(Tag::Generator),
        : is_intrusive_ptr(g.defined()),

            // Note: the undefined generator is not refcounted, so while it
        // is tagged as a generator, is_intrusive_ptr is set to false.
        // This is not an optional optimization: our incref call
        // *will not* do the right thing when called on an
        // undefined generator.
        payload.u.as_intrusive_ptr = null_to_undefined_tensor(g.unsafeReleaseGeneratorImpl());
        */
    }
    
    pub fn is_generator(&self) -> bool {
        
        todo!();
        /*
            return Tag::Generator == tag;
        */
    }
    
    /// for debugging
    pub fn tag_kind(&self) -> String {
        
        todo!();
        /*
            switch (tag) {
    #define DEFINE_CASE(x) \
      case Tag::x:         \
        return #x;
          TORCH_FORALL_TAGS(DEFINE_CASE)
    #undef DEFINE_CASE
        }
        return "InvalidTag(" + to_string(static_cast<int>(tag)) + ")";
        */
    }

    /*
      | generic v.to<Tensor>() implementations
      |
      | that can be used in special functions like
      | pop/push
      |
      | that use template meta-programming.
      |
      | prefer the directly named methods when you
      | can, since they are simpler to understand
      */

    /**
      | ToOptional: convert a IValue to the
      | Optional obj that accepts both T and
      | None
      |
      */
    pub fn to_optional<T>(&mut self) -> Option<T> {
    
        todo!();
        /*
        
        */
    }
    
    pub fn to_optional<T>(&self) -> Option<T> {
    
        todo!();
        /*
        
        */
    }

    /**
      | @private [doxygen private]
      |
      | this is a shallow comparison of two IValues
      | to test the object identity
      */
    pub fn is_same_identity(&self, rhs: &IValue) -> bool {
        
        todo!();
        /*
        
        */
    }

    /**
      | Computes the "official" string representation
      | of an IValue. This produces a TorchScript
      | expression that can be used to recreate an
      | IValue with the same value (e.g. when we are
      | printing constants in the serializer).
      |
      | Callers can use `customFormatter` to override
      | how `repr()` prints out an IValue. This is
      | useful if you have some other environment
      | where you can look up values, and you want to
      | print a reference to that environment (like
      | the serializer's constant table).
      |
      | repr() is not necessarily defined on all
      | objects!
      */
    pub fn repr(&self, 
        stream:           &mut std::io::BufWriter,
        custom_formatter: fn(_0: &mut std::io::BufWriter, v: &IValue) -> bool) -> &mut std::io::BufWriter {
        
        todo!();
        /*
        
        */
    }

    pub fn is_ptr_type(&self) -> bool {
        
        todo!();
        /*
            return (isTensor() && payload.as_tensor.defined()) || is_intrusive_ptr;
        */
    }

    pub fn internal_to_pointer(&self)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(
            isPtrType(), "Can only call internalToPointer() for pointer types");
        if (isTensor()) {
          return payload.as_tensor.unsafeGetTensorImpl();
        } else {
          return payload.u.as_intrusive_ptr != UndefinedTensorImpl::singleton()
            ? payload.u.as_intrusive_ptr : nullptr;
        }
        */
    }
    
    pub fn ty(&self) -> TypePtr {
        
        todo!();
        /*
        
        */
    }

    /**
      | Chechs if this and rhs has a subvalues in
      | common.
      |
      | [t1,t2] and [t2, t3] returns true.
      */
    pub fn overlaps(&self, rhs: &IValue) -> bool {
        
        todo!();
        /*
        
        */
    }

    /**
      | Inserts all subvalues of this in subValues.
      |
      */
    pub fn get_sub_values(&self, sub_values: &mut HashAliasedIValues)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Apply visitor to every subvalue.
      |
      | TODO: There are several places that recurse
      | over IValue. This is fragile.
      |
      | This visitor should be used to recurse over
      | ivalues.
      */
    pub fn visit(&self, visitor: &fn(_0: &IValue) -> bool)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn deepcopy(&self) -> IValue {
        
        todo!();
        /*
        
        */
    }
    
    pub fn deepcopy(&self, memo: &mut HashAliasedIValueMap) -> IValue {
        
        todo!();
        /*
        
        */
    }
    
    pub fn null_to_undefined_tensor(p: *mut IntrusivePtrTarget) -> *mut IntrusivePtrTarget {
        
        todo!();
        /*
            return p ? p : static_cast<intrusive_ptr_target*>(UndefinedTensorImpl::singleton());
        */
    }
    
    pub fn ptr_equal(
        lhs: &IValue,
        rhs: &IValue) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn move_to_intrusive_ptr<T, NullType = IntrusiveTargetDefaultNullType<T>>(&mut self) -> IntrusivePtr<T,NullType> {
    
        todo!();
        /*
        
        */
    }
    
    pub fn to_intrusive_ptr<T, NullType = IntrusiveTargetDefaultNullType<T>>(&self) -> IntrusivePtr<T,NullType> {
    
        todo!();
        /*
        
        */
    }
    
    pub fn destroy(&mut self)  {
        
        todo!();
        /*
            // We carefully construct this call to both 1) avoid UB by using
        // the "wrong" one of as_tensor and as_intrusive_ptr and 2) enable
        // the compiler to generate the same code for each case. It is
        // surprisingly difficult to get this right.
        if (isTensor() || is_intrusive_ptr) {
          intrusive_ptr_target* p = isTensor() ? payload.as_tensor.unsafeGetTensorImpl() : payload.u.as_intrusive_ptr;
          intrusive_ptr<intrusive_ptr_target, UndefinedTensorImpl>::reclaim(p);
          // No need to make this destructor call!
          // payload.as_tensor.~Tensor();
        }
        */
    }

    #[inline(always)] 
    pub fn move_from(&mut self, rhs: IValue)  {
        
        todo!();
        /*
            if (rhs.isTensor()) {
          new (&payload.as_tensor) Tensor(move(rhs.payload.as_tensor));
          // As far as I can tell, omitting the usual explicit destructor call
          // is not UB in and of itself, and it's a slight perf win. The
          // destructor is a no-op, because the moved-from Tensor is
          // effectively an intrusive_ptr in the null state, so we don't need
          // the behavior for correctness reasons either. Leaving this
          // explanatory comment, including commented-out destructor call, to
          // make this abundantly clear.
          //
          // rhs.payload.as_tensor.~Tensor();
        } else {
          payload.u = rhs.payload.u;
        }
        tag = rhs.tag;
        is_intrusive_ptr = rhs.is_intrusive_ptr;
        rhs.clearToNone();
        */
    }
    
    pub fn clear_to_none(&mut self)  {
        
        todo!();
        /*
            payload.u.as_int = 0;
        tag = Tag::None;
        is_intrusive_ptr = false;
        */
    }
    
    pub fn new(
        p: &Payload,
        t: Tag,
        i: bool) -> Self {
    
        todo!();
        /*
        : tag(t),
        : is_intrusive_ptr(i),

            if (isTensor()) {
          new (&payload.as_tensor) Tensor(p.as_tensor);
        } else {
          payload.u = p.u;
        }
        */
    }
}

///---------------------------------
pub type Payload = IValuePayloadTriviallyCopyablePayload;

pub struct WeakIValue {
    payload:          Payload,
    tag:              IValueTag,
    is_intrusive_ptr: bool,
}

impl Default for WeakIValue {
    
    fn default() -> Self {
        todo!();
        /*
        : tag(IValue::Tag::None),
        : is_intrusive_ptr(false),

        
        */
    }
}

impl Drop for WeakIValue {

    fn drop(&mut self) {
        todo!();
        /*
            if (is_intrusive_ptr && payload.as_intrusive_ptr != UndefinedTensorImpl::singleton()) {
          raw::weak_intrusive_ptr::decref(payload.as_intrusive_ptr);
        }
        */
    }
}

impl WeakIValue {
    
    pub fn new(rhs: &WeakIValue) -> Self {
    
        todo!();
        /*


            : payload(rhs.payload),
            tag(rhs.tag),
            is_intrusive_ptr(rhs.is_intrusive_ptr) 

        if (is_intrusive_ptr && payload.as_intrusive_ptr != UndefinedTensorImpl::singleton()) {
          raw::weak_intrusive_ptr::incref(payload.as_intrusive_ptr);
        }
        */
    }
    
    pub fn new(rhs: &IValue) -> Self {
    
        todo!();
        /*


            : tag(rhs.tag),
            is_intrusive_ptr(rhs.is_intrusive_ptr) 

        if (rhs.isTensor()) {
          payload.as_intrusive_ptr = rhs.unsafeToTensorImpl();
          is_intrusive_ptr = true;
        } else {
          payload = rhs.payload.u;
        }
        if (is_intrusive_ptr) {
          if (payload.as_intrusive_ptr != UndefinedTensorImpl::singleton()) {
            raw::weak_intrusive_ptr::incref(payload.as_intrusive_ptr);
          }
        }
        */
    }
    
    pub fn new(rhs: WeakIValue) -> Self {
    
        todo!();
        /*
        : weak_ivalue(),

            swap(rhs);
        */
    }

    pub fn assign_from(&mut self, rhs: WeakIValue) -> &mut WeakIValue {
        
        todo!();
        /*
            WeakIValue(move(rhs)).swap(*this); // this also sets rhs to None
        return *this;
        */
    }
    
    pub fn assign_from(&mut self, rhs: &WeakIValue) -> &mut WeakIValue {
        
        todo!();
        /*
            WeakIValue(rhs).swap(*this);
        return *this;
        */
    }
    
    pub fn swap(&mut self, rhs: &mut WeakIValue)  {
        
        todo!();
        /*
            swap(payload, rhs.payload);
        swap(is_intrusive_ptr, rhs.is_intrusive_ptr);
        swap(tag, rhs.tag);
        */
    }
    
    pub fn is_same_identity(&self, rhs: &WeakIValue) -> bool {
        
        todo!();
        /*
            return payload.as_int == rhs.payload.as_int && tag == rhs.tag &&
            is_intrusive_ptr == rhs.is_intrusive_ptr;
        */
    }
    
    pub fn lock(&self) -> IValue {
        
        todo!();
        /*
            if (!is_intrusive_ptr) {
          IValue::Payload newPayload;
          newPayload.u = payload;
          return IValue(newPayload, tag, false);
        }
        if (IValue::Tag::Tensor == tag) {
          auto temp = weak_intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(
              static_cast<TensorImpl*>(payload.as_intrusive_ptr));
          intrusive_ptr<TensorImpl, UndefinedTensorImpl> ip(temp.lock());
          temp.release();
          if (!ip) {
            return IValue();
          } else {
            return IValue(Tensor(move(ip)));
          }
        } else {
          auto temp = weak_intrusive_ptr<intrusive_ptr_target>::reclaim(
              payload.as_intrusive_ptr == UndefinedTensorImpl::singleton()
              ? nullptr
              : payload.as_intrusive_ptr);
          IValue::Payload pl;
          pl.u.as_intrusive_ptr = temp.lock().release();
          temp.release();
          if (!pl.u.as_intrusive_ptr) {
            return IValue();
          } else {
            return IValue(pl, tag, true);
          }
        }
        */
    }
    
    pub fn use_count(&self) -> usize {
        
        todo!();
        /*
            if (!is_intrusive_ptr) {
          return 1;
        }
        auto temp = weak_intrusive_ptr<intrusive_ptr_target, UndefinedTensorImpl>::reclaim(
            payload.as_intrusive_ptr);
        usize result = temp.use_count();
        temp.release();
        return result;
        */
    }
    
    pub fn weak_use_count(&self) -> usize {
        
        todo!();
        /*
            if (!is_intrusive_ptr) {
          return 1;
        }
        auto temp = weak_intrusive_ptr<intrusive_ptr_target, UndefinedTensorImpl>::reclaim(
            payload.as_intrusive_ptr);
        usize result = temp.weak_use_count();
        temp.release();
        return result;
        */
    }
    
    pub fn hash(&self) -> usize {
        
        todo!();
        /*
            return payload.as_int;
        */
    }
}

/**
  | An owning pointer to a type. When the type is
  | class type, it requires a pair of shared_ptrs
  | to the class type and its owning CU, so that
  | the class type is guaranteed to stay alive as
  | long as we hold this object.
  |
  */
pub struct StrongTypePtr {
    cu: Arc<TorchJitCompilationUnit>,
    ty: Arc<Type>,
}

impl StrongTypePtr {
    
    pub fn new(
        cu: Arc<TorchJitCompilationUnit>,
        ty: Arc<Type>) -> Self {
    
        todo!();
        /*


        
        */
    }
}

pub fn get_custom_class_type_impl<T>() -> ClassTypePtr {

    todo!();
        /*
            auto& tmap = getCustomClassTypeMap();
      auto res = tmap.find(type_index(typeid(T)));
      if (res == tmap.end()) {
        throw Error("Can't find class id in custom class type map", "");
      }
      return res->second;
        */
}

pub fn get_custom_class_type<T>() -> &ClassTypePtr {

    todo!();
        /*
            // Classes are never unregistered from getCustomClassTypeMap and the
      // hash lookup can be a hot path, so just cache.
      // For the same reason, it's fine If this ends up getting duplicated across
      // DSO boundaries for whatever reason.
      static ClassTypePtr cache = getCustomClassTypeImpl<T>();
      return cache;
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/ivalue.cpp]

pub fn fast_equals_for_container(
        lhs: &IValue,
        rhs: &IValue) -> bool {
    
    todo!();
        /*
            if (lhs.is(rhs)) {
        // Like Python, for containers we consider identity equality to be
        // sufficient but not necessary for value equality
        return true;
      }
      return lhs == rhs;
        */
}

/**
  | This is in ivalue.cpp because we need to access
  | Type::annotation_str, which is declared in
  | jit_type.h
  |
  */
pub fn check_custom_class_type(
        expected_type: *const Type,
        actual_type:   *const Type)  {
    
    todo!();
        /*
            // NB: doing pointer comparison here
      // If in the future there ever arises a need to call operator== on custom class
      // Type's, this needs to be changed!
      TORCH_CHECK(actual_type == expected_type,
                  "Tried to convert an IValue of type ",
                  actual_type->repr_str(),
                  " to custom class type ",
                  expected_type->repr_str());
        */
}

impl ConstantString {
    
    pub fn create(&mut self, str_: String) -> IntrusivePtr<ConstantString> {
        
        todo!();
        /*
            return make_intrusive<ConstantString>(move(str_));
        */
    }
    
    pub fn create(&mut self, str_: StringView) -> IntrusivePtr<ConstantString> {
        
        todo!();
        /*
            return make_intrusive<ConstantString>(string(str_));
        */
    }
    
    pub fn create(&mut self, str_: *const u8) -> IntrusivePtr<ConstantString> {
        
        todo!();
        /*
            return make_intrusive<ConstantString>(string(str_));
        */
    }
}

impl PartialEq<Tuple> for Tuple {
    
    #[inline] fn eq(&self, other: &Tuple) -> bool {
        todo!();
        /*
            return lhs.elements_.size() == rhs.elements_.size() &&
          // see [container equality]
          equal(
                 lhs.elements_.cbegin(),
                 lhs.elements_.cend(),
                 rhs.elements_.cbegin(),
                 _fastEqualsForContainer);
        */
    }
}

impl Tuple {
    
    pub fn ty(&self) -> TupleTypePtr {
        
        todo!();
        /*
            if (!type_) {
        type_ = TupleType::create(
            fmap(elements_, [&](const IValue& v) { return v.type(); }));
      }
      return type_;
        */
    }
}

impl PartialEq<EnumHolder> for EnumHolder {
    
    #[inline] fn eq(&self, other: &EnumHolder) -> bool {
        todo!();
        /*
            return lhs.name() == rhs.name() && *rhs.type() == *lhs.type();
        */
    }
}

impl EnumHolder {
    
    pub fn qualified_class_name(&self) -> String {
        
        todo!();
        /*
            return type_->qualifiedClassName().qualifiedName();
        */
    }
    
    pub fn unqualified_class_name(&self) -> String {
        
        todo!();
        /*
            return type_->qualifiedClassName().name();
        */
    }
}

impl IValue {
    
    pub fn ty(&self) -> TypePtr {
        
        todo!();
        /*
            switch (tag) {
        case Tag::None:
          return NoneType::get();
        case Tag::Tensor:
          return TensorType::create(toTensor());
        case Tag::Storage:
          return StorageType::get();
        case Tag::Double:
          return FloatType::get();
        case Tag::ComplexDouble:
          return ComplexType::get();
        case Tag::Int:
          return IntType::get();
        case Tag::Bool:
          return BoolType::get();
        case Tag::String:
          return StringType::get();
        case Tag::Blob:
          return AnyType::get();
        case Tag::GenericDict: {
          auto d = toGenericDict();
          return DictType::create(d.keyType(), d.valueType());
        }
        case Tag::GenericList:
          return ListType::create(toList().elementType());
        case Tag::Future:
          return FutureType::create(toFuture()->elementType());
        case Tag::RRef:
          return RRefType::create(toRRef()->type());
        case Tag::Device:
          return DeviceObjType::get();
        case Tag::Stream:
          return StreamObjType::get();
        case Tag::Object:
          return toObjectRef().type();
        case Tag::PyObject:
          return PyObjectType::get();
        case Tag::Uninitialized:
          return AnyType::get();
        case Tag::Capsule:
          return CapsuleType::get();
        case Tag::Tuple:
          return toTuple()->type();
        case Tag::Generator:
          return GeneratorType::get();
        case Tag::Quantizer:
          return QuantizerType::get();
        case Tag::Enum:
          return toEnumHolder()->type();
      }
      // switch above is complete but this silences compiler warnings
      TORCH_INTERNAL_ASSERT(false, "unhandled case in IValue::type()");
        */
    }
    
    pub fn visit(&self, visitor: &fn(_0: &IValue) -> bool)  {
        
        todo!();
        /*
            if (visitor(*this)) {
        // Shortcut
        return;
      }
      switch (this->tag) {
        case Tag::Tuple:
        case Tag::GenericList: {
          ArrayRef<IValue> elems;
          if (isTuple()) {
            elems = this->toTuple()->elements();
          } else {
            elems = this->toListRef();
          }
          for (auto& elem : elems) {
            elem.visit(visitor);
          }
          break;
        }
        case Tag::GenericDict:
          for (const auto& pair : this->toGenericDict()) {
            pair.value().visit(visitor);
            pair.key().visit(visitor);
          }
          break;
        case Tag::Object: {
          auto obj_type = type()->expect<ClassType>();
          auto obj_value = toObject();
          auto attributes = obj_type->getAttributes();
          for (const auto& attr: attributes) {
            auto attribute = obj_value->getAttr(attr.getName());
            attribute.visit(visitor);
          }
          break;
        }
        case Tag::PyObject: {
          intrusive_ptr<PyObjectHolder> py_obj = toPyObjectHolder();
          auto match = py_obj->tryToInferType();
          if (match.success()) {
            auto contained_value = py_obj->toIValue(match.type());
            contained_value.visit(visitor);
          }
          break;
        }
        default:
          break;
     }
        */
    }
    
    pub fn get_sub_values(&self, sub_values: &mut HashAliasedIValues)  {
        
        todo!();
        /*
            switch (this->tag) {
        case Tag::Tensor:
          subValues.insert(*this);
          return;
        case Tag::Tuple:
        case Tag::GenericList: {
          subValues.insert(*this);
          ArrayRef<IValue> elems;
          if (isTuple()) {
            elems = this->toTuple()->elements();
          } else {
            elems = this->toListRef();
          }
          for (auto& elem : elems) {
            elem.getSubValues(subValues);
          }
          break;
        }
        case Tag::GenericDict:
          subValues.insert(*this);
          for (const auto& pair : this->toGenericDict()) {
            pair.value().getSubValues(subValues);
            pair.key().getSubValues(subValues);
          }
          break;
        case Tag::Object: {
          // Record Object IValue and its attributes.
          subValues.insert(*this);
          auto obj_type = type()->expect<ClassType>();
          auto obj_value = toObject();
          auto attributes = obj_type->getAttributes();
          for (const auto& attr: attributes) {
            auto attribute = obj_value->getAttr(attr.getName());
            attribute.getSubValues(subValues);
          }
          break;
        }
        case Tag::PyObject: {
          subValues.insert(*this);
          intrusive_ptr<PyObjectHolder> py_obj = toPyObjectHolder();
          auto match = py_obj->tryToInferType();
          TORCH_CHECK_TYPE(match.success(),
                "Cannot infer type of ", py_obj->toStr(), ": ", match.reason());
          auto contained_value = py_obj->toIValue(match.type());
          contained_value.getSubValues(subValues);
          break;
        }
        case Tag::Future:
        case Tag::Device:
        case Tag::Uninitialized:
        case Tag::Capsule:
          TORCH_CHECK_TYPE(
              false, "Cannot inspect value of type ", this->tagKind());
          // Fall through
        default:
          // don't record scalars.
          break;
      }
        */
    }
    
    pub fn overlaps(&self, rhs: &IValue) -> bool {
        
        todo!();
        /*
            HashAliasedIValues rhsSubValues, thisSubValues;
      rhs.getSubValues(rhsSubValues);
      getSubValues(thisSubValues);
      for (auto& sub : thisSubValues) {
        if (rhsSubValues.count(sub)) {
          return true;
        }
      }
      return false;
        */
    }
}

impl PartialEq<IValue> for IValue {
    
    #[inline] fn eq(&self, other: &IValue) -> bool {
        todo!();
        /*
            IValue eq = lhs.equals(rhs);
      if (eq.isBool()) {
        return eq.toBool();
      }
      // The only case we don't return bool is for tensor comparison. In Python,
      // `bool()` is called on the return value of `__eq__` if the return value is
      // not a boolean. Mimic that behavior here.
      TORCH_INTERNAL_ASSERT(eq.isTensor());
      return eq.toTensor().is_nonzero();
        */
    }
}

pub fn is_undefined_tensor(iv: &IValue) -> bool {
    
    todo!();
        /*
            return iv.isTensor() && !iv.toTensor().defined();
        */
}

impl IValue {
    
    pub fn ptr_equal(&mut self, 
        lhs: &IValue,
        rhs: &IValue) -> bool {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(lhs.is_intrusive_ptr);
      TORCH_INTERNAL_ASSERT(rhs.is_intrusive_ptr);
      return lhs.tag == rhs.tag &&
          lhs.payload.u.as_intrusive_ptr == rhs.payload.u.as_intrusive_ptr;
        */
    }
    
    pub fn equals(&self, rhs: &IValue) -> IValue {
        
        todo!();
        /*
            const IValue& lhs = *this;
      switch (lhs.tag) {
        case Tag::None:
          // In Python you're not supposed to do this comparison apparently. Not
          // sure if we should warn here or what
          return rhs.isNone();
        case Tag::Tensor:
          if (!rhs.isTensor()) {
            return false;
          }
          return lhs.toTensor().eq(rhs.toTensor());
        case Tag::Storage:
          return rhs.isStorage() && lhs.toStorage().unsafeGetStorageImpl() == rhs.toStorage().unsafeGetStorageImpl();
        case Tag::Double:
          return rhs.isDouble() && lhs.toDouble() == rhs.toDouble();
        case Tag::ComplexDouble:
          return rhs.isComplexDouble() && lhs.toComplexDouble() == rhs.toComplexDouble();
        case Tag::Int:
          return rhs.isInt() && lhs.toInt() == rhs.toInt();
        case Tag::Bool:
          return rhs.isBool() && lhs.toBool() == rhs.toBool();
        case Tag::String:
          return rhs.isString() && lhs.toStringRef() == rhs.toStringRef();
        case Tag::GenericDict:
          return rhs.isGenericDict() && lhs.toGenericDict() == rhs.toGenericDict();
        case Tag::Tuple:
          return rhs.isTuple() && *lhs.toTuple() == *rhs.toTuple();
        case Tag::Stream:
          return rhs.isStream() && lhs.toStream() == rhs.toStream();
        case Tag::Device:
          return rhs.isDevice() && lhs.toDevice() == rhs.toDevice();
        case Tag::GenericList:
          return rhs.isList() && lhs.toList() == rhs.toList();
        case Tag::Blob:
        case Tag::Future:
        case Tag::RRef:
        case Tag::Object:
        case Tag::PyObject:
        case Tag::Capsule:
        case Tag::Generator:
        case Tag::Quantizer:
          return ptrEqual(lhs, rhs);
        case Tag::Enum:
          return lhs.toEnumHolder()->is(*rhs.toEnumHolder());
        case Tag::Uninitialized:
          // Unitialized ivalues show up in no-ops when the compiler can prove a
          // value will never be used. Just return false on any equality comparison.
          return false;
      }
      // the above switch should be exhaustive
      TORCH_INTERNAL_ASSERT(false, "we should never reach here")
        */
    }
    
    pub fn hash(&mut self, v: &IValue) -> usize {
        
        todo!();
        /*
            switch (v.tag) {
        case Tag::None:
          return 0;
        case Tag::Bool:
          return get_hash(v.payload.u.as_bool);
        case Tag::Double:
          return get_hash(v.payload.u.as_double);
        case Tag::Tensor:
          // Tensor __hash__ is equivalent to `id()`, so take the pointer value of
          // the tensor to emulate it
          return get_hash(v.payload.as_tensor.unsafeGetTensorImpl());
        case Tag::Storage:
          return get_hash(v.payload.u.as_int);
        case Tag::Int:
          return get_hash(v.payload.u.as_int);
        case Tag::String:
          return get_hash(v.toStringRef());
        case Tag::Tuple:
          return get_hash(*v.toTuple());
        case Tag::Device:
          return get_hash(v.toDevice());
        case Tag::GenericDict:
        case Tag::GenericList:
        case Tag::Blob:
        case Tag::Future:
        case Tag::RRef:
        case Tag::Object:
        case Tag::PyObject:
        case Tag::Capsule:
        case Tag::Generator:
        case Tag::Quantizer:
        case Tag::ComplexDouble:
        case Tag::Enum:
        case Tag::Stream:
        case Tag::Uninitialized:
          throw runtime_error(
              "unhashable type: '" + v.type()->repr_str() + "'");
      }
      // the above switch should be exhaustive
      TORCH_INTERNAL_ASSERT(false, "we should never reach here")
        */
    }
    
    pub fn is(&self, rhs: &IValue) -> bool {
        
        todo!();
        /*
            const IValue& lhs = *this;
      // Special handling for undefined tensors:
      // 1. Undefined_tensor is None and vice versa.
      if ((isUndefinedTensor(lhs) && rhs.isNone()) ||
          (lhs.isNone() && isUndefinedTensor(rhs))) {
        return true;
      }
      // 2. Undefined_tensor is Undefined_tensor.
      if (isUndefinedTensor(lhs) && isUndefinedTensor(rhs)) {
        return true;
      }

      if (lhs.isTensor()) {
        // Use the standard way of comparing two tensors for identity
        return rhs.isTensor() && lhs.toTensor().is_same(rhs.toTensor());
      }

      if (lhs.is_intrusive_ptr) {
        return rhs.is_intrusive_ptr && ptrEqual(lhs, rhs);
      }
      return lhs == rhs;
        */
    }
}

pub type IValueFormatter = fn(_0: &mut std::io::BufWriter, _1: &IValue) -> ();

pub fn print_list<T>(
        out:       &mut std::io::BufWriter,
        list:      &T,
        start:     String,
        finish:    String,
        formatter: IValueFormatter) -> &mut std::io::BufWriter {

    todo!();
        /*
            out << start;
      for (const auto i : irange(list.size())) {
        if (i > 0) {
          out << ", ";
        }
        formatter(out, IValue(list[i]));
      }
      out << finish;
      return out;
        */
}

/**
  | Properly disambiguate the type of an
  | empty list
  |
  */
pub fn print_maybe_annotated_list(
        out:       &mut std::io::BufWriter,
        the_list:  &IValue,
        formatter: IValueFormatter) -> &mut std::io::BufWriter {
    
    todo!();
        /*
            auto list_elem_type = the_list.type()->expectRef<ListType>().getElementType();
      if (the_list.toListRef().size() == 0 ||
          !elementTypeCanBeInferredFromMembers(list_elem_type)) {
        out << "annotate(" << the_list.type()->annotation_str() << ", ";
        printList(out, the_list.toListRef(), "[", "]", formatter);
        out << ")";
        return out;
      } else {
        return printList(out, the_list.toListRef(), "[", "]", formatter);
      }
        */
}

pub fn print_dict<Dict>(
        out:       &mut std::io::BufWriter,
        v:         &Dict,
        formatter: IValueFormatter) -> &mut std::io::BufWriter {

    todo!();
        /*
            out << "{";

      bool first = true;
      for (const auto& pair : v) {
        if (!first) {
          out << ", ";
        }

        formatter(out, pair.key());
        out << ": ";
        formatter(out, pair.value());
        first = false;
      }

      out << "}";
      return out;
        */
}

/// Properly disambiguate the type of an empty
/// dict
///
pub fn print_maybe_annotated_dict(
        out:       &mut std::io::BufWriter,
        the_dict:  &IValue,
        formatter: IValueFormatter) -> &mut std::io::BufWriter {
    
    todo!();
        /*
            auto value_type = the_dict.type()->castRaw<DictType>()->getValueType();
      if (the_dict.toGenericDict().size() == 0 ||
          !elementTypeCanBeInferredFromMembers(value_type)) {
        out << "annotate(" << the_dict.type()->annotation_str() << ",";
        printDict(out, the_dict.toGenericDict(), formatter) << ")";
      } else {
        return printDict(out, the_dict.toGenericDict(), formatter);
      }
      return out;
        */
}

pub fn print_complex(
        out: &mut std::io::BufWriter,
        v:   &IValue) -> &mut std::io::BufWriter {
    
    todo!();
        /*
            complex<double> d = v.toComplexDouble();
      IValue real(d.real()), imag(abs(d.imag()));
      auto sign = "";
      if (d.imag() >= 0) {
        sign = "+";
      } else {
        sign = "-";
      }
      return out << real << sign << imag << "j";
        */
}

impl IValue {
    
    pub fn repr(&self, 
        out:              &mut std::io::BufWriter,
        custom_formatter: fn(_0: &mut std::io::BufWriter, v: &IValue) -> bool) -> &mut std::io::BufWriter {
        
        todo!();
        /*
            // First check if the caller has provided a custom formatter. Use that if possible.
      if (customFormatter(out, *this)) {
        return out;
      }

      const IValue& v = *this;
      // continue to use custom formatter in recursion
      auto formatter = [&](ostream& out, const IValue& input) {
        input.repr(out, customFormatter);
      };
      switch (v.tag) {
        case IValue::Tag::None:
          return out << v.toNone();
        case IValue::Tag::Double: {
          double d = v.toDouble();
          int c = fpclassify(d);
          if ((c == FP_NORMAL || c == FP_ZERO ) && abs(d) < 1e10) {
            i64 i = i64(d);
            if (double(i) == d) {
              // -0.0 (signed zero) needs to be parsed as -0.
              if (i == 0 && signbit(d)) {
                return out << "-" << i << ".";
              }
              return out << i << ".";
            }
          }
          auto orig_prec = out.precision();
          return out << setprecision(numeric_limits<double>::max_digits10)
                     << d << setprecision(orig_prec);
        }
        case IValue::Tag::ComplexDouble: {
          return printComplex(out, v);
        }
        case IValue::Tag::Int:
          return out << v.toInt();
        case IValue::Tag::Bool:
          return out << (v.toBool() ? "True" : "False");
        case IValue::Tag::Tuple: {
          const auto& elements = v.toTuple()->elements();
          const auto& finish = elements.size() == 1 ? ",)" : ")";
          return printList(out, elements, "(", finish, formatter);
        }
        case IValue::Tag::String:
          printQuotedString(out, v.toStringRef());
          return out;
        case IValue::Tag::GenericList: {
          return printMaybeAnnotatedList(out, *this, formatter);
        }
        case IValue::Tag::Device: {
          stringstream device_stream;
          device_stream << v.toDevice();
          out << "torch.device(";
          printQuotedString(out, device_stream.str());
          return out << ")";
        }
        case IValue::Tag::GenericDict:
          return printMaybeAnnotatedDict(out, v, formatter);
        case IValue::Tag::Enum: {
          auto enum_holder = v.toEnumHolder();
          return out << enum_holder->qualifiedClassName() << "." <<
              enum_holder->name();
        }
        case IValue::Tag::Object: {
          TORCH_INTERNAL_ASSERT(false, "repr() not defined on: ", v.tagKind(), ". Perhaps you've frozen a module with custom classes?");
        }
        default:
          TORCH_INTERNAL_ASSERT(false, "repr() not defined on: ", v.tagKind());
      }
        */
    }
}

pub fn simple_class_type_arg(
        arg: &Argument,
        ty:  &ClassTypePtr) -> bool {
    
    todo!();
        /*
            return arg.type() == type && !arg.kwarg_only() && !arg.default_value();
        */
}

pub fn check_object_sort_schema(
        t:       &ClassTypePtr,
        why_not: &mut StringStream) -> *mut TorchJitFunction {
    
    todo!();
        /*
            if (auto method = t->findMethod("__lt__")) {
          const auto& lt_schema = method->getSchema();
          const auto& schema_args = lt_schema.arguments();
          bool error =
              (schema_args.size() != 2 ||
               !simpleClassTypeArg(schema_args[0], t) ||
               !simpleClassTypeArg(schema_args[1], t) ||
               lt_schema.returns().size() != 1 ||
               lt_schema.returns()[0].type() != BoolType::get());
          if (!error) {
            return method;
          }
        }

        why_not << "To sort a list of " << t->repr_str()
                << " it must define a "
                << "__lt__ method with two inputs of type "
                << t->repr_str() << " that "
                << "returns a bool";
        return nullptr;
        */
}

pub fn get_less_than_comparator(v: &IValue) -> IValueComparator {
    
    todo!();
        /*
            if (v.isTensor()) {
          return [](const IValue& a, const IValue& b) {
            return a.toTensor().lt(b.toTensor()).is_nonzero();
          };
      }

      if (v.isDouble()) {
          return [](const IValue& a, const IValue& b) {
            return a.toDouble() < b.toDouble();
          };
      }

      if (v.isInt()) {
          return [](const IValue& a, const IValue& b) {
            return a.toInt() < b.toInt();
          };
      }

      if (v.isBool()) {
          return [](const IValue& a, const IValue& b) {
            return a.toBool() == false && b.toBool() == true;
          };
      }

      if (v.isString()) {
          return [](const IValue& a, const IValue& b) {
           return a.toStringRef() < b.toStringRef();
          };
      }

      if (v.isTuple()) {
          const auto& elements = v.toTuple()->elements();
          usize n = elements.size();

          vector<IValueComparator> elements_lts;
          elements_lts.reserve(n);
          for (const auto i : irange(n)) {
            elements_lts.push_back(getLessThanComparator(elements[i]));
          }

          return [elements_lts=move(elements_lts), n](const IValue& a, const IValue& b) {
            const auto& a_elements = a.toTuple()->elements();
            const auto& b_elements = b.toTuple()->elements();

            for (const auto i : irange(n)) {
              if (elements_lts[i](a_elements[i], b_elements[i])) {
                return true;
              }
              if (a_elements[i] == b_elements[i]) {
                continue;
              }
              return false;
            }
            // Reaching here means two tuples are equal.
            return false;
          };
      }

      if (v.isObject()) {
        stringstream why_not;
        TorchJitFunction* lt_func =
            checkObjectSortSchema(v.type()->expect<ClassType>(), why_not);
        if (!lt_func) {
          AT_ERROR(why_not.str());
        }

        return [lt_func](const IValue& a, const IValue& b) {
          // Quick pass to satisfy "strict weak ordering" requirement
          if (a.is(b)) {
            return false;
          }
          TorchJitStack sort_stack;
          sort_stack.push_back(a);
          sort_stack.push_back(b);
          lt_func->run(sort_stack);
          return TorchJitpop(sort_stack).toBool();
        };
      }

      AT_ERROR("IValues of type: ", v.tagKind(), " are not comparable");
        */
}

pub fn get_greater_than_comparator(v: &IValue) -> IValueComparator {
    
    todo!();
        /*
            auto lt = getLessThanComparator(v);
      return [lt = move(lt)](const IValue& a, const IValue& b) {
        return lt(b, a);  // gt(a, b) === lt(b, a)
      };
        */
}

impl fmt::Display for &mut EnumHolder {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            out << v.qualifiedClassName() << "." << v.name();
      return out;
        */
    }
}

impl fmt::Display for IValue {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            auto formatter = [&](ostream& out, const IValue& v) {
        out << v;
      };
      switch(v.tag) {
        case IValue::Tag::None:
          return out << v.toNone();
        case IValue::Tag::Tensor:
          return out << v.toTensor();
        case IValue::Tag::Storage:
          return out << v.toStorage().unsafeGetStorageImpl();
        case IValue::Tag::Double: {
          double d = v.toDouble();
          int c = fpclassify(d);
          if (c == FP_NORMAL || c == FP_ZERO) {
            i64 i = i64(d);
            if (double(i) == d) {
              return out << i << ".";
            }
          }
          auto orig_prec = out.precision();
          return out
            << setprecision(numeric_limits<double>::max_digits10)
            << v.toDouble()
            << setprecision(orig_prec);
        } case IValue::Tag::ComplexDouble: {
          return printComplex(out, v);
        } case IValue::Tag::Int:
          return out << v.toInt();
        case IValue::Tag::Bool:
          return out << (v.toBool() ? "True" : "False");
        case IValue::Tag::Tuple: {
          const auto& elements = v.toTuple()->elements();
          const auto& finish = elements.size() == 1 ? ",)" : ")";
          return printList(out, elements, "(", finish, formatter);
        }
        case IValue::Tag::String:
          return out << v.toStringRef();
        case IValue::Tag::Blob:
          return out << *v.toBlob();
        case IValue::Tag::Capsule:
          return out << "Capsule";
        case IValue::Tag::GenericList:
          return printList(out, v.toList(), "[", "]", formatter);
        case IValue::Tag::RRef:
          return out << "RRef";
        case IValue::Tag::Future:
          return out << "Future";
        case IValue::Tag::Uninitialized:
          return out << "Uninitialized";
        case IValue::Tag::Device:
          return out << v.toDevice();
        case IValue::Tag::Stream:
          return out << v.toStream();
        case IValue::Tag::GenericDict:
          return printDict(out, v.toGenericDict(), formatter);
        case IValue::Tag::PyObject: {
          auto py_obj = v.toPyObject();
          return out << "<PyObject at" << py_obj << ">";
        }
        case IValue::Tag::Generator:
          return out << "Generator";
        case IValue::Tag::Quantizer:
          return out << "Quantizer";
        case IValue::Tag::Object: {
          // TODO we should attempt to call __str__ if the object defines it.
          auto obj = v.toObject();
          // print this out the way python would do it
          return out << "<" << obj->name() << " object at " << obj.get() << ">";
        }
        case IValue::Tag::Enum: {
          auto enum_holder = v.toEnumHolder();
          return out << "Enum<" << enum_holder->unqualifiedClassName() << "." <<
              enum_holder->name() << ">";
        }

      }
      AT_ERROR("Tag not found: ", v.tagKind());
        */
    }
}

impl Object {
    
    pub fn ty(&self) -> Arc<ClassType> {
        
        todo!();
        /*
            return type_.type_->expect<ClassType>();
        */
    }
}

impl IValue {
    
    pub fn dump(&self)  {
        
        todo!();
        /*
            cout << *this << "\n";
        */
    }
    
    pub fn deepcopy(&self) -> IValue {
        
        todo!();
        /*
            IValue::HashAliasedIValueMap memo;
      return deepcopy(memo);
        */
    }
    
    pub fn deepcopy(&self, memo: &mut HashAliasedIValueMap) -> IValue {
        
        todo!();
        /*
            if (memo.count(*this)) {
        return memo.at(*this);
      }
      IValue copy;
      switch(tag) {
        case IValue::Tag::Tensor:
          copy = IValue(toTensor().clone());
          break;
        case IValue::Tag::Tuple: {
          vector<IValue> copied_tuple;
          for (const auto& e : toTuple()->elements()) {
            copied_tuple.push_back(e.deepcopy(memo));
          }
          copy = IValue(Tuple::create(copied_tuple));
        }
          break;
        case IValue::Tag::GenericList: {
          auto list = toList();
          auto copied_list = GenericList(list.elementType());
          for (IValue v : list) {
            copied_list.push_back(v.deepcopy(memo));
          }
          copy = IValue(copied_list);
        }
          break;
        case IValue::Tag::GenericDict: {
          auto dict = toGenericDict();
          auto copied_dict = GenericDict(dict.keyType(), dict.valueType());
          for (const auto& entry : dict) {
            copied_dict.insert(entry.key().deepcopy(memo), entry.value().deepcopy(memo));
          }
          copy = IValue(copied_dict);
        }
          break;
        case IValue::Tag::Object: {
          auto class_type = type()->expect<ClassType>();
          if (class_type->hasMethod("__getstate__") &&
              class_type->hasMethod("__setstate__")) {
            copy = Object::create(
                StrongTypePtr(class_type->compilation_unit(), type()),
                class_type->numAttributes());
            auto state = class_type->getMethod("__getstate__")({*this});
            class_type->getMethod("__setstate__")({copy, move(state)});
          } else {
            copy = IValue(toObject()->deepcopy(memo));
          }
        } break;
        case IValue::Tag::String:
        case IValue::Tag::None:
        case IValue::Tag::Double:
        case IValue::Tag::Int:
        case IValue::Tag::Bool:
        case IValue::Tag::Device:
        case IValue::Tag::Uninitialized: {
          copy = *this;
        } break;
        default: {
          AT_ERROR("Can't deepcopy IValue with tag: ", tagKind());
        }
      }
      // NB: this doesn't work if an object contains itself, and it may
      // come up in the future when we expand the object system, we will
      // have a follow up PR to fix this when it becomes an issue.
      if (!isAliasOf(copy)) {
        memo[*this] = copy;
      }
      return copy;
        */
    }
    
    pub fn report_to_tensor_type_error(&self)  {
        
        todo!();
        /*
            TORCH_CHECK(false, "Expected Tensor but got ", tagKind());
        */
    }
}

impl Object {
    
    pub fn name(&self) -> String {
        
        todo!();
        /*
            return type()->name()->qualifiedName();
        */
    }
    
    pub fn get_attr(&self, name: &String) -> IValue {
        
        todo!();
        /*
            const usize slot = type()->getAttributeSlot(name);
      return getSlot(slot);
        */
    }
    
    pub fn set_attr(&mut self, 
        name: &String,
        v:    IValue)  {
        
        todo!();
        /*
            const usize slot = type()->getAttributeSlot(name);
      setSlot(slot, move(v));
        */
    }
    
    pub fn unsafe_remove_attr(&mut self, name: &String)  {
        
        todo!();
        /*
            const usize slot = type()->getAttributeSlot(name);
      unsafeRemoveSlot(slot);
        */
    }
    
    pub fn resize_object(&mut self, slot: usize)  {
        
        todo!();
        /*
            AT_ASSERT(slot < type()->numAttributes());
      slots_.resize(type()->numAttributes());
        */
    }
    
    pub fn copy_(&self) -> IntrusivePtr<Object> {
        
        todo!();
        /*
            auto object = Object::create(StrongTypePtr(type_.cu_, type()), type()->numAttributes());
      for (const auto i : irange(slots_.size())) {
        object->setSlot(i, slots_[i]);
      }
      return object;
        */
    }
    
    pub fn deepcopy(&self) -> IntrusivePtr<Object> {
        
        todo!();
        /*
            IValue::HashAliasedIValueMap memo;
      return deepcopy(memo);
        */
    }
    
    pub fn deepcopy(&self, memo: &mut HashAliasedIValueMap) -> IntrusivePtr<Object> {
        
        todo!();
        /*
            auto object = Object::create(StrongTypePtr(type_.cu_, type()), type()->numAttributes());
      for (const auto i : irange(slots_.size())) {
        if (slots_[i].type() == CapsuleType::get()) {
          // If we've gotten here, it means that we have *not* copied this
          // class via __getstate__ and __setstate__. That fact and the
          // fact that we have a Capsule attribute mean that this is a
          // custom C++ class without serialization methods defined.
          stringstream err;
          err << "Cannot serialize custom bound C++ class";
          if (auto qualname = type()->name()) {
            err << " " << qualname->qualifiedName();
          }
          err << ". Please define serialization methods via def_pickle() for "
                "this class.";
          AT_ERROR(err.str());
        }
        object->setSlot(i, slots_[i].deepcopy(memo));
      }
      return object;
        */
    }
}

impl StrongTypePtr {
    
    pub fn new(
        cu: Arc<TorchJitCompilationUnit>,
        ty: Arc<Type>) -> Self {
    
        todo!();
        /*


            cu_ = move(cu);
      type_ = type;
      TORCH_INTERNAL_ASSERT(type_);
        */
    }
}

pub fn get_custom_class_type_map() -> &mut FlatHashMap<TypeIndex,ClassTypePtr> {
    
    todo!();
        /*
            static ska::flat_hash_map<type_index, ClassTypePtr> tmap;
        return tmap;
        */
}

pub fn get_class_converter() -> &mut HashMap<String,fn(_0: *mut void) -> *mut PyObject> {
    
    todo!();
        /*
            static unordered_map<string, function<PyObject*(void*)>>
          classConverter;
      return classConverter;
        */
}

impl Future {
    
    /// Needs to be in this .cpp file to access
    /// the full definition of PyObjectHolder
    ///
    pub fn extract_data_ptrs(&mut self, value: &IValue) -> Vec<ReferenceWrapper<DataPtr>> {
        
        todo!();
        /*
            vector<reference_wrapper<const DataPtr>> data_ptrs;
      // getSubValues works poorly on Python objects: it only works if they can be
      // converted to a "regular" IValue type hence, for example, it doesn't support
      // custom subclasses. Thus, instead, we extract the tensors through pickling.
      if (value.isPyObject()) {
        vector<Tensor> tensors =
            value.toPyObjectHolder()->extractTensors();
        data_ptrs.reserve(tensors.size());
        for (const Tensor& tensor : tensors) {
          data_ptrs.emplace_back(tensor.storage().data_ptr());
        }
      } else {
        IValue::HashAliasedIValues sub_values;
        // Prefer getSubValues() over visit() as the latter is a silent no-op for
        // some unsupported types, whereas the former at least fails loudly.
        value.getSubValues(sub_values);
        for (const IValue& sub_value : sub_values) {
          if (sub_value.isTensor()) {
            data_ptrs.emplace_back(sub_value.toTensor().storage().data_ptr());
          }
        }
      }
      return data_ptrs;
        */
    }
}

pub fn collect_all(srcs: List<IntrusivePtr<Future>>) -> IntrusivePtr<Future> {
    
    todo!();
        /*
            struct Ctx {
        explicit Ctx(List<intrusive_ptr<Future>> srcs)
            : remaining(srcs.size()),
              srcFutures(move(srcs)),
              asIvalue(srcFutures),
              // No need to pass devices, because dstFuture won't directly contain
              // the value, it will contain the srcFutures (which have no DataPtrs).
              dstFuture(make_intrusive<Future>(asIvalue.type())) {}
        atomic<i32> remaining{0};
        List<intrusive_ptr<Future>> srcFutures;
        IValue asIvalue;
        intrusive_ptr<Future> dstFuture;
      };

      auto ctx = make_shared<Ctx>(move(srcs));
      if (ctx->srcFutures.size() == 0) {
        ctx->dstFuture->markCompleted(ctx->asIvalue);
      } else {
        auto typePtr = ctx->srcFutures.get(0)->elementType();
        for (const auto i : irange(ctx->srcFutures.size())) {

          function<void(Future&)> func = [ctx](Future& fut) {
            // Set error and exit early if encountered.
            if (fut.hasError()) {
              ctx->dstFuture->setErrorIfNeeded(fut.exception_ptr());
              return;
            }

            if (--ctx->remaining == 0 && !ctx->dstFuture->completed()) {
              // No need to pass DataPtrs, because dstFuture won't directly contain
              // the value, it will contain the srcFutures (which have no DataPtrs).
              ctx->dstFuture->markCompleted(ctx->asIvalue);
            }
          };
          ctx->srcFutures.get(i)->addCallback(func);
        }
      }
      return ctx->dstFuture;
        */
}

pub fn format_set_of_devices(devices: &Vec<Device>) -> String {
    
    todo!();
        /*
            ostringstream oss;
      copy(
          devices.begin(),
          devices.end(),
          ostream_iterator<Device>(oss, ", "));
      return oss.str();
        */
}

pub fn collect_any(srcs: List<IntrusivePtr<Future>>) -> IntrusivePtr<Future> {
    
    todo!();
        /*
            if (srcs.empty()) {
        auto res = make_intrusive<Future>(NoneType::get());
        res->markCompleted();
        return res;
      }
      TypePtr typePtr = srcs.get(0)->elementType();
      const vector<Device>& devices = srcs.get(0)->devices();
      for (const auto i : irange(srcs.size())) {
        if (srcs.get(i)->completed()) {
          return srcs.get(i);
        }
        TORCH_CHECK_TYPE(
            i == 0 || (*typePtr == *srcs.get(i)->elementType()),
            "Expected all futures to have the same type, but found ", *typePtr,
            " in position 0 and ", *srcs.get(i)->elementType(), " in position ", i);
        TORCH_CHECK_VALUE(
            i == 0 || (devices == srcs.get(i)->devices()),
            "Expected all futures to have the same devices, but found ",
            formatSetOfDevices(devices), " in position 0 and ",
            formatSetOfDevices(srcs.get(i)->devices()), " in position ", i);
      }
      struct Ctx {
        explicit Ctx(
            List<intrusive_ptr<Future>> srcs,
            TypePtr typePtr,
            vector<Device> devices)
            : srcFutures(move(srcs)),
              dstFuture(make_intrusive<Future>(typePtr, move(devices))) {}
        atomic<bool> done{false};
        List<intrusive_ptr<Future>> srcFutures;
        intrusive_ptr<Future> dstFuture;
      };
      auto ctx = make_shared<Ctx>(move(srcs), typePtr, devices);
      function<void(Future&)> func = [ctx](Future& src) {
        if (!ctx->done.exchange(true)) {
          intrusive_ptr<Future> dst = ctx->dstFuture;
          ctx->dstFuture.reset(); // Once future is satisfied, remove refs.
          ctx->srcFutures =
              List<intrusive_ptr<Future>>(ctx->srcFutures.elementType());
          if (src.hasError()) {
            dst->setError(src.exception_ptr());
          } else {
            dst->markCompleted(src.constValue(), src.dataPtrs());
          }
        }
      };
      for (const auto i : irange(ctx->srcFutures.size())) {
        ctx->srcFutures.get(i)->addCallback(func);
      }
      return ctx->dstFuture;
        */
}
