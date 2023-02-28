/*!
  | TypeIdentifier is a small type containing
  | an id.
  | 
  | Types must be registered using CAFFE_KNOWN_TYPE()
  | for them to have a type id.
  | 
  | If a type is registered, you can also
  | create an object containing meta data
  | like constructor, destructor, stringified
  | name, ... about the type by calling
  | 
  | TypeMeta::Make<T>. This returns a
  | TypeMeta() object, which is basically
  | just a pointer to the type information,
  | so it's cheap to pass around.
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/typeid.h]

/**
  | A type id is a unique id for a given C++
  | type.
  | 
  | You need to register your types using
  | CAFFE_KNOWN_TYPE(MyType) to be able
  | to use TypeIdentifier with custom types.
  | This is for example used to store the
  | dtype of tensors.
  |
  */
#[derive(PartialEq,Eq)]
pub struct TypeIdentifier {
    id_wrapper: TypeIndex,
}

impl TypeIdentifier {

    /**
      | Returns the unique id for the given type
      | T. The id is unique for the type T in the
      | sense that for any two different types,
      | their ids are different; for the same
      | type T, the id remains the same over different
      | calls of the function.
      | 
      | However, this is not guaranteed over
      | different runs, as the id is generated
      | during run-time. Do NOT serialize the
      | id for storage.
      |
      */
    #[C10_HOST_CONSTEXPR]
    pub fn get<T>() -> TypeIdentifier {
    
        todo!();
        /*
            return TypeIdentifier(util::get_type_index<T>());
        */
    }
    
    pub fn uninitialized() -> TypeIdentifier {
        
        todo!();
        /*
            return TypeIdentifier(util::type_index{0});
        */
    }
    
    pub fn new(id: TypeIndex) -> Self {
    
        todo!();
        /*
        : id_wrapper(id),

        
        */
    }
}

impl Ord for TypeIdentifier {
    
    /**
      | Allow usage in map / set
      |
      | TODO Disallow this and rather use
      | unordered_map/set everywhere
      */
    fn cmp(&self, other: &TypeIdentifier) -> Ordering {
        todo!();
        /*
            return lhs.underlyingId() < rhs.underlyingId();
        */
    }
}

impl PartialOrd<TypeIdentifier> for TypeIdentifier {

    fn partial_cmp(&self, other: &TypeIdentifier) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for TypeIdentifier {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            return stream << typeId.underlyingId();
        */
    }
}

pub type DataType = TypeIdentifier;

impl Hash for TypeIdentifier {

    fn hash<H>(&self, state: &mut H) where H: Hasher {
        todo!("hash by ID");
    }
}

pub mod type_meta_data {

    use super::*;

    pub type New             = fn() -> *mut c_void;
    pub type PlacementNew    = fn(*mut c_void, usize) -> ();
    pub type Copy            = fn(*const c_void, *mut c_void, usize) -> ();
    pub type PlacementDelete = fn(*mut c_void, usize) -> ();
    pub type Delete          = fn(*mut c_void) -> ();
}

/**
  | This struct holds the actual type
  | information. There will be one allocated per
  | type. TypeMeta objects will then point to the
  | struct instance for the type they're configured
  | for.
  */
pub struct TypeMetaData {
    itemsize:         usize,
    new:              *mut type_meta_data::New,
    placement_new:    *mut type_meta_data::PlacementNew,
    copy_:            *mut type_meta_data::Copy,
    placement_delete: *mut type_meta_data::PlacementDelete,
    delete:           *mut type_meta_data::Delete,
    id:               TypeIdentifier,
    name:             String,
}

impl Default for TypeMetaData {
    
    fn default() -> Self {
        todo!();
        /*

            : itemsize_(0),
            new_(nullptr),
            placementNew_(nullptr),
            copy_(nullptr),
            placementDelete_(nullptr),
            delete_(nullptr),
            id_(TypeIdentifier::uninitialized()),
            name_("nullptr (uninitialized)")
        */
    }
}

impl TypeMetaData {

    pub fn new(
        itemsize:         usize,
        new_fn:           *mut type_meta_data::New,
        placement_new:    *mut type_meta_data::PlacementNew,
        copy_:            *mut type_meta_data::Copy,
        placement_delete: *mut type_meta_data::PlacementDelete,
        delete_fn:        *mut type_meta_data::Delete,
        id:               TypeIdentifier,
        name:             &str) -> Self {
    
        todo!();
        /*

            : itemsize_(itemsize),
            new_(newFn),
            placementNew_(placementNew),
            copy_(copy),
            placementDelete_(placementDelete),
            delete_(deleteFn),
            id_(id),
            name_(name)
        */
    }
}

/**
  | Placement new function for the type.
  |
  */
#[inline] pub fn placement_new<T>(
        ptr: *mut c_void,
        n:   usize)  {

    todo!();
        /*
            T* typed_ptr = static_cast<T*>(ptr);
      for (size_t i = 0; i < n; ++i) {
        new (typed_ptr + i) T;
      }
        */
}

#[inline] pub fn placement_new_not_default<T>(
        ptr: *mut c_void,
        n:   usize)  {

    todo!();
        /*
            _ThrowRuntimeTypeLogicError(
          "Type " + string(util::get_fully_qualified_type_name<T>()) +
          " is not default-constructible.");
        */
}

lazy_static!{
    /*
    template <
        typename T,
        enable_if_t<is_default_constructible<T>::value>* = nullptr>
    inline constexpr TypeMetaData::PlacementNew* _PickPlacementNew() {
      return (is_fundamental<T>::value || is_pointer<T>::value)
          ? nullptr
          : &_PlacementNew<T>;
    }

    template <
        typename T,
        enable_if_t<!is_default_constructible<T>::value>* = nullptr>
    inline constexpr TypeMetaData::PlacementNew* _PickPlacementNew() {
      static_assert(
          !is_fundamental<T>::value && !is_pointer<T>::value,
          "this should have picked the other SFINAE case");
      return &_PlacementNewNotDefault<T>;
    }
    */
}

#[inline] pub fn new<T>()  {

    todo!();
        /*
            return new T;
        */
}

#[inline] pub fn new_not_default<T>()  {

    todo!();
        /*
            _ThrowRuntimeTypeLogicError(
          "Type " + string(util::get_fully_qualified_type_name<T>()) +
          " is not default-constructible.");
        */
}

lazy_static!{
    /*
    template <
        typename T,
        enable_if_t<is_default_constructible<T>::value>* = nullptr>
    inline constexpr TypeMetaData::New* _PickNew() {
      return &_New<T>;
    }

    template <
        typename T,
        enable_if_t<!is_default_constructible<T>::value>* = nullptr>
    inline constexpr TypeMetaData::New* _PickNew() {
      return &_NewNotDefault<T>;
    }
    */
}

/**
  | Typed copy function for classes.
  |
  */
#[inline] pub fn copy_<T>(
        src: *const c_void,
        dst: *mut c_void,
        n:   usize)  {

    todo!();
        /*
            const T* typed_src = static_cast<const T*>(src);
      T* typed_dst = static_cast<T*>(dst);
      for (size_t i = 0; i < n; ++i) {
        typed_dst[i] = typed_src[i];
      }
        */
}

/**
  | A placeholder function for types that
  | do not allow assignment.
  |
  */
#[inline] pub fn copy_not_allowed<T>(
        src: *const c_void,
        dst: *mut c_void,
        n:   usize)  {

    todo!();
        /*
            _ThrowRuntimeTypeLogicError(
          "Type " + string(util::get_fully_qualified_type_name<T>()) +
          " does not allow assignment.");
        */
}

lazy_static!{
    /*
    template <
        typename T,
        enable_if_t<is_copy_assignable<T>::value>* = nullptr>
    inline constexpr TypeMetaData::Copy* _PickCopy() {
      return (is_fundamental<T>::value || is_pointer<T>::value)
          ? nullptr
          : &_Copy<T>;
    }

    template <
        typename T,
        enable_if_t<!is_copy_assignable<T>::value>* = nullptr>
    inline constexpr TypeMetaData::Copy* _PickCopy() {
      static_assert(
          !is_fundamental<T>::value && !is_pointer<T>::value,
          "this should have picked the other SFINAE case");
      return &_CopyNotAllowed<T>;
    }
    */
}

/**
  | Destructor for non-fundamental types.
  |
  */
#[inline] pub fn placement_delete<T>(
        ptr: *mut c_void,
        n:   usize)  {

    todo!();
        /*
            T* typed_ptr = static_cast<T*>(ptr);
      for (size_t i = 0; i < n; ++i) {
        typed_ptr[i].~T();
      }
        */
}

lazy_static!{
    /*
    template <typename T>
    inline constexpr TypeMetaData::PlacementDelete* _PickPlacementDelete() {
      return (is_fundamental<T>::value || is_pointer<T>::value)
          ? nullptr
          : &_PlacementDelete<T>;
    }
    */
}

#[inline] pub fn delete<T>(ptr: *mut c_void)  {

    todo!();
        /*
            T* typed_ptr = static_cast<T*>(ptr);
      delete typed_ptr;
        */
}

lazy_static!{
    /*
    template <class T>
    inline constexpr TypeMetaData::Delete* _PickDelete()  {
      return &_Delete<T>;
    }
    */
}

pub struct Uninitialized {}

/*
  | note: this is outside TypeMeta bc gcc seems to
  | have trouble with scalarTypeItemSizes as
  | a constexpr static member used by a public
  | inline instance method
  |
  | item sizes for TypeMeta::itemsize() fast path
  */
lazy_static!{
    /*
    static constexpr uint8_t scalarTypeItemSizes[NumScalarTypes] = {
    #define SCALAR_TYPE_SIZE(T, name) sizeof(T),
        AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SCALAR_TYPE_SIZE)
    #undef SCALAR_TYPE_SIZE
            0, // Undefined
    };
    */
}

/**
  | TypeMeta is a thin class that allows
  | us to store the type of a container such
  | as a blob, or the data type of a tensor,
  | with a unique run-time id. It also stores
  | some additional data such as the item
  | size and the name of the type for run-time
  | inspection.
  |
  */
pub struct TypeMeta {

    /**
     | TypeMeta just wraps this index
     |
     */
    index: u16,
}

/**
  | hard limit number of registered types
  |
  | note: constexpr provokes Windows
  | compilation error "member may not be
  | initialized" static constexpr size_t
  | MaxTypeIndex = 32;
  |
  | The reason for this to be 32 and not
  | UINT8_MAX is that the array initialization
  | takes space which is proportional to the
  | size of the array.
  |
  | The compiler seems to add code (or data
  | padding) to initialize the array with empty
  | elements.
  |
  | In practice, this array doesn't hold more
  | than 18 elements (on mobile), so 32 should
  | be plenty for now. Please see
  | https://github.com/pytorch/pytorch/pull/51881
  | for details.
  |
  */
#[cfg(C10_MOBILE)]
pub const MaxTypeIndex: usize = 32;

#[cfg(not(C10_MOBILE))]
pub const MaxTypeIndex: usize = u8::MAX as usize;

lazy_static!{
    /*
    static atomic<uint16_t> nextTypeIndex;
    */
}

impl Default for TypeMeta {
    
    /**
      | Create a dummy TypeMeta object. To create
      | a TypeMeta object for a specific type,
      | use TypeMeta::Make<T>().
      |
      */
    fn default() -> Self {
        todo!();
        /*

        
        */
    }
}

pub mod type_meta {

    use super::*;

    pub type New             = type_meta_data::New;
    pub type PlacementNew    = type_meta_data::PlacementNew;
    pub type Copy            = type_meta_data::Copy;
    pub type PlacementDelete = type_meta_data::PlacementDelete;
    pub type Delete          = type_meta_data::Delete;
}

impl TypeMeta {

    #[inline] pub fn assign_from(&mut self, scalar_type: ScalarType) -> &mut TypeMeta {
        
        todo!();
        /*
            index_ = static_cast<uint16_t>(scalar_type);
        return *this;
        */
    }

    /**
      | TypeMeta can only be created by Make,
      | making sure that we do not create incorrectly
      | mixed up TypeMeta objects.
      |
      */
    pub fn new(index: u16) -> Self {
    
        todo!();
        /*
        : index(index),

        
        */
    }

    /**
      | Returns the type id.
      |
      */
    pub fn id(&self) -> TypeIdentifier {
        
        todo!();
        /*
            return data().id_;
        */
    }

    /**
      | true if we represent some ScalarType
      | type
      |
      */
    #[inline] pub fn is_scalar_type(&self) -> bool {
        
        todo!();
        /*
            return index_ < NumScalarTypes;
        */
    }

    /**
      | true if we represent ScalarType scalar_type
      |
      */
    #[inline] pub fn is_scalar_type_with_arg(&self, scalar_type: ScalarType) -> bool {
        
        todo!();
        /*
            return index_ == static_cast<uint16_t>(scalar_type);
        */
    }

    /**
      | Returns the size of the item.
      |
      */
    #[inline] pub fn itemsize(&self) -> usize {
        
        todo!();
        /*
            if (C10_LIKELY(isScalarType())) {
          return scalarTypeItemSizes[index_];
        }
        return data().itemsize_;
        */
    }

    /**
      | Returns the new function pointer for
      | individual items.
      |
      */
    pub fn new_fn(&self) -> *mut type_meta::New {
        
        todo!();
        /*
            return data().new_;
        */
    }

    /**
      | Returns the placement new function
      | pointer for individual items.
      |
      */
    pub fn placement_new(&self) -> *mut type_meta::PlacementNew {
        
        todo!();
        /*
            return data().placementNew_;
        */
    }

    /**
      | Returns the typed copy function pointer
      | for individual iterms.
      |
      */
    pub fn copy_(&self) -> *mut type_meta::Copy {
        
        todo!();
        /*
            return data().copy_;
        */
    }

    /**
      | Returns the destructor function pointer
      | for individual items.
      |
      */
    pub fn placement_delete(&self) -> *mut type_meta::PlacementDelete {
        
        todo!();
        /*
            return data().placementDelete_;
        */
    }
    
    pub fn delete_fn(&self) -> *mut type_meta::Delete {
        
        todo!();
        /*
            return data().delete_;
        */
    }

    /**
      | Returns a printable name for the type.
      |
      */
    pub fn name(&self) -> &str {
        
        todo!();
        /*
            return data().name_;
        */
    }
    
    pub fn match_<T>(&self) -> bool {
    
        todo!();
        /*
            return (*this == Make<T>());
        */
    }

    /**
      | Below are static functions that can
      | be called by passing a specific type.
      |
      */
    pub const fn id_t<T>() -> TypeIdentifier {
    
        todo!();
        /*
            return TypeIdentifier::Get<T>();
        */
    }
    
    pub fn type_name<'a, T>() -> &'a str {
    
        todo!();
        /*
            return util::get_fully_qualified_type_name<T>();
        */
    }
    
    pub fn item_size<T>() -> usize {
    
        todo!();
        /*
            return sizeof(T);
        */
    }

    /**
      | Returns a TypeMeta object that corresponds
      | to the typename T.
      |
      */
    pub fn make<T>() -> TypeMeta {
    
        todo!();
        /*
            // The instance pointed to is declared here, but defined in a .cpp file.
        // We need to silence the compiler warning about using an undefined
        // variable template. '-Wpragmas' and '-Wunknown-warning-option' has to be
        // disabled for compilers that don't know '-Wundefined-var-template' and
        // would error at our attempt to disable it.
    #ifndef _MSC_VER
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wpragmas"
    #pragma GCC diagnostic ignored "-Wunknown-warning-option"
    #pragma GCC diagnostic ignored "-Wundefined-var-template"
    #endif
        return TypeMeta(_typeMetaData<T>());
    #ifndef _MSC_VER
    #pragma GCC diagnostic pop
    #endif
        */
    }

    /**
      | convert ScalarType enum values to TypeMeta
      | handles
      |
      */
    #[inline] pub fn from_scalar_type(scalar_type: ScalarType) -> TypeMeta {
        
        todo!();
        /*
            const auto index = static_cast<uint16_t>(scalar_type);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            index < NumScalarTypes,
            "Unrecognized Scalartype ",
            scalar_type,
            " (please report this error)");
        return TypeMeta(index);
        */
    }

    /**
      | convert TypeMeta handles to ScalarType
      | enum values
      |
      */
    #[inline] pub fn to_scalar_type(&mut self) -> ScalarType {
        
        todo!();
        /*
            if (C10_LIKELY(isScalarType())) {
          return static_cast<ScalarType>(index_);
        }
        error_unsupported_typemeta(*this);
        */
    }

    pub fn add_type_meta_data<T>() -> u16 {
    
        todo!();
        /*
            const uint16_t index = nextTypeIndex++;
        TORCH_CHECK(
            index <= MaxTypeIndex,
            "Maximum number of CAFFE_KNOWN_TYPE declarations has been exceeded. ",
            "Please report this issue.");
        typeMetaDatas()[index] = TypeMetaData{
            sizeof(T),
            _PickNew<T>(),
            _PickPlacementNew<T>(),
            _PickCopy<T>(),
            _PickPlacementDelete<T>(),
            _PickDelete<T>(),
            TypeIdentifier::Get<T>(),
            util::get_fully_qualified_type_name<T>()};
        return index;
        */
    }

    /**
      | specializations return indexes into
      | typeMetaDataInstances()
      |
      */
    pub fn type_meta_data<T>() -> u16 {
    
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn data(&self) -> &TypeMetaData {
        
        todo!();
        /*
            return typeMetaDatas()[index_];
        */
    }
}

/**
  | specializations of TypeMeta::_typeMetaData
  | for ScalarType types
  |
  */
macro_rules! define_scalar_metadata_instance {
    ($T:ident, $name:ident) => {
        /*
        
          template <>                                                
          constexpr uint16_t TypeMeta::_typeMetaData<T>()  { 
            return static_cast<uint16_t>(ScalarType::name);          
          }
        */
    }
}

at_forall_scalar_types_with_complex_and_qints!{
    DEFINE_SCALAR_METADATA_INSTANCE
}

lazy_static!{
    /*
    constexpr uint16_t TypeMeta::_typeMetaData<_Uninitialized>()  {
      return static_cast<uint16_t>(ScalarType::Undefined);
    }

    inline TypeMeta::TypeMeta() 
    {
        : index_(_typeMetaData<_Uninitialized>()) 
    }
    */
}

lazy_static!{
    /*
    inline bool operator==(const TypeMeta lhs, const TypeMeta rhs)  {
      return (lhs.index_ == rhs.index_);
    }
    */
}

impl fmt::Display for TypeMeta {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            return stream << typeMeta.name();
        */
    }
}

/**
  | Register unique id for a type so it can
  | be used in TypeMeta context, e.g. be
  | used as a type for Blob or for Tensor elements.
  | 
  | CAFFE_KNOWN_TYPE does explicit instantiation
  | of TypeIdentifier::Get<T> template
  | function and thus needs to be put in a
  | single translation unit (.cpp file)
  | for a given type T. Other translation
  | units that use type T as a type of the Blob
  | or element type of Tensor need
  | to depend on the translation unit that
  | contains CAFFE_KNOWN_TYPE declaration
  | via regular linkage dependencies.
  | 
  | -----------
  | @note
  | 
  | the macro needs to be invoked in ::caffe2
  | namespace
  |
  */
lazy_static!{
    /*
    // Implementation note: in MSVC, we will need to prepend the 
    // keyword in order to get things compiled properly. in Linux, gcc seems to
    // create attribute ignored error for explicit template instantiations, see
    //   http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0537r0.html
    //   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51930
    // and as a result, we define these two macros slightly differently.
    #if defined(_MSC_VER) || defined(__clang__)
    #define EXPORT_IF_NOT_GCC C10_EXPORT
    #else
    #define EXPORT_IF_NOT_GCC
    #endif
    */
}

macro_rules! caffe_known_type {
    ($T:ty) => {
        /*
        
          template <>                                                        
          EXPORT_IF_NOT_GCC uint16_t TypeMeta::_typeMetaData<T>()  { 
            static const uint16_t index = addTypeMetaData<T>();              
            return index;                                                    
          }
        */
    }
}

//-------------------------------------------[.cpp/pytorch/c10/util/typeid.cpp]

/**
  | Mechanism for throwing errors which can't be
  | prevented at compile time due to type
  | erasure. E.g. somebody calling TypeMeta::copy()
  | for non-copyable type. Right now just throws
  | exception but is implemented in .cpp to manage
  | dependencies
  */
#[noreturn]
pub fn throw_runtime_type_logic_error(msg: &String)  {
    
    todo!();
        /*
            // In earlier versions it used to be abort() but it's a bit hard-core
      // for a library
      TORCH_CHECK(false, msg);
        */
}

impl TypeMeta {
    
    #[noreturn]
    pub fn error_unsupported_typemeta(&mut self, dtype: TypeMeta)  {
        
        todo!();
        /*
            TORCH_CHECK(
          false,
          "Unsupported TypeMeta in ATen: ",
          dtype,
          " (please report this error)");
        */
    }

    // fixed length array of TypeMetaData instances
    pub fn type_meta_datas(&mut self) -> *mut TypeMetaData {
        
        todo!();
        /*
            // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
      static TypeMetaData instances[MaxTypeIndex + 1] = {
    #define SCALAR_TYPE_META(T, name)        \
      /* ScalarType::name */                 \
      TypeMetaData(                  \
          sizeof(T),                         \
          _PickNew<T>(),             \
          _PickPlacementNew<T>(),    \
          _PickCopy<T>(),            \
          _PickPlacementDelete<T>(), \
          _PickDelete<T>(),          \
          TypeIdentifier::Get<T>(),          \
          util::get_fully_qualified_type_name<T>()),
          AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SCALAR_TYPE_META)
    #undef SCALAR_TYPE_META
          // The remainder of the array is padded with TypeMetaData blanks.
          // The first of these is the entry for ScalarType::Undefined.
          // The rest are consumed by CAFFE_KNOWN_TYPE entries.
      };
      return instances;
        */
    }
}

caffe_known_type!{ String }
caffe_known_type!{ u16 }
caffe_known_type!{ char }
caffe_known_type!{ Box<Mutex> }
caffe_known_type!{ Box<AtomicBool> }
caffe_known_type!{ Vec<i32> }
caffe_known_type!{ Vec<i64> }
caffe_known_type!{ Vec<u64> }
caffe_known_type!{ *mut bool }
caffe_known_type!{ *mut char }
caffe_known_type!{ *mut i32 }

/**
  | For some of the compilers, long is defined
  | separately from int32_t and int64_t. As
  | a result we will need to actually define them
  | separately.
  |
  | It is recommended that one does NOT use long
  | - use int32_t and int64_t explicitly. Explicit
  | long type annotation may go away in the future.
  |
  | details: This hack works by defining
  | a _guard_long_unique type, which is long iff
  | the compiler has a separate long type and is
  | a dummy type otherwise.
  |
  | we then allocate a type id to that
  | _guard_long_unique.
  |
  | If the compiler has a separate long type, this
  | allocates a type id for long. Otherwise, it
  | allocates a type id for the dummy type, which
  | doesn't matter.
  */
lazy_static!{
    /*
    template <class T>
    class _guard_long_unique_dummy final {};
    template <class T>
    using _guard_long_unique = conditional_t<
        is_same<long, int32_t>::value || is_same<long, int64_t>::value,
        _guard_long_unique_dummy<T>,
        T>;
    */
}

caffe_known_type!{ _guard_long_unique<i64> }
caffe_known_type!{ _guard_long_unique<Vec<i64>> }
caffe_known_type!{ *mut f32 }
caffe_known_type!{ *mut f16 }
