crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/TypeIndex.h]

/// TODO Make it work for more compilers
lazy_static!{
    /*
    #if defined(__clang__)

    // except for NVCC
    #if defined(__CUDACC__)
    #define C10_TYPENAME_SUPPORTS_CONSTEXPR 0
    #define C10_TYPENAME_CONSTEXPR
    #else
    #define C10_TYPENAME_SUPPORTS_CONSTEXPR 1
    #define C10_TYPENAME_CONSTEXPR constexpr
    #endif

    // Windows works
    #elif defined(_MSC_VER)

    // except for NVCC
    #if defined(__CUDACC__)
    #define C10_TYPENAME_SUPPORTS_CONSTEXPR 0
    #define C10_TYPENAME_CONSTEXPR
    #else
    #define C10_TYPENAME_SUPPORTS_CONSTEXPR 1
    #define C10_TYPENAME_CONSTEXPR constexpr
    #endif

    // GCC works
    #elif defined(__GNUC__)

    // except when gcc < 9
    #if (__GNUC__ < 9) || defined(__CUDACC__)
    #define C10_TYPENAME_SUPPORTS_CONSTEXPR 0
    #define C10_TYPENAME_CONSTEXPR
    #else
    #define C10_TYPENAME_SUPPORTS_CONSTEXPR 1
    #define C10_TYPENAME_CONSTEXPR constexpr
    #endif

    // some other compiler we don't know about
    #else
    #define C10_TYPENAME_SUPPORTS_CONSTEXPR 1
    #define C10_TYPENAME_CONSTEXPR constexpr
    #endif
    */
}

#[derive(PartialEq,Eq)]
pub struct TypeIndex {
    id_wrapper: u64,
}

impl TypeIndex {
    
    pub fn new(checksum: u64) -> Self {
    
        todo!();
        /*
        : id_wrapper(checksum),

        
        */
    }
}

impl Ord for TypeIndex {
    
    /**
      | Allow usage in map / set
      | 
      | TODO Disallow this and rather use unordered_map/set
      | everywhere
      |
      */
    #[inline] fn cmp(&self, other: &TypeIndex) -> Ordering {
        todo!();
        /*
            return lhs.underlyingId() < rhs.underlyingId();
        */
    }
}

impl PartialOrd<TypeIndex> for TypeIndex {
    
    #[inline] fn partial_cmp(&self, other: &TypeIndex) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for TypeIndex {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            return stream << typeId.underlyingId();
        */
    }
}

#[inline] pub fn extract<'a>(
    prefix: &str,
    suffix: &str,
    str_:   &str) -> &'a str {

    todo!();
        /*
            #if !defined(__CUDA_ARCH__) // Cuda doesn't like logic_error in device code
      return (!str.starts_with(prefix) || !str.ends_with(suffix))
          ? (throw logic_error("Invalid pattern"), string_view())
          : str.substr(prefix.size(), str.size() - prefix.size() - suffix.size());
    #else
      return str.substr(prefix.size(), str.size() - prefix.size() - suffix.size());
    #endif
        */
}

#[inline] pub const fn fully_qualified_type_name_impl<'a, T>() -> &'a str {

    todo!();
        /*
            #if defined(_MSC_VER) && !defined(__clang__)
    #if defined(__NVCC__)
      return extract(
          "basic_string_view<char> util::fully_qualified_type_name_impl<",
          ">()",
          __FUNCSIG__);
    #else
      return extract(
          "class basic_string_view<char> __cdecl util::fully_qualified_type_name_impl<",
          ">(void)",
          __FUNCSIG__);
    #endif
    #elif defined(__clang__)
      return extract(
          "string_view util::fully_qualified_type_name_impl() [T = ",
          "]",
          __PRETTY_FUNCTION__);
    #elif defined(__GNUC__)
      return extract(
    #if C10_TYPENAME_SUPPORTS_CONSTEXPR
          "constexpr string_view util::fully_qualified_type_name_impl() [with T = ",
    #else
          "string_view util::fully_qualified_type_name_impl() [with T = ",
    #endif
          "; string_view = basic_string_view<char>]",
          __PRETTY_FUNCTION__);
    #endif
        */
}

#[cfg(not(__CUDA_ARCH__))]
#[inline] pub fn type_index_impl<T>() -> u64 {

    todo!();
        /*
            // Idea: __PRETTY_FUNCTION__ (or __FUNCSIG__ on msvc) contains a qualified name
    // of this function, including its template parameter, i.e. including the
    // type we want an id for. We use this name and run crc64 on it to get a type
    // id.
    #if defined(_MSC_VER)
      return crc64(__FUNCSIG__, sizeof(__FUNCSIG__)).checksum();
    #elif defined(__clang__)
      return crc64(__PRETTY_FUNCTION__, sizeof(__PRETTY_FUNCTION__)).checksum();
    #elif defined(__GNUC__)
      return crc64(__PRETTY_FUNCTION__, sizeof(__PRETTY_FUNCTION__)).checksum();
    #endif
        */
}

#[inline] pub fn get_type_index<T>() -> TypeIndex {

    todo!();
        /*
            #if !defined(__CUDA_ARCH__)
      // To enforce that this is really computed at compile time, we pass the
      // type index through integral_constant.
      return type_index{integral_constant<
          uint64_t,
          type_index_impl<remove_cv_t<decay_t<T>>>()>::value};
    #else
      // There's nothing in theory preventing us from running this on device code
      // except for nvcc throwing a compiler error if we enable it.
      return (abort(), type_index(0));
    #endif
        */
}

#[inline] pub const fn get_fully_qualified_type_name<'a, T>() -> &'a str {

    todo!();
        /*
            #if C10_TYPENAME_SUPPORTS_CONSTEXPR
      constexpr
    #else
      static
    #endif
          string_view name = fully_qualified_type_name_impl<T>();
      return name;
        */
}

impl Hash for TypeIndex {

    fn hash<H>(&self, state: &mut H) where H: Hasher {
        todo!("hash by ID");
    }
}
