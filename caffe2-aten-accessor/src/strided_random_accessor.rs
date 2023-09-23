/*!
  | (Const)StridedRandomAccessor is a (const)
  | random access iterator defined over a strided
  | array.
  |
  | The traits below are to introduce __restrict__
  | modifier on different platforms.
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/StridedRandomAccessor.h]

type Todo                           = i32;
pub type ConstStridedRandomAccessor = Todo;
pub type StridedRandomAccessor      = Todo;

lazy_static!{
    /*
    template <
      typename T,
      typename Index = i64,
      typename PtrTraits = DefaultPtrTraits
    >
    class ConstStridedRandomAccessor {

      using difference_type = Index;
      using value_type = const T;
      using pointer = const typename PtrTraits<T>::PtrType;
      using reference = const value_type&;
      using iterator_category = random_access_iterator_tag;

      using PtrType = typename PtrTraits<T>::PtrType;
      using index_type = Index;

      // Constructors {
      
      ConstStridedRandomAccessor(PtrType ptr, Index stride)
        : ptr{ptr}, stride{stride}
      {}

      
      explicit ConstStridedRandomAccessor(PtrType ptr)
        : ptr{ptr}, stride{static_cast<Index>(1)}
      {}

      
      ConstStridedRandomAccessor()
        : ptr{nullptr}, stride{static_cast<Index>(1)}
      {}
      // }

      // Pointer-like operations {
      
      reference operator*() const {
        return *ptr;
      }

      
      const value_type* operator->() const {
        return reinterpret_cast<const value_type*>(ptr);
      }

      
      reference operator[](Index idx) const {
        return ptr[idx * stride];
      }
      // }

      // Prefix/postfix increment/decrement {
      
      ConstStridedRandomAccessor& operator++() {
        ptr += stride;
        return *this;
      }

      
      ConstStridedRandomAccessor operator++(int) {
        ConstStridedRandomAccessor copy(*this);
        ++*this;
        return copy;
      }

      
      ConstStridedRandomAccessor& operator--() {
        ptr -= stride;
        return *this;
      }

      
      ConstStridedRandomAccessor operator--(int) {
        ConstStridedRandomAccessor copy(*this);
        --*this;
        return copy;
      }
      // }

      // Arithmetic operations {
      
      ConstStridedRandomAccessor& operator+=(Index offset) {
        ptr += offset * stride;
        return *this;
      }

      
      ConstStridedRandomAccessor operator+(Index offset) const {
        return ConstStridedRandomAccessor(ptr + offset * stride, stride);
      }

      
      friend ConstStridedRandomAccessor operator+(
        Index offset,
        const ConstStridedRandomAccessor& accessor
      ) {
        return accessor + offset;
      }

      
      ConstStridedRandomAccessor& operator-=(Index offset) {
        ptr -= offset * stride;
        return *this;
      }

      
      ConstStridedRandomAccessor operator-(Index offset) const {
        return ConstStridedRandomAccessor(ptr - offset * stride, stride);
      }

      // Note that this operator is well-defined when `this` and `other`
      // represent the same sequences, i.e. when
      // 1. this.stride == other.stride,
      // 2. |other - this| / this.stride is an Integer.
      
      difference_type operator-(const ConstStridedRandomAccessor& other) const {
        return (ptr - other.ptr) / stride;
      }
      // }

      // Comparison operators {
      
      bool operator==(const ConstStridedRandomAccessor& other) const {
        return (ptr == other.ptr) && (stride == other.stride);
      }

      
      bool operator!=(const ConstStridedRandomAccessor& other) const {
        return !(*this == other);
      }

      
      bool operator<(const ConstStridedRandomAccessor& other) const {
        return ptr < other.ptr;
      }

      
      bool operator<=(const ConstStridedRandomAccessor& other) const {
        return (*this < other) || (*this == other);
      }

      
      bool operator>(const ConstStridedRandomAccessor& other) const {
        return !(*this <= other);
      }

      
      bool operator>=(const ConstStridedRandomAccessor& other) const {
        return !(*this < other);
      }
      // }


      PtrType ptr;
      Index stride;
    };
    */
}

lazy_static!{
    /*
    template <
      typename T,
      typename Index = i64,
      template <typename U> class PtrTraits = DefaultPtrTraits
    >
    class StridedRandomAccessor
      : public ConstStridedRandomAccessor<T, Index, PtrTraits> {

      using difference_type = Index;
      using value_type = T;
      using pointer = typename PtrTraits<T>::PtrType;
      using reference = value_type&;

      using BaseType = ConstStridedRandomAccessor<T, Index, PtrTraits>;
      using PtrType = typename PtrTraits<T>::PtrType;

      // Constructors {
      
      StridedRandomAccessor(PtrType ptr, Index stride)
        : BaseType(ptr, stride)
      {}

      
      explicit StridedRandomAccessor(PtrType ptr)
        : BaseType(ptr)
      {}

      
      StridedRandomAccessor()
        : BaseType()
      {}
      // }

      // Pointer-like operations {
      
      reference operator*() const {
        return *this->ptr;
      }

      
      value_type* operator->() const {
        return reinterpret_cast<value_type*>(this->ptr);
      }

      
      reference operator[](Index idx) const {
        return this->ptr[idx * this->stride];
      }
      // }

      // Prefix/postfix increment/decrement {
      
      StridedRandomAccessor& operator++() {
        this->ptr += this->stride;
        return *this;
      }

      
      StridedRandomAccessor operator++(int) {
        StridedRandomAccessor copy(*this);
        ++*this;
        return copy;
      }

      
      StridedRandomAccessor& operator--() {
        this->ptr -= this->stride;
        return *this;
      }

      
      StridedRandomAccessor operator--(int) {
        StridedRandomAccessor copy(*this);
        --*this;
        return copy;
      }
      // }

      // Arithmetic operations {
      
      StridedRandomAccessor& operator+=(Index offset) {
        this->ptr += offset * this->stride;
        return *this;
      }

      
      StridedRandomAccessor operator+(Index offset) const {
        return StridedRandomAccessor(this->ptr + offset * this->stride, this->stride);
      }

      
      friend StridedRandomAccessor operator+(
        Index offset,
        const StridedRandomAccessor& accessor
      ) {
        return accessor + offset;
      }

      
      StridedRandomAccessor& operator-=(Index offset) {
        this->ptr -= offset * this->stride;
        return *this;
      }

      
      StridedRandomAccessor operator-(Index offset) const {
        return StridedRandomAccessor(this->ptr - offset * this->stride, this->stride);
      }

      // Note that here we call BaseType::operator- version
      
      difference_type operator-(const BaseType& other) const {
        return (static_cast<const BaseType&>(*this) - other);
      }
      // }
    };
    */
}
