crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/List.h]

pub type TypePtr  = Arc<Type>;
pub type ListType = Vec<IValue>;

pub struct ListImpl {
    base:         IntrusivePtrTarget,
    list:         ListType,
    element_type: TypePtr,
}

impl ListImpl {
    
    pub fn new(
        list:         ListType,
        element_type: TypePtr) -> Self {
    
        todo!();
        /*
        : list(move(list_)),
        : element_type(move(elementType_)),

        
        */
    }
    
    pub fn copy_(&self) -> IntrusivePtr<ListImpl> {
        
        todo!();
        /*
            return make_intrusive<ListImpl>(list, elementType);
        */
    }
}

pub fn swap(
        lhs: ListElementReference<T,Iterator>,
        rhs: ListElementReference<T,Iterator>)  {
    
    todo!();
        /*
        
        */
}

pub struct ListElementConstReferenceTraits<T> {

}

/**
  | There is no to() overload for optional<string>.
  |
  */
pub mod list_element_const_reference_traits_optional_string {

    use super::*;

    pub type ConstReference = Option<ReferenceWrapper<String>>;
}

pub struct ListElementReference<T,Iterator> {
    iterator: Iterator,
}

impl ListElementReference<T,Iterator> {

    pub fn operator_t(&self) -> T {
        
        todo!();
        /*
        
        */
    }
    
    pub fn assign_from(&mut self, new_value: T) -> &mut ListElementReference {
        
        todo!();
        /*
        
        */
    }
    
    pub fn assign_from(&mut self, new_value: &T) -> &mut ListElementReference {
        
        todo!();
        /*
        
        */
    }

    /// assigning another ref to this assigns the
    /// underlying value
    ///
    pub fn assign_from(&mut self, rhs: ListElementReference) -> &mut ListElementReference {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new(iter: Iterator) -> Self {
    
        todo!();
        /*
        : iterator(iter),

        
        */
    }

    /// allow moving, but only our friends
    /// (i.e. the List class) can move us
    ///
    pub fn assign_from(&mut self, rhs: ListElementReference) -> &mut ListElementReference {
        
        todo!();
        /*
            iterator_ = move(rhs.iterator_);
        return *this;
        */
    }
}

/**
  | this wraps vector::iterator to make sure user
  | code can't rely on it being the type of the
  | underlying vector.
  |
  */
pub struct ListIterator<T,Iterator> {
    base:     Iterator<RandomAccessIteratorTag,T>,
    iterator: Iterator,
}

impl Index<usize> for ListIterator<T,Iterator> {

    type Output = ListElementReference<T,Iterator>;
    
    #[inline] fn index(&self, offset: usize) -> &Self::Output {
        todo!();
        /*
            return {iterator_ + offset};
        */
    }
}

impl PartialEq<ListIterator> for ListIterator {
    
    #[inline] fn eq(&self, other: &ListIterator) -> bool {
        todo!();
        /*
            return lhs.iterator_ == rhs.iterator_;
        */
    }
}

impl Ord<ListIterator> for ListIterator {
    
    #[inline] fn cmp(&self, other: &ListIterator) -> Ordering {
        todo!();
        /*
            return lhs.iterator_ < rhs.iterator_;
        */
    }
}

impl PartialOrd<ListIterator> for ListIterator {

    #[inline] fn partial_cmp(&self, other: &ListIterator) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl ListIterator<T,Iterator> {
    
    pub fn prefix_increment(&mut self) -> &mut ListIterator {
        
        todo!();
        /*
            ++iterator_;
          return *this;
        */
    }
    
    pub fn prefix_increment(&mut self, _0: i32) -> ListIterator {
        
        todo!();
        /*
            ListIterator copy(*this);
          ++*this;
          return copy;
        */
    }
    
    pub fn prefix_decrement(&mut self) -> &mut ListIterator {
        
        todo!();
        /*
            --iterator_;
          return *this;
        */
    }
    
    pub fn prefix_decrement(&mut self, _0: i32) -> ListIterator {
        
        todo!();
        /*
            ListIterator copy(*this);
          --*this;
          return copy;
        */
    }
    
    pub fn operator_plus_equals(&mut self, offset: SizeType) -> &mut ListIterator {
        
        todo!();
        /*
            iterator_ += offset;
          return *this;
        */
    }
    
    pub fn operator_minus_equals(&mut self, offset: SizeType) -> &mut ListIterator {
        
        todo!();
        /*
            iterator_ -= offset;
          return *this;
        */
    }
    
    pub fn operator_plus(&self, offset: SizeType) -> ListIterator {
        
        todo!();
        /*
            return ListIterator{iterator_ + offset};
        */
    }
    
    pub fn operator_minus(&self, offset: SizeType) -> ListIterator {
        
        todo!();
        /*
            return ListIterator{iterator_ - offset};
        */
    }
    
    pub fn operator_minus(&mut self, 
        lhs: &ListIterator,
        rhs: &ListIterator) -> iterator<random_access_iterator_tag, T>::difference_type {
        
        todo!();
        /*
            return lhs.iterator_ - rhs.iterator_;
        */
    }
    
    pub fn operator_mul(&self) -> ListElementReference<T,Iterator> {
        
        todo!();
        /*
            return {iterator_};
        */
    }
    
    pub fn new(iterator: Iterator) -> Self {
    
        todo!();
        /*
        : iterator(move(iterator)),
        */
    }
}

/**
 | An object of this class stores a list of values
 | of type T.
 |
 | This is a pointer type. After a copy, both
 | Lists will share the same storage:
 |
 | > List<int> a;
 | > List<int> b = a;
 | > b.push_back("three");
 | > ASSERT("three" == a.get(0));
 |
 | We use this class in the PyTorch kernel API
 | instead of vector<T>, because that allows us to
 | do optimizations and switch out the underlying
 | list implementation without breaking backwards
 | compatibility for the kernel API.
 */
pub struct List<T> {

    /**
      | This is an intrusive_ptr because List
      | is a pointer type.
      | 
      | Invariant: This will never be a nullptr,
      | there will always be a valid ListImpl.
      |
      */
    impl_: IntrusivePtr<ListImpl>,
}

pub mod list {

    use super::*;

    pub type InternalReferenceType      = ListElementReference<T,ListImpl::list_type::iterator>;
    pub type InternalConstReferenceType = ListElementConstReferenceTraits<T>::const_reference;
    pub type ValueType                  = T;
    pub type SizeType                   = ListImpl::list_type::size_type;
    pub type Iterator                   = ListIterator<T,ListImpl::list_type::iterator>;
    pub type ReverseIterator            = ListIterator<T,ListImpl::list_type::reverse_iterator>;
}

impl<T> Index<SizeType> for List<T> {

    type Output = InternalConstReferenceType;
    
    /**
      | Returns a reference to the element at
      | specified location pos, with bounds checking.
      |
      | If pos is not within the range of the
      | container, an exception of type out_of_range
      | is thrown.
      |
      | You cannot store the reference, but you can
      | read it and assign new values to it:
      |
      |   List<i64> list = ...;
      |   list[2] = 5;
      |   i64 v = list[1];
      */
    #[inline] fn index(&self, pos: SizeType) -> &Self::Output {
        todo!();
        /*
        
        */
    }
}

impl<T> IndexMut<SizeType> for List<T> {
    
    #[inline] fn index_mut(&mut self, pos: SizeType) -> &mut Self::Output {
        todo!();
        /*
        
        */
    }
}

impl List<T> {

    /**
      | Constructs an empty list.
      |
      */
    pub fn new() -> Self {
    
        todo!();
        /*


        
        */
    }

    /**
      | Constructs a list with some initial
      | values.
      | 
      | Example:
      | 
      | List<int> a({2, 3, 4});
      |
      */
    pub fn new(initial_values: InitializerList<T>) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn new(initial_values: &[T]) -> Self {
    
        todo!();
        /*


        
        */
    }

    /**
      | Create a generic list with runtime type
      | information.
      | 
      | This only works for GenericList and
      | is not part of the public API but only
      | supposed to be used internally by PyTorch.
      |
      */
    pub fn new(element_type: TypePtr) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn new(_0: List) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn assign_from(&mut self, _0: List) -> &mut List {
        
        todo!();
        /*
        
        */
    }

    /**
      | Create a new List pointing to a deep copy
      | of the same data.
      | 
      | The List returned is a new list with separate
      | storage.
      | 
      | Changes in it are not reflected in the
      | original list or vice versa.
      |
      */
    pub fn copy_(&self) -> List {
        
        todo!();
        /*
        
        */
    }

    /**
      | Returns the element at specified location
      | pos, with bounds checking.
      | 
      | If pos is not within the range of the container,
      | an exception of type out_of_range is
      | thrown.
      |
      */
    pub fn get(&self, pos: SizeType) -> ValueType {
        
        todo!();
        /*
        
        */
    }

    /**
      | Moves out the element at the specified
      | location pos and returns it, with bounds
      | checking.
      | 
      | If pos is not within the range of the container,
      | an exception of type out_of_range is
      | thrown.
      | 
      | The list contains an invalid element
      | at position pos afterwards. Any operations
      | on it before re-setting it are invalid.
      |
      */
    pub fn extract(&self, pos: SizeType) -> ValueType {
        
        todo!();
        /*
        
        */
    }

    /**
      | Assigns a new value to the element at
      | location pos.
      |
      */
    pub fn set(&self, 
        pos:   SizeType,
        value: &ValueType)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Assigns a new value to the element at
      | location pos.
      |
      */
    pub fn set(&self, 
        pos:   SizeType,
        value: ValueType)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Returns an iterator to the first element
      | of the container.
      | 
      | If the container is empty, the returned
      | iterator will be equal to end().
      |
      */
    pub fn begin(&self) -> Iterator {
        
        todo!();
        /*
        
        */
    }

    /**
      | Returns an iterator to the element following
      | the last element of the container.
      | 
      | This element acts as a placeholder;
      | attempting to access it results in undefined
      | behavior.
      |
      */
    pub fn end(&self) -> Iterator {
        
        todo!();
        /*
        
        */
    }

    /**
      | Checks if the container has no elements.
      |
      */
    pub fn empty(&self) -> bool {
        
        todo!();
        /*
        
        */
    }

    /**
      | Returns the number of elements in the
      | container
      |
      */
    pub fn size(&self) -> SizeType {
        
        todo!();
        /*
        
        */
    }

    /**
      | Increase the capacity of the vector
      | to a value that's greater or equal to
      | new_cap.
      |
      */
    pub fn reserve(&self, new_cap: SizeType)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Erases all elements from the container.
      | After this call, size() returns zero.
      | 
      | Invalidates any references, pointers,
      | or iterators referring to contained
      | elements. Any past-the-end iterators
      | are also invalidated.
      |
      */
    pub fn clear(&self)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Inserts value before pos.
      | 
      | May invalidate any references, pointers,
      | or iterators referring to contained
      | elements. Any past-the-end iterators
      | may also be invalidated.
      |
      */
    pub fn insert(&self, 
        pos:   Iterator,
        value: &T) -> Iterator {
        
        todo!();
        /*
        
        */
    }

    /**
      | Inserts value before pos.
      | 
      | May invalidate any references, pointers,
      | or iterators referring to contained
      | elements. Any past-the-end iterators
      | may also be invalidated.
      |
      */
    pub fn insert(&self, 
        pos:   Iterator,
        value: T) -> Iterator {
        
        todo!();
        /*
        
        */
    }

    /**
      | Inserts a new element into the container
      | directly before pos.
      | 
      | The new element is constructed with
      | the given arguments.
      | 
      | May invalidate any references, pointers,
      | or iterators referring to contained
      | elements. Any past-the-end iterators
      | may also be invalidated.
      |
      */
    pub fn emplace<Args>(&self, 
        pos:   Iterator,
        value: Args) -> Iterator {
    
        todo!();
        /*
        
        */
    }

    /**
      | Appends the given element value to the
      | end of the container.
      | 
      | May invalidate any references, pointers,
      | or iterators referring to contained
      | elements. Any past-the-end iterators
      | may also be invalidated.
      |
      */
    pub fn push_back(&self, value: &T)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Appends the given element value to the
      | end of the container.
      | 
      | May invalidate any references, pointers,
      | or iterators referring to contained
      | elements. Any past-the-end iterators
      | may also be invalidated.
      |
      */
    pub fn push_back(&self, value: T)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Appends the given list to the end of the
      | container. Uses at most one memory allocation.
      | 
      | May invalidate any references, pointers,
      | or iterators referring to contained
      | elements. Any past-the-end iterators
      | may also be invalidated.
      |
      */
    pub fn append(&self, lst: List<T>)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Appends the given element value to the
      | end of the container.
      | 
      | The new element is constructed with
      | the given arguments.
      | 
      | May invalidate any references, pointers,
      | or iterators referring to contained
      | elements. Any past-the-end iterators
      | may also be invalidated.
      |
      */
    pub fn emplace_back<Args>(&self, args: Args)  {
    
        todo!();
        /*
        
        */
    }

    /**
      | Removes the element at pos.
      | 
      | May invalidate any references, pointers,
      | or iterators referring to contained
      | elements. Any past-the-end iterators
      | may also be invalidated.
      |
      */
    pub fn erase(&self, pos: Iterator) -> Iterator {
        
        todo!();
        /*
        
        */
    }

    /**
      | Removes the elements in the range [first,
      | last).
      | 
      | May invalidate any references, pointers,
      | or iterators referring to contained
      | elements. Any past-the-end iterators
      | may also be invalidated.
      |
      */
    pub fn erase(&self, 
        first: Iterator,
        last:  Iterator) -> Iterator {
        
        todo!();
        /*
        
        */
    }

    /**
      | Removes the last element of the container.
      | 
      | Calling pop_back on an empty container
      | is undefined.
      | 
      | May invalidate any references, pointers,
      | or iterators referring to contained
      | elements. Any past-the-end iterators
      | may also be invalidated.
      |
      */
    pub fn pop_back(&self)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Resizes the container to contain count
      | elements.
      | 
      | If the current size is less than count,
      | additional default-inserted elements
      | are appended.
      | 
      | May invalidate any references, pointers,
      | or iterators referring to contained
      | elements. Any past-the-end iterators
      | may also be invalidated.
      |
      */
    pub fn resize(&self, count: SizeType)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Resizes the container to contain count
      | elements.
      | 
      | If the current size is less than count,
      | additional copies of value are appended.
      | 
      | May invalidate any references, pointers,
      | or iterators referring to contained
      | elements. Any past-the-end iterators
      | may also be invalidated.
      |
      */
    pub fn resize(&self, 
        count: SizeType,
        value: &T)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Identity comparison. Returns true
      | if and only if `rhs` represents the same
      | List object as `this`.
      |
      */
    pub fn is(&self, rhs: &List<T>) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn vec(&self) -> Vec<T> {
        
        todo!();
        /*
        
        */
    }

    /**
      | Returns the number of Lists currently
      | pointing to this same list.
      | 
      | If this is the only instance pointing
      | to this list, returns 1. // TODO Test
      | use_count
      |
      */
    pub fn use_count(&self) -> usize {
        
        todo!();
        /*
        
        */
    }
    
    pub fn element_type(&self) -> TypePtr {
        
        todo!();
        /*
        
        */
    }

    /// See [unsafe set type] for why this exists.
    ///
    pub fn unsafe_set_element_type(&mut self, t: TypePtr)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new(elements: IntrusivePtr<ListImpl>) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn new(elements: &IntrusivePtr<ListImpl>) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn to_typed_list(&mut self, _0: List<IValue>) -> List<T> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_list(&mut self, _0: List<T>) -> List<IValue> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_list(&mut self, _0: &List<T>) -> List<IValue> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn ptr_to_first_element(&mut self, list: &List<IValue>) -> *const IValue {
        
        todo!();
        /*
        
        */
    }
}

/**
  | GenericList is how IValue stores lists. It is,
  | however, not part of the public API. Kernels
  | should use Lists with concrete types instead
  | (maybe except for some internal prim ops).
  */
pub type GenericList = List<IValue>;

#[inline] pub fn ptr_to_first_element(list: &GenericList) -> *const IValue {
    
    todo!();
        /*
            return &list.impl_->list[0];
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/List.cpp]

impl PartialEq<ListImpl> for ListImpl {
    
    #[inline] fn eq(&self, other: &ListImpl) -> bool {
        todo!();
        /*
            return *lhs.elementType == *rhs.elementType &&
          lhs.list.size() == rhs.list.size() &&
          // see: [container equality]
          equal(
              lhs.list.cbegin(),
              lhs.list.cend(),
              rhs.list.cbegin(),
              _fastEqualsForContainer);
        */
    }
}
