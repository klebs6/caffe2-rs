crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/Dict.h]

pub type TypePtr = Arc<Type>;

pub type ValidDictKeyTypes = TypeList<i64,String,f64,Complex<f64>,bool,Tensor>;

pub struct DictKeyHash {

}

impl DictKeyHash {
    
    pub fn invoke(&self, ivalue: &IValue) -> usize {
        
        todo!();
        /*
        
        */
    }
}

pub struct DictKeyEqualTo {

}

impl DictKeyEqualTo {

    pub fn invoke(&self, 
        lhs: &IValue,
        rhs: &IValue) -> bool {
        
        todo!();
        /*
        
        */
    }
}

//-------------------------------------
pub struct DictImpl {
    base:          IntrusivePtrTarget,
    dict:          DictMapType,
    element_types: DictElementTypes,
}

pub struct DictElementTypes {
    key_type:   TypePtr,
    value_type: TypePtr,
}

pub mod dict_impl {

    use super::*;

    pub type DictMapType = OrderPreservingFlatHashMap<IValue,IValue,DictKeyHash,DictKeyEqualTo>;
}

impl DictImpl {
    
    pub fn new(
        dict:          DictMapType,
        element_types: DictElementTypes) -> Self {
    
        todo!();
        /*


            : dict(move(dict_))
      , elementTypes(move(elementTypes_))
        */
    }
    
    pub fn copy_(&self) -> IntrusivePtr<DictImpl> {
        
        todo!();
        /*
        
        */
    }
}

/**
  | A reference to an entry in the Dict.
  | 
  | Use the `key()` and `value()` methods
  | to read the element.
  |
  */
pub struct DictEntryRef<Key,Value,Iterator> {
    iterator: Iterator,
}

impl DictEntryRef<Key,Value,Iterator> {

    pub fn new(iterator: Iterator) -> Self {
    
        todo!();
        /*
        : iterator(move(iterator)),

        
        */
    }
    
    pub fn key(&self) -> Key {
        
        todo!();
        /*
            return iterator_->first.template to<Key>();
        */
    }
    
    pub fn value(&self) -> Value {
        
        todo!();
        /*
            return iterator_->second.template to<Value>();
        */
    }
    
    pub fn set_value<Value_>(&self, value: Value_)  {
    
        todo!();
        /*
            static_assert(is_constructible<Value, Value_>::value, "Wrong type for the value argument of setValue()");
        iterator_->second = Value(forward<Value_>(value));
        */
    }
}

/**
  | this wraps map_type::iterator to make sure user
  | code can't rely on it being the type of the
  | underlying map.
  |
  */
pub struct DictIterator<Key,Value,Iterator> {
    base:      Iterator<ForwardIteratorTag,DictEntryRef<Key,Value,Iterator>>,
    entry_ref: DictEntryRef<Key,Value,Iterator>,
}

impl Sub<&DictIterator> for DictIterator {

    type Output = usize;
    
    #[inline]fn sub(self, other: &&DictIterator) -> Self::Output {
        todo!();
        /*
            return lhs.entryRef_.iterator_ - rhs.entryRef_.iterator_;
        */
    }
}

impl PartialEq<DictIterator> for DictIterator {
    
    #[inline] fn eq(&self, other: &DictIterator) -> bool {
        todo!();
        /*
            return lhs.get_iterator_() == rhs.get_iterator_();
        */
    }
}

impl Ord<DictIterator> for DictIterator {
    
    #[inline] fn cmp(&self, other: &DictIterator) -> Ordering {
        todo!();
        /*
            return lhs.get_iterator_() < rhs.get_iterator_();
        */
    }
}

impl PartialOrd<DictIterator> for DictIterator {
    #[inline] fn partial_cmp(&self, other: &DictIterator) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl DictIterator<Key,Value,Iterator> {
    
    pub fn new(rhs: &DictIterator) -> Self {
    
        todo!();
        /*
        : entry_ref(rhs.entryRef_),

        
        */
    }
    
    pub fn new(rhs: DictIterator) -> Self {
    
        todo!();
        /*


            : entryRef_(move(rhs.entryRef_))
        */
    }
    
    pub fn assign_from(&mut self, rhs: &DictIterator) -> &mut DictIterator {
        
        todo!();
        /*
            entryRef_ = rhs.entryRef_;
        return *this;
        */
    }
    
    pub fn assign_from(&mut self, rhs: DictIterator) -> &mut DictIterator {
        
        todo!();
        /*
            entryRef_ = move(rhs.entryRef_);
        return *this;
        */
    }
    
    pub fn prefix_increment(&mut self) -> &mut DictIterator {
        
        todo!();
        /*
            ++entryRef_.iterator_;
          return *this;
        */
    }
    
    pub fn prefix_increment(&mut self, _0: i32) -> DictIterator {
        
        todo!();
        /*
            DictIterator copy(*this);
          ++*this;
          return copy;
        */
    }
    
    pub fn deref(&self) -> &DictEntryRef<Key,Value,Iterator> {
        
        todo!();
        /*
            return entryRef_;
        */
    }
    
    pub fn deref(&self) -> *const DictEntryRef<Key,Value,Iterator> {
        
        todo!();
        /*
            return &entryRef_;
        */
    }
    
    pub fn new(iterator: Iterator) -> Self {
    
        todo!();
        /*
        : entry_ref(move(iterator)),

        
        */
    }
    
    pub fn get_iterator(&self) -> &Iterator {
        
        todo!();
        /*
            return entryRef_.iterator_;
        */
    }
}

lazy_static!{
    /*
    template<class Key, class Value> Dict<Key, Value> toTypedDict(Dict<IValue, IValue> dict);
    template<class Key, class Value> Dict<IValue, IValue> toGenericDict(Dict<Key, Value> dict);
    */
}

/**
 | An object of this class stores a map from Key
 | to Value.
 |
 | This is a pointer type. After a copy, both
 | Dicts will share the same storage:
 |
 | > Dict<int, string> a;
 | > Dict<int, string> b = a;
 | > b.insert(3, "three");
 | > ASSERT("three" == a.at(3));
 |
 | We use this class in the PyTorch kernel API
 | because that allows us to do optimizations and
 | switch out the underlying map implementation
 | without breaking backwards compatibility for
 | the kernel API.
 */
pub struct Dict<Key,Value> {

    /**
      | impl_ stores the underlying map as a
      | ska_ordered::order_preserving_flat_hash_map.
      | We intentionally don't offer conversion
      | from/to order_preserving_flat_hash_map,
      | return references to it or something
      | like that, because such operations
      | would get expensive if we switch out
      | the actual map implementation.
      | 
      | This is an intrusive_ptr because Dict
      | is a pointer type.
      | 
      | Invariant: This will never be a nullptr,
      | there will always be a valid DictImpl.
      |
      */
    impl_: IntrusivePtr<DictImpl>,
}

pub mod dict {

    use super::*;

    lazy_static!{
        /*
        static_assert(
                (is_same<IValue, Key>::value && is_same<IValue, Value>::value) 
                || typelist::contains<valid_dict_key_types, Key>::value, 
                "Invalid Key type for Dict. We only support i64, double, bool, and string."
            );
        */
    }

    pub type KeyType = Key;
    pub type MappedType = Value;
    pub type SizeType = DictImpl::dict_map_type::size_type;
    pub type Iterator = DictIterator<Key,Value,DictImpl::dict_map_type::iterator>;
}

impl Dict<Key,Value> {
    
    pub fn new(impl_: IntrusivePtr<DictImpl>) -> Self {
    
        todo!();
        /*

        
        */
    }

    /**
      | Creates an empty dict.
      |
      */
    pub fn new() -> Self {
    
        todo!();
        /*


        
        */
    }

    /**
      | Create a generic dict with runtime type
      | information.
      | 
      | This only works for GenericDict and
      | is not part of the public API but only
      | supposed to be used internally by PyTorch.
      |
      */
    pub fn new(
        key_type:   TypePtr,
        value_type: TypePtr) -> Self {
    
        todo!();
        /*


        
        */
    }

    /**
      | Create a new Dict pointing to a deep copy
      | of the same data.
      | 
      | The Dict returned is a new dict with separate
      | storage.
      | 
      | Changes in it are not reflected in the
      | original dict or vice versa.
      |
      */
    pub fn copy_(&self) -> Dict {
        
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
      | container.
      |
      */
    pub fn size(&self) -> SizeType {
        
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
      | elements. May also invalidate past-the-end
      | iterators.
      |
      */
    pub fn clear(&self)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Inserts element(s) into the container,
      | if the container doesn't already contain
      | an element with an equivalent key.
      | 
      | May invalidate any references, pointers,
      | or iterators referring to contained
      | elements.
      | 
      | 
      | 
      | -----------
      | @return
      | 
      | A pair consisting of an iterator to the
      | inserted element (or to the element
      | that prevented the insertion) and a
      | bool denoting whether the insertion
      | took place.
      |
      */
    pub fn insert<Key_, Value_>(&self, 
        key:   Key,
        value: Value) -> (Iterator,bool) {
    
        todo!();
        /*
        
        */
    }

    /**
      | If an element with the given key already
      | exists, it is overwritten with the given
      | value.
      | 
      | Otherwise, a new element with the given
      | key and value are inserted.
      | 
      | May invalidate any references, pointers,
      | or iterators referring to contained
      | elements.
      | 
      | -----------
      | @return
      | 
      | The bool component is true if the insertion
      | took place and false if the assignment
      | took place. The iterator component
      | is pointing at the element that was inserted
      | or updated.
      |
      */
    pub fn insert_or_assign<Key_, Value_>(&self, 
        key:   Key,
        value: Value) -> (Iterator,bool) {
    
        todo!();
        /*
        
        */
    }

    /**
      | Removes the element pointed to by iter.
      | 
      | May invalidate any references, pointers,
      | or iterators referring to contained
      | elements.
      | 
      | The iterator iter must be valid and dereferenceable.
      | Thus the end() iterator (which is valid,
      | but is not dereferenceable) cannot
      | be used as a value for iter.
      |
      */
    pub fn erase(&self, iter: Iterator)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Removes the element with the given key,
      | if it exists.
      | 
      | May invalidate any references, pointers,
      | or iterators referring to contained
      | elements.
      | 
      | -----------
      | @return
      | 
      | The number of elements removed. This
      | is either '1' if an element with the key
      | existed, or '0' if it didn't.
      |
      */
    pub fn erase(&self, key: &Key) -> usize {
        
        todo!();
        /*
        
        */
    }

    /**
      | Returns the mapped value of the element
      | with key equivalent to key.
      | 
      | If no such element exists, an exception
      | of type out_of_range is thrown.
      |
      */
    pub fn at(&self, key: &Key) -> Value {
        
        todo!();
        /*
        
        */
    }

    /**
      | Finds an element with key equivalent
      | to key.
      | 
      | -----------
      | @return
      | 
      | Iterator to an element with key equivalent
      | to key.
      | 
      | If no such element is found, past-the-end
      | (see end()) iterator is returned.
      |
      */
    pub fn find(&self, key: &Key) -> Iterator {
        
        todo!();
        /*
        
        */
    }

    /**
      | Checks if there is an element with key
      | equivalent to key in the container.
      | 
      | -----------
      | @return
      | 
      | true if there is such an element, otherwise
      | false.
      |
      */
    pub fn contains(&self, key: &Key) -> bool {
        
        todo!();
        /*
        
        */
    }

    /**
      | Increase the capacity so that at least
      | count elements can be stored without
      | having to reallocate or rehash.
      |
      */
    pub fn reserve(&self, count: SizeType)  {
        
        todo!();
        /*
        
        */
    }

    /*
      | Value equality comparison. This function
      | implements Python-like semantics
      | for equality: two dicts with the same
      | identity (e.g. same pointer) trivially
      | compare equal, otherwise each element
      | is compared for equality.
      |
      */

    /**
      | Identity comparison. Returns true
      | if and only if `rhs` represents the same
      | Dict object as `this`.
      |
      */
    pub fn is(&self, rhs: &Dict) -> bool {
        
        todo!();
        /*
        
        */
    }

    /**
      | private API for now because the return
      | type will change to TypePtr instead
      | of optional<TypePtr> once types are
      | mandatory.
      |
      */
    pub fn key_type(&self) -> TypePtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn value_type(&self) -> TypePtr {
        
        todo!();
        /*
        
        */
    }

    /**
      | [unsafe set type]
      |
      | These functions mutate the tagged type of
      | this dictionary in place.
      |
      | There is no checking that the members of the
      | dictionary are instances of the new types,
      | nor is there a check that other IValues which
      | hold references to this dictionary have the
      | right static type.
      |
      | This functionality is used only in the
      | unpickler, where at creation type the real
      | type of the dictionary is unknown, but then
      | later recovered from the static type
      | information of the unpickled object.
      */
    pub fn unsafe_set_key_type(&mut self, t: TypePtr)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn unsafe_set_value_type(&mut self, t: TypePtr)  {
        
        todo!();
        /*
        
        */
    }
}

/**
  | GenericDict is how IValue stores dicts. It is,
  | however, not part of the public API. Kernels
  | should use Dicts with concrete Key, Value types
  | instead (maybe except for some internal prim
  | ops).
  */
pub type GenericDict = Dict<IValue,IValue>;

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/Dict.cpp]

impl PartialEq<DictImpl> for DictImpl {
    
    #[inline] fn eq(&self, other: &DictImpl) -> bool {
        todo!();
        /*
            bool isEqualFastChecks =
          *lhs.elementTypes.keyType == *rhs.elementTypes.keyType &&
          *lhs.elementTypes.valueType == *rhs.elementTypes.valueType &&
          lhs.dict.size() == rhs.dict.size();
      if (!isEqualFastChecks) {
        return false;
      }

      // Dict equality should not care about ordering.
      for (const auto& pr : lhs.dict) {
        auto it = rhs.dict.find(pr.first);
        if (it == rhs.dict.cend()) {
          return false;
        }
        // see: [container equality]
        if (!_fastEqualsForContainer(it->second, pr.second)) {
          return false;
        }
      }

      return true;
        */
    }
}
