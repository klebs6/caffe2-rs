crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/Dict_inl.h]

impl DictKeyEqualTo {
    
    #[inline] pub fn invoke(&self, 
        lhs: &IValue,
        rhs: &IValue) -> bool {
        
        todo!();
        /*
            if (lhs.isTensor() && rhs.isTensor()) {
        // for tensors, we compare only by identity (following how it's done in Python).
        return lhs.is(rhs);
      }
      // Otherwise, we first compare by identity for efficiency, then by value (see:
      // [container equality])
      return _fastEqualsForContainer(lhs, rhs);
        */
    }
}

pub fn get_type_ptr() -> TypePtr {
    
    todo!();
        /*
        
        */
}

pub fn to_string(type_ptr: TypePtr) -> String {
    
    todo!();
        /*
        
        */
}

pub fn to_typed_dict<Key, Value>(dict: GenericDict) -> Dict<Key,Value> {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(*getTypePtr<Key>() == *dict.impl_->elementTypes.keyType, "Tried to cast a Dict<", toString(dict.impl_->elementTypes.keyType), ", ", toString(dict.impl_->elementTypes.valueType) ,"> to a Dict<", toString(getTypePtr<Key>()), ", ", toString(getTypePtr<Value>()), ">. Key types mismatch.");
      TORCH_INTERNAL_ASSERT(*getTypePtr<Value>() == *dict.impl_->elementTypes.valueType, "Tried to cast a Dict<", toString(dict.impl_->elementTypes.keyType), ", ", toString(dict.impl_->elementTypes.valueType) ,"> to a Dict<", toString(getTypePtr<Key>()), ", ", toString(getTypePtr<Value>()), ">. Value types mismatch.");

      return Dict<Key, Value>(move(dict.impl_));
        */
}

pub fn to_generic_dict<Key, Value>(dict: Dict<Key,Value>) -> GenericDict {

    todo!();
        /*
            return GenericDict(move(dict.impl_));
        */
}

impl DictKeyHash {
    
    #[inline] pub fn invoke(&self, ivalue: &IValue) -> usize {
        
        todo!();
        /*
            if (ivalue.isInt()) {
        return hash<i64>()(ivalue.toInt());
      } else if (ivalue.isString()) {
        return hash<string_view>()(ivalue.toStringView());
      } else if (ivalue.isDouble()) {
        return hash<double>()(ivalue.toDouble());
      } else if (ivalue.isComplexDouble()) {
        return hash<complex<double>>()(ivalue.toComplexDouble());
      } else if (ivalue.isBool()) {
        return hash<bool>()(ivalue.toBool());
      } else if (ivalue.isTensor()) {
        return hash<TensorImpl*>()(ivalue.toTensor().unsafeGetTensorImpl());
      } else {
        throw runtime_error(
            "Can't hash IValues with tag '" + ivalue.tagKind() + "'");
      }
        */
    }
}

impl DictImpl {
    
    #[inline] pub fn copy_(&self) -> IntrusivePtr<DictImpl> {
        
        todo!();
        /*
            return make_intrusive<DictImpl>(dict, elementTypes);
        */
    }
}


impl<K,V> Dict<K,V> {

    pub fn new() -> Self {
    
        todo!();
        /*


            :Dict(make_intrusive<DictImpl>(
          DictImpl::dict_map_type(),
          DictImpl::DictElementTypes{getTypePtr<Key>(), getTypePtr<Value>()})) 

      static_assert(!is_same<Key, IValue>::value, "This constructor is not valid for Dict<IValue, _>. Please use GenericDict(keyType, valueType) instead, or if you absolutely have to, use GenericDict(deprecatedUntypedDict()).");
      static_assert(!is_same<Value, IValue>::value, "This constructor is not valid for Dict<_, IValue>. Please use GenericDict(keyType, valueType) instead, or if you absolutely have to, use GenericDict(deprecatedUntypedDict()).");
        */
    }
    
    pub fn new(
        key_type:   TypePtr,
        value_type: TypePtr) -> Self {
    
        todo!();
        /*


            : Dict(make_intrusive<DictImpl>(
        DictImpl::dict_map_type(),
        DictImpl::DictElementTypes {move(keyType), move(valueType)})) 

      static_assert(is_same<Key, IValue>::value, "This constructor is only valid for GenericDict.");
      static_assert(is_same<Value, IValue>::value, "This constructor is only valid for GenericDict.");
        */
    }
    
    pub fn new(rhs: Dict) -> Self {
    
        todo!();
        /*


            : impl_(move(rhs.impl_)) 

      rhs.impl_ = make_intrusive<DictImpl>(DictImpl::dict_map_type(), impl_->elementTypes);
        */
    }
    
    pub fn new(impl_: IntrusivePtr<DictImpl>) -> Self {
    
        todo!();
        /*


            : impl_(move(impl))
        */
    }
    
    pub fn assign_from(&mut self, rhs: Dict) -> &mut Dict<Key,Value> {
        
        todo!();
        /*
            impl_ = move(rhs.impl_);
      rhs.impl_ = make_intrusive<DictImpl>(DictImpl::dict_map_type(), impl_->elementTypes);
      return *this;
        */
    }
    
    pub fn copy_(&self) -> Dict<Key,Value> {
        
        todo!();
        /*
            return Dict<Key, Value>(impl_->copy());
        */
    }
    
    pub fn begin(&self) -> DictIterator {
        
        todo!();
        /*
            return iterator{impl_->dict.begin()};
        */
    }
    
    pub fn end(&self) -> DictIterator {
        
        todo!();
        /*
            return iterator{impl_->dict.end()};
        */
    }
    
    pub fn empty(&self) -> bool {
        
        todo!();
        /*
            return impl_->dict.empty();
        */
    }
    
    pub fn size(&self) -> DictSizeType {
        
        todo!();
        /*
            return impl_->dict.size();
        */
    }
    
    pub fn clear(&self)  {
        
        todo!();
        /*
            impl_->dict.clear();
        */
    }
    
    pub fn insert<Key_, Value_>(&self, 
        key:   Key,
        value: Value) -> (DictIterator,bool) {
    
        todo!();
        /*
            static_assert(is_constructible<Key, Key_>::value, "Wrong type for the key argument of Dict::insert");
      static_assert(is_constructible<Value, Value_>::value, "Wrong type for the value argument of Dict::insert");
      auto inserted = impl_->dict.insert(pair<IValue, IValue>{
        Key(forward<Key_>(key)),
        Value(forward<Value_>(value))});
      return {iterator{inserted.first}, inserted.second};
        */
    }
    
    pub fn insert_or_assign<Key_, Value_>(&self, 
        key:   Key,
        value: Value) -> (DictIterator,bool) {
    
        todo!();
        /*
            static_assert(is_constructible<Key, Key_>::value, "Wrong type for the key argument of Dict::insert_or_assign");
      static_assert(is_constructible<Value, Value_>::value, "Wrong type for the value argument of Dict::insert_or_assign");
      auto inserted = impl_->dict.insert_or_assign(
        Key(forward<Key_>(key)),
        Value(forward<Value_>(value)));
      return {iterator{inserted.first}, inserted.second};
        */
    }
    
    pub fn erase<Key, Value>(&self, iter: Iterator)  {
    
        todo!();
        /*
            impl_->dict.erase(iter.entryRef_.iterator_);
        */
    }
    
    pub fn erase(&self, key: &Key) -> usize {
        
        todo!();
        /*
            return impl_->dict.erase(key);
        */
    }
    
    pub fn at(&self, key: &Key) -> Value {
        
        todo!();
        /*
            return impl_->dict.at(key).template to<Value>();
        */
    }
    
    pub fn find(&self, key: &Key) -> DictIterator {
        
        todo!();
        /*
            return iterator{impl_->dict.find(key)};
        */
    }
    
    pub fn contains(&self, key: &Key) -> bool {
        
        todo!();
        /*
            return end() != find(key);
        */
    }
    
    pub fn reserve(&self, count: DictSizeType)  {
        
        todo!();
        /*
            impl_->dict.reserve(count);
        */
    }

    pub fn key_type(&self) -> TypePtr {
        
        todo!();
        /*
            return impl_->elementTypes.keyType;
        */
    }
    
    pub fn value_type(&self) -> TypePtr {
        
        todo!();
        /*
            return impl_->elementTypes.valueType;
        */
    }
    
    pub fn unsafe_set_key_type(&mut self, t: TypePtr)  {
        
        todo!();
        /*
            impl_->elementTypes.keyType = move(t);
        */
    }
    
    pub fn unsafe_set_value_type(&mut self, t: TypePtr)  {
        
        todo!();
        /*
            impl_->elementTypes.valueType = move(t);
        */
    }
    
    pub fn is(&self, rhs: &Dict) -> bool {
        
        todo!();
        /*
            return this->impl_ == rhs.impl_;
        */
    }
}

impl PartialEq<Dict> for Dict {
    
    #[inline] fn eq(&self, other: &Dict) -> bool {
        todo!();
        /*
            // Dicts with the same identity trivially compare equal.
      if (lhs.impl_ == rhs.impl_) {
        return true;
      }

      // Otherwise compare the values
      return *lhs.impl_ == *rhs.impl_;
        */
    }
}
