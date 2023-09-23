crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/List_inl.h]

pub fn get_type_ptr<T>() -> TypePtr {
    
    todo!();
        /*
        
        */
}

pub fn to_string(type_ptr: TypePtr) -> String {
    
    todo!();
        /*
        
        */
}

impl List {
    
    pub fn new<T>(elements: IntrusivePtr<ListImpl>) -> Self {
    
        todo!();
        /*


            : impl_(move(elements))
        */
    }
    
    pub fn new<T>(elements: &IntrusivePtr<ListImpl>) -> Self {
    
        todo!();
        /*
        : impl_(elements),

        
        */
    }
    
    pub fn new<T>() -> Self {
    
        todo!();
        /*


            : List(make_intrusive<ListImpl>(
      typename ListImpl::list_type(),
      getTypePtr<T>())) 
      static_assert(!is_same<T, IValue>::value, "This constructor is not valid for List<IValue>. Please use GenericList(elementType) instead.");
        */
    }
    
    pub fn new<T>(values: &[T]) -> Self {
    
        todo!();
        /*


            : List(make_intrusive<ListImpl>(
        typename ListImpl::list_type(),
        getTypePtr<T>())) 

      static_assert(!is_same<T, IValue>::value, "This constructor is not valid for List<IValue>. Please use GenericList(elementType).");
      impl_->list.reserve(values.size());
      for (const T& element : values) {
        impl_->list.push_back(element);
      }
        */
    }
    
    pub fn new<T>(initial_values: InitializerList<T>) -> Self {
    
        todo!();
        /*


            : List(ArrayRef<T>(initial_values)) 

      static_assert(!is_same<T, IValue>::value, "This constructor is not valid for List<IValue>. Please use GenericList(elementType).");
        */
    }
    
    pub fn new<T>(element_type: TypePtr) -> Self {
    
        todo!();
        /*


            : List(make_intrusive<ListImpl>(
        typename ListImpl::list_type(),
        move(elementType))) 
      static_assert(is_same<T, IValue>::value || is_same<T, intrusive_ptr<Future>>::value,
                    "This constructor is only valid for GenericList or List<Future>.");
        */
    }
}

pub fn to_typed_list<T>(list: GenericList) -> List<T> {

    todo!();
        /*
            // If there's other instances of the list (i.e. list.use_count() > 1), then we have to be invariant
      // because upcasting would allow people to add types into the new list that would break the old list.
      // However, if there aren't any other instances of this list (i.e. list.use_count() == 1), then we can
      // allow upcasting. This can be a perf improvement since we can cast List<T> to List<optional<T>>
      // without having to copy it. This is also used to provide backwards compatibility with some old models
      // that serialized the index arguments to index, index_put, index_put_ and index_put_impl_
      // as List<Tensor> before we changed that argument to be List<optional<Tensor>>. When deserializing, we
      // have list.use_count() == 1 and can deserialize the List<Tensor> directly as List<optional<Tensor>>.
      TORCH_CHECK(*list.impl_->elementType == *getTypePtr<T>()
        || (list.use_count() == 1 && list.impl_->elementType->isSubtypeOf(getTypePtr<T>()))
        , "Tried to cast a List<", toString(list.impl_->elementType), "> to a List<", toString(getTypePtr<T>()), ">. Types mismatch.");
      return List<T>(move(list.impl_));
        */
}

pub fn to_list<T>(list: &List<T>) -> GenericList {

    todo!();
        /*
            return GenericList(list.impl_);
        */
}

impl List {
    
    pub fn new<T>(rhs: List) -> Self {
    
        todo!();
        /*


            rhs.impl_ = make_intrusive<ListImpl>(vector<IValue>{}, impl_->elementType);
        */
    }
    
    pub fn assign_from<T>(&mut self, rhs: List) -> &mut List<T> {
    
        todo!();
        /*
            impl_ = move(rhs.impl_);
      rhs.impl_ = make_intrusive<ListImpl>(vector<IValue>{}, impl_->elementType);
      return *this;
        */
    }
    
    pub fn copy_<T>(&self) -> List<T> {
    
        todo!();
        /*
            return List<T>(impl_->copy());
        */
    }
}

pub fn list_element_to<I,T>(element: I) -> T {

    /*
    pub fn list_element_to<T>(element: &IValue) -> T {

        todo!();
            /*
                return element.template to<T>();
            */
    }

    pub fn list_element_to<T>(element: IValue) -> T {

        todo!();
            /*
                return move(element).template to<T>();
            */
    }
    */

    todo!();
        /*
            return element;
        */
}

impl ListElementReference {
    
    pub fn operator_t<T, Iterator>(&self) -> T {
    
        todo!();
        /*
            return list_element_to<T>(*iterator_);
        */
    }
    
    pub fn assign_from<T, Iterator>(&mut self, new_value: T) -> &mut ListElementReference<T,Iterator> {
    
        todo!();
        /*
            *iterator_ = ListElementFrom<T>::from(move(new_value));
      return *this;
        */
    }
    
    pub fn assign_from<T, Iterator>(&mut self, new_value: &T) -> &mut ListElementReference<T,Iterator> {
    
        todo!();
        /*
            *iterator_ = ListElementFrom<T>::from(move(new_value));
      return *this;
        */
    }
    
    pub fn assign_from<T, Iterator>(&mut self, rhs: ListElementReference<T,Iterator>) -> &mut ListElementReference<T,Iterator> {
    
        todo!();
        /*
            *iterator_ = *rhs.iterator_;
      return *this;
        */
    }
}

pub fn swap<T, Iterator>(
        lhs: ListElementReference<T,Iterator>,
        rhs: ListElementReference<T,Iterator>)  {

    todo!();
        /*
            swap(*lhs.iterator_, *rhs.iterator_);
        */
}

impl<T> PartialEq<ListElementReference> for T {
    
    #[inline] fn eq(&self, other: &ListElementReference) -> bool {
        todo!();
        /*
            T lhs_tmp = lhs;
      return lhs_tmp == rhs;
        */
    }
}


impl PartialEq<T> for ListElementReference {
    
    fn eq(&self, other: &T) -> bool {
        todo!();
        /*
            return rhs == lhs;
        */
    }
}

#[inline] pub fn list_element_to_const_ref<T>(element: &IValue) -> ListElementConstReferenceTraits<T>::const_reference {

    todo!();
        /*
            return element.template to<T>();
        */
}

#[inline] pub fn list_element_to_const_ref_optional_string(element: &IValue) -> ListElementConstReferenceTraits<optional<string>>::const_reference {
    
    todo!();
        /*
            return element.toOptionalStringRef();
        */
}

impl List {
    
    pub fn set<T>(&self, 
        pos:   ListSizeType,
        value: &ValueType)  {
    
        todo!();
        /*
            impl_->list.at(pos) = ListElementFrom<T>::from(value);
        */
    }
    
    pub fn set<T>(&self, 
        pos:   ListSizeType,
        value: ValueType)  {
    
        todo!();
        /*
            impl_->list.at(pos) = ListElementFrom<T>::from(move(value));
        */
    }
    
    pub fn get<T>(&self, pos: ListSizeType) -> List<T>::value_type {
    
        todo!();
        /*
            return list_element_to<T>(impl_->list.at(pos));
        */
    }
}

impl Index<ListSizeType> for List {

    type Output = List<T>::internal_const_reference_type;
    
    #[inline] fn index(&self, pos: ListSizeType) -> &Self::Output {
        todo!();
        /*
            return list_element_to_const_ref<T>(impl_->list.at(pos));
        */
    }
}

impl IndexMut<ListSizeType> for List {
    
    #[inline] fn index_mut(&mut self, pos: ListSizeType) -> &mut Self::Output {
        todo!();
        /*
            static_cast<void>(impl_->list.at(pos)); // Throw the exception if it is out of range.
      return {impl_->list.begin() + pos};
        */
    }
}

impl List {
    
    pub fn extract(&self, pos: ListSizeType) -> List<T>::value_type {
    
        todo!();
        /*
            auto& elem = impl_->list.at(pos);
      auto result = list_element_to<T>(move(elem));
      // Reset the list element to a T() instead of None to keep it correctly typed
      elem = ListElementFrom<T>::from(T{});
      return result;
        */
    }
    
    pub fn begin(&self) -> List<T>::iterator {
    
        todo!();
        /*
            return iterator(impl_->list.begin());
        */
    }
    
    pub fn end(&self) -> List<T>::iterator {
    
        todo!();
        /*
            return iterator(impl_->list.end());
        */
    }
    
    pub fn empty(&self) -> bool {
    
        todo!();
        /*
            return impl_->list.empty();
        */
    }
    
    pub fn size(&self) -> List<T>::size_type {
    
        todo!();
        /*
            return impl_->list.size();
        */
    }
    
    pub fn reserve(&self, new_cap: ListSizeType)  {
    
        todo!();
        /*
            impl_->list.reserve(new_cap);
        */
    }
    
    pub fn clear(&self)  {
    
        todo!();
        /*
            impl_->list.clear();
        */
    }
    
    pub fn insert(&self, 
        pos:   Iterator,
        value: &T) -> List<T>::iterator {
        
        todo!();
        /*
            return iterator { impl_->list.insert(pos.iterator_, ListElementFrom<T>::from(value)) };
        */
    }
    
    pub fn insert(&self, 
        pos:   Iterator,
        value: T) -> List<T>::iterator {
    
        todo!();
        /*
            return iterator { impl_->list.insert(pos.iterator_, ListElementFrom<T>::from(move(value))) };
        */
    }
    
    pub fn emplace(&self, 
        pos:   Iterator,
        value: Args) -> List<T>::iterator {
        
        todo!();
        /*
            // TODO Use list_element_from?
      return iterator { impl_->list.emplace(pos.iterator_, forward<Args>(value)...) };
        */
    }
    
    pub fn push_back(&self, value: &T)  {
        
        todo!();
        /*
            impl_->list.push_back(ListElementFrom<T>::from(value));
        */
    }
    
    pub fn push_back(&self, value: T)  {
        
        todo!();
        /*
            impl_->list.push_back(ListElementFrom<T>::from(move(value)));
        */
    }
    
    pub fn append(&self, b: List<T>)  {
        
        todo!();
        /*
            if (b.use_count() == 1) {
        impl_->list.insert(impl_->list.end(), make_move_iterator(b.impl_->list.begin()), make_move_iterator(b.impl_->list.end()));
      } else {
        impl_->list.insert(impl_->list.end(), b.impl_->list.begin(), b.impl_->list.end());
      }
        */
    }
    
    pub fn emplace_back(&self, args: Args)  {
        
        todo!();
        /*
            // TODO Use list_element_from?
      impl_->list.push_back(T(forward<Args>(args)...));
        */
    }
    
    pub fn erase(&self, pos: Iterator) -> List<T>::iterator {
        
        todo!();
        /*
            return iterator { impl_->list.erase(pos.iterator_) };
        */
    }
    
    pub fn erase(&self, 
        first: Iterator,
        last:  Iterator) -> List<T>::iterator {
        
        todo!();
        /*
            return iterator { impl_->list.erase(first.iterator_, last.iterator_) };
        */
    }
    
    pub fn pop_back(&self)  {
        
        todo!();
        /*
            impl_->list.pop_back();
        */
    }
    
    pub fn resize(&self, count: ListSizeType)  {
        
        todo!();
        /*
            impl_->list.resize(count, T{});
        */
    }
    
    pub fn resize(&self, 
        count: ListSizeType,
        value: &T)  {
        
        todo!();
        /*
            impl_->list.resize(count, value);
        */
    }
}

impl PartialEq<List> for List {
    
    #[inline] fn eq(&self, other: &List) -> bool {
        todo!();
        /*
            // Lists with the same identity trivially compare equal.
      if (lhs.impl_ == rhs.impl_) {
        return true;
      }

      // Otherwise, just compare values directly.
      return *lhs.impl_ == *rhs.impl_;
        */
    }
}

impl List {
    
    pub fn is(&self, rhs: &List<T>) -> bool {
        
        todo!();
        /*
            return this->impl_ == rhs.impl_;
        */
    }
    
    pub fn vec(&self) -> Vec<T> {
        
        todo!();
        /*
            vector<T> result(begin(), end());
      return result;
        */
    }
    
    pub fn use_count(&self) -> usize {
        
        todo!();
        /*
            return impl_.use_count();
        */
    }
    
    pub fn element_type(&self) -> TypePtr {
        
        todo!();
        /*
            return impl_->elementType;
        */
    }
    
    pub fn unsafe_set_element_type(&mut self, t: TypePtr)  {
        
        todo!();
        /*
            impl_->elementType = move(t);
        */
    }
}
