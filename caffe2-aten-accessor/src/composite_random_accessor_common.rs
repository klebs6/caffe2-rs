crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/CompositeRandomAccessorCommon.h]

/**
  | operator_brackets_proxy is used in
  | CompositeRandomAccessor in place of
  | operator[].
  |
  | For some iterators, references
  | returned by operator[] could become invalid,
  | operator_brackets_proxy tries to resolve that
  | by making accessor[n] to be equivalent to
  | *(accessor + n).
  |
  */
pub struct OperatorBracketsProxy<Accessor> {
    accessor: Accessor,
}

//pub type reference = iterator_traits<Accessor>::reference;
//pub type value_type = iterator_traits<Accessor>::value_type;

impl OperatorBracketsProxy<Accessor> {
  
    pub fn new(accessor: &Accessor) -> Self {
    
        todo!();
        /*
        : accessor(accessor),

        
        */
    }
    
    pub fn operator_reference(&mut self) -> Reference {
        
        todo!();
        /*
            return *accessor;
        */
    }
    
    pub fn reference_operator_star(&mut self) -> Reference {
        
        todo!();
        /*
            return *accessor;
        */
    }
    
    pub fn assign_from(&mut self, val: &ValueType) -> &mut OperatorBracketsProxy {
        
        todo!();
        /*
            *accessor = val;
        return *this;
        */
    }
}

/**
  | references_holder is used as a surrogate for
  | the references type from iterator_traits in
  | CompositeRandomAccessor.
  |
  | It is assumed in CompositeRandomAccessor that
  | References = tuple<Types&...>,
  |
  | Values = tuple<Types...> by default, but they
  | could be anything as long as References could
  | be cast to Values.
  |
  | If you plan to use it with STL, for example,
  | you will need to define 'swap` and `get`(aka
  | get) methods.
  */
pub struct ReferencesHolder<Values,References> {
    refs: References,
}

//pub type values = Values;
//pub type references = References;

impl ReferencesHolder<Values,References> {
    
    pub fn new(refs: References) -> Self {
    
        todo!();
        /*


            : refs{refs}
        */
    }
    
    pub fn operator_references(&mut self) -> References {
        
        todo!();
        /*
            return refs;
        */
    }
    
    pub fn operator_values(&mut self) -> Values {
        
        todo!();
        /*
            return refs;
        */
    }
    
    pub fn assign_from(&mut self, vals: Values) -> &mut ReferencesHolder {
        
        todo!();
        /*
            refs = vals;
        return *this;
        */
    }
    
    pub fn data(&mut self) -> &mut References {
        
        todo!();
        /*
            return refs;
        */
    }
}

/**
  | CompositeRandomAccessor is essentially
  | a simplified version of a random access
  | iterator over two random access iterators.
  |
  | TupleInfo should contain a variadic type
  | `tuple`, and a method `tie`, which constructs
  | a tuple of references from a variadic list of
  | arguments.
  |
  */
#[derive(Default)]
pub struct CompositeRandomAccessor<KeyAccessor,ValueAccessor,TupleInfo> {
    keys:   KeyAccessor,
    values: ValueAccessor,
}

pub mod composite_random_accessor {

    use super::*;

    lazy_static!{
        /*
        using self_type = CompositeRandomAccessor<KeyAccessor, ValueAccessor, TupleInfo>;

          using key_accessor_value_type =
            typename iterator_traits<KeyAccessor>::value_type;
          using value_accessor_value_type =
            typename iterator_traits<ValueAccessor>::value_type;
          using key_accessor_reference_type =
            typename iterator_traits<KeyAccessor>::reference;
          using value_accessor_reference_type =
            typename iterator_traits<ValueAccessor>::reference;

          using composite_value_type = typename TupleInfo::template tuple<
            key_accessor_value_type,
            value_accessor_value_type>;
          using composite_reference = typename TupleInfo::template tuple<
            key_accessor_reference_type,
            value_accessor_reference_type>;


          using value_type = composite_value_type;
          using reference = references_holder<composite_value_type, composite_reference>;
          // Note that CompositeRandomAccessor does not hold key and values
          // in a specific datastrcture, which means that a pointer to a (key, value)
          // is not defined. Hence we just use a pointer type of the KeyAccessor.
          using pointer = typename iterator_traits<KeyAccessor>::pointer;
          using difference_type = typename iterator_traits<KeyAccessor>::difference_type;
          using iterator_category = random_access_iterator_tag;
        */
    }
}

impl IndexMut<DifferenceType> for CompositeRandomAccessor<KeyAccessor,ValueAccessor,TupleInfo> {
    
    #[inline] fn index_mut(&mut self, idx: DifferenceType) -> &mut Self::Output {
        todo!();
        /*
            return operator_brackets_proxy<self_type>(
          CompositeRandomAccessor(keys + idx, values + idx)
        );
        */
    }
}

impl AddAssign<DifferenceType> for CompositeRandomAccessor {

    //type Output = CompositeRandomAccessor;
    
    #[inline]fn add_assign(&mut self, other: &DifferenceType) {
        todo!();
        /*
            keys += offset;
        values += offset;
        return *this;
        */
    }
}

impl Add<DifferenceType> for CompositeRandomAccessor {

    type Output = CompositeRandomAccessor;
    
    #[inline]fn add(self, other: &DifferenceType) -> Self::Output {
        todo!();
        /*
            return CompositeRandomAccessor(keys + offset, values + offset);
        */
    }
}

impl Add<DifferenceType> for CompositeRandomAccessor {

    type Output = CompositeRandomAccessor;
    
    #[inline]fn add(self, other: &DifferenceType) -> Self::Output {
        todo!();
        /*
            return accessor + offset;
        */
    }
}


impl SubAssign<DifferenceType> for CompositeRandomAccessor {

    //type Output = CompositeRandomAccessor;
    
    #[inline]fn sub_assign(&mut self, other: &DifferenceType) {
        todo!();
        /*
            keys -= offset;
        values -= offset;
        return *this;
        */
    }
}


impl Sub<DifferenceType> for CompositeRandomAccessor {

    type Output = CompositeRandomAccessor;
    
    #[inline]fn sub(self, other: &DifferenceType) -> Self::Output {
        todo!();
        /*
            return CompositeRandomAccessor(keys - offset, values - offset);
        */
    }
}


impl Sub<&CompositeRandomAccessor> for CompositeRandomAccessor {

    type Output = DifferenceType;
    
    #[inline]fn sub(self, other: &&CompositeRandomAccessor) -> Self::Output {
        todo!();
        /*
            return keys - other.keys;
        */
    }
}


impl PartialEq<CompositeRandomAccessor> for CompositeRandomAccessor {
    
    #[inline] fn eq(&self, other: &CompositeRandomAccessor) -> bool {
        todo!();
        /*
            return keys == other.keys;
        */
    }
}




impl Ord<CompositeRandomAccessor> for CompositeRandomAccessor {
    
    #[inline] fn cmp(&self, other: &CompositeRandomAccessor) -> Ordering {
        todo!();
        /*
            return keys < other.keys;
        */
    }
}

impl PartialOrd<CompositeRandomAccessor> for CompositeRandomAccessor {
    #[inline] fn partial_cmp(&self, other: &CompositeRandomAccessor) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}


impl CompositeRandomAccessor<KeyAccessor,ValueAccessor,TupleInfo> {
    
    pub fn new(
        keys:   KeyAccessor,
        values: ValueAccessor) -> Self {
    
        todo!();
        /*


            : keys(keys), values(values)
        */
    }

    /// Pointer-like operations {
    pub fn operator_star(&self) -> Reference {
        
        todo!();
        /*
            return TupleInfo::tie(*keys, *values);
        */
    }

    /**
      | operator->() is supposed to return a pointer
      | type.
      |
      | Since CompositeRandomAccessor does not hold
      | pointers to pairs, we just return a pointer
      | to a key.
      */
    pub fn operator_arrow(&self) -> *mut Auto {
        
        todo!();
        /*
            return keys.operator->();
        */
    }

    /// Prefix/postfix increment/decrement {
    pub fn prefix_increment(&mut self) -> &mut CompositeRandomAccessor {
        
        todo!();
        /*
            ++keys;
        ++values;
        return *this;
        */
    }
    
    pub fn prefix_increment(&mut self, _0: i32) -> CompositeRandomAccessor {
        
        todo!();
        /*
            CompositeRandomAccessor copy(*this);
        ++*this;
        return copy;
        */
    }
    
    pub fn prefix_decrement(&mut self) -> &mut CompositeRandomAccessor {
        
        todo!();
        /*
            --keys;
        --values;
        return *this;
        */
    }
    
    pub fn prefix_decrement(&mut self, _0: i32) -> CompositeRandomAccessor {
        
        todo!();
        /*
            CompositeRandomAccessor copy(*this);
        --*this;
        return copy;
        */
    }
}
