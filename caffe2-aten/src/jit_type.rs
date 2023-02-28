crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/jit_type.h]

pub type OptNameList = Option<Vec<String>>;
pub type AnyTypePtr  = Arc<AnyType>;

/**
  | Any is the top of the type hierarchy,
  | all other types are subtypes T <: Any,
  | forall T
  |
  */
pub struct AnyType {
    base: Type,
}

pub mod any_type {

    pub const KIND: TypeKind = TypeKind::AnyType;
}

impl PartialEq<Type> for AnyType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

impl Default for AnyType {
    
    fn default() -> Self {
        todo!();
        /*
        : ty(TypeKind::AnyType),

        
        */
    }
}

impl AnyType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "Any";
        */
    }

    /// global singleton
    pub fn get() -> AnyTypePtr {
        
        todo!();
        /*
        
        */
    }
}

#[inline] pub fn to_string(type_ptr: TypePtr) -> String {
    
    todo!();
        /*
            return typePtr->str();
        */
}


// common base for all types that have a single sub element
// e.g. Future[T], Optional[T], List[T]
pub struct SingleElementType<const K: TypeKind,T> {
    base: Type,
    elem: TypePtr,
}

pub mod single_element_type {
    pub const Kind: TypeKind = K;
}

impl PartialEq<Type> for SingleElementType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            if (auto rhs_ = rhs.cast<T>()) {
          return *getElementType() == *rhs_->getElementType();
        }
        return false;
        */
    }
}

impl<const K: TypeKind,T> SingleElementType<K,T> {
    
    pub fn get_element_type(&self) -> TypePtr {
        
        todo!();
        /*
            return elem;
        */
    }
    
    pub fn has_free_variables(&self) -> bool {
        
        todo!();
        /*
            return getElementType()->hasFreeVariables();
        */
    }
    
    pub fn contained_types(&self) -> &[TypePtr] {
        
        todo!();
        /*
            return elem;
        */
    }
    
    pub fn new(elem: TypePtr) -> Self {
    
        todo!();
        /*
        : ty(Kind),
        : elem(move(elem)),

            if (!this->elem) {
          throw runtime_error(str(
                "Can not create ", typeKindToString(Kind), " with None type"));
        }
        */
    }
}

pub type OptionalTypePtr = Arc<OptionalType>;

/**
  | This type represents an optional type, for each
  | element type.
  |
  | Optional[T] can accept both T and None(nullopt
  | in C++) Subtype hierarchy for Optional:
  |
  | 1. Optional[T] <: Optional[R] iff T <: R
  |
  | 2. T <: Optional[R] if T <: R
  |
  | 3. None <: Optional[T] for all T
  |
  */
pub struct OptionalType {
    base: SingleElementType<TypeKind_OptionalType,OptionalType>,
}

impl OptionalType {
    
    pub fn create(element: TypePtr) -> OptionalTypePtr {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(element, "OptionalType requires valid TypePtr");
        // Optional is a union of [None, T], so Optional[[Optional[T]]] ->
        // Optional[T]
        if (auto opt_ptr = element->cast<OptionalType>()) {
          return opt_ptr;
        }
        return OptionalTypePtr(
            new OptionalType(move(element))); // NOLINT(modernize-make-shared)
        */
    }
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            stringstream ss;
        ss << getElementType()->str() << "?";
        return ss.str();
        */
    }
    
    pub fn create_with_contained(&self, contained_types: Vec<TypePtr>) -> TypePtr {
        
        todo!();
        /*
            AT_ASSERT(contained_types.size() == 1);
        return create(contained_types[0]);
        */
    }
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
            if (Type::isSubtypeOfExt(rhs, why_not)) {
          return true;
        }
        if (auto rhs_ = rhs->cast<OptionalType>()) {
          return getElementType()->isSubtypeOfExt(rhs_->getElementType(), why_not);
        }
        return false;
        */
    }

    /// common cast Optional[Tensor] for undefined
    /// tensor type
    ///
    pub fn of_tensor() -> OptionalTypePtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new(elem: TypePtr) -> Self {
    
        todo!();
        /*
        : single_element_type(elem),

        
        */
    }
    
    pub fn annotation_str_impl(&self, printer: TypePrinter) -> String {
        let printer: TypePrinter = printer.unwrap_or(nullptr);

        todo!();
        /*
            stringstream ss;
        ss << "Optional[" << getElementType()->annotation_str(printer) << "]";
        return ss.str();
        */
    }
}

#[inline] pub fn merge_primitive_option_t<T>(
        a: &Option<T>,
        b: &Option<T>) -> Option<T> {

    todo!();
        /*
            if (a.has_value() && b.has_value() && a.value() == b.value()) {
        return a;
      }
      return optional<T>{};
        */
}

/**
  | If we see `a + b + c`  and know that a, b, and
  | c are the same size and have two dimensions
  | (WxH), then we can generate a fused kernel for
  | them. That fused kernel would likely have
  | indexing math to handling both the W and
  | H dimensions. However, if we knew the WxH
  | dimensions were contiguous, we can pretend like
  | we only have a single dimension, simplifying
  | the indexing logic.
  |
  | This can be performed even if the dimensions
  | are transposed, as long as a, b, and c are
  | transposed in the same way.
  |
  | We'd like to have the compiler be able to do
  | this dimensionality reduction, but simply
  | knowing sizes is not enough.
  |
  | We can extend profiling to also record stride
  | information.
  |
  | Rather than recording specific strides, we can
  | simply order the strides from smallest to
  | largest with `stride_indices` A contiguity
  | marker on the smallest stride (c0) indicates
  | the stride is precisely 1, otherwise
  | a contiguity marker means that $stride_n
  | = size_{n-1}*stride_{n-1}$
  |
  */
pub struct Stride {
    stride_index: Option<usize>,
    contiguous:   Option<bool>,
    stride:       Option<usize>,
}

impl PartialEq<Stride> for Stride {
    
    #[inline] fn eq(&self, other: &Stride) -> bool {
        todo!();
        /*
            return stride_index_ == b.stride_index_ && contiguous_ == b.contiguous_ &&
            stride_ == b.stride_;
        */
    }
}

impl Stride {
    
    pub fn new(
        stride_index: &Option<usize>,
        contiguous:   &Option<bool>,
        stride:       &Option<usize>) -> Self {
    
        todo!();
        /*
        : stride_index(stride_index),
        : contiguous(contiguous),
        : stride(stride),

        
        */
    }
    
    pub fn is_complete(&self) -> bool {
        
        todo!();
        /*
            return stride_index_ && contiguous_ && stride_;
        */
    }
}

#[inline] pub fn merge_primitive_option_stride(
        a: &Option<Stride>,
        b: &Option<Stride>) -> Option<Stride> {
    
    todo!();
        /*
            optional<Stride> left = a;
      optional<Stride> right = b;
      if (!left.has_value()) {
        left = {Stride()};
      }
      if (!right.has_value()) {
        right = {Stride()};
      }

      auto merged_index =
          merge_primitive(left->stride_index_, right->stride_index_);
      auto merged_cont = merge_primitive(left->contiguous_, right->contiguous_);
      auto merged_stride = merge_primitive(left->stride_, right->stride_);
      auto r = Stride(merged_index, merged_cont, merged_stride);
      // normalize
      if (!r.stride_index_.has_value() && !r.contiguous_.has_value() &&
          !r.stride_.has_value()) {
        return optional<Stride>{};
      }

      return r;
        */
}

pub struct ShapeSymbol {
    value: i64,
}

pub mod shape_symbol {

    use super::*;

    lazy_static!{
        static ref num_symbols: AtomicUsize = AtomicUsize::new(0);
    }
}

impl Default for ShapeSymbol {
    
    /// needed for use in `map`
    fn default() -> Self {
        todo!();
        /*
        : value(-1),

        
        */
    }
}

impl PartialEq<ShapeSymbol> for ShapeSymbol {
    
    #[inline] fn eq(&self, other: &ShapeSymbol) -> bool {
        todo!();
        /*
            return value_ == b.value_;
        */
    }
}

impl Ord<ShapeSymbol> for ShapeSymbol {
    
    #[inline] fn cmp(&self, other: &ShapeSymbol) -> Ordering {
        todo!();
        /*
            return value_ < b.value_;
        */
    }
}

impl PartialOrd<ShapeSymbol> for ShapeSymbol {

    #[inline] fn partial_cmp(&self, other: &ShapeSymbol) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl ShapeSymbol {

    /// is this symbol a fixed/static dimension
    ///
    pub fn is_static(&self) -> bool {
        
        todo!();
        /*
            return value_ >= 0;
      }{
        */
    }
    
    pub fn from_static_size(val: i64) -> ShapeSymbol {
        
        todo!();
        /*
            return ShapeSymbol(val);
        */
    }
    
    pub fn static_size(&self) -> i64 {
        
        todo!();
        /*
            TORCH_CHECK(is_static());
        return value_;
      }{
        */
    }
    
    pub fn value(&self) -> i64 {
        
        todo!();
        /*
            return value_;
      }{
        */
    }
    
    pub fn new_symbol() -> ShapeSymbol {
        
        todo!();
        /*
            return fromStaticSize(-static_cast<i64>(++num_symbols));
      }{
        */
    }
    
    pub fn new(val: i64) -> Self {
    
        todo!();
        /*
        : value(val),

        
        */
    }
}

#[inline] pub fn merge_primitive_shape_symbol(
        a: &ShapeSymbol,
        b: &ShapeSymbol) -> ShapeSymbol {
    
    todo!();
        /*
            if (a.is_static() && b.is_static() && a == b) {
        return a;
      }
      return ShapeSymbol::newSymbol();
        */
}

/**
  | Shape of a Tensor represented with
  | ShapeSymbol's. Unranked, ranked unknown dims,
  | partially known and fully known shapes are all
  | supported.
  |
  */
pub struct SymbolicShape {
    dims: Option<Vec<ShapeSymbol>>,
}

impl Default for SymbolicShape {
    
    /// Unranked shape constructor.
    fn default() -> Self {
        todo!();
        /*
        : dims(nullopt),

        
        */
    }
}

impl Index<usize> for SymbolicShape {

    type Output = ShapeSymbol;
    
    #[inline] fn index(&self, i: usize) -> &Self::Output {
        todo!();
        /*
            if (!dims_) {
          throw runtime_error("Rank isn't fixed");
        }
        return (*dims_).at(i);
        */
    }
}

impl SymbolicShape {

    /// Known rank but unknown dimentions.
    ///
    pub fn new(rank: Option<usize>) -> Self {
    
        todo!();
        /*
        : dims(nullopt),

            if(!rank) {
          return;
        }

        vector<ShapeSymbol> shape_symbols;
        shape_symbols.reserve(*rank);
        for(usize i = 0; i < *rank; ++i) {
          shape_symbols.push_back(ShapeSymbol::newSymbol());
        }
        dims_ = shape_symbols;
        */
    }

    /// Mix of known and unknown ranks
    ///
    pub fn new(dims: &Vec<Option<i64>>) -> Self {
    
        todo!();
        /*


            vector<ShapeSymbol> shape_symbols;
        shape_symbols.reserve(dims.size());
        for(optional<i64> dim: dims) {
          if(!dim) {
            shape_symbols.push_back(ShapeSymbol::newSymbol());
          } else {
            shape_symbols.push_back(ShapeSymbol::fromStaticSize(*dim));
          }
        }
        dims_ = shape_symbols;
        */
    }
    
    pub fn dump(&self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new(dims: Vec<ShapeSymbol>) -> Self {
    
        todo!();
        /*
        : dims(move(dims)),

        
        */
    }
    
    pub fn new(dims: &[i32]) -> Self {
    
        todo!();
        /*


            vector<ShapeSymbol> shape_symbols;
        shape_symbols.reserve(dims.size());
        for(i64 dim : dims) {
          shape_symbols.push_back(ShapeSymbol::fromStaticSize(dim));
        }
        dims_ = shape_symbols;
        */
    }
    
    pub fn at(&self, i: usize) -> ShapeSymbol {
        
        todo!();
        /*
            if (!dims_) {
          throw runtime_error("Rank isn't fixed");
        }
        return (*dims_).at(i);
        */
    }

    /// Returns rank or nullopt in case of
    /// unranked shape.
    ///
    pub fn rank(&self) -> Option<usize> {
        
        todo!();
        /*
            if(!dims_) {
          return nullopt;
        }
        return dims_->size();
        */
    }
    
    pub fn sizes(&self) -> Option<Vec<ShapeSymbol>> {
        
        todo!();
        /*
            return dims_;
        */
    }

    /**
      | Checks whether the shape is fully
      | defined/complete, ie. rank and sizes of every
      | dimension are known.
      |
      */
    pub fn is_complete(&self) -> bool {
        
        todo!();
        /*
            if(!dims_) {
          return false;
        }
        for(auto d : *dims_) {
          if(!d.is_static()) {
            return false;
          }
        }
        return true;
        */
    }

    /**
      | Create new SymbolicShape that is result of
      | merging self and another SymbolicShape. Only
      | dimensions that are static and equal will be
      | preserved.
      |
      | If either of two shapes are of unknown rank
      | or they have unmatching rank, result will be
      | unranked.
      |
      */
    pub fn merge(&self, other: &SymbolicShape) -> SymbolicShape {
        
        todo!();
        /*
        
        */
    }
}

#[inline] pub fn is_complete(s: &Stride) -> bool {
    
    todo!();
        /*
            return s.isComplete();
        */
}

///----------------------------------------
pub struct VaryingShape<T> {
    dims: Option<ListOfOptionalElements>,
}

pub mod varying_shape {

    pub type ListOfOptionalElements<T> = Vec<Option<T>>;
}

impl PartialEq<VaryingShape> for VaryingShape {
    
    #[inline] fn eq(&self, other: &VaryingShape) -> bool {
        todo!();
        /*
            return dims_ == other.dims_;
        */
    }
}

impl Index<usize> for VaryingShape {

    type Output = Option<T>;
    
    #[inline] fn index(&self, i: usize) -> &Self::Output {
        todo!();
        /*
            if (!dims_) {
          throw runtime_error("Rank isn't fixed");
        }
        return (*dims_).at(i);
        */
    }
}

impl VaryingShape<T> {
    
    pub fn new(vec: &Vec<T>) -> Self {
    
        todo!();
        /*
        : varying_shape(ListOfOptionalElements(vec.begin(), vec.end())),

        
        */
    }
    
    pub fn new(vec: &[T]) -> Self {
    
        todo!();
        /*
        : varying_shape(ListOfOptionalElements(vec.begin(), vec.end())),

        
        */
    }
    
    pub fn new(size: Option<usize>) -> Self {
        let size: Option<usize> = size.unwrap_or(nullopt);
        todo!();
        /*
        : dims(nullopt),

            if (size) {
          dims_ = ListOfOptionalElements(*size);
        }
        */
    }
    
    pub fn new(dims: ListOfOptionalElements) -> Self {
    
        todo!();
        /*
        : dims(move(dims)),

        
        */
    }
    
    pub fn new(size: usize) -> Self {
    
        todo!();
        /*


            : VaryingShape(optional<usize>(size))
        */
    }
    
    pub fn size(&self) -> Option<usize> {
        
        todo!();
        /*
            if (!dims_) {
          return nullopt;
        }
        const auto& dims = dims_.value();
        return dims.size();
        */
    }
    
    pub fn sizes(&self) -> &Option<ListOfOptionalElements> {
        
        todo!();
        /*
            return dims_;
        */
    }
    
    pub fn merge(&self, other: &VaryingShape) -> VaryingShape {
        
        todo!();
        /*
        
        */
    }
    
    pub fn concrete_sizes(&self) -> Option<Vec<T>> {
        
        todo!();
        /*
            if (!dims_) {
          return nullopt;
        }
        vector<T> sizes;
        for (auto d : *dims_) {
          if (!d) {
            return nullopt;
          }
          sizes.push_back(d.value());
        }
        return sizes;
        */
    }
    
    pub fn is_complete(&self) -> bool {
        
        todo!();
        /*
            if (!dims_) {
          return false;
        }
        for (auto d : *dims_) {
          if (!d || !isComplete(*d)) {
            return false;
          }
        }
        return true;
        */
    }
}

pub type TensorTypePtr = Arc<TensorType>;

/**
  | This type represents a single Tensor
  | with a specific size
  |
  */
pub struct TensorType {

    base:          Type,

    scalar_type:   Option<ScalarType>,
    device:        Option<Device>,
    sizes:         SymbolicShape,
    strides:       VaryingShape<Stride>,
    requires_grad: Option<bool>,

    /**
      | we exploit the fact certain tensors
      | must be zero in the autograd to optimize
      | gradient computation. Such zero tensors
      | are currently implemented with `UndefinedTensorImpl.`
      | They can be handled only by special operators
      | (e.g. `AutogradAdd`) and their `Tensor::defined()`
      | property returns false.
      | 
      | Normally, `undefined_` is set to false,
      | unless a type was created with `withUndefined`
      | 
      | This will also mean that `undefined`
      | tensors will fail `subtypeOf(TensorType::get())`
      | check undefined_ may become `nullopt`
      | if the tensor was observed to be both
      | defined and undefined. However, no
      | tensor type starts out with `undefined_`
      | set to `nullopt`
      |
      */
    undefined:     Option<bool>,


    /**
      | Represents whether or not this type
      | was inferred.
      |
      */
    is_inferred:   bool, // default = false
}

pub mod tensor_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::TensorType;
}

impl TensorType {
    
    pub fn create(t: &Tensor) -> TensorTypePtr {
        
        todo!();
        /*
        
        */
    }

    /**
      | used by TensorType::create(usize
      | dim) which in turn used by shape_analysis.cpp
      |
      */
    pub fn create(
        scalar_type:       Option<ScalarType>,
        device:            Option<Device>,
        sizes:             &VaryingShape<i64>,
        strides:           &VaryingShape<i64>,
        requires_grad:     Option<bool>,
        undefined:         Option<bool>,
        tensor_contiguity: bool) -> TensorTypePtr {

        let undefined: Option<bool> = undefined.unwrap_or(false);
        let tensor_contiguity: bool = tensor_contiguity.unwrap_or(false);

        todo!();
        /*
        
        */
    }
    
    pub fn create(
        scalar_type:   Option<ScalarType>,
        device:        Option<Device>,
        sizes:         &SymbolicShape,
        stride:        &VaryingShape<Stride>,
        requires_grad: Option<bool>,
        undefined:     Option<bool>) -> TensorTypePtr {
        let undefined: Option<bool> = undefined.unwrap_or(false);

        todo!();
        /*
        
        */
    }
    
    pub fn create(
        scalar_type:   Option<ScalarType>,
        device:        Option<Device>,
        dim:           Option<usize>,
        requires_grad: Option<bool>) -> TensorTypePtr {
        
        todo!();
        /*
        
        */
    }

    /**
      | overloaded create variadic template
      | argument as it could not distinguish
      | initializer list
      |
      */
    pub fn create_contiguous(
        scalar_type: ScalarType,
        device:      Device,
        sizes:       &[i32]) -> TensorTypePtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn from_number_type(typ: TypePtr) -> TypePtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn from_bool_type() -> TypePtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn dim(&self) -> Option<usize> {
        
        todo!();
        /*
            return sizes().size();
        */
    }
    
    pub fn sizes(&self) -> VaryingShape<i64> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn strides(&self) -> VaryingShape<i64> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn stride_properties(&self) -> &VaryingShape<Stride> {
        
        todo!();
        /*
            return strides_;
        */
    }
    
    pub fn device(&self) -> Option<Device> {
        
        todo!();
        /*
            return device_;
        */
    }
    
    pub fn scalar_type(&self) -> Option<ScalarType> {
        
        todo!();
        /*
            return scalar_type_;
        */
    }
    
    pub fn requires_grad(&self) -> Option<bool> {
        
        todo!();
        /*
            return requires_grad_;
        */
    }
    
    pub fn requires_grad(&self) -> bool {
        
        todo!();
        /*
            return requires_grad_ ? *requires_grad_ : true;
        */
    }
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
        
        */
    }
    
    pub fn repr_str(&self) -> String {
        
        todo!();
        /*
            return str() + (isInferredType() ? " (inferred)" : "");
        */
    }
    
    pub fn numel(&self) -> Option<usize> {
        
        todo!();
        /*
            usize prod = 1;
        const auto& shape = sizes();

        for (usize i = 0; i < shape.size(); i++) {
          if (!shape[i]) {
            return optional<usize>{};
          }
          prod *= shape[i].value();
        }
        return prod;
        */
    }
    
    pub fn with_requires_grad(&mut self, s: Option<bool>) -> TensorTypePtr {
        
        todo!();
        /*
            auto copy = clone();
        copy->requires_grad_ = s;
        return copy;
        */
    }
    
    pub fn with_scalar_type(&mut self, st: Option<ScalarType>) -> TensorTypePtr {
        
        todo!();
        /*
            auto copy = clone();
        copy->scalar_type_ = st;
        return copy;
        */
    }
    
    pub fn with_dim(&mut self, d: Option<usize>) -> TensorTypePtr {
        
        todo!();
        /*
            auto copy = clone();
        // withDim is only used by the legacy executor
        // that only cares about the rank, so create dummy symbols)) :
        copy->sizes_ = SymbolicShape(d);
        copy->strides_ = VaryingShape<Stride>(d);
        return copy;
        */
    }
    
    pub fn with_sizes_strides(&self, 
        sizes:   &[i32],
        strides: &[i32]) -> TensorTypePtr {
        
        todo!();
        /*
            auto cloned = clone();
        auto ssizes = SymbolicShape(sizes);
        cloned->sizes_ = ssizes;
        cloned->strides_ = computeStrideProps(sizes, strides);
        return cloned;
        */
    }
    
    pub fn with_symbolic_shapes(&self, ssizes: SymbolicShape) -> TensorTypePtr {
        
        todo!();
        /*
            auto cloned = clone();
        cloned->sizes_ = move(ssizes);
        return cloned;
        */
    }
    
    pub fn with_sizes(&self, sizes: &[i32]) -> TensorTypePtr {
        
        todo!();
        /*
            return withSizesStrides(
            sizes, contiguousStridesOf(sizes));
        */
    }
    
    pub fn dimensioned_only(&self) -> TensorTypePtr {
        
        todo!();
        /*
            auto copy = clone();
        copy->sizes_ = SymbolicShape(sizes().size());
        copy->strides_ = VaryingShape<Stride>(sizes().size());
        return copy;
        */
    }
    
    pub fn contiguous(&self) -> TensorTypePtr {
        
        todo!();
        /*
            auto cloned = clone();
        TORCH_INTERNAL_ASSERT(sizes().concrete_sizes().has_value());
        auto strides = computeStrideProps(
            *sizes().concrete_sizes(),
            contiguousStridesOf(*sizes().concrete_sizes()));
        cloned->strides_ = strides;
        return cloned;
        */
    }
    
    pub fn symbolic_sizes(&self) -> &SymbolicShape {
        
        todo!();
        /*
        
        */
    }
    
    pub fn merge(&self, 
        other:       &TensorType,
        merge_sizes: bool) -> TensorTypePtr {
        let merge_sizes: bool = merge_sizes.unwrap_or(true);

        todo!();
        /*
        
        */
    }
    
    pub fn match_tensor(&mut self, t: &Tensor) -> bool {
        
        todo!();
        /*
        
        */
    }

    /**
      | is all information about the type specified
      | except for autograd?
      |
      | This replaces the notion of
      | a 'CompleteTensorType' that used to exist in
      | the type-hierarchy. Excluding require_grad
      | and undefined allows this to match the old
      | behavior.
      |
      */
    pub fn is_complete(&self) -> bool {
        
        todo!();
        /*
            return scalar_type_ && device_ && sizes_.isComplete() && strides_.isComplete();
        */
    }
    
    pub fn is_inferred_type(&self) -> bool {
        
        todo!();
        /*
            return is_inferred_;
        */
    }
    
    pub fn get_inferred() -> TensorTypePtr {
        
        todo!();
        /*
            static auto valueInferred = TensorType::create(
            /*scalar_type=*/{},
            /*device=*/{},
            /*sizes=*/SymbolicShape(),
            /*stride=*/VaryingShape<Stride>{},
            /*requires_grad=*/{},
            /*undefined=*/false);
        valueInferred->is_inferred_ = true;
        return valueInferred;
        */
    }

    /**
      | this property is used by GuardElimination
      | please see `checkInputs` for more details
      |
      */
    pub fn is_summarized(&self) -> bool {
        
        todo!();
        /*
            return !(isComplete() && requiresGrad().has_value() &&
                 undefined().has_value());
        */
    }
    
    pub fn with_undefined(&mut self) -> TensorTypePtr {
        
        todo!();
        /*
            auto r = clone();
        r->undefined_ = true;
        return r;
        */
    }
    
    pub fn with_possibly_undefined(&mut self) -> TensorTypePtr {
        
        todo!();
        /*
            auto r = clone();
        r->undefined_ = nullopt;
        return r;
        */
    }
    
    pub fn undefined(&self) -> Option<bool> {
        
        todo!();
        /*
            return undefined_;
        */
    }
    
    pub fn get() -> TensorTypePtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn contiguous_strides_of(sizes: &[i32]) -> Vec<i64> {
        
        todo!();
        /*
            vector<i64> strides(sizes.size());
        if (sizes.empty()) // zero-dim case
          return strides;
        strides.back() = 1;
        for (usize i = strides.size() - 1; i > 0; i--) {
          strides[i - 1] = strides[i] * sizes[i];
        }
        return strides;
        */
    }
    
    pub fn new(
        scalar_type:   Option<ScalarType>,
        device:        Option<Device>,
        sizes:         &SymbolicShape,
        strides:       &VaryingShape<Stride>,
        requires_grad: Option<bool>,
        undefined:     Option<bool>) -> Self {
        let undefined: Option<bool> = undefined.unwrap_or(false);
        todo!();
        /*


        
        */
    }
    
    pub fn clone(&self) -> TensorTypePtr {
        
        todo!();
        /*
            return TensorTypePtr(new TensorType(
            scalar_type_, device_, sizes_, strides_, requires_grad_, undefined_));
        */
    }
    
    pub fn compute_stride_props(
        sizes:             &[i32],
        strides:           &[i32],
        tensor_contiguity: bool) -> VaryingShape<Stride> {
        let tensor_contiguity: bool = tensor_contiguity.unwrap_or(false);

        todo!();
        /*
        
        */
    }
}

pub type ListTypePtr = Arc<ListType>;

pub struct ListType {
    base: SingleElementType<TypeKind_ListType,ListType>,
}

impl ListType {

    /**
      | It's not exactly a singleton, but there
      | should be exactly one instance of List[T]
      | for every T
      |
      */
    pub fn create<T>(all: T) -> ListTypePtr {
    
        todo!();
        /*
            return ListTypePtr(
            new ListType(forward<T>(all)...)); // NOLINT(modernize-make-shared)
        */
    }
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            stringstream ss;
        ss << getElementType()->str() << "[]";
        return ss.str();
        */
    }
    
    pub fn create_with_contained(&self, contained_types: Vec<TypePtr>) -> TypePtr {
        
        todo!();
        /*
            return create(contained_types.at(0));
        */
    }
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
        
        */
    }

    /// common cast List[Tensor]
    ///
    pub fn of_tensors() -> ListTypePtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn of_ints() -> ListTypePtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn of_floats() -> ListTypePtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn of_complex_doubles() -> ListTypePtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn of_bools() -> ListTypePtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn of_strings() -> ListTypePtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new(elem: TypePtr) -> Self {
    
        todo!();
        /*
        : single_element_type(elem),

        
        */
    }
    
    pub fn annotation_str_impl(&self, printer: TypePrinter) -> String {
        let printer: TypePrinter = printer.unwrap_or(nullptr);

        todo!();
        /*
            stringstream ss;
        ss << "List[" << getElementType()->annotation_str(printer) << "]";
        return ss.str();
        */
    }
}

pub type DictTypePtr = Arc<DictType>;

pub struct DictType {
    base:               Type,
    types:              Vec<TypePtr>,
    has_free_variables: bool,
}

pub mod dict_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::DictType;
}

impl PartialEq<Type> for DictType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            if (auto dict_rhs = rhs.cast<DictType>()) {
          return *getKeyType() == *(dict_rhs->getKeyType()) &&
              *getValueType() == *(dict_rhs->getValueType());
        }
        return false;
        */
    }
}

impl DictType {

    pub fn create(
        key:   TypePtr,
        value: TypePtr) -> DictTypePtr {
        
        todo!();
        /*
            switch (key->kind()) {
          case TypeKind::AnyType:
          case TypeKind::IntType:
          case TypeKind::BoolType:
          case TypeKind::FloatType:
          case TypeKind::ComplexType:
          case TypeKind::StringType:
          case TypeKind::TensorType:
            return DictTypePtr(new DictType(key, value));
          default:
            AT_ERROR(
                "Cannot create dict for key type '",
                key->str(),
                "', only int, float, complex, Tensor and string keys are supported");
        }
        */
    }

    /// aligned with the format in FunctionSchema
    ///
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            stringstream ss;
        ss << "Dict(" << getKeyType()->str() << ", " << getValueType()->str()
           << ")";
        return ss.str();
        */
    }
    
    pub fn create_with_contained(&self, contained_types: Vec<TypePtr>) -> TypePtr {
        
        todo!();
        /*
            if (contained_types.size() != 2) {
          throw runtime_error("Expected 2 contained types");
        }
        return create(contained_types.at(0), contained_types.at(1));
        */
    }
    
    pub fn get_key_type(&self) -> TypePtr {
        
        todo!();
        /*
            return types.at(0);
        */
    }
    
    pub fn get_value_type(&self) -> TypePtr {
        
        todo!();
        /*
            return types.at(1);
        */
    }
    
    pub fn has_free_variables(&self) -> bool {
        
        todo!();
        /*
            return has_free_variables;
        */
    }
    
    pub fn contained_types(&self) -> &[TypePtr] {
        
        todo!();
        /*
            return types;
        */
    }
    
    pub fn new(
        key:   TypePtr,
        value: TypePtr) -> Self {
    
        todo!();
        /*


            : Type(TypeKind::DictType),
            types({key, value}),
            has_free_variables(
                key->hasFreeVariables() || value->hasFreeVariables())
        */
    }
    
    pub fn annotation_str_impl(&self, printer: TypePrinter) -> String {
        let printer: TypePrinter = printer.unwrap_or(nullptr);

        todo!();
        /*
            stringstream ss;
        ss << "Dict[" << getKeyType()->annotation_str(printer) << ", "
           << getValueType()->annotation_str(printer) << "]";
        return ss.str();
        */
    }
}

pub type FutureTypePtr = Arc<FutureType>;

pub struct FutureType {
    base: SingleElementType<TypeKind_FutureType,FutureType>,
}

impl FutureType {
    
    pub fn create<T>(elem: TypePtr) -> FutureTypePtr {
    
        todo!();
        /*
            return FutureTypePtr(
            new FutureType(move(elem))); // NOLINT(modernize-make-shared)
        */
    }
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            stringstream ss;
        ss << "Future(" << getElementType()->str() << ")";
        return ss.str();
        */
    }
    
    pub fn create_with_contained(&self, contained_types: Vec<TypePtr>) -> TypePtr {
        
        todo!();
        /*
            return create(contained_types.at(0));
        */
    }
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
            if (Type::isSubtypeOfExt(rhs, why_not)) {
          return true;
        }
        if (auto rhs_ = rhs->cast<FutureType>()) {
          return getElementType()->isSubtypeOfExt(rhs_->getElementType(), why_not);
        }
        return false;
        */
    }
    
    pub fn new(elem: TypePtr) -> Self {
    
        todo!();
        /*
        : single_element_type(elem),

        
        */
    }
    
    pub fn annotation_str_impl(&self, printer: TypePrinter) -> String {
        let printer: TypePrinter = printer.unwrap_or(nullptr);

        todo!();
        /*
            stringstream ss;
        ss << "Future[" << getElementType()->annotation_str(printer) << "]";
        return ss.str();
        */
    }
}

pub type RRefTypePtr = Arc<RRefType>;

pub struct RRefType {
    base: SingleElementType<TypeKind_RRefType,RRefType>,
}

impl RRefType {
    
    pub fn create<T>(elem: TypePtr) -> RRefTypePtr {
    
        todo!();
        /*
            return RRefTypePtr(
            new RRefType(move(elem))); // NOLINT(modernize-make-shared)
        */
    }
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            stringstream ss;
        ss << "RRef(" << getElementType()->str() << ")";
        return ss.str();
        */
    }
    
    pub fn create_with_contained(&self, contained_types: Vec<TypePtr>) -> TypePtr {
        
        todo!();
        /*
            return create(contained_types.at(0));
        */
    }
    
    pub fn new(elem: TypePtr) -> Self {
    
        todo!();
        /*
        : single_element_type(elem),

        
        */
    }
    
    pub fn annotation_str_impl(&self, printer: TypePrinter) -> String {
        let printer: TypePrinter = printer.unwrap_or(nullptr);

        todo!();
        /*
            stringstream ss;
        ss << "RRef[" << getElementType()->annotation_str(printer) << "]";
        return ss.str();
        */
    }
}

pub type NamedTypePtr      = Arc<NamedType>;
pub type ConstNamedTypePtr = Arc<NamedType>;

pub struct NamedType {
    base: Type,
    name: Option<QualifiedName>,
}

impl NamedType {
    
    pub fn new(
        tk:   TypeKind,
        name: Option<QualifiedName>) -> Self {
    
        todo!();
        /*
        : ty(tk),
        : name(move(name)),

            TORCH_INTERNAL_ASSERT(
            tk == TypeKind::TupleType || tk == TypeKind::FunctionType ||
            tk == TypeKind::ClassType || tk == TypeKind::InterfaceType ||
            tk == TypeKind::EnumType,
            "If you add a new kind of NamedType, ",
            "please update the cast<NamedType> specialization and this assert");
        */
    }

    /**
      | Fully qualified name of type
      |
      | Looks like: "foo.bar.Baz".
      |
      */
    pub fn name(&self) -> &Option<QualifiedName> {
        
        todo!();
        /*
            return name_;
        */
    }
}

/**
  | Any should never appear in a named type like
  | a class, namedtuple or interface. If it does,
  | then dynamic type information will be lost in
  | the Pickler, leading to hard-to-track-down bugs
  | that will only occur after saving or loading
  | a model.
  |
  | This is because we rely on the static types in
  | named types to reconstruct type tags of loaded
  | values. Lifting this restriction requires
  | solving the serialization problem first.
  */
pub fn check_no_any(
        base:     &Type,
        what:     *const u8,
        attrname: &String,
        attrtype: &TypePtr)  {
    
    todo!();
        /*
        
        */
}

pub type TupleTypePtr = Arc<TupleType>;
pub type NameList     = Vec<String>;

/**
  | This type represents a Tuple
  |
  */
pub struct TupleType {
    base:               NamedType,
    elements:           Vec<TypePtr>,
    has_free_variables: bool,
    schema:             Arc<FunctionSchema>,
}

pub mod tuple_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::TupleType;
}

impl TupleType {

    pub fn create_named(
        name:        &Option<QualifiedName>,
        field_names: &Vec<String>,
        types:       &Vec<TypePtr>) -> TupleTypePtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn create(types: Vec<TypePtr>) -> TupleTypePtr {
        
        todo!();
        /*
            return TupleTypePtr(new TupleType(
            move(types),
            nullopt,
            nullptr)); // NOLINT(modernize-make-shared)
        */
    }
    
    pub fn create() -> TupleTypePtr {
        
        todo!();
        /*
            return create({});
        */
    }
    
    pub fn elements(&self) -> &[TypePtr] {
        
        todo!();
        /*
            return elements_;
        */
    }
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
        
        */
    }
    
    pub fn has_free_variables(&self) -> bool {
        
        todo!();
        /*
            return has_free_variables_;
        */
    }
    
    pub fn contained_types(&self) -> &[TypePtr] {
        
        todo!();
        /*
            return elements_;
        */
    }
    
    pub fn create_with_contained(&self, contained_types: Vec<TypePtr>) -> TypePtr {
        
        todo!();
        /*
            return shared_ptr<TupleType>(
            new TupleType(move(contained_types), name(), schema()));
        */
    }
    
    pub fn schema(&self) -> &Arc<FunctionSchema> {
        
        todo!();
        /*
            return schema_;
        */
    }
    
    pub fn new(
        elements: Vec<TypePtr>,
        name:     Option<QualifiedName>,
        schema:   Arc<FunctionSchema>) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn compare(&self, 
        rhs: &Type,
        fn_: fn(a: &Type, b: &Type) -> bool) -> bool {
        
        todo!();
        /*
            if (rhs.kind() != kind()) {
          return false;
        }

        const auto& l_elements = elements();
        const auto& r_elements = rhs.castRaw<TupleType>()->elements();
        if (l_elements.size() != r_elements.size())
          return false;
        for (usize i = 0; i < l_elements.size(); ++i) {
          if (!fn(l_elements[i], r_elements[i]))
            return false;
        }
        return true;
        */
    }
    
    pub fn annotation_str_impl(&self, printer: TypePrinter) -> String {
        let printer: TypePrinter = printer.unwrap_or(nullptr);

        todo!();
        /*
        
        */
    }
}

pub type EnumTypePtr   = Arc<EnumType>;
pub type EnumNameValue = (String,IValue);

pub struct EnumType {
    base:              NamedType,
    value_type:        TypePtr,
    enum_names_values: Vec<EnumNameValue>,
    cu:                Weak<TorchJitCompilationUnit>,
}

pub mod enum_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::EnumType;
}

impl PartialEq<Type> for EnumType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            if (auto enum_rhs = rhs.cast<EnumType>()) {
          return name().value() == enum_rhs->name().value() &&
              *getValueType() == *(enum_rhs->getValueType()) &&
              this->compilation_unit() == enum_rhs->compilation_unit();
        }
        return false;
        */
    }
}

impl EnumType {
    
    pub fn create(
        qualified_class_name: &QualifiedName,
        value:                TypePtr,
        enum_names_values:    Vec<EnumNameValue>,
        cu:                   Weak<TorchJitCompilationUnit>) -> EnumTypePtr {
        
        todo!();
        /*
            switch (value->kind()) {
          case TypeKind::IntType:
          case TypeKind::FloatType:
          case TypeKind::StringType:
            return EnumTypePtr(new EnumType(qualified_class_name, move(value), move(enum_names_values), move(cu)));
          default:
            AT_ERROR(
                "Cannot create Enum with value type '",
                value->str(),
                "', only int, float and string are supported");
        }
        */
    }
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "Enum<" + annotation_str() + ">";
        */
    }
    
    pub fn repr_str(&self) -> String {
        
        todo!();
        /*
            return str();
        */
    }
    
    pub fn get_value_type(&self) -> TypePtr {
        
        todo!();
        /*
            return value_type_;
        */
    }
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn compilation_unit(&self) -> Arc<TorchJitCompilationUnit> {
        
        todo!();
        /*
            auto cu = cu_.lock();
        return cu;
        */
    }
    
    pub fn qualified_class_name(&self) -> QualifiedName {
        
        todo!();
        /*
            return name().value();
        */
    }
    
    pub fn contained_types(&self) -> &[TypePtr] {
        
        todo!();
        /*
            return value_type_;
        */
    }
    
    pub fn enum_names_values(&self) -> &[EnumNameValue] {
        
        todo!();
        /*
            return enum_names_values_;
        */
    }
    
    pub fn new(
        qualified_class_name: QualifiedName,
        value_type:           TypePtr,
        enum_names_values:    Vec<EnumNameValue>,
        cu:                   Weak<TorchJitCompilationUnit>) -> Self {
    
        todo!();
        /*
        : named_type(TypeKind::EnumType, move(qualified_class_name)),
        : value_type(move(value_type)),
        : enum_names_values(move(enum_names_values)),
        : cu(cu),

        
        */
    }
    
    pub fn annotation_str_impl(&self, printer: TypePrinter) -> String {
        let printer: TypePrinter = printer.unwrap_or(nullptr);

        todo!();
        /*
            const auto& n = name().value();
        return n.qualifiedName();
        */
    }
}

// the common supertype of all Enums, only used in
// operator registraion. EnumType <: AnyEnumType
// for all Enums
//
pub type AnyEnumTypePtr = Arc<AnyEnumType>;

pub struct AnyEnumType {
    base: Type,
}

pub mod any_enum_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::AnyEnumType;
}

impl PartialEq<Type> for AnyEnumType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

impl Default for AnyEnumType {
    
    fn default() -> Self {
        todo!();
        /*
        : ty(TypeKind::AnyEnumType),

        
        */
    }
}

impl AnyEnumType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "AnyEnumType";
        */
    }

    /// global singleton
    ///
    pub fn get() -> AnyEnumTypePtr {
        
        todo!();
        /*
        
        */
    }
}

pub type NumberTypePtr = Arc<NumberType>;

/**
  | This type represents a Python number
  |
  | Subtype hierarchy for Number Types (NumberType
  | as the base type):
  |
  | IntType <: NumberType
  |
  | FloatType <: NumberType
  |
  | ComplexType <:NumberType
  |
  */
pub struct NumberType {
    base: Type,
}

impl PartialEq<Type> for NumberType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

pub mod number_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::NumberType;
}

impl NumberType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "Scalar"; // match what PythonArgParser says for clarity
        */
    }

    /// global singleton
    ///
    pub fn get() -> NumberTypePtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new(kind: TypeKind) -> Self {
        let kind: TypeKind = kind.unwrap_or(TypeKind_NumberType);
        todo!();
        /*
        : ty(kind),

        
        */
    }
    
    pub fn annotation_str_impl(&self, printer: TypePrinter) -> String {
        let printer: TypePrinter = printer.unwrap_or(nullptr);

        todo!();
        /*
            return "number"; // technically not a valid python type, but
                         // we need to use it when parsing back in annotations
                         // for implicit conversions
        */
    }
}

pub type FloatTypePtr = Arc<FloatType>;

/**
  | This type represents a Python float
  | number
  |
  */
pub struct FloatType {
    base: NumberType,
}

pub mod float_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::FloatType;
}

impl PartialEq<Type> for FloatType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

impl Default for FloatType {
    
    fn default() -> Self {
        todo!();
        /*
        : number_type(TypeKind::FloatType),

        
        */
    }
}

impl FloatType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "float";
        */
    }
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
            return rhs->kind() == TypeKind::NumberType || NumberType::isSubtypeOfExt(rhs, why_not);
        */
    }

    /// global singleton
    ///
    pub fn get() -> FloatTypePtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn annotation_str_impl(&self, printer: TypePrinter) -> String {
        let printer: TypePrinter = printer.unwrap_or(nullptr);

        todo!();
        /*
            return "float";
        */
    }
}

pub type ComplexTypePtr = Arc<ComplexType>;

/**
  | This type represents a Python float
  | number
  |
  */
pub struct ComplexType {
    base: NumberType,
}

pub mod complex_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::ComplexType;
}

impl PartialEq<Type> for ComplexType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

impl Default for ComplexType {
    
    fn default() -> Self {
        todo!();
        /*
        : number_type(TypeKind::ComplexType),

        
        */
    }
}

impl ComplexType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "complex";
        */
    }
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
            return rhs->kind() == TypeKind::NumberType || NumberType::isSubtypeOfExt(rhs, why_not);
        */
    }

    /// global singleton
    pub fn get() -> ComplexTypePtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn annotation_str_impl(&self, printer: TypePrinter) -> String {
        let printer: TypePrinter = printer.unwrap_or(nullptr);

        todo!();
        /*
            return "complex";
        */
    }
}

pub type IntTypePtr = Arc<IntType>;

/**
  | This type represents a Python int number
  |
  */
pub struct IntType {
    base: NumberType,
}

impl PartialEq<Type> for IntType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

pub mod int_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::IntType;
}

impl Default for IntType {
    
    fn default() -> Self {
        todo!();
        /*
        : number_type(TypeKind::IntType),

        
        */
    }
}

impl IntType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "int";
        */
    }
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
            return rhs->kind() == TypeKind::NumberType || NumberType::isSubtypeOfExt(rhs, why_not);
        */
    }

    /// global singleton
    ///
    pub fn get() -> IntTypePtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn annotation_str_impl(&self, printer: TypePrinter) -> String {
        let printer: TypePrinter = printer.unwrap_or(nullptr);

        todo!();
        /*
            return "int";
        */
    }
}

pub type BoolTypePtr = Arc<BoolType>;

/**
  | This node represents a Python bool value
  |
  */
pub struct BoolType {
    base: Type,
}

impl PartialEq<Type> for BoolType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

impl Default for BoolType {
    
    fn default() -> Self {
        todo!();
        /*
        : ty(TypeKind::BoolType),

        
        */
    }
}

pub mod bool_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::BoolType;
}

impl BoolType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "bool";
        */
    }

    /// global singleton
    ///
    pub fn get() -> BoolTypePtr {
        
        todo!();
        /*
        
        */
    }
}

pub type StringTypePtr = Arc<StringType>;

/// This type represents a Python string
pub struct StringType {
    base: Type,
}

impl PartialEq<Type> for StringType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

pub mod string_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::StringType;
}

impl Default for StringType {
    
    fn default() -> Self {
        todo!();
        /*
        : ty(TypeKind::StringType),

        
        */
    }
}

impl StringType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            // we only use "str" (not "string") in both FunctionSchema and script
        return annotation_str();
        */
    }
    
    pub fn annotation_str_impl(&self, printer: TypePrinter) -> String {
        let printer: TypePrinter = printer.unwrap_or(nullptr);

        todo!();
        /*
            return "str";
        */
    }

    /// global singleton
    ///
    pub fn get() -> StringTypePtr {
        
        todo!();
        /*
        
        */
    }
}

pub type StorageTypePtr = Arc<StorageType>;

pub struct StorageType {
    base: Type,
}

impl PartialEq<Type> for StorageType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

pub mod storage_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::StorageType;
}

impl Default for StorageType {
    
    fn default() -> Self {
        todo!();
        /*
        : ty(TypeKind::StorageType),

        
        */
    }
}

impl StorageType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return annotation_str();
        */
    }
    
    pub fn annotation_str_impl(&self, printer: TypePrinter) -> String {
        let printer: TypePrinter = printer.unwrap_or(nullptr);

        todo!();
        /*
            return "Storage";
        */
    }

    /// global singleton
    ///
    pub fn get() -> StorageTypePtr {
        
        todo!();
        /*
        
        */
    }
}

pub type FunctionTypePtr = Arc<FunctionType>;

pub struct FunctionType {
    base:     NamedType,
    function: *mut TorchJitFunction,
}

impl PartialEq<Type> for FunctionType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            if (auto func_type = rhs.cast<FunctionType>()) {
          return func_type->function_ == function_;
        }

        return false;
        */
    }
}

pub mod function_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::FunctionType;
}

impl FunctionType {
    
    pub fn create(function: *mut TorchJitFunction) -> FunctionTypePtr {
        
        todo!();
        /*
            return FunctionTypePtr(
            new FunctionType(function)); // NOLINT(modernize-make-shared)
        */
    }
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "Function";
        */
    }
    
    pub fn function(&self) -> *mut TorchJitFunction {
        
        todo!();
        /*
            return function_;
        */
    }
    
    pub fn new(function: *mut TorchJitFunction) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn annotation_str_impl(&self, printer: TypePrinter) -> String {
        let printer: TypePrinter = printer.unwrap_or(nullptr);

        todo!();
        /*
            const auto& n = name().value();
        return n.qualifiedName();
        */
    }
}

pub type NoneTypePtr = Arc<NoneType>;

/**
  | This type represents a Python None
  |
  */
pub struct NoneType {
    base: Type,
}

pub mod none_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::NoneType;
}

impl PartialEq<Type> for NoneType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

impl Default for NoneType {
    
    fn default() -> Self {
        todo!();
        /*
        : ty(TypeKind::NoneType),

        
        */
    }
}

impl NoneType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "NoneType";
        */
    }
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
            if (rhs->kind() == OptionalType::Kind) {
          return true;
        }
        return Type::isSubtypeOfExt(rhs, why_not);
        */
    }

    /// global singleton
    ///
    pub fn get() -> NoneTypePtr {
        
        todo!();
        /*
        
        */
    }
}

pub type GeneratorTypePtr = Arc<GeneratorType>;

/**
  | This type represents a Generator
  |
  */
pub struct GeneratorType {
    base: Type,
}

pub mod generator_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::GeneratorType;
}

impl PartialEq<Type> for GeneratorType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

impl Default for GeneratorType {
    
    fn default() -> Self {
        todo!();
        /*
        : ty(TypeKind::GeneratorType),

        
        */
    }
}

impl GeneratorType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "Generator";
        */
    }

    /// global singleton
    pub fn get() -> GeneratorTypePtr {
        
        todo!();
        /*
        
        */
    }
}

pub type QuantizerTypePtr = Arc<QuantizerType>;

/**
  | This type represents a Quantizer
  |
  */
pub struct QuantizerType {
    base: Type,
}

impl PartialEq<Type> for QuantizerType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

pub mod quantizer_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::QuantizerType;
}

impl Default for QuantizerType {
    
    fn default() -> Self {
        todo!();
        /*
        : ty(TypeKind::QuantizerType),

        
        */
    }
}

impl QuantizerType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "Quantizer";
        */
    }

    /// global singleton
    ///
    pub fn get() -> QuantizerTypePtr {
        
        todo!();
        /*
        
        */
    }
}

pub type QSchemeTypePtr = Arc<QSchemeType>;

/**
  | This type represents a QScheme
  |
  */
pub struct QSchemeType {
    base: Type,
}

impl PartialEq<Type> for QSchemeType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

pub mod qscheme_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::QSchemeType;
}

impl Default for QSchemeType {
    
    fn default() -> Self {
        todo!();
        /*
        : ty(TypeKind::QSchemeType),

        
        */
    }
}

impl QSchemeType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "QScheme";
        */
    }

    /// global singleton
    pub fn get() -> QSchemeTypePtr {
        
        todo!();
        /*
        
        */
    }
}

pub type DeviceObjTypePtr = Arc<DeviceObjType>;

/**
  | This type represents a Device
  |
  */
pub struct DeviceObjType {
    base: Type,
}

pub mod device_obj_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::DeviceObjType;
}

impl PartialEq<Type> for DeviceObjType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

impl Default for DeviceObjType {
    
    fn default() -> Self {
        todo!();
        /*
        : ty(TypeKind::DeviceObjType),

        
        */
    }
}

impl DeviceObjType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "Device";
        */
    }

    /// global singleton
    ///
    pub fn get() -> DeviceObjTypePtr {
        
        todo!();
        /*
        
        */
    }
}

pub type StreamObjTypePtr = Arc<StreamObjType>;

/**
  | This type represents a Generator
  |
  */
pub struct StreamObjType {
    base: Type,
}

impl PartialEq<Type> for StreamObjType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

pub mod stream_obj_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::StreamObjType;
}

impl Default for StreamObjType {
    
    fn default() -> Self {
        todo!();
        /*
        : ty(TypeKind::StreamObjType),

        
        */
    }
}

impl StreamObjType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "Stream";
        */
    }

    /// global singleton
    pub fn get() -> StreamObjTypePtr {
        
        todo!();
        /*
        
        */
    }
}

pub type VarTypePtr = Arc<VarType>;

/**
  | This type represents a type variable,
  | used in
  | FunctionSchema
  |
  */
pub struct VarType {
    base: Type,
    name: String,
}

impl PartialEq<Type> for VarType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

pub mod var_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::VarType;
}

impl VarType {

    pub fn create(name: String) -> VarTypePtr {
        
        todo!();
        /*
            return VarTypePtr(new VarType(move(name_)));
        */
    }
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return name();
        */
    }
    
    pub fn name(&self) -> &String {
        
        todo!();
        /*
            return name_;
        */
    }
    
    pub fn has_free_variables(&self) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    pub fn new(name: String) -> Self {
    
        todo!();
        /*
        : ty(TypeKind::VarType),
        : name(move(name_)),

        
        */
    }
}

pub type CapsuleTypePtr = Arc<CapsuleType>;

/**
  | This type represents a Python Capsule.
  |
  | It does not appear in the IR and is only used
  | during runtime
  |
  */
pub struct CapsuleType {
    base: Type,
}

impl PartialEq<Type> for CapsuleType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

pub mod capsule_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::CapsuleType;
}

impl Default for CapsuleType {
    
    fn default() -> Self {
        todo!();
        /*
        : ty(TypeKind::CapsuleType),

        
        */
    }
}

impl CapsuleType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "Capsule";
        */
    }

    /// global singleton
    pub fn get() -> CapsuleTypePtr {
        
        todo!();
        /*
        
        */
    }
}

pub type PyObjectTypePtr = Arc<PyObjectType>;

/**
  | This type represents a PyObject Type
  |
  */
pub struct PyObjectType {
    base: Type,
}

impl PartialEq<Type> for PyObjectType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

impl Default for PyObjectType {
    
    fn default() -> Self {
        todo!();
        /*
        : ty(TypeKind::PyObjectType),

        
        */
    }
}

pub mod py_object_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::PyObjectType;
}

impl PyObjectType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "PyObject";
        */
    }

    /// global singleton
    pub fn get() -> PyObjectTypePtr {
        
        todo!();
        /*
        
        */
    }
}

pub enum TypeVerbosity {
    None,
    Type,
    TypeAndStride,
    Full,
    Symbolic,
    Default = Full,
}

pub fn type_verbosity() -> TypeVerbosity {
    
    todo!();
        /*
        
        */
}

impl fmt::Display for SymbolicShape {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
        
        */
    }
}

impl fmt::Display for ShapeSymbol {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
        
        */
    }
}

impl fmt::Display for Stride {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
        
        */
    }
}

/*
  | what is the type, ignoring extra size/shape
  | information?
  |
  | e.g. Tensor(2x3) -> Dynamic, and
  | Tuple(Tensor(2x3),...) -> Tuple(Dynamic,...)
  */

/**
  | xxx: be careful with calls because this
  | can be very slow. If calling this on a
  | graph use `EraseShapeInformation`
  | in shape_analysis.h
  |
  */
#[inline] pub fn unshaped_type(ty: &TypePtr) -> TypePtr {
    
    todo!();
        /*
            if (type->isSubtypeOf(TensorType::get())) {
        return TensorType::get();
      }
      return type->withContained(fmap(type->containedTypes(), unshapedType));
        */
}

impl TensorType {
    
    #[inline] pub fn from_number_type(&mut self, typ: TypePtr) -> TypePtr {
        
        todo!();
        /*
            if (typ->isSubtypeOf(IntType::get())) {
        return TensorType::createContiguous(kLong, kCPU, {});
      } else if (typ->isSubtypeOf(FloatType::get())) {
        return TensorType::createContiguous(kDouble, kCPU, {});
      } else if (typ->isSubtypeOf(BoolType::get())) {
        return TensorType::createContiguous(kBool, kCPU, {});
      } else if (typ->kind() == NumberType::Kind) {
        return TensorType::create(nullopt, kCPU, {}, nullopt);
      }
      TORCH_CHECK(false, "Unknown number type: ", typ->str());
        */
    }
    
    #[inline] pub fn from_bool_type(&mut self) -> TypePtr {
        
        todo!();
        /*
            return TensorType::createContiguous(kBool, kCPU, {});
        */
    }
}

#[inline] pub fn try_scalar_type_from_jit_type(ty: &TypePtr) -> Option<ScalarType> {
    
    todo!();
        /*
            if (type == FloatType::get()) {
        return typeMetaToScalarType(get_default_dtype());
      } else if (type == IntType::get()) {
        return ScalarType::Long;
      } else if (type == BoolType::get()) {
        return ScalarType::Bool;
      }
      return nullopt;
        */
}

#[inline] pub fn scalar_type_from_jit_type(ty: &TypePtr) -> ScalarType {
    
    todo!();
        /*
            auto result = tryScalarTypeFromJitType(type);
      TORCH_CHECK(
          result,
          "Add new condition, expected Float, Complex, Int, or Bool but got",
          type->str());
      return *result;
        */
}

/**
  | Attempt to find the correct supertype of t1 and
  | t2.
  |
  | If none is found then nullopt will be returned
  | if default_to_any is false, and Any will be
  | returned if it is true.
  |
  | If t1 == t2, or t1 is a type refinement of t2,
  | then t2 will be returned (and vice versa).
  |
  | Two different tensortypes will return dynamic.
  |
  | Currently we chose not to support returning
  | a NumberType for a float & int input because of
  | a lack of operator support for NumberType
  |
  */
pub fn unify_types(
        t1:             &TypePtr,
        t2:             &TypePtr,
        default_to_any: bool) -> Option<TypePtr> {
    let default_to_any: bool = default_to_any.unwrap_or(false);

    todo!();
        /*
        
        */
}

pub fn unify_type_list(
        elements: &[TypePtr],
        why_not:  &mut std::io::BufWriter) -> Option<TypePtr> {
    
    todo!();
        /*
        
        */
}

pub struct GetTypePtr<T> {

}

impl GetTypePtr<T> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            TypePtr res = []() {
          try {
            return getCustomClassType<T>();
          } catch(const Error&) {
            TORCH_CHECK(
                false,
                "Type ",
                util::get_fully_qualified_type_name<T>(),
                " could not be converted to any of the known types."
            );
          }
        }();
        return dynamic_pointer_cast<Type>(move(res));
        */
    }
}

impl GetTypePtr_<IValue> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return AnyType::get();
        */
    }
}

impl GetTypePtr<Tensor> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return TensorType::get();
        */
    }
}

impl GetTypePtr<Storage> {

    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return StorageType::get();
        */
    }
}

impl GetTypePtr<Stream> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return StreamObjType::get();
        */
    }
}

impl GetTypePtr<double> {

    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return FloatType::get();
        */
    }
}

impl GetTypePtr<complex<double>> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return ComplexType::get();
        */
    }
}

impl GetTypePtr<i64> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return IntType::get();
        */
    }
}

impl GetTypePtr<ScalarType> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return IntType::get();
        */
    }
}

impl GetTypePtr<Device> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return DeviceObjType::get();
        */
    }
}


impl GetTypePtr<Layout> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return IntType::get();
        */
    }
}

impl GetTypePtr<MemoryFormat> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return IntType::get();
        */
    }
}

impl GetTypePtr<bool> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return BoolType::get();
        */
    }
}

impl GetTypePtr<Scalar> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return NumberType::get();
        */
    }
}

impl GetTypePtr<QScheme> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return QSchemeType::get();
        */
    }
}

impl GetTypePtr<Generator> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return OptionalType::create(GeneratorType::get());
        */
    }
}

impl GetTypePtr<string> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return StringType::get();
        */
    }
}

impl GetTypePtr<string_view> {

    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return StringType::get();
        */
    }
}

impl GetTypePtr<Dimname> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return StringType::get();
        */
    }
}

impl GetTypePtr<Vec<T>> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            static auto type = ListType::create(getTypePtr_<T>::call());
        return type;
        */
    }
}

impl GetTypePtr<&[T]> {

    pub fn call() -> TypePtr {
        
        todo!();
        /*
            static auto type = ListType::create(getTypePtr_<T>::call());
        return type;
        */
    }
}

impl GetTypePtr<List<T>> {

    pub fn call() -> TypePtr {
        
        todo!();
        /*
            static auto type = ListType::create(getTypePtr_<T>::call());
        return type;
        */
    }
}

impl GetTypePtr<array<T, N>> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            static auto type = ListType::create(getTypePtr_<T>::call());
        return type;
        */
    }
}

impl GetTypePtr<unordered_map<K, V>> {

    pub fn call() -> TypePtr {
        
        todo!();
        /*
            static auto type =
            DictType::create(getTypePtr_<K>::call(), getTypePtr_<V>::call());
        return type;
        */
    }
}

impl GetTypePtr<Dict<K, V>> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            static auto type =
            DictType::create(getTypePtr_<K>::call(), getTypePtr_<V>::call());
        return type;
        */
    }
}

impl GetTypePtr<optional<T>> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            static auto type = OptionalType::create(getTypePtr_<T>::call());
        return type;
        */
    }
}

impl GetTypePtr<tuple<Contained>> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            vector<TypePtr> contained_types = {
          (getTypePtr_<Contained>::call())...
        };
        return TupleType::create(move(contained_types));
        */
    }
}

impl GetTypePtr<void> {
    
    pub fn call() -> TypePtr {
        
        todo!();
        /*
            return NoneType::get();
        */
    }
}

#[inline] pub fn get_type_ptr<T>() -> TypePtr {

    todo!();
        /*
            // TODO: static_assert that a templated function exists, and throw a friendly
      // error message if not
      return getTypePtr_<T>::call();
        */
}

pub type TypeEnv = HashMap<String,TypePtr>;

pub struct MatchTypeReturn {

    /**
      | is there is no match, this contains the
      | reason
      |
      */
    reason: Option<String>,
}

impl Default for MatchTypeReturn {
    
    fn default() -> Self {
        todo!();
        /*
        : reason(nullopt),

        
        */
    }
}

impl MatchTypeReturn {

    pub fn new(reason: String) -> Self {
    
        todo!();
        /*
        : reason(move(reason)),

        
        */
    }
    
    pub fn success() -> MatchTypeReturn {
        
        todo!();
        /*
            return MatchTypeReturn();
        */
    }
    
    pub fn success(&self) -> bool {
        
        todo!();
        /*
            return !reason_.has_value();
        */
    }
    
    pub fn reason(&self) -> &String {
        
        todo!();
        /*
            return reason_.value();
        */
    }
}

/**
  | attempt to match the type variables in formal
  | to actual, adding them to type_env.
  |
  | If no match is possible this returns
  | a MatchTypeReturn with r.success() == false and
  | a r.reason() that describes why it could not
  | match.
  |
  | note: It is possible to successfully match
  | a formal, but for type variables in the formal
  | to still not be defined. In particular, None
  | matches Optional[T] but does not define the
  | value of T.
  |
  */
pub fn match_type_variables(
        formal:   TypePtr,
        actual:   TypePtr,
        type_env: &mut TypeEnv) -> MatchTypeReturn {
    
    todo!();
        /*
        
        */
}

/**
  | replace type variables appearing in `type` with
  | the values in `type_env`. Returns nullptr if
  | a variable used in `type` does not appear in
  | `type_env`
  |
  */
pub fn try_eval_type_variables(
        ty:       TypePtr,
        type_env: &mut TypeEnv) -> TypePtr {
    
    todo!();
        /*
        
        */
}

pub fn element_type_can_be_inferred_from_members(elem_type: &TypePtr) -> bool {
    
    todo!();
        /*
        
        */
}

/**
  | This enumerator represents the 'kind' of an
  | attribute - a buffer, a paramter, or neither.
  |
  | This state is mutually exclusive. Buffers and
  | Parameters can only appear on modules.
  |
  */
pub enum AttributeKind {
    BUFFER,
    PARAMETER,
    REGULAR_ATTRIBUTE
}

/**
  | This structure represents all notional booking
  | entities in a class attribute: name, kind (see:
  | AttributeKind), and type (see: TypePtr).
  |
  | Note: This structure does not represent the
  | value of the attribute.
  |
  */
pub struct ClassAttribute {
    kind:           AttributeKind,
    attribute_type: TypePtr,
    attribute_name: String,
}

impl ClassAttribute {
    
    pub fn new(
        kind:           AttributeKind,
        attribute_type: TypePtr,
        attribute_name: String) -> Self {
    
        todo!();
        /*
        : kind(kind),
        : attribute_type(attributeType),
        : attribute_name(move(attributeName)),

        
        */
    }
    
    pub fn get_kind(&self) -> AttributeKind {
        
        todo!();
        /*
            return kind_;
        */
    }
    
    pub fn get_type(&self) -> TypePtr {
        
        todo!();
        /*
            return attributeType_;
        */
    }
    
    pub fn get_name(&self) -> &String {
        
        todo!();
        /*
            return attributeName_;
        */
    }
}
  
/* -------------- User Defined Types  -------------- */

/**
  | This represents an attribute of a class;
  | a name associated with an attribute, and
  | a getter and (optional) setter for that
  | attribute.
  |
  */
pub struct ClassTypeProperty {
    name:   String,
    getter: *mut TorchJitFunction,
    setter: *mut TorchJitFunction,
}

pub type ClassTypePtr = Arc<ClassType>;

/**
  | This represents a class in TorchScript.
  |
  */
pub struct ClassType {

    base: NamedType,

    /**
      | Mapping of constant names -> their value.
      |
      */
    constant_names:              Vec<String>,
    constant_values:             Vec<IValue>,

    /**
      | Holds method attributes
      |
      */
    compilation_unit:            Weak<CompilationUnit>,


    /**
      | Holds all atrributes, attribute details
      | are found on ClassAttribute
      |
      */
    attributes:                  Vec<ClassAttribute>,


    /**
      | Construct mirroring attributes_,
      | only around due to the fact that `containedTypes()`
      | method returns an ArrayRef.
      | 
      | Never fill this without using the appropriate
      | provideNewClassAttribute method
      |
      */
    attribute_types:             Vec<TypePtr>,


    /**
      | List of methods associated with this
      | class.
      |
      */
    methods:                     Vec<*mut TorchJitFunction>,

    staticmethods:               Vec<*mut TorchJitFunction>,

    /**
      | List of hooks to be run before/after
      | forward.
      |
      */
    forward_hooks:               Vec<*mut TorchJitFunction>,

    forward_pre_hooks:           Vec<*mut TorchJitFunction>,

    /**
      | List of properties exposed by this class.
      |
      */
    properties:                  Vec<Property>,

    is_module:                   bool, // default = false

    /**
      | Doc string of class.
      |
      */
    doc_string:                  String, // default = ""


    /**
      | For error reporting accesses to class
      | level attributes.
      |
      */
    unresolved_class_attributes: Vec<String>,
}

impl PartialEq<Type> for ClassType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            if (auto user_rhs = rhs.cast<ClassType>()) {
          const auto& lhs_name = name().value();
          const auto& rhs_name = user_rhs->name().value();

          return lhs_name == rhs_name &&
              this->compilation_unit() == user_rhs->compilation_unit();
        }
        return false;
        */
    }
}

pub mod class_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::ClassType;
}

impl ClassType {

    /**
      | Create a class type with name `name`
      | and its methods stored in `cu`.
      |
      */
    pub fn create(
        qualified_name:              Option<QualifiedName>,
        cu:                          Weak<CompilationUnit>,
        is_module:                   bool,
        doc_string:                  String,
        unresolved_class_attributes: Vec<String>) -> ClassTypePtr {

        let is_module: bool = is_module.unwrap_or(false);
        let doc_string: String = doc_string.unwrap_or("");

        todo!();
        /*
        
        */
    }
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return annotation_str();
        */
    }
    
    pub fn repr_str(&self) -> String {
        
        todo!();
        /*
            stringstream ss;
        ss << str()
           << " (of Python compilation unit at: " << compilation_unit().get() << ")";
        return ss.str();
        */
    }
    
    pub fn methods(&self) -> &Vec<*mut TorchJitFunction> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn find_attribute(&self, name: &String) -> TypePtr {
        
        todo!();
        /*
            usize pos = 0;
        for (const auto& attr : attributes_) {
          if (name == attr.getName()) {
            break;
          }
          ++pos;
        }

        if (pos >= attributes_.size()) {
          return nullptr;
        }
        return attributes_[pos].getType();
        */
    }
    
    pub fn get_attribute(&self, name: &String) -> TypePtr {
        
        todo!();
        /*
            auto type = findAttribute(name);
        TORCH_CHECK(
            type,
            repr_str(),
            " does not have an attribute with name '",
            name,
            "'");
        return type;
        */
    }
    
    pub fn num_attributes(&self) -> usize {
        
        todo!();
        /*
            return attributes_.size();
        */
    }
    
    pub fn get_attribute(&self, slot: usize) -> TypePtr {
        
        todo!();
        /*
            AT_ASSERT(slot < attributes_.size());
        return attributes_.at(slot).getType();
        */
    }
    
    pub fn get_attribute_name(&self, slot: usize) -> String {
        
        todo!();
        /*
            AT_ASSERT(slot < attributes_.size());
        return attributes_[slot].getName();
        */
    }
    
    pub fn check_not_exist(&self, 
        name: &String,
        what: &String)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Attributes are stored in a specific slot at
      | runtime for effiency.
      |
      | When emitting instructions we specify the
      | slot so that attribute access is a constant
      | lookup
      */
    pub fn find_attribute_slot(&self, name: &String) -> Option<usize> {
        
        todo!();
        /*
            usize slot = 0;
        for (const auto& attr : attributes_) {
          if (name.compare(attr.getName()) == 0) {
            return slot;
          }
          slot++;
        }
        return nullopt;
        */
    }
    
    pub fn get_attribute_slot(&self, name: &String) -> usize {
        
        todo!();
        /*
            if (auto r = findAttributeSlot(name)) {
          return *r;
        }
        TORCH_CHECK(
            false,
            repr_str(),
            " does not have an attribute with name '",
            name,
            "'");
        */
    }
    
    pub fn has_attribute(&self, name: &String) -> bool {
        
        todo!();
        /*
            return find_if(
                   attributes_.cbegin(),
                   attributes_.cend(),
                   [&](const ClassAttribute& attr) { return attr.getName() == name; }) !=
            attributes_.cend();
        */
    }
    
    pub fn is_unresolved_class_attribute(&self, name: &String) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn contained_types(&self) -> &[TypePtr] {
        
        todo!();
        /*
            return attributeTypes_;
        */
    }
    
    pub fn add_attribute(&mut self, 
        name:         &String,
        ty:           &TypePtr,
        is_parameter: bool,
        is_buffer:    bool) -> usize {

        let is_parameter: bool = is_parameter.unwrap_or(false);
        let is_buffer:    bool = is_buffer.unwrap_or(false);

        todo!();
        /*
        
        */
    }

    /**
      | [Internal Only] Remove attribute from the
      | ClassType, caller is responsible to make
      | sure the modification is safe:
      |
      | it is unsafe to having existing allocations
      | of this object around anymore, and any code
      | that works on the attribute is now
      | invalid. Only newly created code is valid
      | again.
      */
    pub fn unsafe_remove_attribute(&mut self, name: &String)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | [Internal Only] Change the type of an
      | attribute of the ClassType,
      |
      | The caller is responsible to make sure the
      | modification is safe: it is unsafe to
      | maintain uses of the old type of the
      | attribute, and any code that works on the
      | attribute is now invalid.
      |
      | Only newly created code is valid again.
      */
    pub fn unsafe_change_attribute_type(&mut self, 
        name:   &String,
        new_ty: TypePtr)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Add attribute \p NAME if it doesn't exist or
      | verify that it has a compatible type
      | otherwise.
      |
      */
    pub fn add_or_check_attribute(&mut self, 
        name:         &String,
        ty:           TypePtr,
        is_parameter: bool,
        is_buffer:    bool) -> usize {

        let is_parameter: bool = is_parameter.unwrap_or(false);
        let is_buffer:    bool = is_buffer.unwrap_or(false);

        todo!();
        /*
            auto slot_idx = findAttributeSlot(name);
        if (!slot_idx) {
          return addAttribute(name, ty, is_parameter, is_buffer);
        }

        TORCH_CHECK(
            is_parameter == this->is_parameter(*slot_idx),
            "Parameter field mismatch for the field '",
            name,
            "'");
        TypePtr atype = getAttribute(*slot_idx);
        TORCH_CHECK(
          ty->isSubtypeOf(atype),
          ty->repr_str(),
          " is not compatible with the type ",
          atype->repr_str(),
          " for the field '",
          name,
          "'");
        return *slot_idx;
        */
    }

    /**
      | Get the property with the given \p name,
      | if it exists on the class.
      |
      */
    pub fn get_property(&mut self, name: &String) -> Option<ClassType_Property> {
        
        todo!();
        /*
        
        */
    }

    /**
      | Add a property named \p name with \p getter
      | and \p setter as its getter and setter.
      |
      */
    pub fn add_property(&mut self, 
        name:   &String,
        getter: *mut TorchJitFunction,
        setter: *mut TorchJitFunction)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn properties(&self) -> Vec<Property> {
        
        todo!();
        /*
            return properties_;
        */
    }
    
    pub fn has_constant(&self, name: &String) -> bool {
        
        todo!();
        /*
            return find_if(
                   constantNames_.cbegin(),
                   constantNames_.cend(),
                   [&](const string& constant) { return constant == name; }) !=
            constantNames_.cend();
        */
    }
    
    pub fn add_constant(&mut self, 
        name:  &String,
        value: &IValue) -> usize {
        
        todo!();
        /*
        
        */
    }
    
    pub fn find_constant_slot(&self, name: &String) -> Option<usize> {
        
        todo!();
        /*
            TORCH_CHECK(constantNames_.size() == constantValues_.size());
        usize slot = 0;
        for (const auto& constant : constantNames_) {
          if (name == constant) {
            return slot;
          }
          slot++;
        }
        return nullopt;
        */
    }
    
    pub fn get_constant_slot(&self, name: &String) -> usize {
        
        todo!();
        /*
            if (auto r = findConstantSlot(name)) {
          return *r;
        }
        TORCH_CHECK(
            false,
            repr_str(),
            " does not have constant field with the name '",
            name,
            "'");
        */
    }
    
    pub fn get_constant_name(&self, slot: usize) -> &String {
        
        todo!();
        /*
            TORCH_CHECK(constantNames_.size() == constantValues_.size());
        TORCH_CHECK(slot < constantNames_.size());
        return constantNames_[slot];
        */
    }
    
    pub fn doc_string(&self) -> &String {
        
        todo!();
        /*
            return doc_string_;
        */
    }
    
    pub fn get_constant(&self, name: &String) -> IValue {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_constant(&self, slot: usize) -> IValue {
        
        todo!();
        /*
        
        */
    }
    
    pub fn find_constant(&self, name: &String) -> Option<IValue> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn num_constants(&self) -> usize {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(constantNames_.size() == constantValues_.size());
        return constantNames_.size();
        */
    }
    
    pub fn constant_names(&self) -> &[String] {
        
        todo!();
        /*
            return constantNames_;
        */
    }
    
    pub fn constant_values(&self) -> &[IValue] {
        
        todo!();
        /*
            return constantValues_;
        */
    }

    /**
      | [Internal Only] Remove constant from the
      | ClassType caller is responsible to make sure
      | the modification is safe: it is unsafe to
      | having existing allocations of this object
      | around anymore, and any code that works on
      | the attribute is now invalid. Only newly
      | created code is valid again.
      */
    pub fn unsafe_remove_constant(&mut self, name: &String)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn create_with_contained(&self, contained_types: Vec<TypePtr>) -> TypePtr {
        
        todo!();
        /*
            auto ptr = ClassType::create(name(), compilation_unit_, is_module());
        AT_ASSERT(numAttributes() == contained_types.size());
        for(usize i = 0; i < attributes_.size(); ++i) {
          AT_ASSERT(attributes_[i].getType()->isSubtypeOf(contained_types[i]));
          ptr->addAttribute(attributes_[i].getName(), contained_types[i]);
        }
        // Copy methods over
        for (const auto& method : methods()) {
          ptr->addMethod(method);
        }
        return ptr;
        */
    }
    
    pub fn is_module(&self) -> bool {
        
        todo!();
        /*
            return isModule_;
        */
    }
    
    pub fn get_attributes(&self) -> &Vec<ClassAttribute> {
        
        todo!();
        /*
            return attributes_;
        */
    }
    
    pub fn is_parameter(&self, slot: usize) -> bool {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(
            is_module(), "asking for parameterSlots of non-Module");
        return attributes_.at(slot).getKind() == AttributeKind::PARAMETER;
        */
    }
    
    pub fn is_buffer(&self, slot: usize) -> bool {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(
            is_module(), "asking for bufferWrittenSlots of non-Module");
        return attributes_.at(slot).getKind() == AttributeKind::BUFFER;
        */
    }
    
    pub fn add_forward_pre_hook(&mut self, pre_hook_ptr: *mut TorchJitFunction)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn add_forward_hook(&mut self, hook_ptr: *mut TorchJitFunction)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn find_forward_pre_hook(&self, name: &String) -> *mut TorchJitFunction {
        
        todo!();
        /*
        
        */
    }
    
    pub fn find_forward_hook(&self, name: &String) -> *mut TorchJitFunction {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_forward_hooks(&self) -> &Vec<*mut TorchJitFunction> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_forward_pre_hooks(&self) -> &Vec<*mut TorchJitFunction> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn check_forward_pre_hook_schema(&self, 
        pre_hook_idx:    i32,
        pre_hook_schema: &FunctionSchema)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn check_forward_hook_schema(&self, 
        hook_idx:    i32,
        hook_schema: &FunctionSchema)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn add_method(&mut self, method: *mut TorchJitFunction)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn find_method(&self, name: &String) -> *mut TorchJitFunction {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_method(&self, name: &String) -> &mut TorchJitFunction {
        
        todo!();
        /*
        
        */
    }
    
    pub fn find_hook(&self, name: &String) -> *mut TorchJitFunction {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_hook(&self, name: &String) -> &mut TorchJitFunction {
        
        todo!();
        /*
        
        */
    }
    
    pub fn has_method(&self, name: &String) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn find_static_method(&self, name: &String) -> *mut TorchJitFunction {
        
        todo!();
        /*
        
        */
    }
    
    pub fn add_static_method(&mut self, method: *mut TorchJitFunction)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | [Internal Only] Remove method from the
      | ClassType caller is responsible to make sure
      | the modification is safe:
      |
      | it is unsafe to having existing allocations
      | of this object around anymore, and any code
      | that works on the attribute is now
      | invalid. Only newly created code is valid
      | again.
      |
      | Note this method is intended for freezing
      | only.
      */
    pub fn unsafe_remove_method(&mut self, name: &String)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn compilation_unit(&mut self) -> Arc<CompilationUnit> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn compilation_unit(&self) -> Arc<CompilationUnit> {
        
        todo!();
        /*
        
        */
    }

    /**
      | generate a refined version of this class.
      |
      | It has the same name but the slot Types are
      | subtypes of the original slots. It is only
      | valid to refine a class type in a context
      | where it is know that there are not
      | assignments to the objects slots that would
      | invalidate the refinement.
      |
      | These variants are not registered in the
      | global class table.
      */
    pub fn refine(&self, refined_slots: &[TypePtr]) -> ClassTypePtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new(
        name:                        Option<QualifiedName>,
        cu:                          Weak<CompilationUnit>,
        is_module:                   bool,
        doc_string:                  String,
        unresolved_class_attributes: Vec<String>) -> Self {
        let is_module: bool = is_module.unwrap_or(false);
        let doc_string: String = doc_string.unwrap_or("");
        todo!();
        /*


        
        */
    }
    
    pub fn annotation_str_impl(&self, printer: TypePrinter) -> String {
        let printer: TypePrinter = printer.unwrap_or(nullptr);

        todo!();
        /*
            const auto& n = name().value();
        return n.qualifiedName();
        */
    }
    
    pub fn add_attribute(&mut self, class_attribute: ClassAttribute)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_forward_pre_hook_error_message(&self, pre_hook_idx: i32) -> String {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_forward_hook_error_message(&self, hook_idx: i32) -> String {
        
        todo!();
        /*
        
        */
    }

    /*
      | Mapping of attribute names -> their type.
      |
      | NOTE: this does not contain methods, which
      | are stored in the module
      |
      | TODO: once modules support arbitrary ivalue
      | attributes, we don't need this anymore.
      |
      | TODO: This is better represented as an
      | OrderedDict, but alas it is not yet available
      | from c10
      */
}

pub type InterfaceTypePtr = Arc<InterfaceType>;

/**
  | Interfaces are a list of abstract methods that
  | a class might meet.
  |
  | If a class provides those methods, it
  | implicitly meets the interface.
  |
  | Subtype relations for Interface with ClassType:
  | lhs (ClassType or InterfaceType) is a subtype
  | of rhs if:
  |
  | 1. lhs methods are a superset of rhs methods
  |
  | 2. if rhs is module interface, the lhs must be
  | module interface or module itself
  |
  */
pub struct InterfaceType {

    base:      NamedType,

    /**
      | shared_ptr so that this header does
      | not have to depend on
      | FunctionSchema.h
      |
      */
    methods:   Arc<Vec<FunctionSchema>>,

    /**
      | flag to distinguish if it's an interface
      | type from a module or not
      |
      */
    is_module: bool,
}

impl PartialEq<Type> for InterfaceType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            if (auto user_rhs = rhs.cast<InterfaceType>()) {
          return isSubTypeImpl(*this, *user_rhs, nullptr) &&
              isSubTypeImpl(*user_rhs, *this, nullptr);
        }
        return false;
        */
    }
}

pub mod interface_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::InterfaceType;
}

impl InterfaceType {
    
    pub fn create(
        qualified_name: QualifiedName,
        is_module:      bool) -> InterfaceTypePtr {
        let is_module: bool = is_module.unwrap_or(false);

        todo!();
        /*
        
        */
    }
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return string("InterfaceType<") + name()->name() + ">";
        */
    }
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
        
        */
    }

    /**
      | try to find a method of this interface,
      | 
      | returns nullptr if not found.
      |
      */
    pub fn get_method(&self, name: &String) -> *const FunctionSchema {
        
        todo!();
        /*
        
        */
    }
    
    pub fn add_method(&mut self, schema: FunctionSchema)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn methods(&mut self) -> &Vec<FunctionSchema> {
        
        todo!();
        /*
            return *methods_;
        */
    }
    
    pub fn is_module(&self) -> bool {
        
        todo!();
        /*
            return is_module_;
        */
    }
    
    pub fn new(
        name:      QualifiedName,
        is_module: bool) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn is_sub_type_impl(
        lhs:     &InterfaceType,
        rhs:     &InterfaceType,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn annotation_str_impl(&self, printer: TypePrinter) -> String {
        let printer: TypePrinter = printer.unwrap_or(nullptr);

        todo!();
        /*
            return name()->qualifiedName();
        */
    }
}

pub struct EnumerationType<const K: TypeKind> {
    base: Type,
}

pub mod enumeration_type {

    use super::*;

    pub const Kind: TypeKind = K;
}

impl Default for EnumerationType {
    
    fn default() -> Self {
        todo!();
        /*
        : ty(Kind),

        
        */
    }
}

impl PartialEq<Type> for EnumerationType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

pub type LayoutTypePtr = Arc<LayoutType>;

/**
  | This type represents a Generator
  |
  */
pub struct LayoutType {
    base: EnumerationType<TypeKind_LayoutType>,
}

pub mod layout_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::LayoutType;
}

impl Default for LayoutType {
    
    fn default() -> Self {
        todo!();
        /*
        : enumeration_type(),
        */
    }
}

impl LayoutType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "Layout";
        */
    }

    /// global singleton
    pub fn get() -> LayoutTypePtr {
        
        todo!();
        /*
        
        */
    }
}

pub type ScalarTypeTypePtr = Arc<ScalarTypeType>;

// This type represents a Generator
//
pub struct ScalarTypeType {
    base: EnumerationType<TypeKind_ScalarTypeType>,
}

pub mod scalar_type_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::ScalarTypeType;
}

impl Default for ScalarTypeType {
    
    fn default() -> Self {
        todo!();
        /*
        : enumeration_type(),

        
        */
    }
}

impl ScalarTypeType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "ScalarType";
        */
    }

    /// global singleton
    pub fn get() -> ScalarTypeTypePtr {
        
        todo!();
        /*
        
        */
    }
}

/**
  | the common supertype of all lists,
  | List[T] <: AnyList for all T
  |
  */
pub type AnyListTypePtr = Arc<AnyListType>;

pub struct AnyListType {
    base: Type,
}

impl PartialEq<Type> for AnyListType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

pub mod any_list_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::AnyListType;
}

impl Default for AnyListType {
    
    fn default() -> Self {
        todo!();
        /*
        : ty(TypeKind::AnyListType),

        
        */
    }
}

impl AnyListType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "list";
        */
    }

    /// global singleton
    pub fn get() -> AnyListTypePtr {
        
        todo!();
        /*
        
        */
    }
}

/**
  | the common supertype of all tuples,
  | Tuple[T...] <: AnyTuple for all T
  |
  */
pub type AnyTupleTypePtr = Arc<AnyTupleType>;

pub struct AnyTupleType {
    base: Type,
}

impl PartialEq<Type> for AnyTupleType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

pub mod any_tuple_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::AnyTupleType;
}

impl AnyTupleType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "tuple";
        */
    }

    /// global singleton
    pub fn get() -> AnyTupleTypePtr {
        
        todo!();
        /*
        
        */
    }
}

impl Default for AnyTupleType {
    
    fn default() -> Self {
        todo!();
        /*
        : ty(TypeKind::AnyTupleType),

        
        */
    }
}

/**
  | the common supertype of all classes,
  | ClassType <: AnyClassType for all classes
  |
  */
pub type AnyClassTypePtr = Arc<AnyClassType>;

pub struct AnyClassType {
    base: Type,
}

impl PartialEq<Type> for AnyClassType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            return rhs.kind() == kind();
        */
    }
}

pub mod any_class_type {

    use super::*;

    pub const Kind: TypeKind = TypeKind::AnyClassType;
}

impl Default for AnyClassType {
    
    fn default() -> Self {
        todo!();
        /*
        : ty(TypeKind::AnyClassType),

        
        */
    }
}

impl AnyClassType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "AnyClassType";
        */
    }

    /// global singleton
    pub fn get() -> AnyClassTypePtr {
        
        todo!();
        /*
        
        */
    }
}

impl IValue {
    
    #[inline] pub fn is_double_list(&self) -> bool {
        
        todo!();
        /*
            // note: avoids calling type() to avoid extra referencing counting for the returned type.
      return isList() && static_cast<ListImpl*>(payload.u.as_intrusive_ptr)->elementType->kind() == FloatType::Kind;
        */
    }
    
    #[inline] pub fn is_complex_double_list(&self) -> bool {
        
        todo!();
        /*
            // note: avoids calling type() to avoid extra referencing counting for the returned type.
      return isList() && static_cast<ListImpl*>(payload.u.as_intrusive_ptr)->elementType->kind() == ComplexType::Kind;
        */
    }
    
    #[inline] pub fn is_tensor_list(&self) -> bool {
        
        todo!();
        /*
            return isList() && static_cast<ListImpl*>(payload.u.as_intrusive_ptr)->elementType->kind() == TensorType::Kind;
        */
    }
    
    #[inline] pub fn is_int_list(&self) -> bool {
        
        todo!();
        /*
            return isList() && static_cast<ListImpl*>(payload.u.as_intrusive_ptr)->elementType->kind() == IntType::Kind;
        */
    }
    
    #[inline] pub fn is_bool_list(&self) -> bool {
        
        todo!();
        /*
            return isList() && static_cast<ListImpl*>(payload.u.as_intrusive_ptr)->elementType->kind() == BoolType::Kind;
        */
    }
}


impl Type {
    
    #[inline] pub fn cast(&mut self) -> Arc<NamedType> {
        
        todo!();
        /*
            if (kind() == TypeKind::TupleType || kind() == TypeKind::FunctionType ||
          kind() == TypeKind::ClassType || kind() == TypeKind::InterfaceType) {
        return static_pointer_cast<NamedType>(shared_from_this());
      }
      return nullptr;
        */
    }
    
    #[inline] pub fn cast_named_type(&self) -> Arc<NamedType> {
        
        todo!();
        /*
            if (kind() == TypeKind::TupleType || kind() == TypeKind::FunctionType ||
          kind() == TypeKind::ClassType || kind() == TypeKind::InterfaceType) {
        return static_pointer_cast<const NamedType>(shared_from_this());
      }
      return nullptr;
        */
    }
}

/**
  | Used as a return type when inferring
  | the IValue type of a Python object.
  |
  */
pub struct InferredType {
    ty:     TypePtr,
    reason: String,
}

impl InferredType {
    
    pub fn new(ty: TypePtr) -> Self {
    
        todo!();
        /*
        : ty(move(type)),

        
        */
    }
    
    pub fn new(reason: String) -> Self {
    
        todo!();
        /*
        : ty(nullptr),
        : reason(move(reason)),

        
        */
    }
    
    pub fn ty(&self) -> TypePtr {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(type_);
        return type_;
        */
    }
    
    pub fn success(&self) -> bool {
        
        todo!();
        /*
            return type_ != nullptr;
        */
    }
    
    pub fn reason(&self) -> &String {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(!type_);
        return reason_;
        */
    }
}
