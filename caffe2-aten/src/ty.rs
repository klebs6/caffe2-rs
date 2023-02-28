// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/type.cpp]

pub fn type_verbosity() -> TypeVerbosity {
    
    todo!();
        /*
            static const char* c_verbosity = getenv("PYTORCH_JIT_TYPE_VERBOSITY");
      static TypeVerbosity verbosity = c_verbosity ?
        static_cast<TypeVerbosity>(stoi(c_verbosity)) : TypeVerbosity::Default;
      return verbosity;
        */
}

impl fmt::Display for Type {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            if (auto value = t.cast<TensorType>()) {
        if  (value->scalarType().has_value()) {
          out << toString(*value->scalarType());
          if (!value->sizes().size().has_value()) {
            out << "Tensor";
          }
        } else {
          out << "Tensor";
        }
        if (auto ndim = value->sizes().size()) {
          bool has_valid_strides_info = *ndim > 0 &&
              value->strides().isComplete() && value->strides().size() == ndim;

          out << "(";
          usize i = 0;
          bool symbolic = type_verbosity() == TypeVerbosity::Symbolic;
          for (i = 0; i < *ndim; ++i) {
            if (i > 0) {
              out << ", ";
            }
            if (auto s = value->sizes()[i]) {
              out << *s;
            } else if (symbolic) {
              out << value->symbolic_sizes().at(i);
            } else {
              out << "*";
            }
          }
          if (has_valid_strides_info &&
              type_verbosity() >= TypeVerbosity::TypeAndStride) {
            out << ", strides=[";
            for (usize i = 0; i < *ndim; ++i) {
              if (i > 0) {
                out << ", ";
              }
              out << *value->strides()[i];
            }
            out << "]";
          }
          if (type_verbosity() >= TypeVerbosity::Full) {
            if (value->requiresGrad()) {
              if (i++ > 0) {
                out << ", ";
              }
              out << "requires_grad=" << *value->requiresGrad();
            }
            if (value->device()) {
              if (i++ > 0) {
                out << ", ";
              }
              out << "device=" << *value->device();
            }
          }
          out << ")";
        } else {
          if (type_verbosity() >= TypeVerbosity::Full) {
            usize i = 0;
            if (value->requiresGrad()) {
              out << "("
                  << "requires_grad=" << *value->requiresGrad();
              i++;
            }
            if (value->device()) {
              out << ((i++ > 0) ? ", " : "(") << "device=" << *value->device();
            }
            if (i > 0) {
              out << ")";
            }
          }
        }

        if (value->undefined() && *value->undefined()) {
          out << "[Undefined]";
        }
      } else if(t.kind() == TypeKind::ListType) {
        auto prim = t.castRaw<ListType>()->getElementType();
        out << *prim << "[]";
      } else if (t.kind() == TypeKind::OptionalType) {
        auto prim = t.castRaw<OptionalType>()->getElementType();
        out << *prim << "?";
      } else if(t.kind() == TypeKind::FutureType) {
        auto elem = t.castRaw<FutureType>()->getElementType();
        out << "Future[" << *elem << "]";
      } else if(t.kind() == TypeKind::RRefType) {
        auto elem = t.castRaw<RRefType>()->getElementType();
        out << "RRef[" << *elem << "]";
      } else if(auto tup = t.cast<TupleType>()) {
        if (tup->schema()) {
          out << "NamedTuple";
        }
        out << "(";
        for(usize i = 0; i < tup->elements().size(); ++i) {
          if(i > 0)
            out << ", ";
          if (tup->schema()) {
            out << tup->schema()->arguments()[i].name() << " : ";
          }
          out << *(tup->elements()[i]);
        }
        out << ")";
      } else if (t.kind() == TypeKind::FunctionType) {
        out << "Function";
      } else {
         out << t.str();
      }
      return out;
        */
    }
}

impl AnyType {
    
    pub fn get(&mut self) -> AnyTypePtr {
        
        todo!();
        /*
            static AnyTypePtr value(new AnyType());
      return value;
        */
    }
}

impl TensorType {
    
    pub fn get(&mut self) -> TensorTypePtr {
        
        todo!();
        /*
            static auto value = TensorType::create(
          {}, {}, SymbolicShape(), VaryingShape<Stride>{}, {});
      return value;
        */
    }
}

impl NumberType {
    
    pub fn get(&mut self) -> NumberTypePtr {
        
        todo!();
        /*
            static NumberTypePtr value(new NumberType());
      return value;
        */
    }
}

impl IntType {
    
    pub fn get(&mut self) -> IntTypePtr {
        
        todo!();
        /*
            static IntTypePtr value(new IntType());
      return value;
        */
    }
}

impl FloatType {
    
    pub fn get(&mut self) -> FloatTypePtr {
        
        todo!();
        /*
            static FloatTypePtr value(new FloatType());
      return value;
        */
    }
}

impl ComplexType {
    
    pub fn get(&mut self) -> ComplexTypePtr {
        
        todo!();
        /*
            static ComplexTypePtr value(new ComplexType());
      return value;
        */
    }
}

impl BoolType {
    
    pub fn get(&mut self) -> BoolTypePtr {
        
        todo!();
        /*
            static BoolTypePtr value(new BoolType());
      return value;
        */
    }
}

impl StorageType {
    
    pub fn get(&mut self) -> StorageTypePtr {
        
        todo!();
        /*
            static StorageTypePtr value(new StorageType());
      return value;
        */
    }
}

impl NoneType {
    
    pub fn get(&mut self) -> NoneTypePtr {
        
        todo!();
        /*
            static NoneTypePtr value(new NoneType());
      return value;
        */
    }
}

impl GeneratorType {
    
    pub fn get(&mut self) -> GeneratorTypePtr {
        
        todo!();
        /*
            static GeneratorTypePtr value(new GeneratorType());
      return value;
        */
    }
}

impl QuantizerType {
    
    pub fn get(&mut self) -> QuantizerTypePtr {
        
        todo!();
        /*
            static QuantizerTypePtr value(new QuantizerType());
      return value;
        */
    }
}

impl QSchemeType {
    
    pub fn get(&mut self) -> QSchemeTypePtr {
        
        todo!();
        /*
            static QSchemeTypePtr value(new QSchemeType());
      return value;
        */
    }
}

impl StringType {
    
    pub fn get(&mut self) -> StringTypePtr {
        
        todo!();
        /*
            static StringTypePtr value(new StringType());
      return value;
        */
    }
}

impl DeviceObjType {
    
    pub fn get(&mut self) -> DeviceObjTypePtr {
        
        todo!();
        /*
            static DeviceObjTypePtr value(new DeviceObjType());
      return value;
        */
    }
}

impl StreamObjType {
    
    pub fn get(&mut self) -> StreamObjTypePtr {
        
        todo!();
        /*
            static StreamObjTypePtr value(new StreamObjType());
      return value;
        */
    }
}

impl ScalarTypeType {
    
    pub fn get(&mut self) -> ScalarTypeTypePtr {
        
        todo!();
        /*
            static ScalarTypeTypePtr value(new ScalarTypeType());
    return value;
        */
    }
}

impl LayoutType {
    
    pub fn get(&mut self) -> LayoutTypePtr {
        
        todo!();
        /*
            static LayoutTypePtr value(new LayoutType());
    return value;
        */
    }
}

impl OptionalType {
    
    pub fn of_tensor(&mut self) -> OptionalTypePtr {
        
        todo!();
        /*
            static auto value = OptionalType::create(TensorType::get());
      return value;
        */
    }
}

impl PyObjectType {
    
    pub fn get(&mut self) -> PyObjectTypePtr {
        
        todo!();
        /*
            static PyObjectTypePtr value(new PyObjectType());
      return value;
        */
    }
}

impl CapsuleType {
    
    pub fn get(&mut self) -> CapsuleTypePtr {
        
        todo!();
        /*
            static CapsuleTypePtr value(new CapsuleType());
      return value;
        */
    }
}

impl ListType {
    
    pub fn of_tensors(&mut self) -> ListTypePtr {
        
        todo!();
        /*
            static auto value = ListType::create(TensorType::get());
      return value;
        */
    }
    
    pub fn of_ints(&mut self) -> ListTypePtr {
        
        todo!();
        /*
            static auto value = ListType::create(IntType::get());
      return value;
        */
    }
    
    pub fn of_complex_doubles(&mut self) -> ListTypePtr {
        
        todo!();
        /*
            static auto value = ListType::create(ComplexType::get());
      return value;
        */
    }
    
    pub fn of_floats(&mut self) -> ListTypePtr {
        
        todo!();
        /*
            static auto value = ListType::create(FloatType::get());
      return value;
        */
    }
    
    pub fn of_bools(&mut self) -> ListTypePtr {
        
        todo!();
        /*
            static auto value = ListType::create(BoolType::get());
      return value;
        */
    }
    
    pub fn of_strings(&mut self) -> ListTypePtr {
        
        todo!();
        /*
            static auto value = ListType::create(StringType::get());
      return value;
        */
    }
}


impl AnyListType {

    pub fn get(&mut self) -> AnyListTypePtr {
        
        todo!();
        /*
            static AnyListTypePtr value(new AnyListType());
      return value;
        */
    }
}

impl AnyTupleType {
    
    pub fn get(&mut self) -> AnyTupleTypePtr {
        
        todo!();
        /*
            static AnyTupleTypePtr value(new AnyTupleType());
      return value;
        */
    }
}

impl AnyClassType {
    
    pub fn get(&mut self) -> AnyClassTypePtr {
        
        todo!();
        /*
            static AnyClassTypePtr value(new AnyClassType());
      return value;
        */
    }
}

impl AnyEnumType {
    
    pub fn get(&mut self) -> AnyEnumTypePtr {
        
        todo!();
        /*
            static AnyEnumTypePtr value(new AnyEnumType());
      return value;
        */
    }
}

pub fn unify_types_impl(
        t1: &TypePtr,
        t2: &TypePtr) -> Option<TypePtr> {
    
    todo!();
        /*
            // check direct subtyping relation
      if (t1->isSubtypeOf(t2)) {
        return t2;
      } else if (t2->isSubtypeOf(t1)) {
        return t1;
      }

      // Handle non-container types which do not subtype each other and unify
      if (t1->kind() == TensorType::Kind && t2->kind() == TensorType::Kind) {
        return t1->expectRef<TensorType>().merge(*t2->expect<TensorType>());
      }

      if (t1->isSubtypeOf(NoneType::get()) && !t2->isSubtypeOf(NoneType::get())) {
        return OptionalType::create(t2);
      } else if (t2->isSubtypeOf(NoneType::get()) && !t1->isSubtypeOf(NoneType::get())) {
        return OptionalType::create(t1);
      }

      // NB: we do not return NumberType because there is not currently enough
      // operator support for it

      // Attempt to unify Complete Tensor Types for immutable type containers

      // unify(Optional[t1], t2) => Optional[unify(t1, t2)]
      if (auto opt_t1 = t1->cast<OptionalType>()) {
        if (auto elem = unifyTypes(opt_t1->getElementType(), t2)) {
          return OptionalType::create(*elem);
        }
      } else if (auto opt_t2 = t2->cast<OptionalType>()) {
        if (auto elem = unifyTypes(opt_t2->getElementType(), t1)) {
          return OptionalType::create(*elem);
        }
      }

      if (t1->cast<TupleType>() && t2->cast<TupleType>()) {
        auto tuple1 = t1->cast<TupleType>();
        auto tuple2 = t2->cast<TupleType>();
        if (tuple1->elements().size() != tuple2->elements().size()) {
          return nullopt;
        }
        vector<TypePtr> elements;
        for (usize i = 0; i < tuple1->elements().size(); i++) {
          if (auto elem = unifyTypes(tuple1->elements().at(i), tuple2->elements().at(i))) {
            elements.push_back(*elem);
          } else {
            return nullopt;
          }
        }
        return static_cast<TypePtr>(TupleType::create(elements));
      }

      if (t1->cast<FutureType>() && t2->cast<FutureType>()) {
        if (auto elem = unifyTypes(
                t1->castRaw<FutureType>()->getElementType(),
                t2->castRaw<FutureType>()->getElementType())) {
          return FutureType::create(*elem);
        }
      }

      // Check direct subtyping relations again with Unshaped Types,
      // to handle unification of mutable container types which might contain two different
      // specialized tensors (ListType / DictType)
      auto t1_unshaped = unshapedType(t1);
      auto t2_unshaped = unshapedType(t2);

      if (t1_unshaped->isSubtypeOf(t2_unshaped)) {
        return t2_unshaped;
      } else if (t2_unshaped->isSubtypeOf(t1_unshaped)) {
        return t1_unshaped;
      }

      return nullopt;
        */
}

pub fn unify_types(
        t1:             &TypePtr,
        t2:             &TypePtr,
        default_to_any: bool) -> Option<TypePtr> {
    
    todo!();
        /*
            auto unified = unifyTypesImpl(t1, t2);

      if (default_to_any && !unified) {
        return AnyType::get();
      }

      return unified;
        */
}

pub fn unify_type_list(
        elements: &[TypePtr],
        why_not:  &mut std::io::BufWriter) -> Option<TypePtr> {
    
    todo!();
        /*
            if (elements.size() == 0) {
        why_not << "Cannot get unified type from empty list";
        return nullopt;
      }

      TypePtr ret_type = elements.at(0);
      for (usize i = 1; i < elements.size() && ret_type; ++i) {
        auto maybe_unified = unifyTypes(ret_type, elements.at(i));
        if (!maybe_unified) {
          why_not << "Could not unify type list since element " << i << " of type "
                  << elements.at(i)->repr_str()
                  << " did not match the types before it ("
                  << ret_type->repr_str() << ")";
          return nullopt;
        }
        ret_type = maybe_unified.value();
      }

      return ret_type;
        */
}

pub fn match_type_variables(
    formal:   TypePtr,
    actual:   TypePtr,
    type_env: &mut TypeEnv) -> MatchTypeReturn {
    
    todo!();
        /*
            if (!formal->hasFreeVariables()) {
        return MatchTypeReturn::Success();
      }

      if (auto vt = formal->cast<VarType>()) {
        auto it = type_env.find(vt->name());
        if (it == type_env.end()) {
          type_env[vt->name()] = actual;
          return MatchTypeReturn::Success();
        } else if (auto unified = unifyTypes(it->second, actual)) {
          // note: unifyTypes allows subtyping in either direction, so actual
          // may be a supertype of the current binding. we're not responsible
          // for reporting the error, only for keeping type_env stable
          return MatchTypeReturn::Success();
        }
        stringstream ss;
        ss << "Type variable '" << vt->name() << "' previously matched to type "
           << it->second->repr_str() << " is matched to type "
           << actual->repr_str();
        return ss.str();
      } else if (auto lt_formal = formal->cast<ListType>()) {
        if (auto lt_actual = actual->cast<ListType>()) {
          const auto innerMatch = matchTypeVariables(
              lt_formal->getElementType(), lt_actual->getElementType(), type_env);
          if (!innerMatch.success()) {
            // propagate the errMsg onward
            return innerMatch;
          }
          return MatchTypeReturn::Success();
        } else if (auto tup_type = actual->cast<TupleType>()) {
          stringstream ss;
          auto maybe_tuple_unified = unifyTypeList(tup_type->elements(), ss);
          if (maybe_tuple_unified) {
            return matchTypeVariables(
                lt_formal->getElementType(), *maybe_tuple_unified, type_env);
          }
        }

        stringstream ss;
        ss << "Cannot match " << lt_formal->repr_str() << " to "
           << actual->repr_str();
        return ss.str();
      } else if (auto tp_formal = formal->cast<TupleType>()) {
        if (auto tp_actual = actual->cast<TupleType>()) {
          if (tp_formal->elements().size() != tp_actual->elements().size()) {
            return MatchTypeReturn("Cannot match tuples of mismatched size");
          }
          for (usize i = 0; i < tp_formal->elements().size(); ++i) {
            const auto result = matchTypeVariables(
                tp_formal->elements()[i], tp_actual->elements()[i], type_env);
            if (!result.success()) {
              return result;
            }
          }
          return MatchTypeReturn::Success();
        } else {
          stringstream ss;
          ss << "Cannot match a tuple to " << actual->repr_str();
          return MatchTypeReturn(ss.str());
        }
      } else if (auto lt_formal = formal->cast<FutureType>()) {
        if (auto lt_actual = actual->cast<FutureType>()) {
          const auto innerMatch = matchTypeVariables(
              lt_formal->getElementType(), lt_actual->getElementType(), type_env);
          if (!innerMatch.success()) {
            return innerMatch;
          }
          return MatchTypeReturn::Success();
        } else {
          stringstream ss;
          ss << "Cannot match a future to " << actual->repr_str();
          return ss.str();
        }
      } else if (auto lt_formal = formal->cast<RRefType>()) {
        if (auto lt_actual = actual->cast<RRefType>()) {
          const auto innerMatch = matchTypeVariables(
              lt_formal->getElementType(), lt_actual->getElementType(), type_env);
          if (!innerMatch.success()) {
            return innerMatch;
          }
          return MatchTypeReturn::Success();
        } else {
          stringstream ss;
          ss << "Cannot match a rref to " << actual->repr_str();
          return ss.str();
        }
      } else if (auto opt_formal = formal->cast<OptionalType>()) {
        if (auto opt_actual = actual->cast<OptionalType>()) {
          const auto optionedMatch = matchTypeVariables(
              opt_formal->getElementType(), opt_actual->getElementType(), type_env);
          if (!optionedMatch.success()) {
            return optionedMatch;
          }
        } else if (!actual->isSubtypeOf(NoneType::get())) {
          // If the actual type is a non-optional, allow matching to the formal if
          // its element type matches the actual.
          // Don't match None because it is already an optional (but one of
          // unknown type).
          return matchTypeVariables(opt_formal->getElementType(), actual, type_env);
        }
        // note: if actual was None here we potentially did not fill in the type
        // variables contained in the formal. It is still a valid match because None
        // matches Optional[T] later error checking on tryEvalTypeVariables will
        // report the problem if we never match variables in type T
        return MatchTypeReturn::Success();
      } else if (auto dict_formal = formal->cast<DictType>()) {
        if (auto dict_actual = actual->cast<DictType>()) {
          auto key_match = matchTypeVariables(
              dict_formal->getKeyType(), dict_actual->getKeyType(), type_env);
          if (!key_match.success()) {
            return key_match;
          }
          auto value_match = matchTypeVariables(
              dict_formal->getValueType(), dict_actual->getValueType(), type_env);
          if (!value_match.success()) {
            return value_match;
          }
          return MatchTypeReturn::Success();
        } else {
          stringstream ss;
          ss << "Cannot match a dict to " << actual->repr_str();
          return ss.str();
        }
      }

      AT_ERROR("Unhandled free variable container: ", formal->repr_str());
        */
}

/**
  | change return types like List[List[t]]
  | into List[List[int]]
  |
  */
pub fn try_eval_type_variables(
        ty:       TypePtr,
        type_env: &mut HashMap<String,TypePtr>) -> TypePtr {
    
    todo!();
        /*
            if (!type->hasFreeVariables()) {
        return type;
      }

      if (auto vt = type->cast<VarType>()) {
        auto it = type_env.find(vt->name());
        if (it == type_env.end()) {
          return nullptr;
        }
        return it->second;
      } else {
        vector<TypePtr> new_contained;
        new_contained.reserve(type->containedTypes().size());
        for (const TypePtr& t : type->containedTypes()) {
          TypePtr r = tryEvalTypeVariables(t, type_env);
          if (!r) {
            return nullptr;
          }
          new_contained.push_back(r);
        }
        return type->withContained(move(new_contained));
      }
        */
}

pub fn element_type_can_be_inferred_from_members(elem_type: &TypePtr) -> bool {
    
    todo!();
        /*
            if (elem_type->kind() == OptionalType::Kind ||
          elem_type->kind() == NumberType::Kind) {
        // Builtin Union types
        return false;
      }
      if (elem_type->kind() == InterfaceType::Kind) {
        // since classes can be members of multiple interfaces, we cannot
        // construct which interface the list holds from the members alone
        return false;
      }
      if (elem_type->kind() == AnyType::Kind) {
        // List of Any can contains heterogenous types
        return false;
      }
      return true;
        */
}


pub fn type_kind_to_string(kind: TypeKind) -> *const u8 {
    
    todo!();
        /*
            #define CASE_TYPE(T) case TypeKind::T: return #T;
      switch(kind) {
        C10_FORALL_TYPES(CASE_TYPE)
      }
    #undef CASE_TYPE
      return "";
        */
}

impl Type {
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
            if (rhs->kind() == TypeKind::AnyType || *this == *rhs) {
        return true;
      }
      if(auto rhs_ = rhs->cast<OptionalType>()) {
        return this->isSubtypeOfExt(rhs_->getElementType(), why_not);
      }
      return false;
        */
    }
    
    pub fn is_module(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}

impl TensorType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            return "Tensor";
        */
    }
}

impl VaryingShape {
    
    pub fn merge<T>(&self, other: &VaryingShape<T>) -> VaryingShape<T> {
    
        todo!();
        /*
            if (!dims_ || !other.dims_ || dims_->size() != other.dims_->size()) {
        return VaryingShape<T>();
      }
      ListOfOptionalElements dims;
      for (usize i = 0, n = dims_->size(); i < n; i++) {
        dims.push_back(merge_primitive((*dims_)[i], (*other.dims_)[i]));
      }
      return VaryingShape<T>(move(dims));
        */
    }
}

impl TensorType {
    
    pub fn sizes(&self) -> VaryingShape<i64> {
        
        todo!();
        /*
            if (!sizes_.rank()) {
        return VaryingShape<i64>();
      }
      return VaryingShape<i64>(
          fmap(*sizes_.sizes(), [](ShapeSymbol ss) {
            // we turn symbolic shapes into unknowns
            return ss.is_static()
                ? optional<i64>(ss.static_size())
                : nullopt;
          }));
        */
    }
    
    pub fn merge(&self, 
        other:       &TensorType,
        merge_sizes: bool) -> TensorTypePtr {
        
        todo!();
        /*
            auto scalar_type = merge_primitive(scalarType(), other.scalarType());
      auto dev = merge_primitive(device(), other.device());
      auto sprops = stride_properties().merge(other.stride_properties());
      auto gr = merge_primitive(requiresGrad(), other.requiresGrad());
      auto undef = merge_primitive(undefined(), other.undefined());
      return TensorType::create(
          scalar_type,
          dev,
          merge_sizes ? symbolic_sizes().merge(other.symbolic_sizes())
                      : symbolic_sizes(),
          sprops,
          gr,
          undef);
        */
    }
}

pub fn is_null_or_equal<T>(
        a: Option<T>,
        b: &[i32]) -> bool {

    todo!();
        /*
            return !a.has_value() || a.value() == b;
        */
}

impl TensorType {
    
    pub fn match_tensor(&mut self, t: &Tensor) -> bool {
        
        todo!();
        /*
            bool undef = undefined().value_or(!t.defined());
      if (undef != !t.defined()) {
        // When the followings are true, we consider it's not a match:
        // - undefined().has_value() == true
        // - undefined().value() != !t.defined()
        return false;
      } else if (!t.defined()) {
        // When the followings are true, we consider it's a match:
        // - t is not defined
        // - undefined() == null or undefined().value() == true
        return true;
      }
      // Here we know t.defined() == true and compare all other properties.
      bool rg = GradMode::is_enabled() && t.requires_grad();
      bool matched_strides = (!stride_properties().size()) ||
          (!t.has_storage() && !stride_properties().isComplete()) ||
          stride_properties() ==
              computeStrideProps(t.sizes(), t.strides(), t.is_contiguous());
      return scalarType().value_or(t.scalar_type()) == t.scalar_type()
        && device().value_or(t.device()) == t.device()
        && requiresGrad().value_or(rg) == rg
        && matched_strides
        && is_null_or_equal(sizes().concrete_sizes(), t.sizes());
        */
    }
}

impl PartialEq<Type> for TensorType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            if (rhs.kind() != kind()) {
        return false;
      }
      auto rt = rhs.expect<TensorType>();

      return scalar_type_ == rt->scalarType() && sizes() == rt->sizes() &&
          stride_properties() == rt->stride_properties() &&
          device() == rt->device() && requiresGrad() == rt->requiresGrad() &&
          undefined() == rt->undefined();
        */
    }
}

impl fmt::Display for VaryingShape<T> {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            out << "(";
      if (!vs.size()) {
        out << "*)";
        return out;
      }

      for (usize i = 0; i < vs.size(); i++) {
        if (i > 0) {
          out << ", ";
        }
        if (vs[i].has_value()) {
          out << vs[i].value();
        } else {
          out << "*";
        }
      }
      out << ")";
      return out;
        */
    }
}

impl fmt::Display for SymbolicShape {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            // TODO: Unranked SymbolicShape printing is ambiguous with that of
      // dynamic-shaped vector.
      if(!ss.rank()) {
        os << "(*)";
        return os;
      }

      auto sizes = ss.sizes().value();

      os << "(";
      for (usize i = 0; i < ss.rank().value(); i++) {
        if (i > 0) {
          os << ", ";
        }
        if(sizes[i].is_static()) {
          os << sizes[i];
        } else {
          os << "*";
        }
      }
      os << ")";

      return os;
        */
    }
}

impl fmt::Display for ShapeSymbol {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            if (s.value_ >= 0) {
        os << s.value_;
      } else {
        os << "SS(" << s.value_ << ')';
      }
      return os;
        */
    }
}

impl fmt::Display for Stride {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            os << "{";
      if (s.stride_index_.has_value()) {
        os << *s.stride_index_;
      } else {
        os << "*";
      }
      os << ":";
      if (s.stride_.has_value()) {
        os << *s.stride_;
      } else {
        os << "*";
      }
      os << '}';
      return os;
        */
    }
}

impl TupleType {
    
    pub fn create_named(&mut self, 
        qual_name:   &Option<QualifiedName>,
        field_names: &Vec<String>,
        field_types: &Vec<TypePtr>) -> TupleTypePtr {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(field_names.size() == field_types.size());
      vector<Argument> arguments;
      for (usize i = 0; i < field_names.size(); ++i) {
        arguments.emplace_back(
            /*name=*/field_names[i],
            /*type=*/field_types[i],
            /*N=*/i);
      }

      auto schema = make_shared<FunctionSchema>(
          /*name=*/qualName.value_or(QualifiedName()).name(),
          /*overload_name=*/string(""),
          /*arguments=*/arguments,
          /*returns=*/vector<Argument>{});
      return shared_ptr<TupleType>(new TupleType(
          field_types, qualName, schema)); // NOLINT(modernize-make-shared)
        */
    }
    
    pub fn new(
        elements: Vec<TypePtr>,
        name:     Option<QualifiedName>,
        schema:   Arc<FunctionSchema>) -> Self {
    
        todo!();
        /*


            : NamedType(TypeKind::TupleType, move(name)),
          elements_(move(elements)),
          schema_(move(schema)) 
      has_free_variables_ =
          any_of(elements_.begin(), elements_.end(), [](TypePtr v) {
            if (!v) {
              throw runtime_error("Can not create tuple with None type");
            }
            return v->hasFreeVariables();
          });
      if (schema_) {
        for (const Argument& arg : schema_->arguments()) {
          checkNoAny(*this, "attribute", arg.name(), arg.type());
        }
      }
        */
    }
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
            if (Type::isSubtypeOfExt(rhs_, why_not)) {
        return true;
      }
      if (rhs_->kind() == AnyTupleType::Kind) {
        return true;
      }
      auto rhs = rhs_->cast<TupleType>();
      if (!rhs)
        return false;
      // unnamed tuple is not a subtype of nametuple
      if (!schema() && rhs->schema())
        return false;
      // namedtuple may be a subtype of unnamed tuple
      auto test_names_match = [&](const shared_ptr<FunctionSchema>& lhs, const shared_ptr<FunctionSchema>& rhs) {
        const auto& args_lhs = lhs->arguments();
        const auto& args_rhs = rhs->arguments();
        if (args_lhs.size() != args_rhs.size()) {
          return false;
        }

        for (usize i = 0; i < args_lhs.size(); ++i) {
          if (args_lhs[i].name() != args_rhs[i].name()) {
            return false;
          }
        }
        return true;
      };
      bool names_match = !rhs->schema() || test_names_match(schema(), rhs->schema());
      // co-variant rules for tuples
      return names_match && compare(*rhs, [&](const TypePtr a, const TypePtr b) {
        return a->isSubtypeOfExt(b, why_not);
      });
        */
    }
}

impl ListType {
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
            if (Type::isSubtypeOfExt(rhs_, why_not)) {
        return true;
      }
      if (rhs_->kind() == AnyListType::Kind) {
        return true;
      }
      return false;
        */
    }
}

impl PartialEq<Type> for TupleType {
    
    #[inline] fn eq(&self, other: &Type) -> bool {
        todo!();
        /*
            bool typesSame =
           compare(rhs, [](const TypePtr a, const TypePtr b) { return *a == *b; });
       if (!typesSame) {
         return false;
      }

      // `compare` guarantees that rhs is always a TupleType.
      auto rhsTuple = rhs.expect<TupleType>();
      if (schema_ == nullptr && rhsTuple->schema_ == nullptr) {
        return typesSame;
      }
      if (schema_ == nullptr || rhsTuple->schema_ == nullptr) {
        return false;
      }
      return *schema_ == *rhsTuple->schema_;
        */
    }
}

impl TupleType {
    
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            stringstream ss;
      if (schema_ && name()) {
        ss << name()->qualifiedName();
      } else {
        ss << "(";
        for(usize i = 0; i < elements().size(); ++i) {
          if(i > 0)
            ss << ", ";
          ss << elements()[i]->str();
        }
        ss << ")";
      }
      return ss.str();
        */
    }
    
    pub fn annotation_str_impl(&self, printer: TypePrinter) -> String {
        
        todo!();
        /*
            stringstream ss;
      if (schema_ && name()) {
        ss << name()->qualifiedName();
      } else {
        ss << "Tuple[";
        if (elements().size() == 0) {
          // `typing.Tuple` special-cases the annotation syntax for empty tuple
          // with `typing.Tuple[()]`. See
          // https://docs.python.org/3/library/typing.html#typing.Tuple
          ss << "()";
        } else {
          for (usize i = 0; i < elements().size(); ++i) {
            if (i > 0)
              ss << ", ";
            ss << elements()[i]->annotation_str(printer);
          }
        }
        ss << "]";
      }
      return ss.str();
        */
    }
}

impl TensorType {
    
    pub fn strides(&self) -> VaryingShape<i64> {
        
        todo!();
        /*
            if (!strides_.size().has_value()) {
        return VaryingShape<i64>();
      }
      vector<optional<i64>> ss(*strides_.size());
      for (usize i = 0; i < *strides_.size(); i++) {
        if (!strides_[i].has_value()) {
          continue;
        }
        auto s = *strides_[i];
        if (s.stride_index_.has_value() && s.stride_.has_value()) {
          ss[*s.stride_index_] = *s.stride_;
        }
      }
      return VaryingShape<i64>(ss);
        */
    }
    
    pub fn compute_stride_props(&mut self, 
        sizes:             &[i32],
        strides:           &[i32],
        tensor_contiguity: bool) -> VaryingShape<Stride> {
        
        todo!();
        /*
            vector<usize> stride_indices(sizes.size());
      iota(stride_indices.begin(), stride_indices.end(), 0);

      sort(
          stride_indices.begin(),
          stride_indices.end(),
          [&strides](const int& a, const int& b) {
            // break ties in case of unsqueezed dims
            // i.e. (1, 1, 5)
            if (strides[a] == strides[b]) {
              return a > b;
            }
            return strides[a] < strides[b];
          });

      vector<Stride> stride_properties;
      for (usize i = 0; i < stride_indices.size(); i++) {
        bool contiguous_ = tensor_contiguity;
        if (!contiguous_) {
          // innermost stride expected to be 1
          // TODO: turn contiguous_ into an enum CONTIGUOUS, NONCONTIGUOUS,
          // BROADCASTED
          if (i == 0) {
            contiguous_ = strides[stride_indices[i]] == 1;
          } else {
            contiguous_ = strides[stride_indices[i]] == 1 ||
                (strides[stride_indices[i]] != 0 &&
                 strides[stride_indices[i]] ==
                     strides[stride_indices[i - 1]] * sizes[stride_indices[i - 1]]);
          }
        }
        stride_properties.emplace_back(stride_indices[i], contiguous_, strides[stride_indices[i]]);
      }

      return VaryingShape<Stride>{stride_properties};
        */
    }
}

lazy_static!{
    /*
    atomic<usize> ShapeSymbol::num_symbols{1};
    */
}

impl TensorType {
    
    pub fn new(
        scalar_type:   Option<ScalarType>,
        device:        Option<Device>,
        sizes:         &SymbolicShape,
        strides:       &VaryingShape<Stride>,
        requires_grad: Option<bool>,
        undefined:     Option<bool>) -> Self {
    
        todo!();
        /*


            : Type(TypeKind::TensorType),
          scalar_type_(scalar_type),
          device_(device),
          sizes_(sizes),
          strides_(strides),
          requires_grad_(requires_grad),
          undefined_(undefined)
        */
    }
    
    pub fn create(&mut self, t: &Tensor) -> TensorTypePtr {
        
        todo!();
        /*
            VaryingShape<bool> contiguity;
      VaryingShape<usize> stride_indices;
      VaryingShape<i64> strides;
      VaryingShape<i64> sizes;
      if (!t.is_mkldnn() && !t.is_sparse()) {
        sizes = VaryingShape<i64>{t.sizes().vec()};
        strides = VaryingShape<i64>{t.strides().vec()};
        return TensorType::create(
            t.scalar_type(), t.device(), sizes, strides, t.requires_grad(), false, t.is_contiguous());
      }

      return TensorType::create(
          t.scalar_type(),
          t.device(),
          SymbolicShape(),
          VaryingShape<Stride>{},
          t.requires_grad(),
          false);
        */
    }
    
    pub fn create(&mut self, 
        scalar_type:       Option<ScalarType>,
        device:            Option<Device>,
        sizes:             &VaryingShape<i64>,
        strides:           &VaryingShape<i64>,
        requires_grad:     Option<bool>,
        undefined:         Option<bool>,
        tensor_contiguity: bool) -> TensorTypePtr {
        
        todo!();
        /*
            if(strides.concrete_sizes() && strides.concrete_sizes().has_value()){
        // handles case where strides are set
        TORCH_INTERNAL_ASSERT(sizes.concrete_sizes()->size() == strides.concrete_sizes()->size());
        auto sprops = strides.concrete_sizes().has_value()
          ? computeStrideProps(*sizes.concrete_sizes(), *strides.concrete_sizes(), tensor_contiguity)
          : VaryingShape<Stride>();
        auto symbol_sizes = SymbolicShape(*sizes.concrete_sizes());
        return TensorType::create(
          scalar_type, device, symbol_sizes, sprops, requires_grad, undefined);
      } else {
        // strides are all null, but still have number of strides equal to number of ranks
        TORCH_INTERNAL_ASSERT(sizes.sizes() && sizes.size());
        auto symbol_sizes = SymbolicShape(*sizes.sizes());
        return TensorType::create(
          scalar_type, device, symbol_sizes, VaryingShape<Stride>(*sizes.size()), requires_grad, undefined);
      }
        */
    }
    
    pub fn create(&mut self, 
        scalar_type:   Option<ScalarType>,
        device:        Option<Device>,
        sizes:         &SymbolicShape,
        strides:       &VaryingShape<Stride>,
        requires_grad: Option<bool>,
        undefined:     Option<bool>) -> TensorTypePtr {
        
        todo!();
        /*
            auto pt = TensorTypePtr(new TensorType(
          scalar_type, device, sizes, strides, requires_grad, undefined));
      return pt;
        */
    }
    
    pub fn create(&mut self, 
        scalar_type:   Option<ScalarType>,
        device:        Option<Device>,
        dim:           Option<usize>,
        requires_grad: Option<bool>) -> TensorTypePtr {
        
        todo!();
        /*
            return TensorType::create(
          scalar_type,
          device,
          SymbolicShape(dim),
          VaryingShape<Stride>(dim),
          requires_grad);
        */
    }
    
    pub fn create_contiguous(&mut self, 
        scalar_type: ScalarType,
        device:      Device,
        sizes:       &[i32]) -> TensorTypePtr {
        
        todo!();
        /*
            auto strides = contiguousStridesOf(sizes);
      TORCH_INTERNAL_ASSERT(strides.size() == sizes.size());
      return create(
          scalar_type,
          device,
          VaryingShape<i64>(sizes),
          VaryingShape<i64>(strides),
          nullopt);
        */
    }
    
    pub fn symbolic_sizes(&self) -> &SymbolicShape {
        
        todo!();
        /*
            return sizes_;
        */
    }
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
            if (auto rhs_p = rhs->cast<TensorType>()) {
        // if we have the same pointer, avoid computing the merge
        if (this == rhs_p.get()) {
          return true;
        }
        return *merge(*rhs_p) == *rhs_p;
      }
      return Type::isSubtypeOfExt(rhs, why_not);
        */
    }
}

impl InterfaceType {
    
    pub fn create(&mut self, 
        qualified_name: QualifiedName,
        is_module:      bool) -> InterfaceTypePtr {
        
        todo!();
        /*
            return InterfaceTypePtr(
          new InterfaceType(move(qualifiedName), is_module));
        */
    }
}

impl ClassType {
    
    pub fn add_method(&mut self, method: *mut TorchJitFunction)  {
        
        todo!();
        /*
            TORCH_CHECK(
          findMethod(method->name()) == nullptr,
          "Can't redefine method: ",
          method->name(),
          " on class: ",
          repr_str());
      methods_.push_back(method);
        */
    }
    
    pub fn get_forward_hooks(&self) -> &Vec<*mut TorchJitFunction> {
        
        todo!();
        /*
            return forward_hooks_;
        */
    }
    
    pub fn get_forward_pre_hooks(&self) -> &Vec<*mut TorchJitFunction> {
        
        todo!();
        /*
            return forward_pre_hooks_;
        */
    }
    
    pub fn add_forward_pre_hook(&mut self, pre_hook_ptr: *mut TorchJitFunction)  {
        
        todo!();
        /*
            forward_pre_hooks_.emplace_back(pre_hook_ptr);
        */
    }
    
    pub fn add_forward_hook(&mut self, hook_ptr: *mut TorchJitFunction)  {
        
        todo!();
        /*
            forward_hooks_.emplace_back(hook_ptr);
        */
    }
    
    pub fn find_forward_pre_hook(&self, name: &String) -> *mut TorchJitFunction {
        
        todo!();
        /*
            for (const auto& pre_hook : forward_pre_hooks_) {
        if (name == pre_hook->name()) {
          return pre_hook;
        }
      }
      return nullptr;
        */
    }
    
    pub fn find_forward_hook(&self, name: &String) -> *mut TorchJitFunction {
        
        todo!();
        /*
            for (const auto& hook : forward_hooks_) {
        if (name == hook->name()) {
          return hook;
        }
      }
      return nullptr;
        */
    }
}

pub fn get_schema_input_types_string(schema: &FunctionSchema) -> String {
    
    todo!();
        /*
            stringstream input_types;
      const vector<Argument>& forward_args = schema.arguments();
      for (const auto i : irange(1, forward_args.size())) {
        input_types << forward_args[i].type()->annotation_str();
        if (forward_args.size() - 1 != i) {
          input_types << ", ";
        }
      }
      if (forward_args.size() == 1) {
        input_types << "()";
      }
      return input_types.str();
        */
}

impl ClassType {
    
    pub fn get_forward_pre_hook_error_message(&self, pre_hook_idx: i32) -> String {
        
        todo!();
        /*
            const string& pre_hook_name = forward_pre_hooks_[pre_hook_idx]->name();
      const FunctionSchema& forward_schema = getMethod("forward").getSchema();
      string input_types = getSchemaInputTypesString(forward_schema);
      const vector<Argument>& forward_args = forward_schema.arguments();

      string single_output = "";
      if (forward_args.size() == 2 &&
          forward_args[1].type()->cast<TupleType>() == nullptr) {
        // if the output type is a single tuple, it needs to be wrapped in an outer tuple
        // to match eager's behavior
        single_output = ", '" + forward_args[1].type()->annotation_str() + "',";
      }
      string pre_hook_schema =
          pre_hook_name + "(self, input: Tuple[" + input_types + "])";
      string return_string =
          "This error occured while scripting the forward pre-hook '" +
          pre_hook_name + "' on module '" + name()->name() +
          "'. If you did not want to script this pre-hook remove it from the "
          "original NN module before scripting. Pre-hooks for module '" +
          name()->name() + "' are expected to have the following signature: "
          + pre_hook_schema + " with a return type of either 'None'" +
          single_output + " or 'Tuple[" + input_types + "]'.";
      return return_string;
        */
    }
    
    pub fn get_forward_hook_error_message(&self, hook_idx: i32) -> String {
        
        todo!();
        /*
            const string& hook_name = forward_hooks_[hook_idx]->name();
      const FunctionSchema& forward_schema = getMethod("forward").getSchema();
      string input_types = getSchemaInputTypesString(forward_schema);

      // create expected output types string
      const Argument& pre_output =
          (hook_idx == 0)
              ? forward_schema.returns()[0]
              : forward_hooks_[hook_idx - 1]->getSchema().returns()[0];
      string output_types = pre_output.type()->annotation_str();
      // create error message
      string hook_schema = hook_name + "(self, input: Tuple[" +
                                input_types + "], output: " + output_types + ")";
      string return_string =
          "This error occured while scripting the forward hook '"
          + hook_name + "' on module " + name()->name() +
          ". If you did not want to script this hook remove it from" +
          " the original NN module before scripting. This hook was" +
          " expected to have the following signature: " + hook_schema +
          ". The type of the output arg is the returned type from" +
          " either the forward method or the previous hook if it exists. " +
          "Note that hooks can return anything, but if the hook is " +
          "on a submodule the outer module is expecting" +
          " the same return type as the submodule's forward.";
      return return_string;
        */
    }
    
    pub fn is_unresolved_class_attribute(&self, name: &String) -> bool {
        
        todo!();
        /*
            return find(
          unresolved_class_attributes_.begin(),
          unresolved_class_attributes_.end(),
          name) != unresolved_class_attributes_.end();
        */
    }
    
    pub fn check_forward_pre_hook_schema(&self, 
        pre_hook_idx:    i32,
        pre_hook_schema: &FunctionSchema)  {
        
        todo!();
        /*
            const TorchJitFunction* pre_hook = forward_pre_hooks_[pre_hook_idx];
      string hook_id =
          "Pre-hook '" + pre_hook->name() + "' on module '" + name()->name() + "' ";
      string pre_hook_err_msg = getForwardPreHookErrorMessage(pre_hook_idx) + "\n";

      // Pre-hooks are expecting two inputs: self, and a Tuple containing the
      // non-self arguments passed to Forward
      TORCH_CHECK(
          pre_hook_schema.arguments().size() == 2,
          hook_id,
          "was expected to only have exactly 2 inputs but it had ",
          pre_hook_schema.arguments().size(),
          " inputs. ",
          pre_hook_err_msg
       );

      const FunctionSchema& forward_schema = getMethod("forward").getSchema();
      const vector<Argument>& forward_args = forward_schema.arguments();
      checkForwardHookInputArguments(forward_schema, pre_hook_schema, hook_id, pre_hook_err_msg);

      // check return type, expected to be either None, the same type as the input,
      // or the contained single type if the input was a tuple containing a single
      // type.
      TORCH_CHECK(
                pre_hook_schema.returns().size() != 0,
                hook_id,
                "is missing a return annotation. Return annotations are required, please add one.\n",
                pre_hook_err_msg
      );
      const Argument return_arg = pre_hook_schema.returns()[0];
      string wrong_type_returned_err_msg = hook_id +
          "returned the wrong type of: '" +
          return_arg.type()->annotation_str() + "'.";

      if (return_arg.type()->kind() == NoneType::get()->kind()) {
        return;
      }
      if (forward_args.size() == 2 && *forward_args[1].type() == *return_arg.type()) {
        // TORCH_CHECK below is for the edge case where forward's input is a tuple and the
        // pre-hook returns a matching tuple. Eager doesn't support this- the working eager return
        // for a tuple type is the forward's input tuple wrapped inside of another tuple.
        TORCH_CHECK(
            return_arg.type()->cast<TupleType>() == nullptr,
            wrong_type_returned_err_msg,
            " When forward has a single tuple input argument, the return needs",
            " to be 'None' or a nested tuple containing forward's input tuple",
            " argument as in: 'Tuple[",
            forward_args[1].type()->annotation_str(),
            "]'.\n",
            pre_hook_err_msg
        );
        return;
      }
      // return can only be tuple of nested types now
      // check to make sure return is of tuple type
      TORCH_CHECK(
          return_arg.type()->cast<TupleType>() != nullptr,
          wrong_type_returned_err_msg,
          pre_hook_err_msg
      );
      const ArrayRef<TypePtr> return_tuple_types =
          return_arg.type()->castRaw<TupleType>()->elements();
      // check for edge case of Tuple[()] for when forward has no arguments
      if (forward_args.size() == 1) {
        TORCH_CHECK(
            return_tuple_types.size() == 0,
            wrong_type_returned_err_msg,
            " Was expecting either 'None' or 'Tuple[()]' since forward had ",
            "no arguments.\n",
            pre_hook_err_msg
        );
        return;
      }

      // check that tuple has proper number of contained types
      TORCH_CHECK(
          return_tuple_types.size() == forward_args.size() - 1,
          wrong_type_returned_err_msg,
          " The returned tuple contains the wrong number of contained types.\n",
          pre_hook_err_msg
      );
      // check that contained types match forward types
      for (const auto i : irange(1, forward_args.size())) {
        if (*forward_args[i].type() != *return_tuple_types[i - 1]) {
          TORCH_CHECK(
              false,
              wrong_type_returned_err_msg,
              " The returned tuple contains the wrong inner types.\n",
              pre_hook_err_msg);
        }
      }
        */
    }
    
    pub fn check_forward_hook_schema(&self, 
        hook_idx:    i32,
        hook_schema: &FunctionSchema)  {
        
        todo!();
        /*
            const TorchJitFunction* hook = forward_hooks_[hook_idx];
      string hook_id =
          "Hook '" + hook->name() + "' on module '" + name()->name() + "' ";
      string hook_err_msg = getForwardHookErrorMessage(hook_idx) + "\n";
      // Hooks are expecting three inputs: self, a Tuple containing the non-self
      // arguments passed to Forward, and the output of either Forward or the
      // previous hook
      TORCH_CHECK(
          hook_schema.arguments().size() == 3,
          hook_id,
          "was expected to only have exactly 3 inputs but it had ",
          hook_schema.arguments().size(),
          " inputs. ",
          hook_err_msg
      );

      const FunctionSchema& forward_schema = getMethod("forward").getSchema();
      checkForwardHookInputArguments(forward_schema, hook_schema, hook_id, hook_err_msg);

      // check output tuple
      const Argument& prev_output = (hook_idx == 0)
                ? forward_schema.returns()[0]
                : forward_hooks_[hook_idx - 1]->getSchema().returns()[0];
      const Argument return_arg = hook_schema.arguments()[2];

      // output tuple needs to match prev_output's return exactly
      TORCH_CHECK(
          *prev_output.type() == *return_arg.type(),
          hook_id,
          "has the wrong type for the output argument. Received type: '",
          return_arg.type()->annotation_str(),
          "'. Expected type: '",
          prev_output.type()->annotation_str(),
          "'.\n",
          hook_err_msg
      );
        */
    }
    
    pub fn find_method(&self, name: &String) -> *mut TorchJitFunction {
        
        todo!();
        /*
            for (auto method : methods_) {
        if (name == method->name()) {
          return method;
        }
      }
      return nullptr;
        */
    }
    
    pub fn get_method(&self, name: &String) -> &mut TorchJitFunction {
        
        todo!();
        /*
            auto method = findMethod(name);
      TORCH_CHECK(
          method != nullptr,
          "Couldn't find method: '",
          name,
          "' on class: '",
          repr_str(),
          "'");
      return *method;
        */
    }
    
    pub fn find_hook(&self, name: &String) -> *mut TorchJitFunction {
        
        todo!();
        /*
            auto hook = findForwardHook(name);
      if (hook == nullptr) {
        hook = findForwardPreHook(name);
      }
      return hook;
        */
    }
    
    pub fn get_hook(&self, name: &String) -> &mut TorchJitFunction {
        
        todo!();
        /*
            TorchJitFunction* function = findHook(name);
      TORCH_CHECK(
          function != nullptr,
          "Couldn't find: '",
          name,
          "' on class: '",
          repr_str(),
          "'as forward hook or forward pre_hook.");
      return *function;
        */
    }
    
    pub fn has_method(&self, name: &String) -> bool {
        
        todo!();
        /*
            return findMethod(name) != nullptr;
        */
    }
    
    pub fn add_static_method(&mut self, method: *mut TorchJitFunction)  {
        
        todo!();
        /*
            TORCH_CHECK(
          findStaticMethod(method->name()) == nullptr &&
              findMethod(method->name()) == nullptr, "Can't redefine method: ",
          method->name(),
          " on class: ",
          repr_str());
      staticmethods_.emplace_back(method);
        */
    }
    
    pub fn find_static_method(&self, name: &String) -> *mut TorchJitFunction {
        
        todo!();
        /*
            for (auto method : staticmethods_) {
        if (name == method->name()) {
          return method;
        }
      }
      return nullptr;
        */
    }
    
    pub fn unsafe_remove_method(&mut self, name: &String)  {
        
        todo!();
        /*
            usize slot = 0;
      for (auto method : methods_) {
        if (method->name() == name) {
          methods_.erase(methods_.begin() + slot);
          return;
        }
        slot++;
      }
      TORCH_CHECK(
          false,
          "Can't delete undefined method ",
          name,
          " on class: ",
          repr_str());
        */
    }
    
    pub fn refine(&self, refined_slots: &[TypePtr]) -> ClassTypePtr {
        
        todo!();
        /*
            auto ptr = ClassType::create(name(), compilation_unit_, is_module());
      AT_ASSERT(numAttributes() == refined_slots.size());
      for (usize i = 0; i < attributes_.size(); ++i) {
        AT_ASSERT(refined_slots[i]->isSubtypeOf(attributes_[i].getType()));
        ptr->addAttribute(attributes_[i].getName(), refined_slots[i], (attributes_[i].getKind() == AttributeKind::PARAMETER),
        (attributes_[i].getKind() == AttributeKind::BUFFER));
      }
      // Copy methods over
      for (const auto& method : methods()) {
        ptr->addMethod(method);
      }
      return ptr;
        */
    }
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
            if (rhs->cast<AnyClassType>()) {
        return true;
      }
      // to improve performance, this check can be cached
      if (auto iface = rhs->cast<InterfaceType>()) {
        // ClassType is not a subtype of InterfaceType if the InterfaceType is a
        // Module Interface Type but the Class Type is not a Module Class Type
        if (!is_module() && iface->is_module()) {
          if (why_not) {
            *why_not << "Class '" << repr_str() << "' is not a subtype of "
                     << "the module interface '" << rhs->repr_str()
                     << "' , only ScriptModule class can be subtype of module"
                     << " interface.\n";
          }
          return false;
        }
        for (const FunctionSchema& schema : iface->methods()) {
          auto self_method = findMethod(schema.name());
          if (!self_method) {
            if (why_not) {
              *why_not << "Class '" << repr_str() << "' does not have method '"
                       << schema.name() << "' but '" << rhs->repr_str()
                       << "' does.\n";
            }
            return false;
          }
          if (!self_method->getSchema().isSubtypeOf(
                  schema, /*is_method=*/true, why_not)) {
            if (why_not) {
              *why_not << "Method on class '" << repr_str()
                       << "' (1) is not compatible with interface '"
                       << rhs->repr_str() << "' (2)\n"
                       << "  (1) " << self_method->getSchema() << "\n"
                       << "  (2) " << schema << "\n";
            }
            return false;
          }
        }
        return true;
      }
      return Type::isSubtypeOfExt(rhs, why_not);
        */
    }
}

impl FunctionType {
    
    pub fn new(function: *mut TorchJitFunction) -> Self {
    
        todo!();
        /*


            : NamedType(TypeKind::FunctionType, function->qualname()),
        function_(function)
        */
    }
}

impl InterfaceType {
    
    pub fn is_sub_type_impl(&mut self, 
        lhs:     &InterfaceType,
        rhs:     &InterfaceType,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
            if (!lhs.is_module() && rhs.is_module()) {
        if (why_not) {
          *why_not << "Interface '" << lhs.repr_str() << "' is not a subtype of "
                   << "the module interface '" << rhs.repr_str() << "'.\n";
        }
        return false;
      }
        for (const FunctionSchema& schema : *rhs.methods_) {
          auto self_schema = lhs.getMethod(schema.name());
          if (!self_schema) {
            if (why_not) {
              *why_not << "Interface '" << lhs.repr_str()
                       << "' does not have method '" << schema.name() << "' but interface '"
                       << rhs.repr_str() << "' does.\n";
            }
            return false;
          }
          if (!self_schema->isSubtypeOf(schema, /*is_method=*/true, why_not)) {
            if (why_not) {
              *why_not << "Method on interface '" << lhs.repr_str()
                       << "' (1) is not compatible with interface '"
                       << rhs.repr_str() << "' (2)\n"
                       << "  (1) " << *self_schema << "\n"
                       << "  (2) " << schema << "\n";
              return false;
            }
            return false;
          }
        }
        return true;
        */
    }
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
            // to improve performance this check can be cached
      if (auto iface = rhs->cast<InterfaceType>()) {
        return isSubTypeImpl(*this, *iface, why_not);
      }
      return Type::isSubtypeOfExt(rhs, why_not);
        */
    }
    
    pub fn get_method(&self, name: &String) -> *const FunctionSchema {
        
        todo!();
        /*
            for (const FunctionSchema& method : *methods_) {
        if (method.name() == name) {
          return &method;
        }
      }
      return nullptr;
        */
    }
    
    pub fn add_method(&mut self, schema: FunctionSchema)  {
        
        todo!();
        /*
            methods_->emplace_back(move(schema));
        */
    }
    
    pub fn new(
        name:      QualifiedName,
        is_module: bool) -> Self {
    
        todo!();
        /*


            : NamedType(InterfaceType::Kind, move(name)),
          methods_(make_shared<vector<FunctionSchema>>()),
          is_module_(is_module)
        */
    }
}

impl ClassType {
    
    pub fn create(&mut self, 
        qualified_name:              Option<QualifiedName>,
        cu:                          Weak<CompilationUnit>,
        is_module:                   bool,
        doc_string:                  String,
        unresolved_class_attributes: Vec<String>) -> ClassTypePtr {
        
        todo!();
        /*
            return ClassTypePtr(new ClassType(
          move(qualifiedName),
          move(cu),
          is_module,
          move(doc_string),
          move(unresolved_class_attributes)));
        */
    }
    
    pub fn new(
        name:                        Option<QualifiedName>,
        cu:                          Weak<CompilationUnit>,
        is_module:                   bool,
        doc_string:                  String,
        unresolved_class_attributes: Vec<String>) -> Self {
    
        todo!();
        /*


            : NamedType(TypeKind::ClassType, move(name)),
          compilation_unit_(move(cu)),
          isModule_(is_module),
          doc_string_(move(doc_string)),
          unresolved_class_attributes_(move(unresolved_class_attributes))
        */
    }
    
    pub fn methods(&self) -> &Vec<*mut TorchJitFunction> {
        
        todo!();
        /*
            return methods_;
        */
    }
    
    pub fn check_not_exist(&self, 
        name: &String,
        what: &String)  {
        
        todo!();
        /*
            // Check no overlap with existing constants
      for (usize i = 0; i < constantNames_.size(); ++i) {
        TORCH_CHECK(
            name != constantNames_[i],
            "attempting to add ",
            what,
            " '",
            name,
            "' to ",
            repr_str(),
            " but a constant field of the same name already exists with value ",
            constantValues_[i]);
      }

      // Check no overlap with existing attributes
      for (usize i = 0; i < attributes_.size(); ++i) {
        TORCH_CHECK(
            name != attributes_[i].getName(),
            "attempting to add ",
            what,
            " '",
            name,
            "' to ",
            repr_str(),
            " but an attribute field of the same name already exists with type ",
            attributes_[i].getType()->repr_str());
      }
        */
    }
    
    pub fn add_attribute(&mut self, class_attribute: ClassAttribute)  {
        
        todo!();
        /*
            attributes_.push_back(classAttribute);
        attributeTypes_.push_back(classAttribute.getType());
        AT_ASSERT(attributes_.size() == attributeTypes_.size());
        */
    }
    
    pub fn add_attribute(&mut self, 
        name:         &String,
        ty:           &TypePtr,
        is_parameter: bool,
        is_buffer:    bool) -> usize {
        
        todo!();
        /*
            if (is_parameter && is_buffer){
        TORCH_INTERNAL_ASSERT(false, "Attribute cannot be both a parameter and a buffer!");
      }

      string what = is_parameter ? "parameter" : "attribute";
      what += (is_buffer? "buffer" : "not buffer");
      checkNotExist(name, what);

      usize slot = attributes_.size();

      AttributeKind kind = AttributeKind::REGULAR_ATTRIBUTE;
      if (is_parameter) {
        kind = AttributeKind::PARAMETER;
      } else if (is_buffer) {
        kind = AttributeKind::BUFFER;
      }

      ClassAttribute ClassAttribute(kind, type, name);

      addAttribute(ClassAttribute);

      if (is_parameter || is_buffer) {
        TORCH_INTERNAL_ASSERT(is_module(), "adding a parameter or buffer to a non module");
        TORCH_CHECK(
            (type->kind() == TensorType::Kind) ||
                (type->kind() == OptionalType::Kind &&
                type->expectRef<OptionalType>().getElementType()->kind() ==
                    TensorType::Kind) ||
                (type->kind() == NoneType::Kind),
            "Expecting parameter or buffer to have either None, Tensor or Optional[Tensor] type, but got: ",
            toString(type));
      }

      return slot;
        */
    }
    
    pub fn unsafe_remove_attribute(&mut self, name: &String)  {
        
        todo!();
        /*
            auto slot = getAttributeSlot(name);
      attributes_.erase(attributes_.begin() + slot);
      attributeTypes_.erase(attributeTypes_.begin() + slot);
      AT_ASSERT(attributes_.size() == attributeTypes_.size());
        */
    }
    
    pub fn unsafe_change_attribute_type(&mut self, 
        name:   &String,
        new_ty: TypePtr)  {
        
        todo!();
        /*
            auto slot = getAttributeSlot(name);
      auto old_attr_info = attributes_[slot];
      AT_ASSERT(old_attr_info.getKind() == AttributeKind::REGULAR_ATTRIBUTE);
      attributes_[slot] = ClassAttribute(old_attr_info.getKind(), new_ty, old_attr_info.getName());
      attributeTypes_[slot] = new_ty;
        */
    }
    
    pub fn add_constant(&mut self, 
        name:  &String,
        value: &IValue) -> usize {
        
        todo!();
        /*
            checkNotExist(name, "constant");
      usize slot = constantNames_.size();
      constantNames_.push_back(name);
      constantValues_.push_back(value);
      return slot;
        */
    }
    
    pub fn get_constant(&self, name: &String) -> IValue {
        
        todo!();
        /*
            const auto& v = findConstant(name);
      TORCH_CHECK(
          v.has_value(),
          repr_str(),
          " does not have a constant field with name '",
          name,
          "'");
      return *v;
        */
    }
    
    pub fn get_constant(&self, slot: usize) -> IValue {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(constantNames_.size() == constantValues_.size());
      TORCH_CHECK(
          slot < constantValues_.size(),
          repr_str(),
          " does not have a constant slot of index ",
          slot);
      return constantValues_[slot];
        */
    }
    
    pub fn find_constant(&self, name: &String) -> Option<IValue> {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(constantNames_.size() == constantValues_.size());
      usize pos = 0;
      for (const auto& c : constantNames_) {
        if (name == c) {
          break;
        }
        ++pos;
      }

      if (pos >= constantNames_.size()) {
        return nullopt;
      }
      return constantValues_[pos];
        */
    }
    
    pub fn unsafe_remove_constant(&mut self, name: &String)  {
        
        todo!();
        /*
            auto slot = getConstantSlot(name);
      constantNames_.erase(constantNames_.begin() + slot);
      constantValues_.erase(constantValues_.begin() + slot);
        */
    }
    
    pub fn compilation_unit(&mut self) -> Arc<CompilationUnit> {
        
        todo!();
        /*
            auto cu = compilation_unit_.lock();
      return cu;
        */
    }
    
    pub fn compilation_unit(&self) -> Arc<CompilationUnit> {
        
        todo!();
        /*
            auto cu = compilation_unit_.lock();
      return cu;
        */
    }
    
    pub fn get_property(&mut self, name: &String) -> Option<ClassType_Property> {
        
        todo!();
        /*
            for (auto& prop : properties_) {
        if (name == prop.name) {
          return prop;
        }
      }

      return nullopt;
        */
    }
    
    pub fn add_property(&mut self, 
        name:   &String,
        getter: *mut TorchJitFunction,
        setter: *mut TorchJitFunction)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(!getProperty(name), "Property named ", name, " already exists!");
      properties_.push_back({name, getter, setter});
        */
    }
}

pub fn contains_any(ty: &TypePtr) -> bool {
    
    todo!();
        /*
            vector<TypePtr> to_scan = { type };
      while (!to_scan.empty()) {
        const auto typ = to_scan.back();
        to_scan.pop_back();
        if (typ->kind() == AnyType::Kind) {
          return true;
        }
        for (const TypePtr& sub : typ->containedTypes()) {
          to_scan.emplace_back(sub);
        }
      }
      return false;
        */
}

pub fn check_no_any(
    base:     &Type,
    what:     *const u8,
    attrname: &String,
    attrtype: &TypePtr)  {
    
    todo!();
        /*
            TORCH_CHECK(
          !containsAny(attrtype),
          "attempting to add ",
          what,
          " '",
          attrname,
          "' of type ",
          attrtype->repr_str(),
          " to '",
          base.repr_str(),
          "' but it contains an Any type. Any types cannot be members of modules, classes, or named tuples.");
        */
}

impl SymbolicShape {
    
    pub fn merge(&self, other: &SymbolicShape) -> SymbolicShape {
        
        todo!();
        /*
            if (!dims_ || !other.dims_ || dims_->size() != other.dims_->size()) {
        return SymbolicShape();
      }
      vector<ShapeSymbol> dims;
      for (usize i = 0, n = dims_->size(); i < n; i++) {
        dims.push_back(merge_primitive((*dims_)[i], (*other.dims_)[i]));
      }
      return SymbolicShape(move(dims));
        */
    }
    
    pub fn dump(&self)  {
        
        todo!();
        /*
            cout << *this << "\n";
        */
    }
}

impl EnumType {
    
    pub fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
            return rhs->kind() == TypeKind::AnyType ||
          rhs->kind() == TypeKind::AnyEnumType || *this == *rhs;
        */
    }
}

pub fn check_forward_hook_input_arguments(
        forward_schema: &FunctionSchema,
        hook_schema:    &FunctionSchema,
        hook_id:        &String,
        hook_err_msg:   &String)  {
    
    todo!();
        /*
            // check for proper tuple input types
      const vector<Argument>& forward_args = forward_schema.arguments();
      const Argument input_arg = hook_schema.arguments()[1];
      TORCH_CHECK(
          input_arg.type()->cast<TupleType>() != nullptr,
          hook_id,
          "expected the input argument to be typed as a Tuple but found type: '",
          input_arg.type()->annotation_str(),
          "' instead.\n",
          hook_err_msg
       );

      const ArrayRef<TypePtr> input_tuple_types = input_arg.type()->castRaw<TupleType>()->elements();
      if (forward_args.size() == 1) {
        // check for empty forward case
        TORCH_CHECK(
            input_tuple_types.size() == 0,
            hook_id,
            "was expecting Tuple[()] as the input type. Received type: '",
            input_arg.type()->annotation_str(),
            "'.\n",
            hook_err_msg
          );
      } else {
        // check input tuple for correct size and correct contained types
        TORCH_CHECK(
            input_tuple_types.size() == forward_args.size() - 1,
            hook_id,
            "has the wrong number of contained types for the",
            " input argument's Tuple. Received type: '",
            input_arg.type()->annotation_str(),
            "'.\n",
            hook_err_msg
        );

        for (const auto i : irange(1, forward_args.size())) {
          if (*forward_args[i].type() != *input_tuple_types[i - 1]) {
            TORCH_CHECK(
                false,
                hook_id,
                "has the wrong inner types for the input tuple argument. Received type: '",
                input_arg.type()->annotation_str(),
                "'.\n",
                hook_err_msg
            );
          }
        }
      }
        */
}
