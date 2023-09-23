crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/dispatch/CppSignature.h]

/**
  | A CppSignature object holds RTTI information
  | about a C++ function signature at runtime
  | and can compare them or get a debug-printable
  | name.
  |
  */
pub struct CppSignature {
    signature: TypeIndex,
}

impl PartialEq<CppSignature> for CppSignature {
    
    #[inline] fn eq(&self, other: &CppSignature) -> bool {
        todo!();
        /*
            if (lhs.signature_ == rhs.signature_) {
                return true;
            }
            // Without RTLD_GLOBAL, the type_index comparison could yield false because
            // they point to different instances of the RTTI data, but the types would
            // still be the same. Let's check for that case too.
            // Note that there still is a case where this might not work, i.e. when
            // linking libraries of different compilers together, they might have
            // different ways to serialize a type name. That, together with a missing
            // RTLD_GLOBAL, would still fail this.
            if (lhs.name() == rhs.name()) {
                return true;
            }

            return false;
        */
    }
}

impl CppSignature {
    
    pub fn make<FuncType>() -> CppSignature {
    
        todo!();
        /*
            // Normalize functors, lambdas, function pointers, etc. into the plain function type
            // The first argument of the schema might be of type DispatchKeySet, in which case we remove it.
            // We do this to guarantee that all CppSignature's for an operator will match, even if they're registered
            // with different calling conventions.
            // See Note [Plumbing Keys Through The Dispatcher]
            using decayed_function_type = typename remove_DispatchKeySet_arg_from_func<decay_t<FuncType>>::func_type;

            return CppSignature(type_index(typeid(decayed_function_type)));
        */
    }
    
    pub fn name(&self) -> String {
        
        todo!();
        /*
            return demangle(signature_.name());
        */
    }
    
    pub fn new(signature: TypeIndex) -> Self {
    
        todo!();
        /*
        : signature(move(signature)),

        
        */
    }
}
