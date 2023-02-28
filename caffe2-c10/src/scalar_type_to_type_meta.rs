/*!
  | these just expose TypeMeta/ScalarType bridge
  | functions in c10
  |
  | TODO move to typeid.h (or codemod away) when
  | TypeMeta et al are moved from caffe2 to c10
  | (see note at top of typeid.h)
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/ScalarTypeToTypeMeta.h]

/**
  | convert ScalarType enum values to TypeMeta
  | handles
  |
  */
#[inline] pub fn scalar_type_to_type_meta(scalar_type: ScalarType) -> TypeMeta {
    
    todo!();
        /*
            return TypeMeta::fromScalarType(scalar_type);
        */
}

/**
  | convert TypeMeta handles to ScalarType
  | enum values
  |
  */
#[inline] pub fn type_meta_to_scalar_type(dtype: TypeMeta) -> ScalarType {
    
    todo!();
        /*
            return dtype.toScalarType();
        */
}

/**
  | typeMetaToScalarType(), lifted to
  | optional
  |
  */
#[inline] pub fn opt_type_meta_to_scalar_type(type_meta: Option<TypeMeta>) -> Option<ScalarType> {
    
    todo!();
        /*
            if (!type_meta.has_value()) {
        return nullopt;
      }
      return type_meta->toScalarType();
        */
}

/**
  | convenience: equality across TypeMeta/ScalarType
  | conversion
  |
  */
lazy_static!{
    /*
    static inline bool operator==(ScalarType t, TypeMeta m) {
      return m.isScalarType(t);
    }

    static inline bool operator==(TypeMeta m, ScalarType t) {
      return t == m;
    }

    static inline bool operator!=(ScalarType t, TypeMeta m) {
      return !(t == m);
    }

    static inline bool operator!=(TypeMeta m, ScalarType t) {
      return !(t == m);
    }
    */
}
