crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/jit_type_base.h]

#[macro_export] macro_rules! c10_forall_types {
    ($_:ident) => {
        /*
        
          _(AnyType)                
          _(EnumType)               
          _(AnyEnumType)            
          _(TensorType)             
          _(StorageType)            
          _(TupleType)              
          _(ListType)               
          _(DictType)               
          _(NumberType)             
          _(FloatType)              
          _(ComplexType)      
          _(FutureType)             
          _(RRefType)               
          _(IntType)                
          _(NoneType)               
          _(StringType)             
          _(GeneratorType)          
          _(QuantizerType)          
          _(BoolType)               
          _(OptionalType)           
          _(VarType)                
          _(DeviceObjType)          
          _(StreamObjType)          
          _(FunctionType)           
          _(ClassType)              
          _(PyObjectType)           
          _(CapsuleType)            
          _(InterfaceType)          
          _(QSchemeType)            
          _(LayoutType)             
          _(ScalarTypeType)         
          _(AnyListType)            
          _(AnyTupleType)           
          _(AnyClassType)
        */
    }
}

pub enum TypeKind {
    AnyType,
    EnumType,
    AnyEnumType,
    TensorType,
    StorageType,
    TupleType,
    ListType,
    DictType,
    NumberType,
    FloatType,
    ComplexType,
    FutureType,
    RRefType,
    IntType,
    NoneType,
    StringType,
    GeneratorType,
    QuantizerType,
    BoolType,
    OptionalType,
    VarType,
    DeviceObjType,
    StreamObjType,
    FunctionType,
    ClassType,
    PyObjectType,
    CapsuleType,
    InterfaceType,
    QSchemeType,
    LayoutType,
    ScalarTypeType,
    AnyListType,
    AnyTupleType,
    AnyClassType,
}

pub fn type_kind_to_string(kind: TypeKind) -> *const u8 {
    
    todo!();
        /*
        
        */
}

pub type TypePtr      = Arc<Type>;
pub type ConstTypePtr = Arc<Type>;

/**
  | Use this to customize how a Type is printed
  | using `annotation_str()`. If nullopt is
  | returned, `annotation_str()` falls through to
  | its default implementation.
  |
  */
pub type TypePrinter = fn(_0: &ConstTypePtr) -> Option<String>;

pub struct Type {
    base: EnableSharedFromThis<Type>,
    kind: TypeKind,
}

impl Type {
    
    pub fn new(kind: TypeKind) -> Self {
    
        todo!();
        /*
        : kind(kind),

        
        */
    }
    
    pub fn annotation_str_impl(&self, printer: TypePrinter) -> String {
        
        todo!();
        /*
            return str();
        */
    }
    
    /**
      | create a new version of this type, replacing
      | its contained types with contained_types
      |
      */
    pub fn with_contained(&mut self, contained_types: Vec<TypePtr>) -> TypePtr {
        
        todo!();
        /*
            auto current_contained = containedTypes();
        AT_ASSERT(current_contained.size() == contained_types.size());
        if (current_contained.equals(contained_types)) {
          return shared_from_this();
        }
        return createWithContained(move(contained_types));
        */
    }

    /**
      | Dynamically cast this object to the
      | subclass indicated by the template
      | variable, returning nullptr if the
      | cast is invalid.
      |
      */
    pub fn cast<T>(&mut self) -> Arc<T> {
    
        todo!();
        /*
            if (T::Kind == kind()) {
          return static_pointer_cast<T>(shared_from_this());
        }
        return nullptr;
        */
    }
    
    pub fn cast<T>(&self) -> Arc<T> {
    
        todo!();
        /*
            if (T::Kind == kind()) {
          return static_pointer_cast<const T>(shared_from_this());
        }
        return nullptr;
        */
    }
    
    pub fn cast_raw<T>(&mut self) -> *mut T {
    
        todo!();
        /*
            if (T::Kind == kind()) {
          return static_cast<T*>(this);
        }
        return nullptr;
        */
    }
    
    pub fn cast_raw<T>(&self) -> *const T {
    
        todo!();
        /*
            if (T::Kind == kind()) {
          return static_cast<const T*>(this);
        }
        return nullptr;
        */
    }
    
    pub fn expect<T>(&mut self) -> Arc<T> {
    
        todo!();
        /*
            auto r = cast<T>();
        AT_ASSERT(r);
        return r;
        */
    }
    
    pub fn expect<T>(&self) -> Arc<T> {
    
        todo!();
        /*
            auto r = cast<const T>();
        AT_ASSERT(r);
        return r;
        */
    }
    
    pub fn expect_ref<T>(&mut self) -> &mut T {
    
        todo!();
        /*
            auto* r = castRaw<T>();
        AT_ASSERT(r);
        return *r;
        */
    }
    
    pub fn expect_ref<T>(&self) -> &T {
    
        todo!();
        /*
            auto* r = castRaw<const T>();
        AT_ASSERT(r);
        return *r;
        */
    }
    
    pub fn is_subtype_of(&self, rhs: &TypePtr) -> bool {
        
        todo!();
        /*
            return isSubtypeOfExt(rhs, nullptr);
        */
    }

    /**
      | How this type will appear as if it were a type
      |   annotation in Python which is sometimes
      |   different than how it appears in declarations
      |   (e.g. int[] vs List[int])
      |
      | Takes a custom printer that users can pass in
      | to customize the output of this method.
      */
    pub fn annotation_str(&self, printer: TypePrinter) -> String {
        
        todo!();
        /*
            if (printer) {
          // the printer can return nullopt to fall through to the default impl
          if (auto renamed = printer(shared_from_this())) {
            return *renamed;
          }
        }
        return annotation_str_impl(printer);
        */
    }
    
    pub fn annotation_str(&self) -> String {
        
        todo!();
        /*
            // Overload instead of define a default value for `printer` to help
        // debuggers out.
        return annotation_str(nullptr);
        */
    }
    
    pub fn kind(&self) -> TypeKind {
        
        todo!();
        /*
            return kind_;
        */
    }
}

pub trait TypeInterface:
PartialEq<Type>
+ IsSubtypeOfExt
+ IsModule
+ ReprStr
+ RequiresGrad
+ HasFreeVariables
+ ContainedTypes
+ CreateWithContained {}
 
/**
  | subtyping relation. By default, we
  | return true for the case when the type
  | is exactly equal or if this <: T where
  | rhs = Optional[T]
  |
  */
pub trait IsSubtypeOfExt {

    /**
      | if this returns false and the why_not stream
      | is non-null, it contains additional details
      | that describe why this is not a subtype of
      | 'rhs'.
      |
      | This additional information should only
      | contain details that are not obvious from the
      | annotation_str() that describes the type. For
      | instance it is clear that `int <: str` is
      | false but not clear why `Foo <: InterfaceBar`
      | might be false.
      */
    fn is_subtype_of_ext(&self, 
        rhs:     &TypePtr,
        why_not: *mut std::io::BufWriter) -> bool;
}

pub trait IsModule {
    
    fn is_module(&self) -> bool;
}

pub trait Str {
    
    /// How this type will appear in FunctionSchema declarations
    fn str_(&self) -> String;
}

pub trait ReprStr {

    /**
      | Returns a human readable string that includes
      | additional information like "type is inferred
      | rather than explictly defined" to help
      | construct more user-friendly messages.
      */
    fn repr_str(&self) -> String {
        
        todo!();
        /*
            return annotation_str();
        */
    }
}

pub trait RequiresGrad {
    
    fn requires_grad(&self) -> bool {
        
        todo!();
        /*
            for (const auto& ct : containedTypes()) {
          if (ct->requires_grad()) {
            return true;
          }
        }
        return false;
        */
    }
}

pub trait HasFreeVariables {
    
    fn has_free_variables(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}

pub trait ContainedTypes {
    
    /**
      | list of types this type contains, e.g.
      | for a List then element type of a list
      | for a tuple, the types of the tuple elements
      |
      */
    fn contained_types(&self) -> &[TypePtr] {
        
        todo!();
        /*
            return {};
        */
    }
}

pub trait CreateWithContained {
    
    /**
      | per-type constructor, you only need
      | to override this if the containedTypes()
      | is not empty
      |
      */
    fn create_with_contained(&self, contained_types: Vec<TypePtr>) -> TypePtr {
        
        todo!();
        /*
            AT_ERROR(
            "type with contained types did not overload createWithContained: ",
            str());
        */
    }
}
