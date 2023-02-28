/*!
  | Simple registry implementation that
  | uses static variables to register object
  | creators during program initialization
  | time.
  |
  | NB: This Registry works poorly when you have
  | other namespaces.
  |
  | Make all macro invocations from inside the at
  | namespace.
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/Registry.h]

#[inline] pub fn key_str_repr_a<KeyType>(key: &KeyType) -> String {

    todo!();
        /*
            return "[key type printing not supported]";
        */
}

#[inline] pub fn key_str_repr_b(key: &String) -> String {
    
    todo!();
        /*
            return key;
        */
}

pub enum RegistryPriority {
    REGISTRY_FALLBACK  = 1,
    REGISTRY_DEFAULT   = 2,
    REGISTRY_PREFERRED = 3,
}

/**
  | -----------
  | @brief
  | 
  | A template class that allows one to register
  | classes by keys.
  | 
  | The keys are usually a string specifying
  | the name, but can be anything that can
  | be used in a map.
  | 
  | You should most likely not use the Registry
  | class explicitly, but use the helper
  | macros below to declare specific registries
  | as well as registering objects.
  |
  */
pub struct Registry<SrcType,ObjectPtrType,Args> {
    registry:       HashMap<SrcType,RegistryCreator<Args,ObjectPtrType>>,
    priority:       HashMap<SrcType,RegistryPriority>,
    terminate:      bool,
    warning:        bool,
    help_message:   HashMap<SrcType,String>,
    register_mutex: RawMutex,
}

pub type RegistryCreator<Args,ObjectPtrType> = fn(_0: Args) -> ObjectPtrType;

impl<SrcType,ObjectPtrType,Args> Registry<SrcType,ObjectPtrType,Args> {
    
    pub fn new(warning: Option<bool>) -> Self {

        let warning: bool = warning.unwrap_or(true);

        todo!();
        /*


            : registry_(), priority_(), terminate_(true), warning_(warning)
        */
    }
    
    pub fn register(&mut self, 
        key:      &SrcType,
        creator:  RegistryCreator<Args,ObjectPtrType>,
        priority: Option<RegistryPriority>)  {

        let priority: RegistryPriority = priority.unwrap_or(RegistryPriority::REGISTRY_DEFAULT);

        todo!();
        /*
            lock_guard<mutex> lock(register_mutex_);
        // The if statement below is essentially the same as the following line:
        // CHECK_EQ(registry_.count(key), 0) << "Key " << key
        //                                   << " registered twice.";
        // However, CHECK_EQ depends on google logging, and since registration is
        // carried out at static initialization time, we do not want to have an
        // explicit dependency on glog's initialization function.
        if (registry_.count(key) != 0) {
          auto cur_priority = priority_[key];
          if (priority > cur_priority) {
    #ifdef DEBUG
            string warn_msg =
                "Overwriting already registered item for key " + KeyStrRepr(key);
            fprintf(stderr, "%s\n", warn_msg.c_str());
    #endif
            registry_[key] = creator;
            priority_[key] = priority;
          } else if (priority == cur_priority) {
            string err_msg =
                "Key already registered with the same priority: " + KeyStrRepr(key);
            fprintf(stderr, "%s\n", err_msg.c_str());
            if (terminate_) {
              exit(1);
            } else {
              throw runtime_error(err_msg);
            }
          } else if (warning_) {
            string warn_msg =
                "Higher priority item already registered, skipping registration of " +
                KeyStrRepr(key);
            fprintf(stderr, "%s\n", warn_msg.c_str());
          }
        } else {
          registry_[key] = creator;
          priority_[key] = priority;
        }
        */
    }
    
    pub fn register_a(&mut self, 
        key:      &SrcType,
        creator:  RegistryCreator<Args,ObjectPtrType>,
        help_msg: &String,
        priority: Option<RegistryPriority>)  {

        let priority: RegistryPriority = priority.unwrap_or(RegistryPriority::REGISTRY_DEFAULT);

        todo!();
        /*
            Register(key, creator, priority);
        help_message_[key] = help_msg;
        */
    }
    
    #[inline] pub fn has(&mut self, key: &SrcType) -> bool {
        
        todo!();
        /*
            return (registry_.count(key) != 0);
        */
    }
    
    pub fn create(&mut self, 
        key:  &SrcType,
        args: Args) -> ObjectPtrType {
        
        todo!();
        /*
            if (registry_.count(key) == 0) {
          // Returns nullptr if the key is not registered.
          return nullptr;
        }
        return registry_[key](args...);
        */
    }

    /**
      | Returns the keys currently registered
      | as a vector.
      |
      */
    pub fn keys(&self) -> Vec<SrcType> {
        
        todo!();
        /*
            vector<SrcType> keys;
        for (const auto& it : registry_) {
          keys.push_back(it.first);
        }
        return keys;
        */
    }
    
    #[inline] pub fn help_message(&self) -> &HashMap<SrcType,String> {
        
        todo!();
        /*
            return help_message_;
        */
    }
    
    pub fn help_message_a(&self, key: &SrcType) -> *const u8 {
        
        todo!();
        /*
            auto it = help_message_.find(key);
        if (it == help_message_.end()) {
          return nullptr;
        }
        return it->second.c_str();
        */
    }

    /**
      | Used for testing, if terminate is unset,
      | 
      | Registry throws instead of calling
      | exit
      |
      */
    pub fn set_terminate(&mut self, terminate: bool)  {
        
        todo!();
        /*
            terminate_ = terminate;
        */
    }
}

pub struct Registerer {

}

impl Registerer {

    pub fn new_a<SrcType,ObjectPtrType,Args>(
        key:      &SrcType,
        registry: *mut Registry<SrcType,ObjectPtrType,Args>,
        creator:  RegistryCreator<Args, ObjectPtrType>,
        help_msg: Option<&str>) -> Self {

        todo!();
        /*
            registry->Register(key, creator, help_msg);
        */
    }
    
    pub fn new_b<SrcType,ObjectPtrType,Args>(
        key:      &SrcType,
        priority: RegistryPriority,
        registry: *mut Registry<SrcType,ObjectPtrType,Args>,
        creator:  RegistryCreator<Args,ObjectPtrType>,
        help_msg: Option<&str>) -> Self {

        todo!();
        /*
            registry->Register(key, creator, help_msg, priority);
        */
    }
    
    pub fn default_creator<DerivedType,Args,ObjectPtrType>(args: Args) -> ObjectPtrType {
    
        todo!();
        /*
            return ObjectPtrType(new DerivedType(args...));
        */
    }
}

/**
  | C10_DECLARE_TYPED_REGISTRY is a macro
  | that expands to a function declaration,
  | as well as creating a convenient typename
  | for its corresponding registerer.
  |
  | Note on C10_IMPORT and C10_EXPORT below: we
  | need to explicitly mark DECLARE as import and
  | DEFINE as export, because these registry macros
  | will be used in downstream shared libraries as
  | well, and one cannot use *_API - the API macro
  | will be defined on a per-shared-library
  | basis. Semantically, when one declares a typed
  | registry it is always going to be IMPORT, and
  | when one defines a registry (which should
  | happen ONLY ONCE and ONLY IN SOURCE FILE), the
  | instantiation unit is always going to be
  | exported.
  |
  | The only unique condition is when in the same
  | file one does DECLARE and DEFINE - in Windows
  | compilers, this generates a warning that
  | dllimport and dllexport are mixed, but the
  | warning is fine and linker will be properly
  | exporting the symbol. Same thing happens in the
  | gflags flag declaration and definition caes.
  */
#[macro_export] macro_rules! c10_declare_typed_registry {
    ($RegistryName:ident, $SrcType:ident, $ObjectType:ident, $PtrType:ident, $($arg:ident),*) => {
        /*
        
          C10_IMPORT ::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>* 
          RegistryName();                                                          
          typedef ::Registerer<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>   
              Registerer##RegistryName
        */
    }
}

#[macro_export] macro_rules! c10_define_typed_registry {
    ($RegistryName:ident, $SrcType:ident, $ObjectType:ident, $PtrType:ident, $($arg:ident),*) => {
        /*
        
          C10_EXPORT ::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>* 
          RegistryName() {                                                         
            static ::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*   
                registry = new ::                                             
                    Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>();       
            return registry;                                                       
          }
        */
    }
}

#[macro_export] macro_rules! c10_define_typed_registry_without_warning {
    ($RegistryName:ident, $SrcType:ident, $ObjectType:ident, $PtrType:ident, $($arg:ident),*) => {
        /*
        
          C10_EXPORT ::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*    
          RegistryName() {                                                            
            static ::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*      
                registry =                                                            
                    new ::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>( 
                        false);                                                       
            return registry;                                                          
          }
        */
    }
}

/**
  | Note(Yangqing): The __VA_ARGS__ below allows
  | one to specify a templated creator with comma
  | in its templated arguments.
  |
  */
#[macro_export] macro_rules! c10_register_typed_creator {
    ($RegistryName:ident, $key:ident, $($arg:ident),*) => {
        /*
        
          static Registerer##RegistryName C10_ANONYMOUS_VARIABLE(g_##RegistryName)( 
              key, RegistryName(), ##__VA_ARGS__);
        */
    }
}

#[macro_export] macro_rules! c10_register_typed_creator_with_priority {
    ($RegistryName:ident, $key:ident, $priority:ident, $($arg:ident),*) => {
        /*
        
          static Registerer##RegistryName C10_ANONYMOUS_VARIABLE(g_##RegistryName)( 
              key, priority, RegistryName(), ##__VA_ARGS__);
        */
    }
}

#[macro_export] macro_rules! c10_register_typed_class {
    ($RegistryName:ident, $key:ident, $($arg:ident),*) => {
        /*
        
          static Registerer##RegistryName C10_ANONYMOUS_VARIABLE(g_##RegistryName)( 
              key,                                                                  
              RegistryName(),                                                       
              Registerer##RegistryName::DefaultCreator<__VA_ARGS__>,                
              ::demangle_type<__VA_ARGS__>());
        */
    }
}

#[macro_export] macro_rules! c10_register_typed_class_with_priority {
    ($RegistryName:ident, $key:ident, $priority:ident, $($arg:ident),*) => {
        /*
        
          static Registerer##RegistryName C10_ANONYMOUS_VARIABLE(g_##RegistryName)( 
              key,                                                                  
              priority,                                                             
              RegistryName(),                                                       
              Registerer##RegistryName::DefaultCreator<__VA_ARGS__>,                
              ::demangle_type<__VA_ARGS__>());
        */
    }
}

/**
  | c10_declare_registry and c10_define_registry
  | are hard-wired to use string as the key type,
  | because that is the most commonly used cases.
  |
  */
#[macro_export] macro_rules! c10_declare_registry {
    ($RegistryName:ident, $ObjectType:ident, $($arg:ident),*) => {
        /*
        
          C10_DECLARE_TYPED_REGISTRY(                               
              RegistryName, string, ObjectType, unique_ptr, ##__VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! c10_define_registry {
    ($RegistryName:ident, $ObjectType:ident, $($arg:ident),*) => {
        /*
        
          C10_DEFINE_TYPED_REGISTRY(                               
              RegistryName, string, ObjectType, unique_ptr, ##__VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! c10_declare_shared_registry {
    ($RegistryName:ident, $ObjectType:ident, $($arg:ident),*) => {
        /*
        
          C10_DECLARE_TYPED_REGISTRY(                                      
              RegistryName, string, ObjectType, shared_ptr, ##__VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! c10_define_shared_registry {
    ($RegistryName:ident, $ObjectType:ident, $($arg:ident),*) => {
        /*
        
          C10_DEFINE_TYPED_REGISTRY(                                      
              RegistryName, string, ObjectType, shared_ptr, ##__VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! c10_define_shared_registry_without_warning {
    ($RegistryName:ident, $ObjectType:ident, $($arg:ident),*) => {
        /*
        
          C10_DEFINE_TYPED_REGISTRY_WITHOUT_WARNING(        
              RegistryName, string, ObjectType, shared_ptr, ##__VA_ARGS__)
        */
    }
}

/**
  | C10_REGISTER_CREATOR and C10_REGISTER_CLASS
  | are hard-wired to use string as the key
  | type, because that is the most commonly
  | used cases.
  |
  */
#[macro_export] macro_rules! c10_register_creator {
    ($RegistryName:ident, $key:ident, $($arg:ident),*) => {
        /*
        
          C10_REGISTER_TYPED_CREATOR(RegistryName, #key, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! c10_register_creator_with_priority {
    ($RegistryName:ident, $key:ident, $priority:ident, $($arg:ident),*) => {
        /*
        
          C10_REGISTER_TYPED_CREATOR_WITH_PRIORITY(                                  
              RegistryName, #key, priority, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! c10_register_class {
    ($RegistryName:ident, $key:ident, $($arg:ident),*) => {
        /*
        
          C10_REGISTER_TYPED_CLASS(RegistryName, #key, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! c10_register_class_with_priority {
    ($RegistryName:ident, $key:ident, $priority:ident, $($arg:ident),*) => {
        /*
        
          C10_REGISTER_TYPED_CLASS_WITH_PRIORITY(                                  
              RegistryName, #key, priority, __VA_ARGS__)

        */
    }
}
