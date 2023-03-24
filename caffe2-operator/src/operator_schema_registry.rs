crate::ix!();

/**
  | @brief
  | 
  | A registry to hold all the operator schemas.
  | 
  | OpSchemaRegistry should not need to
  | be instantiated.
  |
  */
pub struct OpSchemaRegistry {

}

impl OpSchemaRegistry {
    
    #[inline] pub fn new_schema<'a>(
        key:  &String,
        file: &String,
        line: i32) -> &'a mut OpSchema 
    {
        todo!();
        /*
            auto& m = map();
        auto it = m.find(key);
        if (it != m.end()) {
          const auto& schema = it->second;
          std::ios_base::Init init;
          std::cerr << "Trying to register schema with name " << key
                    << " from file " << file << " line " << line
                    << ", but it is already registered from file " << schema.file()
                    << " line " << schema.line();
          abort();
        }
        m.emplace(std::make_pair(key, OpSchema(key, file, line)));
        return m[key];
        */
    }
    
    #[inline] pub fn schema(key: &String) -> *const OpSchema {
        
        todo!();
        /*
            auto& m = map();
        auto it = m.find(key);
        if (it != m.end()) {
          return &it->second;
        } else {
          return nullptr;
        }
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Returns the underlying string to
      | 
      | OpSchema map.
      | 
      | You should not manually manipulate
      | the map object returned. Instead, use
      | the macros defined such as OPERATOR_SCHEMA
      | to register your operator schema.
      | 
      | We wrap it inside a function to avoid
      | the static initialization order fiasco.
      |
      */
    #[inline] pub fn map<'a>(&'a mut self) -> &'a mut HashMap<String,OpSchema> {
        
        todo!();
        /*
            static CaffeMap<string, OpSchema> map;
      return map;
        */
    }
}
