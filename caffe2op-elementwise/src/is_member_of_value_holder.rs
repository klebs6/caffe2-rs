crate::ix!();

///---------------------------------
pub struct IsMemberOfValueHolder {
    int32_values:  HashSet<i32>,
    int64_values:  HashSet<i64>,
    bool_values:   HashSet<bool>,
    string_values: HashSet<String>,
    has_values:    bool,
}

impl IsMemberOfValueHolder {

    #[inline] pub fn set<T>(&mut self, args: &Vec<T>) {
        todo!();
        /*
            has_values_ = true;
            auto& values = get<T>();
            values.insert(args.begin(), args.end());
        */
    }
    
    #[inline] pub fn has_values(&self) -> bool {
        
        todo!();
        /*
            return has_values_;
        */
    }

    #[inline] pub fn get_i32<'a>(&'a mut self) -> &'a HashSet<i32> { 
        &self.int32_values 
    }

    #[inline] pub fn get_i64<'a>(&'a mut self) -> &'a HashSet<i64> { 
        &self.int64_values 
    }
    #[inline] pub fn get_bool<'a>(&'a mut self) -> &'a HashSet<bool> { 
        &self.bool_values 
    }

    #[inline] pub fn get_string<'a>(&'a mut self) -> &'a HashSet<String> { 
        &self.string_values 
    }

    #[inline] pub fn get<'a, T>(&'a mut self) -> &'a HashSet<T> { 
        todo!();
        //dispatch
    }
}

