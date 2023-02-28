crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/utils/ParamsHash.h]

/**
  | Hashing machinery for Params
  |
  | Fowler–Noll–Vo hash function see
  | https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
  |
  | Params must be a POD because we read out its memory
  | contenst as char* when hashing
  */
pub struct ParamsHash<Params: PlainOldData> {

}

impl ParamsHash<Params: PlainOldData> {
    
    pub fn invoke(&self, params: &Params) -> usize {
        
        todo!();
        /*
            auto ptr = reinterpret_cast<const u8*>(&params);
        u32 value = 0x811C9DC5;
        for (int i = 0; i < (int)sizeof(Params); ++i) {
          value ^= ptr[i];
          value *= 0x01000193;
        }
        return (usize)value;
        */
    }
}

/**
  | Params must be a POD because we read out
  | its memory contenst as char* when comparing
  |
  */
pub struct ParamsEqual<Params: PlainOldData> {

}

impl ParamsEqual<Params: PlainOldData> {
    
    pub fn invoke(&self, 
        a: &Params,
        b: &Params) -> bool {
        
        todo!();
        /*
            auto ptr1 = reinterpret_cast<const u8*>(&a);
        auto ptr2 = reinterpret_cast<const u8*>(&b);
        return memcmp(ptr1, ptr2, sizeof(Params)) == 0;
        */
    }
}
