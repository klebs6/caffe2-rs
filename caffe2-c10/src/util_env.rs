crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/env.h]

/**
  | Reads an environment variable and returns
  | - optional<true>,              if set equal to "1"
  | - optional<false>,             if set equal to "0"
  | - nullopt,   otherwise
  |
  | NB:
  | Issues a warning if the value of the
  | environment variable is not 0 or 1.
  */
pub fn check_env(name: *const u8) -> Option<bool> {
    
    todo!();
        /*
            auto envar = getenv(name);
      if (envar) {
        if (strcmp(envar, "0") == 0) {
          return false;
        }
        if (strcmp(envar, "1") == 0) {
          return true;
        }
        TORCH_WARN(
            "Ignoring invalid value for boolean flag ",
            name,
            ": ",
            envar,
            "valid values are 0 or 1.");
      }
      return nullopt;
        */
}


