crate::ix!();

/**
  | These #defines are useful when writing passes
  | as the collapse
  |
  | if (!cond) {
  |   continue; // or break; or return;
  | }
  |
  | into a single line without negation
  */
#[macro_export] macro_rules! nom_require_or_ {
    ($_cond:ident, $_expr:ident) => {
        todo!();
        /*
        
          if (!(_cond)) {                     
            _expr;                            
          }
        */
    }
}

#[macro_export] macro_rules! nom_require_or_cont {
    ($_cond:ident) => {
        todo!();
        /*
                NOM_REQUIRE_OR_(_cond, continue)
        */
    }
}

#[macro_export] macro_rules! nom_require_or_break {
    ($_cond:ident) => {
        todo!();
        /*
                NOM_REQUIRE_OR_(_cond, break)
        */
    }
}

#[macro_export] macro_rules! nom_require_or_ret_null {
    ($_cond:ident) => {
        todo!();
        /*
                NOM_REQUIRE_OR_(_cond, return nullptr)
        */
    }
}

#[macro_export] macro_rules! nom_require_or_ret_false {
    ($_cond:ident) => {
        todo!();
        /*
                NOM_REQUIRE_OR_(_cond, return false)
        */
    }
}

#[macro_export] macro_rules! nom_require_or_ret_true {
    ($_cond:ident) => {
        todo!();
        /*
                NOM_REQUIRE_OR_(_cond, return true)
        */
    }
}

#[macro_export] macro_rules! nom_require_or_ret {
    ($_cond:ident) => {
        todo!();
        /*
                NOM_REQUIRE_OR_(_cond, return )
        */
    }
}
