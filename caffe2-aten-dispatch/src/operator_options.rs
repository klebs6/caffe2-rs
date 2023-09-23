crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/dispatch/OperatorOptions.h]

#[repr(u8)]
pub enum AliasAnalysisKind {

    INTERNAL_SPECIAL_CASE,

    /// The most conservative alias analysis type,
    /// assumes side-effects. This is the default
    /// analysis.
    ///
    CONSERVATIVE, 
    FROM_SCHEMA,
    PURE_FUNCTION
}

#[inline] pub fn to_string(alias_analysis_kind: AliasAnalysisKind) -> *const u8 {
    
    todo!();
        /*
            return (aliasAnalysisKind == AliasAnalysisKind::CONSERVATIVE)
          ? "CONSERVATIVE"
          : (aliasAnalysisKind == AliasAnalysisKind::FROM_SCHEMA)
              ? "FROM_SCHEMA"
              : (aliasAnalysisKind == AliasAnalysisKind::PURE_FUNCTION)
                  ? "PURE_FUNCTION"
                  : (aliasAnalysisKind == AliasAnalysisKind::INTERNAL_SPECIAL_CASE)
                      ? "INTERNAL_SPECIAL_CASE"
                      : "UNKNOWN";
        */
}
