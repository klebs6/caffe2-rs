crate::ix!();

#[macro_export] macro_rules! use_ideep_def_aliases {
    () => {
        /* the tensor type created/handled by iDEEP  */                              
        pub type itensor = ideep::tensor;                                               

        /* the date layout of iDEEP tensor */                                        
        pub type iformat = ideep::format_tag;                                           

        /* the scales for iDEEP tensor with different data type */                   
        pub type iscale = ideep::scale_t;                                               

        /* the detial algorithm for iDEEP operators, e.g. winograd */                
        pub type ialgo = ideep::algorithm;                                              

        /* the kind of propagation for iDEEP operators, e.g. forward, training */    
        pub type iprop = ideep::prop_kind;                                              

        /* the kind of low precision operators, e.g. signed/unsigned activation */   
        pub type ilowp_kind = ideep::lowp_kind;                                         

        /* the data type of iDEEP tensor, e.g. f32, u8, s8 */                        
        pub type idtype = ideep::tensor::data_type;                                     

        /* the descriptor of iDEEP tensor */                                         
        pub type itdesc = ideep::tensor::descriptor;                                    

        /* the attribute for operator to describe the details of inputs&fusion */    
        pub type iattr = ideep::attr_t;                                                 

        /* the detail flags for batch normalization */                               
        pub type ibn_flag = ideep::batch_normalization_flag;
    }
}

