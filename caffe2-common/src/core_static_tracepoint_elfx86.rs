crate::ix!();

/**
  | Default constraint for the probe arguments
  | as operands.
  |
  */
#[cfg(not(caffe_sdt_arg_constraint))]
#[macro_export] macro_rules! caffe_sdt_arg_constraint {
    () => {
        todo!();
        /*
                "nor"
        */
    }
}

/// Instruction to emit for the probe.
#[macro_export] macro_rules! caffe_sdt_nop {
    () => {
        todo!();
        /*
                nop
        */
    }
}

/// Note section properties.
#[macro_export] macro_rules! caffe_sdt_note_name {
    () => {
        todo!();
        /*
                "stapsdt"
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_note_type {
    () => {
        todo!();
        /*
                3
        */
    }
}

/// Size of address depending on platform.
#[cfg(__lp64__)]
#[macro_export] macro_rules! caffe_sdt_asm_addr {
    () => {
        todo!();
        /*
                .8byte
        */
    }
}

#[cfg(not(__lp64__))]
#[macro_export] macro_rules! caffe_sdt_asm_addr {
    () => {
        todo!();
        /*
                .4byte
        */
    }
}

// Assembler helper Macros.
#[macro_export] macro_rules! caffe_sdt_s {
    ($x:ident) => {
        todo!();
        /*
                #x
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_asm_1 {
    ($x:ident) => {
        todo!();
        /*
                CAFFE_SDT_S(x) "\n"
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_asm_2 {
    ($a:ident, $b:ident) => {
        todo!();
        /*
                CAFFE_SDT_S(a) "," CAFFE_SDT_S(b) "\n"
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_asm_3 {
    ($a:ident, $b:ident, $c:ident) => {
        todo!();
        /*
                CAFFE_SDT_S(a) "," CAFFE_SDT_S(b) ","    
                                              CAFFE_SDT_S(c) "\n"
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_asm_string {
    ($x:ident) => {
        todo!();
        /*
                CAFFE_SDT_ASM_1(.asciz CAFFE_SDT_S(x))
        */
    }
}


// Helper to determine the size of an argument.
#[macro_export] macro_rules! caffe_sdt_isarray {
    ($x:ident) => {
        todo!();
        /*
                (__builtin_classify_type(x) == 14)
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_argsize {
    ($x:ident) => {
        todo!();
        /*
                (CAFFE_SDT_ISARRAY(x) ? sizeof(void*) : sizeof(x))
        */
    }
}

/**
  | Format of each probe arguments as operand.
  |
  | Size of the argument tagged with CAFFE_SDT_Sn,
  | with "n" constraint.
  |
  | Value of the argument tagged with CAFFE_SDT_An,
  | with configured constraint.
  */
#[macro_export] macro_rules! caffe_sdt_arg {
    ($n:ident, $x:ident) => {
        todo!();
        /*
        
          [CAFFE_SDT_S##n] "n"                ((size_t)CAFFE_SDT_ARGSIZE(x)),          
          [CAFFE_SDT_A##n] CAFFE_SDT_ARG_CONSTRAINT (x)
        */
    }
}

// Templates to append arguments as operands.
#[macro_export] macro_rules! caffe_sdt_operands_0 {
    () => {
        todo!();
        /*
                [__sdt_dummy] "g" (0)
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_operands_1 {
    ($_1:ident) => {
        todo!();
        /*
                CAFFE_SDT_ARG(1, _1)
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_operands_2 {
    ($_1:ident, $_2:ident) => {
        todo!();
        /*
        
          CAFFE_SDT_OPERANDS_1(_1), CAFFE_SDT_ARG(2, _2)
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_operands_3 {
    ($_1:ident, $_2:ident, $_3:ident) => {
        todo!();
        /*
        
          CAFFE_SDT_OPERANDS_2(_1, _2), CAFFE_SDT_ARG(3, _3)
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_operands_4 {
    ($_1:ident, $_2:ident, $_3:ident, $_4:ident) => {
        todo!();
        /*
        
          CAFFE_SDT_OPERANDS_3(_1, _2, _3), CAFFE_SDT_ARG(4, _4)
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_operands_5 {
    ($_1:ident, $_2:ident, $_3:ident, $_4:ident, $_5:ident) => {
        todo!();
        /*
        
          CAFFE_SDT_OPERANDS_4(_1, _2, _3, _4), CAFFE_SDT_ARG(5, _5)
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_operands_6 {
    ($_1:ident, $_2:ident, $_3:ident, $_4:ident, $_5:ident, $_6:ident) => {
        todo!();
        /*
        
          CAFFE_SDT_OPERANDS_5(_1, _2, _3, _4, _5), CAFFE_SDT_ARG(6, _6)
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_operands_7 {
    ($_1:ident, $_2:ident, $_3:ident, $_4:ident, $_5:ident, $_6:ident, $_7:ident) => {
        todo!();
        /*
        
          CAFFE_SDT_OPERANDS_6(_1, _2, _3, _4, _5, _6), CAFFE_SDT_ARG(7, _7)
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_operands_8 {
    ($_1:ident, $_2:ident, $_3:ident, $_4:ident, $_5:ident, $_6:ident, $_7:ident, $_8:ident) => {
        todo!();
        /*
        
          CAFFE_SDT_OPERANDS_7(_1, _2, _3, _4, _5, _6, _7), CAFFE_SDT_ARG(8, _8)
        */
    }
}

/**
  | Templates to reference the arguments
  | from operands in note section.
  |
  */
#[macro_export] macro_rules! caffe_sdt_argfmt {
    ($no:ident) => {
        todo!();
        /*
                %n[CAFFE_SDT_S##no]@%[CAFFE_SDT_A##no]
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_arg_template_0 {
    () => {
        todo!();
        /*
                /*No arguments*/
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_arg_template_1 {
    () => {
        todo!();
        /*
                CAFFE_SDT_ARGFMT(1)
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_arg_template_2 {
    () => {
        todo!();
        /*
                CAFFE_SDT_ARG_TEMPLATE_1 CAFFE_SDT_ARGFMT(2)
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_arg_template_3 {
    () => {
        todo!();
        /*
                CAFFE_SDT_ARG_TEMPLATE_2 CAFFE_SDT_ARGFMT(3)
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_arg_template_4 {
    () => {
        todo!();
        /*
                CAFFE_SDT_ARG_TEMPLATE_3 CAFFE_SDT_ARGFMT(4)
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_arg_template_5 {
    () => {
        todo!();
        /*
                CAFFE_SDT_ARG_TEMPLATE_4 CAFFE_SDT_ARGFMT(5)
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_arg_template_6 {
    () => {
        todo!();
        /*
                CAFFE_SDT_ARG_TEMPLATE_5 CAFFE_SDT_ARGFMT(6)
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_arg_template_7 {
    () => {
        todo!();
        /*
                CAFFE_SDT_ARG_TEMPLATE_6 CAFFE_SDT_ARGFMT(7)
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_arg_template_8 {
    () => {
        todo!();
        /*
                CAFFE_SDT_ARG_TEMPLATE_7 CAFFE_SDT_ARGFMT(8)
        */
    }
}


// Structure of note section for the probe.
#[macro_export] macro_rules! caffe_sdt_note_content {
    ($provider:ident, $name:ident, $arg_template:ident) => {
        todo!();
        /*
        
          CAFFE_SDT_ASM_1(990: CAFFE_SDT_NOP)                                          
          CAFFE_SDT_ASM_3(     .pushsection .note.stapsdt,"","note")                   
          CAFFE_SDT_ASM_1(     .balign 4)                                              
          CAFFE_SDT_ASM_3(     .4byte 992f-991f, 994f-993f, CAFFE_SDT_NOTE_TYPE)       
          CAFFE_SDT_ASM_1(991: .asciz CAFFE_SDT_NOTE_NAME)                             
          CAFFE_SDT_ASM_1(992: .balign 4)                                              
          CAFFE_SDT_ASM_1(993: CAFFE_SDT_ASM_ADDR 990b)                                
          CAFFE_SDT_ASM_1(     CAFFE_SDT_ASM_ADDR 0) /*Reserved for Semaphore address*/
          CAFFE_SDT_ASM_1(     CAFFE_SDT_ASM_ADDR 0) /*Reserved for Semaphore name*/   
          CAFFE_SDT_ASM_STRING(provider)                                               
          CAFFE_SDT_ASM_STRING(name)                                                   
          CAFFE_SDT_ASM_STRING(arg_template)                                           
          CAFFE_SDT_ASM_1(994: .balign 4)                                              
          CAFFE_SDT_ASM_1(     .popsection)
        */
    }
}

// Main probe Macro.
#[macro_export] macro_rules! caffe_sdt_probe {
    ($provider:ident, $name:ident, $n:ident, $arglist:ident) => {
        todo!();
        /*
        
            __asm__ __volatile__ (                                                     
              CAFFE_SDT_NOTE_CONTENT(provider, name, CAFFE_SDT_ARG_TEMPLATE_##n)       
              :: CAFFE_SDT_OPERANDS_##n arglist                                        
            )                                                                          
        */
    }
}

// Helper Macros to handle variadic arguments.
#[macro_export] macro_rules! caffe_sdt_narg_ {
    ($_0:ident, $_1:ident, $_2:ident, $_3:ident, $_4:ident, $_5:ident, $_6:ident, $_7:ident, $_8:ident, $N:ident, $($arg:ident),*) => {
        todo!();
        /*
                N
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_narg {
    (, $($arg:ident),*) => {
        todo!();
        /*
        
          CAFFE_SDT_NARG_(__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0)
        */
    }
}

#[macro_export] macro_rules! caffe_sdt_probe_n {
    ($provider:ident, $name:ident, $N:ident, $($arg:ident),*) => {
        todo!();
        /*
        
          CAFFE_SDT_PROBE(provider, name, N, (__VA_ARGS__))
        */
    }
}
