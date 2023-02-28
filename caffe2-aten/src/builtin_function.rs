crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/builtin_function.h]

pub mod jit {

    use super::*;

    pub struct BuiltinOpFunction {
        base:       Function,
        name:       QualifiedName,
        callable:   fn(_0: &mut Stack) -> (),
        schema:     FunctionSchema,
        doc_string: String,
    }

    impl BuiltinOpFunction {
        
        pub fn new(
            qualname:   QualifiedName,
            schema:     FunctionSchema,
            callable:   fn(_0: &mut Stack) -> (),
            doc_string: String) -> Self {
            let doc_string: String = doc_string.unwrap_or("");
            todo!();
            /*
            : name(std::move(qualname)),
            : callable(std::move(callable)),
            : schema(std::move(schema)),
            : doc_string(std::move(doc_string)),

                TORCH_INTERNAL_ASSERT(schema_.returns().size() == 1);
            */
        }
        
        pub fn doc_string(&self) -> &String {
            
            todo!();
            /*
                return doc_string_;
            */
        }
        
        pub fn is_graph_function(&self) -> bool {
            
            todo!();
            /*
                return false;
            */
        }
        
        pub fn run(&mut self, stack: &mut Stack)  {
            
            todo!();
            /*
                callable_(stack);
            */
        }
        
        pub fn run(&mut self, stack: Stack)  {
            
            todo!();
            /*
                callable_(stack);
            */
        }
        
        pub fn run_async(&mut self, 
            stack:    &mut Stack,
            not_used: TaskLauncher) -> IntrusivePtr<ivalue::Future> {
            
            todo!();
            /*
                run(stack);
            auto res = c10::make_intrusive<c10::ivalue::Future>(stack.front().type());
            res->markCompleted(std::move(stack.front()));
            return res;
            */
        }
        
        pub fn invoke(&mut self, 
            stack:  Vec<IValue>,
            kwargs: &Kwargs) -> IValue {
            
            todo!();
            /*
                getSchema().checkAndNormalizeInputs(stack, kwargs);
            callable_(stack);
            return stack.front();
            */
        }
        
        pub fn qualname(&self) -> &QualifiedName {
            
            todo!();
            /*
                return name_;
            */
        }
        
        pub fn name(&self) -> &String {
            
            todo!();
            /*
                return name_.name();
            */
        }

        /**
         | if this isn't yet defined, run its method_creator
         | function
         |
         */
        pub fn ensure_defined(&mut self)  {
            
            todo!();
            /*
                // nop
            */
        }
        
        pub fn graph(&self) -> Arc<Graph> {
            
            todo!();
            /*
                TORCH_INTERNAL_ASSERT(false , "BuiltinFunction had a graph requested "
              "from it. This probably indicates that the JIT calling context needs a "
              "special case on Function::isGraphFunction()");
            */
        }
        
        pub fn optimized_graph(&self) -> Arc<Graph> {
            
            todo!();
            /*
                TORCH_INTERNAL_ASSERT(false , "BuiltinFunction had a graph requested "
              "from it. This probably indicates that the JIT calling context needs a "
              "special case on Function::isGraphFunction()");
            */
        }
        
        pub fn clear_execution_info(&mut self)  {
            
            todo!();
            /*
                TORCH_INTERNAL_ASSERT(false , "BuiltinFunction had a graph requested "
              "from it. This probably indicates that the JIT calling context needs a "
              "special case on Function::isGraphFunction()");
            */
        }
        
        pub fn get_executor(&mut self) -> &mut GraphExecutor {
            
            todo!();
            /*
                TORCH_INTERNAL_ASSERT(false , "BuiltinFunction had a GraphExecutor requested "
              "from it. This probably indicates that the JIT calling context needs a "
              "special case on Function::isGraphFunction()");
            */
        }
        
        pub fn get_schema(&self) -> &FunctionSchema {
            
            todo!();
            /*
                return schema_;
            */
        }
        
        pub fn num_inputs(&self) -> usize {
            
            todo!();
            /*
                return schema_.arguments().size();
            */
        }
        
        pub fn check_single_output(&mut self)  {
            
            todo!();
            /*
                TORCH_CHECK(schema_.returns().size() == 1);
            */
        }
        
        pub fn pretty_print_schema(&self) -> String {
            
            todo!();
            /*
                #ifdef __NVCC__
            // Disable the "statement is unreachable" warning
            #pragma diag_suppress code_is_unreachable
            #endif

            TORCH_INTERNAL_ASSERT(false);
            return "";

            #ifdef __NVCC__
            #pragma diag_default code_is_unreachable
            #endif
            */
        }
        
        pub fn set_schema(&mut self, schema: FunctionSchema) -> &mut Function {
            
            todo!();
            /*
                schema_ = std::move(schema);
            return *this;
            */
        }
    }
}
