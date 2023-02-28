crate::ix!();

use crate::{
    NNModule
};

#[inline] pub fn dead_code_elim<T,U>(nn: *mut NNModule<T,U>)  {
    
    todo!();
    /*
        // Iteratively remove unconsumed non-external outputs.
      bool changed = false;
      do {
        changed = false;
        for (const auto& node : nn->dataFlow.getMutableNodes()) {
          NOM_REQUIRE_OR_CONT(nn::is<repr::NeuralNetOperator>(node));

          bool isUsed = false;
          for (const auto& output : nn::getOutputs(node)) {
            if (nn::hasConsumer(output) || nn->outputs.count(output)) {
              isUsed = true;
              break;
            }
          }

          NOM_REQUIRE_OR_CONT(!isUsed);

          // No outputs are used, delete them and the node itself.
          for (const auto& output : nn::getOutputs(node)) {
            nn->dataFlow.deleteNode(output);
          }
          nn->dataFlow.deleteNode(node);
          changed = true;
          break;
        }
      } while (changed);
    */
}

register_opt_pass_from_func!{DeadCodeElim, deadCodeElim}
