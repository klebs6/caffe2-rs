crate::ix!();

use crate::{
    Workspace,
};

pub struct WorkspaceTestFoo {}

caffe_known_type!{WorkspaceTestFoo}

#[test] fn workspace_test_blob_access() {
    todo!();
    /*
      Workspace ws;

      EXPECT_FALSE(ws.HasBlob("nonexisting"));
      EXPECT_EQ(ws.GetBlob("nonexisting"), nullptr);

      EXPECT_EQ(ws.GetBlob("newblob"), nullptr);
      EXPECT_NE(nullptr, ws.CreateBlob("newblob"));
      EXPECT_NE(nullptr, ws.GetBlob("newblob"));
      EXPECT_TRUE(ws.HasBlob("newblob"));

      // Different names should still be not created.
      EXPECT_FALSE(ws.HasBlob("nonexisting"));
      EXPECT_EQ(ws.GetBlob("nonexisting"), nullptr);

      // Check if the returned Blob is OK for all operations
      Blob* blob = ws.GetBlob("newblob");
      int* int_unused CAFFE2_UNUSED = blob->GetMutable<int>();
      EXPECT_TRUE(blob->IsType<int>());
      EXPECT_FALSE(blob->IsType<WorkspaceTestFoo>());
      EXPECT_NE(&blob->Get<int>(), nullptr);

      // Re-creating the blob does not change the content as long as it already
      // exists.
      EXPECT_NE(nullptr, ws.CreateBlob("newblob"));
      EXPECT_TRUE(blob->IsType<int>());
      EXPECT_FALSE(blob->IsType<WorkspaceTestFoo>());
      // When not null, we should only call with the right type.
      EXPECT_NE(&blob->Get<int>(), nullptr);

      // Re-creating the blob through CreateLocalBlob does not change the content
      // either.
      EXPECT_NE(nullptr, ws.CreateLocalBlob("newblob"));
      EXPECT_TRUE(blob->IsType<int>());
      EXPECT_NE(&blob->Get<int>(), nullptr);

      // test removing blob
      EXPECT_FALSE(ws.HasBlob("nonexisting"));
      EXPECT_FALSE(ws.RemoveBlob("nonexisting"));
      EXPECT_TRUE(ws.HasBlob("newblob"));
      EXPECT_TRUE(ws.RemoveBlob("newblob"));
      EXPECT_FALSE(ws.HasBlob("newblob"));
  */
}


#[test] fn workspace_test_run_empty_plan() {
    todo!();
    /*
      PlanDef plan_def;
      Workspace ws;
      EXPECT_TRUE(ws.RunPlan(plan_def));
  */
}


#[test] fn workspace_test_sharing() {
    todo!();
    /*
      Workspace parent;
      EXPECT_FALSE(parent.HasBlob("a"));
      EXPECT_TRUE(parent.CreateBlob("a"));
      EXPECT_TRUE(parent.GetBlob("a"));
      {
        Workspace child(&parent);
        // Child can access parent blobs
        EXPECT_TRUE(child.HasBlob("a"));
        EXPECT_TRUE(child.GetBlob("a"));
        // Child can create local blobs
        EXPECT_FALSE(child.HasBlob("b"));
        EXPECT_FALSE(child.GetBlob("b"));
        EXPECT_TRUE(child.CreateBlob("b"));
        EXPECT_TRUE(child.GetBlob("b"));
        // Parent cannot access child blobs
        EXPECT_FALSE(parent.GetBlob("b"));
        EXPECT_FALSE(parent.HasBlob("b"));
        // Parent can create duplicate names
        EXPECT_TRUE(parent.CreateBlob("b"));
        // But child has local overrides
        EXPECT_NE(child.GetBlob("b"), parent.GetBlob("b"));
        // Child can create a blob that already exists in the parent
        EXPECT_TRUE(child.CreateBlob("a"));
        EXPECT_EQ(child.GetBlob("a"), parent.GetBlob("a"));
        // Child can create a local blob for the blob already exists in the parent
        EXPECT_TRUE(child.CreateLocalBlob("a"));
        // But the local blob will be different from the one in parent workspace
        EXPECT_NE(child.GetBlob("a"), parent.GetBlob("a"));
      }
  */
}


#[test] fn workspace_test_blob_mapping() {
    todo!();
    /*
      Workspace parent;
      EXPECT_FALSE(parent.HasBlob("a"));
      EXPECT_TRUE(parent.CreateBlob("a"));
      EXPECT_TRUE(parent.GetBlob("a"));
      {
        std::unordered_map<string, string> forwarded_blobs;
        forwarded_blobs["inner_a"] = "a";
        Workspace child(&parent, forwarded_blobs);
        EXPECT_FALSE(child.HasBlob("a"));
        EXPECT_TRUE(child.HasBlob("inner_a"));
        EXPECT_TRUE(child.GetBlob("inner_a"));
        Workspace ws;
        EXPECT_TRUE(ws.CreateBlob("b"));
        forwarded_blobs.clear();
        forwarded_blobs["inner_b"] = "b";
        child.AddBlobMapping(&ws, forwarded_blobs);
        EXPECT_FALSE(child.HasBlob("b"));
        EXPECT_TRUE(child.HasBlob("inner_b"));
        EXPECT_TRUE(child.GetBlob("inner_b"));
      }
  */
}

/**
  | Checks that Workspace::ForEach(f)
  | applies f on the specified set of workspaces
  | in any order.
  |
  */
#[inline] pub fn for_each_check(workspaces: &[*mut Workspace])  {
    
    todo!();
    /*
        std::unordered_set<Workspace*> expected(workspaces);
      std::unordered_set<Workspace*> actual;
      Workspace::ForEach([&](Workspace* ws) {
        auto inserted = actual.insert(ws).second;
        EXPECT_TRUE(inserted);
      });
      EXPECT_EQ(actual, expected);
    */
}

#[test] fn workspace_test_for_each() {
    todo!();
    /*
      forEachCheck({});

      {
        Workspace ws1;
        forEachCheck({&ws1});

        {
          Workspace ws2;
          forEachCheck({&ws1, &ws2});
        }

        forEachCheck({&ws1});
      }

      forEachCheck({});
  */
}
