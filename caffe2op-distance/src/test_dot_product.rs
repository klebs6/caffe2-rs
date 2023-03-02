crate::ix!();

#[test] fn dot_product_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "DotProduct",
        ["X",  "Y"],
        ["Z"]
    )

    workspace.FeedBlob("X", np.random.randint(20, size=(5)).astype(np.float32))
    workspace.FeedBlob("Y", np.random.randint(20, size=(5)).astype(np.float32))
    print("X:\n", workspace.FetchBlob("X"))
    print("Y:\n", workspace.FetchBlob("Y"))
    workspace.RunOperatorOnce(op)
    print("Z:\n", workspace.FetchBlob("X"))


    workspace.ResetWorkspace()
    workspace.FeedBlob("X", np.random.randint(10, size=(3,3)).astype(np.float32))
    workspace.FeedBlob("Y", np.random.randint(10, size=(3,3)).astype(np.float32))
    print("X:\n", workspace.FetchBlob("X"))
    print("Y:\n", workspace.FetchBlob("Y"))
    workspace.RunOperatorOnce(op)
    print("Z:\n", workspace.FetchBlob("Z"))

    **Result**

    X:
     [ 2. 15.  2.  7. 12.]
    Y:
     [ 3. 12.  9.  3. 18.]
    Z:
     [ 2. 15.  2.  7. 12.]
    X:
     [[2. 0. 4.]
     [7. 7. 4.]
     [7. 9. 9.]]
    Y:
     [[2. 0. 8.]
     [9. 6. 1.]
     [7. 8. 0.]]
    Z:
     [ 36. 109. 121.]

    */
}
