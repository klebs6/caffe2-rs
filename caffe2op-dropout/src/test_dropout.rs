crate::ix!();

///----------------------------------------
#[test] fn dropout_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Dropout",
        ["X"],
        ["Y"] + ["mask"],
        ratio=0.5,
        is_test=0
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(5, 5)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))
    print("mask:", workspace.FetchBlob("mask"))

    **Result**

    X: [[5. 4. 3. 6. 9.]
     [2. 1. 8. 0. 9.]
     [7. 3. 0. 6. 3.]
     [1. 8. 2. 6. 4.]
     [6. 2. 6. 4. 0.]]
    Y: [[ 0.  0.  0. 12. 18.]
     [ 0.  0. 16.  0.  0.]
     [ 0.  0.  0. 12.  6.]
     [ 0.  0.  4.  0.  0.]
     [12.  0.  0.  0.  0.]]
    mask: [[False False False  True  True]
     [False False  True  True False]
     [False False  True  True  True]
     [False False  True False False]
     [ True False False False False]]
    */
}
