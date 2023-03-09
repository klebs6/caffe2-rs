crate::ix!();

#[test] fn remove_padding_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    addpad_op = core.CreateOperator(
        "AddPadding",
        ["X", "lengths_add"],
        ["Y", "lengths_out_add"],
        padding_width=1
    )

    rmpad_op = core.CreateOperator(
        "RemovePadding",
        ["Y", "lengths_rm"],
        ["Z", "lengths_out_rm"],
        padding_width=1
    )

    workspace.FeedBlob("X", (np.random.randint(20, size=(3,5))))
    workspace.FeedBlob("lengths_add", np.array([3]).astype(np.int32))
    workspace.FeedBlob("lengths_rm", np.array([5]).astype(np.int32))

    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(addpad_op)
    print("Y:", workspace.FetchBlob("Y"))
    print("lengths_out_add:", workspace.FetchBlob("lengths_out_add"))

    workspace.RunOperatorOnce(rmpad_op)
    print("Z:", workspace.FetchBlob("Z"))
    print("lengths_out_rm:", workspace.FetchBlob("lengths_out_rm"))
    ```

    **Result**

    ```
    X: [[17 19  1  9  1]
     [19  3  5 19  1]
     [16  0  0  0  4]]
    Y: [[ 0  0  0  0  0]
     [17 19  1  9  1]
     [19  3  5 19  1]
     [16  0  0  0  4]
     [ 0  0  0  0  0]]
    lengths_out_add: [5]
    Z: [[17 19  1  9  1]
     [19  3  5 19  1]
     [16  0  0  0  4]]
    lengths_out_rm: [3]
    */
}
