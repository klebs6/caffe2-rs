crate::ix!();

#[test] fn clip_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Clip",
        ["X"],
        ["Y"],
        min=20.0,
        max=60.0

    )

    workspace.FeedBlob("X", (np.random.randint(100, size=(5,5))).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X: [[45. 16. 59. 99. 48.]
     [12. 44. 46. 82. 28.]
     [ 1. 91. 18.  9. 71.]
     [24. 37. 61. 12. 81.]
     [36. 38. 30. 84. 40.]]

    Y: [[45. 20. 59. 60. 48.]
     [20. 44. 46. 60. 28.]
     [20. 60. 20. 20. 60.]
     [24. 37. 60. 20. 60.]
     [36. 38. 30. 60. 40.]]
    */
}
