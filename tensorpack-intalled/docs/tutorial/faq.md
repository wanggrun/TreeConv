
# FAQs

## Does it support data format X / augmentation Y / layer Z?

The library tries to __support__ everything, but it could not really __include__ everything.

The interface attempts to be flexible enough so you can put any XYZ on it.
You can either implement them under the interface or simply wrap some existing Python code.
See [Extend Tensorpack](index.html#extend-tensorpack)
for more details.

If you think:
1. The framework has limitation in its interface so your XYZ cannot be supported, OR
2. Your XYZ is super common / very well-defined / very useful, so it would be nice to include it.

Then it is a good time to open an issue.

## How to print/dump intermediate results during training

1. Learn `tf.Print`. Most of the times, adding one line in between:

   ```python
   tensor = obtain_a_tensor()
   tensor = tf.Print(tensor, [tf.shape(tensor), tensor], tensor.name, summarize=100)
   use_the_tensor(tensor)
   ```
   is sufficient.

2. Know [DumpTensors](../modules/callbacks.html#tensorpack.callbacks.DumpTensors),
	[ProcessTensors](../modules/callbacks.html#tensorpack.callbacks.ProcessTensors) callbacks.
	And it's also easy to write your own version of them.

3. The [ProgressBar](../modules/callbacks.html#tensorpack.callbacks.ProgressBar)
	 callback can print some scalar statistics, though not enabled by default.

4. Read [Summary and Logging](summary.html) for more options on logging.

## How to freeze some variables in training

1. Learn `tf.stop_gradient`. You can simply use `tf.stop_gradient` in your model code in many situations (e.g. to freeze first several layers).
	 Note that it stops the gradient flow in the current Tensor but your variables may still contribute to the
	 final loss through other tensors (e.g., weight decay).

2. [varreplace.freeze_variables](../modules/tfutils.html#tensorpack.tfutils.varreplace.freeze_variables) returns a context where variables are freezed.
	It is implemented by `custom_getter` argument of `tf.variable_scope` -- learn it to gain more control over what & how variables are freezed.

3. [ScaleGradient](../modules/tfutils.html#tensorpack.tfutils.gradproc.ScaleGradient) can be used to set the gradients of some variables to 0.
	But it may be slow, since variables still have gradients.

Note that the above methods only prevent variables being updated by SGD.
Some variables may be updated by other means,
e.g., BatchNorm statistics are updated through the `UPDATE_OPS` collection and the [RunUpdateOps](../modules/callbacks.html#tensorpack.callbacks.RunUpdateOps) callback.

## The model does not run on CPUs?

Some TensorFlow ops are not implemented on CPUs.
For example, it does not support many ops in NCHW format on CPUs.
Note that if you use MKL-enabled version of TensorFlow, it supports more NCHW ops.

In general, you need to implement the model in a way your version of TensorFlow supports.

## My training seems slow. Why?

Checkout the [Performance Tuning tutorial](performance-tuning.html)
