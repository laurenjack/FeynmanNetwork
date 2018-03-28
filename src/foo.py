import tensorflow as tf

@tf.RegisterGradient("CustomClipGrad")
def _clip_grad(unused_op, grad):
  return tf.clip_by_value(grad, -0.1, 0.1)

input = tf.Variable([3.0], dtype=tf.float32)

g = tf.get_default_graph()
with g.gradient_override_map({"Exp": "CustomClipGrad"}):
  output_clip = tf.exp(input, name="Exp")
grad_clip = tf.gradients(output_clip, input)

# output without gradient clipping in the backwards pass for comparison:
output = tf.exp(input)
grad = tf.gradients(output, input)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print("with clipping:", sess.run(grad_clip)[0])
  print("without clipping:", sess.run(grad)[0])