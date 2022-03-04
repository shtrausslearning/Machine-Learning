def run_model_setup(self, inputs, labels, is_training):
    logits = self.model_layers(inputs, is_training)

    # convert logits to probabilities with softmax activation
    self.probs = tf.nn.softmax(logits, name='probs')
    # round probabilities
    self.predictions = tf.math.argmax(
        self.probs, axis=-1, name='predictions')
    class_labels = tf.math.argmax(labels, axis=-1)
    # find which predictions were correct
    is_correct = tf.math.equal(
        self.predictions, class_labels)
    is_correct_float = tf.cast(
        is_correct,
        tf.float32)
    
    # compute ratio of correct to incorrect predictions
    self.accuracy = tf.math.reduce_mean(
        is_correct_float)
    
    # train model
    if self.is_training:
        labels_float = tf.cast(
            labels, tf.float32)
        # compute the loss using cross_entropy
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels_float,
            logits=logits)
        self.loss = tf.math.reduce_mean(
            cross_entropy)
        # use adam to train model
        adam = tf.compat.v1.train.AdamOptimizer()
        self.train_op = adam.minimize(
            self.loss, global_step=self.global_step)
