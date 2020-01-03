import tensorflow as tf
import hparams
import model
import model_helper
import data_loader

hp = hparams.HParams().hparams
tf.logging.set_verbosity(tf.logging.INFO)


def get_model_fn():
    def model_fn(features, mode):
        xs = [features["in_ids"], features["length"]]

        m = model.Transformer()
        if mode == tf.estimator.ModeKeys.PREDICT:
            predict = m.predict(xs)
            prediction = {
                "y_hat": predict
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=prediction
            )
            return output_spec
        ys = [features["out_ids"], features["y"]]
        loss, train_op = m.get_loss_train_op(xs, ys)
        train_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)
        return train_spec
    return model_fn


def main():
    output_dir = r"./output"
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)
    run_config = model_helper.get_configure()
    model_fn = get_model_fn()
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       config=run_config)
    if hp.model_mode == "train":
        input_path = r"./data/train.recoder"
        train_input_fn = data_loader.file_based_input_fn_builder(input_file_path=input_path,
                                                                 is_training=True,
                                                                 drop_remainder=True,
                                                                 mode=hp.model_mode)
        estimator.train(input_fn=train_input_fn, max_steps=hp.train_steps)
    elif hp.model_mode == "predict":
        input_path = r"./data/test.recoder"
        result_path = r"./output/result.txt"
        vocdict = data_loader.DataLoader().voacb_list
        assert tf.gfile.Exists(input_path)
        train_input_fn = data_loader.file_based_input_fn_builder(input_file_path=input_path,
                                                                 is_training=False,
                                                                 drop_remainder=True,
                                                                 mode=hp.model_mode)
        with tf.gfile.Open(result_path, mode="w") as f:

            for result in estimator.predict(input_fn=train_input_fn,
                                            yield_single_examples=True,
                                            checkpoint_path=hp.predict_ckpt):
                f.write(model_helper.id2sentence(result["y_hat"], vocdict)+"\n")



if __name__ == "__main__":
    main()
