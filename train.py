import get_data
import get_model


def train(data_name, model_name):
    X, y = get_data.get_training_xy("~/tracktrack/" + data_name + "/")
    model = get_model.get_model(model_name)

    try:
        model.fit(X[0::3], y[0::3], verbose=1, batch_size=64, nb_epoch=300,
                  validation_data=(X[1::3], y[1::3]))
    except KeyboardInterrupt:
        pass

    get_model.save_model(model, model_name)
