import get_model
import get_data
import track


def loop(model_name):
    model = get_model.get_model(model_name)

    for image, _, _ in track.yield_images():
        im = get_data.prep_images(image)
        x, y = model.predict(im)[0]
        print(x, y)
        # predict x, y and set mouse accordingly
        track.set_mouse_position(x, y)
