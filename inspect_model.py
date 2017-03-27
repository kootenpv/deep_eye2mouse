import matplotlib.pyplot
import numpy as np
import get_model
import get_data

X, y = get_data.get_training_xy()
model = get_model.get_model("deep_eye2mouse")

for i in range(3):
    print(i, np.mean((model.predict(X[i::3]) - y[i::3])**2))

for j in range(100):
    print(model.predict(X[i::3])[j])
    print(y[i::3][j])
    print()

matplotlib.pyplot.scatter(model.predict(X[2::3])[:100, 0],
                          model.predict(X[2::3])[:100, 1], c="red")
matplotlib.pyplot.scatter(y[2::3][:100, 0], y[2::3][:100, 1], c="blue")
for v in range(60):
    v = v + 100
    matplotlib.pyplot.scatter(model.predict(X[2::3])[v, 0], model.predict(X[2::3])[v, 1], c="red")
    matplotlib.pyplot.scatter(y[2::3][v, 0], y[2::3][v, 1], c="blue")
    plt.pause(0.05)
