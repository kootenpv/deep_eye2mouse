## Deep Eye2Mouse

### Installation

Install the stuff in requirements, opencv2 is tough, good luck with tensorflow and keras as well.

### track (to learn)

```bash
python main.py track livin_room
```

Press "s" to start learning. Follow the mouse around and keep your eyes focused on it.
Press "q" to quit (might be difficult :D just make it crash)

### train (model)

```bash
python main.py train livin_room deep1
```

Trains on your data.

### predict (go live with the model)

```bash
python main.py predict
```

Press "s" to start predicting and moving the mouse around.
Press "q" to quit (might be difficult :D just make it crash)
