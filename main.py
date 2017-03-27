import sys
import train
import track
import predict

command = sys.argv[1]
args = sys.argv[2:]

if command == "train":
    train.train(*args)
elif command == "track":
    track.record(*args)
elif command == "predict":
    predict.loop(*args)
