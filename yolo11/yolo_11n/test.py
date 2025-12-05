from ultralytics import YOLO

model = YOLO("yolo11n.pt")

def main():
    model.train(
        data="",
        epoch = 100,
        batch = 16,
        imasz = 320
    )
    
if __name__ == "__main__" :
    main()    