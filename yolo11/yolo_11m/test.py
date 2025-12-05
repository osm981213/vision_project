from ultralytics import YOLO

model = YOLO("yolo11m.pt")

def main():
    model.train(
        data="",
        epoch = 100,
        batch = 16,
        imasz = 320,
        device=0
    )
    
if __name__ == "__main__" :
    main()    