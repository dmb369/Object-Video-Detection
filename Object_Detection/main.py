from ultralytics import YOLO

model=YOLO('best.pt')    #this was done by first training the model in roboflow:)
results=model(source='test.jpg', show=True, conf=0.4, save=True)
