class Task():
    def __init__(self, classes, label_to_task, scale=1):
        self.classes = classes
        self.label_to_task = label_to_task
        self.size = len(classes)
        self.scale = scale