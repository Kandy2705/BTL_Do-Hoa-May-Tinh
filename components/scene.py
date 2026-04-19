class Scene:
    def __init__(self):
        self.objects = []

        self.selected_objects = []

    def add_object(self, obj):
        self.objects.append(obj)

    def remove_object(self, obj):
        if obj in self.objects:
            self.objects.remove(obj)
        if obj in self.selected_objects:
            self.selected_objects.remove(obj)

    def remove_objects(self, objects):
        remove_ids = {id(obj) for obj in objects}
        self.objects[:] = [obj for obj in self.objects if id(obj) not in remove_ids]
        self.selected_objects[:] = [
            obj for obj in self.selected_objects if id(obj) not in remove_ids
        ]

    def select_object(self, obj, multi_select=False):
        if multi_select:
            if obj in self.selected_objects:
                self.selected_objects.remove(obj)
            else:
                self.selected_objects.append(obj)
        else:
            self.selected_objects = [obj]

    def clear_selection(self):
        self.selected_objects.clear()
