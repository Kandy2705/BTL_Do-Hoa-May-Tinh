class Scene:
    def __init__(self):
        self.objects = []

        self.selected_objects = []

    def add_object(self, obj):
        self.objects.append(obj)

    def remove_object(self, obj):
        self.objects.remove(obj)

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

