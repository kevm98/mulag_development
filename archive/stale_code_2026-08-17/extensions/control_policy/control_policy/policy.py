class ControlPolicy:
    def __init__(self, config=None):
        self.config = config or {}

    def compute_action(self, observation):
        # Replace with your policy logic
        return observation
