class Scenario:
    def __init__(self, scenario_id=0) -> None:
        self.graph = None
        self.reverted_graph = None
        self.id = scenario_id
        self.K_path = None
        self.ig_mask = None
        self.counts = None
        self.thresh_dict = {}
        self.data_dict = {}
        self.similarity_masks = None
        self.auto_masks = {}
