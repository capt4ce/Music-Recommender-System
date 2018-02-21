import numpy as np
import pandas as pd

class Popularity():
    def __init__(self):
        self.training_data = None
        self.item_id_col = None
        self.pivot_id_col = None

    def create(self, training_data, item_id_col, pivot_id_col):
        self.item_id_col = item_id_col
        self.pivot_id_col = pivot_id_col
        self.training_data = training_data.groupby([item_id_col]).agg({pivot_id_col: 'sum'}).sort_values([self.pivot_id_col], ascending=False).reset_index()

    def recommend(self, no_of_recommendations):
        return self.training_data.head(no_of_recommendations)
        