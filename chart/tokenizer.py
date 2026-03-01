#
# file: chart/tokenizer.py
# desc: Provides functionality for tokenizing vectorized chart data
# auth: Benjamin Piro (brp8396@rit.edu)
# date: 2026-02-28
#

class ChartTokenizer:
    def __init__(self):
        pass

    @staticmethod
    def tokenize_chart(vectorized_chart_data):
        """
        Tokenizes vectorized chart data into a sequence of event vectors, which can be used as input for mapping to discrete token IDs for training data preparation.

        Args:
            - vectorized_chart_data: The vectorized representation of the chart data to be tokenized.
        """
        raise NotImplementedError("Chart tokenization logic not yet implemented.")
    
