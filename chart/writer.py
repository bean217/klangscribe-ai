#
# file: chart/writer.py
# desc: Provides functionality for constructing .chart files from alternative data formats,
#       such as vectorized or tokenized representations of chart data.
# auth: Benjamin Piro (brp8396@rit.edu)
# date: 2026-02-28
#

import numpy as np
import torch as th

class ChartWriter:
    def __init__(self):
        pass


######################
#                    #
#   Public Methods   #
#                    #
######################


    def write_chart(vectorized_data, output_path: str):
        """
        Constructs a .chart file from vectorized chart data and writes it to the specified output path.

        Args:
            - vectorized_data: The vectorized representation of the chart data to be written.
            - output_path (str): The file path where the constructed .chart file should be saved.
        """
        raise NotImplementedError("Chart writing logic not yet implemented.")
    