#
# file: chart/vocab.py
# desc: Defines the vocabulary and tokenization logic for representing chart data in a format suitable for training KlangScribe models.
# auth: Benjamin Piro (brp8396@rit.edu)
# date: 2026-03-04
#

import math
import numpy as np


#########################
#                       #
#   Token Definitions   #
#                       #
#########################

# Special Case Tokens -- fixed tokens representing specific events/conditions

SPECIAL_TOKENS = (
    "BOS",          # Beginning of Sequence token, used to indicate the start of a chart window
    "EOS",          # End of Sequence token, used to indicate the end of a chart window
    "PAD",          # Padding token, used to fill in empty time steps in a chart window (training only)
)

# Lane On/Off Tokens

LANE_ON_TOKENS = (
    "ON_GREEN",         # Indicates the onset of a note event in the green lane
    "ON_RED",           # Indicates the onset of a note event in the red lane
    "ON_YELLOW",        # Indicates the onset of a note event in the yellow lane
    "ON_BLUE",          # Indicates the onset of a note event in the blue lane
    "ON_ORANGE",        # Indicates the onset of a note event in the orange lane
    "ON_OPEN",          # Indicates the onset of an open note event (i.e., no button press required)
)

LANE_OFF_TOKENS = (
    "OFF_GREEN",        # Indicates the offset of a note event in the green lane
    "OFF_RED",          # Indicates the offset of a note event in the red lane  
    "OFF_YELLOW",       # Indicates the offset of a note event in the yellow lane
    "OFF_BLUE",         # Indicates the offset of a note event in the blue lane
    "OFF_ORANGE",       # Indicates the offset of a note event in the orange lane
    "OFF_OPEN",         # Indicates the offset of an open note event (i.e., no button press required)
)

# Onset Modified Tokens

NOTE_MOD_TOKENS = (
    "NOTEMOD_HOPO",     # Indicates a hammer-on/pull-off note modification
    "NOTEMOD_TAP",      # Indicates a tap note modification
)

# Tie Tokens

TIE_TOKENS = (
    "TIE_GREEN",        # Indicates that a note event in the green lane is tied to the previous green note event
    "TIE_RED",          # Indicates that a note event in the red lane is tied to the previous red note event
    "TIE_YELLOW",       # Indicates that a note event in the yellow lane is tied to the previous yellow note event
    "TIE_BLUE",         # Indicates that a note event in the blue lane is tied to the previous blue note event
    "TIE_ORANGE",       # Indicates that a note event in the orange lane is tied to the previous orange note event
    "TIE_OPEN",         # Indicates that an open note event (i.e., no button press required) is tied to the previous open note event
)

# Time-Shift Tokens (template)

TIME_SHIFT_TOKEN = lambda x: f"TIME_SHIFT_{x}"


#####################################
#                                   #
#    Chart Vocabulary Definition    #
#                                   #
#####################################

class ChartVocab:
    """
    Defines the vocabulary for representing chart data.
    
    Provides methods for converting between raw chart events and their corresponding token representations.

    :param grid_size: The number of discrete time steps in a single fixed-length window of chart data.
    """

    def __init__(
        self, 
        grid_size: int = 100
    ):
        assert grid_size > 0, "Grid size must be a positive integer."
        self.grid_size = grid_size
        self.vocab_tokens = None
        self.vocab_size = 0

        # create the sparse absolute-time note-event vocabulary based on the specified grid size
        self.__initialize_vocab()
    
    def __initialize_vocab(self):
        """
        Initializes the vocabulary tokens based on the specified grid size.
        """
        # define the tokens in the vocabulary
        self.vocab_tokens = list(SPECIAL_TOKENS) + list(LANE_ON_TOKENS) + list(LANE_OFF_TOKENS) + list(NOTE_MOD_TOKENS) + list(TIE_TOKENS)
        self.vocab_tokens += [TIME_SHIFT_TOKEN(i) for i in range(0, self.grid_size)]
        # create a mapping from tokens to their corresponding integer IDs
        self._token_map = {token: idx for idx, token in enumerate(self.vocab_tokens)}
        # store the size of the vocabulary for easy access
        self.vocab_size = len(self.vocab_tokens)

    @property
    def size(self):
        """
        Returns the size of the vocabulary (i.e., the total number of unique tokens).
        """
        if not self.vocab_tokens:
            self.__initialize_vocab()
        return self.vocab_size
    
    @property
    def BOS(self) -> int:
        """
        Returns the integer ID corresponding to the Beginning of Sequence (BOS) token in the vocabulary.
        """
        return self._token_map["BOS"]
    
    @property
    def EOS(self) -> int:
        """
        Returns the integer ID corresponding to the End of Sequence (EOS) token in the vocabulary.
        """
        return self._token_map["EOS"]
    
    @property
    def PAD(self) -> int:
        """
        Returns the integer ID corresponding to the Padding (PAD) token in the vocabulary.
        """
        return self._token_map["PAD"]
    
    def token_to_id(self, token: str) -> int:
        """
        Converts a given token to its corresponding integer ID based on the vocabulary mapping.
        """
        if not self.vocab_tokens:
            self.__initialize_vocab()
        return self._token_map[token]
    
    def tokens_to_ids(self, tokens: list) -> np.ndarray:
        """
        Converts a list of tokens to their corresponding integer IDs based on the vocabulary mapping.
        """
        return np.vectorize(self.token_to_id)(tokens)

    def id_to_token(self, idx: int) -> str:
        """
        Converts a given integer ID back to its corresponding token based on the vocabulary mapping.
        """
        if not self.vocab_tokens:
            self.__initialize_vocab()
        return self.vocab_tokens[idx]
    
    def ids_to_tokens(self, ids: np.ndarray) -> list:
        """
        Converts a list of integer IDs back to their corresponding tokens based on the vocabulary mapping.
        """
        return np.vectorize(self.id_to_token)(ids)


if __name__ == "__main__":
    # Example usage of the ChartVocab class
    window_size_sec = 2.0       # 2s window duration
    timestep_len_sec = 0.02     # 20ms time step langth

    grid_size = math.ceil(window_size_sec / timestep_len_sec)

    vocab = ChartVocab(grid_size=grid_size)
    print(f"Vocabulary size: {vocab.size}")
    print(f"Token to ID mapping: {vocab._token_map}")