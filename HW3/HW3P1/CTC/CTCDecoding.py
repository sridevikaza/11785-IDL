import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1
        

        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        symbols_len, seq_len, batch_size = y_probs.shape
        self.symbol_set = ["-"] + self.symbol_set
        for batch_itr in range(batch_size):
            
            path = " "
            path_prob = 1
            for i in range(seq_len):
                max_idx = np.argmax(y_probs[:, i, batch_itr])
                path_prob *= y_probs[max_idx, i, batch_itr]
                if path[-1] != self.symbol_set[max_idx]:
                    path += self.symbol_set[max_idx]
        
            path = path.replace('-', '')
            decoded_path.append(path[1:])

        return path[1:], path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """
        self.symbol_set = ['-'] + self.symbol_set
        symbols_len, seq_len, batch_size = y_probs.shape
        bestPaths = dict()
        tempBestPaths = dict()
        bestPaths['-'] = 1

        # iterate over sequence len
        for t in range(seq_len):
            sym_probs = y_probs[:, t]
            tempBestPaths = dict()

            # iterate best paths
            for path, score in bestPaths.items():

                # iterate symbols
                for r, prob in enumerate(sym_probs):
                    new_path = path

                    # make new path
                    if path[-1] == '-':
                        new_path = new_path[:-1] + self.symbol_set[r]
                    elif (path[-1] != self.symbol_set[r]) and not (t==seq_len-1 and self.symbol_set[r]=='-'):
                        new_path += self.symbol_set[r]

                    # update probabilities in temp paths
                    if new_path in tempBestPaths:
                        tempBestPaths[new_path] += prob * score
                    else:
                        tempBestPaths[new_path] = prob * score
                    

            # get top k paths and reset temp
            if len(tempBestPaths) >= self.beam_width:
                bestPaths = dict(sorted(tempBestPaths.items(), key=lambda x: x[1], reverse=True)[:self.beam_width])

        # get the highest score
        bestPath = max(bestPaths, key=bestPaths.get)
        finalPaths = dict()
        for path, score in tempBestPaths.items():
            if path[-1] == '-':
                finalPaths[path[:-1]] = score
            else:
                finalPaths[path] = score
        return bestPath, finalPaths