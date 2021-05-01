208345561
208271767
*****
Comments:
the better evaluation function is a linear combination of the max_tile value, the score
number of empty_tiles, number of equal adjacent tiles with respect to their values and number of
high value tiles on the edges.
The more empty tiles there are the more space is left on the board to play, equal adjacent tiles
can be merged for a higher score and so they indicate a good state. High value tiles on edges make
sure that when the opponent randomly places new low-valued tiles on the board, the high tiles on edges
that cannot be merged with the new low-valued tiles won't conflict.
The weights of the linear combination came from trial and error and we feel that these weights gave the
best average results.
