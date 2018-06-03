WEEK 1:
Initial subdirectory setup

Created skeleton data structures for the decision nodes and all possible tests.  Although not much code has been written yet, it is definitely more important to have a solid foundation to your code, which is what we wanted to begin focusing on.

WEEK 2:
Successfully parsed out the data and began the recursive algorithm for building the ID3 tree.  There was definitely a learning curve involved in getting the mldata script to do what I wanted it to do, especially getting the type of each attribute.  Also, I really want to use a numpy array, because it has the convient column sort functionality, but it takes about 30 seconds to convert the data from 'spam' to a numpy array. I'm trying to figure out if it's worth the added time overhead or not.


WEEK 3:
Honestly, did almost the entire project this week.  I rewrote the continuous split algorithm 3 times to decrease runtime from O(n^2) to O(n), which made a huge difference on runtime. Then, I hade to put it all together and actually implement the algorithm, and make sure that the main method in dtree was user friendly.  This is the first python project I've done in a while, and I'm sure that there are a lot of code chunks that seem convoluted and overly-complicated. I'd love any tips of improvment for future projects!



HOW TO RUN:

simply type in py dtree.py
The dataset must be in P1 for it to work (I didn't have time to see if it would work for a general path, but maybe it does)

FOr all of the data generated for the write up, run writeup.py