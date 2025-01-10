import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CountImages import count

print("Train count:")
count('./FER2013/train')

print("Test count:")
count('./FER2013/test')