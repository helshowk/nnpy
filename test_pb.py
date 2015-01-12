import sys
import time

for i in range(0,1000000):
	time.sleep(0.1)
	pt = str(i) + " / 1,000,000"
	sys.stdout.write("\b" * (len(pt)+1))
	sys.stdout.flush()
	sys.stdout.write(pt)
	sys.stdout.flush()

sys.stdout.write('\n')
