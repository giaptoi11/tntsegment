import time
import threading
def go(n):
    time.sleep(n)
    return "done"

run1 = threading.Thread(target=go, args=(10,))
run1.start()
run1.join()
print(run1)