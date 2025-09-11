import time

class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.running = False

    def start(self):
        if not self.running:
            self.start_time = time.perf_counter()
            self.running = True
            print("Stopwatch started.")

    def stop(self):
        if self.running:
            end_time = time.perf_counter()
            elapsed_time = end_time - self.start_time # type: ignore
            self.running = False
            print(f"Stopwatch stopped. Elapsed time: {elapsed_time:.2f} seconds.")
            return elapsed_time
        else:
            print("Stopwatch is not running.")
            return 0
        
    @staticmethod
    def create_and_start():
        sw = Stopwatch()
        sw.start()
        return sw

    def reset(self):
        self.start_time = None
        self.running = False
        print("Stopwatch reset.")
