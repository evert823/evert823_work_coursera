from datetime import datetime

class KMeansCluster:
    def __init__(self):
        pass

    def print_with_tms(self, message):
        mytimestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{mytimestamp}|{message}")
