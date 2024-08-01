# Logging --> errorfile.txt
from datetime import datetime

def log_error(error_id, function, reason):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("errorfile.txt", "a") as f:
        f.write(f"Timestamp: {timestamp}, Error ID: {error_id}, Function: {function}, Reason: {reason}\n")