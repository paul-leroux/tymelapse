def log_message(log_file, message):
    log_file.write(message + "\n")
    log_file.flush()