import os
import subprocess
import signal
import threading
import time

print("Launching 4 servers and tracker in another process")

def launch_rpc_server(device_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    process = subprocess.Popen(
        ["python", "-m", "tvm.exec.rpc_server", "--key", "h100", "--tracker", "0.0.0.0:9089"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True  # Ensures output is handled as text, not bytes
    )
    print(f"RPC server {device_id} launched with PID {process.pid}")
    return process

def launch_main_rpc_server():
    process = subprocess.Popen(
        ["python", "-m", "tvm.exec.rpc_tracker", "--host", "0.0.0.0", "--port", "9089"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True  # Ensures output is handled as text, not bytes
    )
    print(f"Main RPC tracker launched with PID {process.pid}")
    return process

def stream_output(process, label):
    """Continuously stream process output to the main program."""
    def stream(pipe, stream_type):
        for line in iter(pipe.readline, ''):
            print(f"[{label} {stream_type}] {line.strip()}")
        pipe.close()

    threading.Thread(target=stream, args=(process.stdout, "stdout"), daemon=True).start()
    threading.Thread(target=stream, args=(process.stderr, "stderr"), daemon=True).start()

def kill_servers_on_ctrl_c():
    processes = [launch_main_rpc_server()]
    time.sleep(3)
    for i in range(4):
        processes.append(launch_rpc_server(i))
        time.sleep(1)
    for idx, proc in enumerate(processes):
        stream_output(proc, f"Process {idx}")

    try:
        while True:
            time.sleep(1)  # Prevent immediate exit
    except KeyboardInterrupt:
        print("\nKilling servers...")
        for proc in processes:
            if proc.poll() is None:  # Check if still running
                os.kill(proc.pid, signal.SIGINT)
                print(f"Sent SIGINT to process {proc.pid}")
        print("All servers killed.")

kill_servers_on_ctrl_c()
