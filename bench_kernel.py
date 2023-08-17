#! /usr/bin/env python3

import socket
from pathlib import Path
from subprocess import *
import shlex
from time import *
import os, shutil
import re
import sys
import argparse
import struct


def start_kernel():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", dest="run_string", action="store", default="")
    parser.add_argument(
        "-b", "--build", dest="build_string", action="store", default=""
    )
    parser.add_argument(
        "-e",
        "--embedding-length",
        type=int,
        dest="emb_length",
        action="store",
        default=47,
    )
    parser.add_argument(
        "-n", "--name", dest="bench_name", action="store", required=True
    )
    parser.add_argument(
        "-i", "--instance-num", type=int, dest="instance", action="store", required=True
    )
    parser.add_argument(
        "--repeats", type=int, dest="bench_repeats", action="store", default=1
    )
    parser.add_argument(
        "-p", "--plugin", dest="plugin_path", action="store", default="plugin"
    )
    args = parser.parse_args()

    socket_name = "kernel.soc"

    build_str = (
        f"$AARCH_GCC -fplugin={args.plugin_path} -O2 -fplugin-arg-plugin-dyn_replace=learning "
        f"-fplugin-arg-plugin-remote_socket={socket_name} "
        f"{args.build_string} -o main.elf"
    )

    gprof_build_str = (
        f"$AARCH_GCC -fplugin={args.plugin_path} -O2 -fplugin-arg-plugin-dyn_replace=learning "
        f"-fplugin-arg-plugin-remote_socket={socket_name} -pg "
        f"{args.build_string} -o pg_main.elf"
    )

    with socket.socket(
        socket.AF_UNIX, socket.SOCK_DGRAM, 0
    ) as gcc_socket, socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM, 0) as env_socket:
        gcc_socket.bind(socket_name)
        env_socket.bind(f"\0{args.bench_name}:backend_{args.instance}")

        afk_env = set()
        while True:  # Bench kernels works until there is at least one env using it
            gcc_instance = Popen(
                build_str, stderr=PIPE, shell=True
            )  # Compile without gprof do all the compilation stuff

            while not os.path.exists("gcc_plugin.soc"):
                gcc_instance.poll()
                if gcc_instance.returncode != None:
                    print(
                        f"gcc failed: return code {gcc_instance.returncode}\n",
                        file=sys.stderr,
                    )
                    print(
                        gcc_instance.communicate()[1].decode("utf-8"), file=sys.stderr
                    )
                    exit(1)
                pass

            active_funcs_lists = {}
            while gcc_instance.poll() == None:
                try:
                    fun_name = gcc_socket.recv(4096, socket.MSG_DONTWAIT).decode(
                        "utf-8"
                    )
                    func_env_address = f"\0{args.bench_name}:{fun_name}_{args.instance}"
                    try:
                        # Send message with fun name only if we have not done it on prev cycle
                        if fun_name not in afk_env:
                            env_socket.sendto(
                                fun_name.encode("utf-8"), func_env_address
                            )
                        pass_list, rem_address = env_socket.recvfrom(
                            4096, socket.MSG_DONTWAIT
                        )
                        if func_env_address != rem_address.decode("utf-8"):
                            print(
                                (
                                    f"Received msg from unexpected address. Expected [{func_env_address}] "
                                    f"got [{rem_address.decode('utf-8')}]"
                                ),
                                file=sys.stderr,
                            )
                            continue
                        list_msg = fun_name.ljust(100, "\0") + pass_list.decode("utf-8")
                        gcc_socket.sendto(
                            list_msg.encode("utf-8"), "gcc_plugin.soc".encode("utf-8")
                        )
                        embedding = gcc_socket.recv(1024)  # surely will be enough
                        try:
                            env_socket.sendto(
                                embedding, func_env_address.encode("utf-8")
                            )
                        except (ConnectionError, FileNotFoundError):
                            print(
                                f"Environment [{func_env_address}] unexpectedly died",
                                file=sys.stderr,
                            )
                            continue
                        active_funcs_lists[fun_name] = list_msg
                        if fun_name in afk_env:
                            afk_env.remove(fun_name)
                    except (
                        ConnectionError,
                        FileNotFoundError,
                        BlockingIOError,
                        ConnectionRefusedError,
                    ) as e:
                        if isinstance(e, BlockingIOError):
                            try:
                                env_socket.sendto("".encode("utf-8"), func_env_address)
                                afk_env = afk_env | {fun_name}
                            except ConnectionRefusedError:
                                afk_env = afk_env - {fun_name}
                        else:
                            afk_env = afk_env - {fun_name}
                        list_msg = fun_name.ljust(100, "\0") + bytes(3996).decode(
                            "utf-8"
                        )
                        gcc_socket.sendto(
                            list_msg.encode("utf-8"), "gcc_plugin.soc".encode("utf-8")
                        )
                        embedding = gcc_socket.recv(1024)  # surely will be enough
                        continue
                except BlockingIOError:
                    pass

            if (
                gcc_instance.returncode != 0
            ):  # We do not expect gcc to break, but who knows
                print(
                    f"gcc failed: return code {gcc_instance.returncode}\n",
                    file=sys.stderr,
                )
                print(gcc_instance.communicate()[1].decode("utf-8"), file=sys.stderr)
                exit(1)

            size_info = (
                run(
                    "nm --demangle --print-size --size-sort --radix=d main.elf",
                    shell=True,
                    capture_output=True,
                )
                .stdout.decode("utf-8")
                .splitlines()
            )
            sizes = {}
            for line in size_info:
                pieces = line.split()
                if pieces[2] != "t" and pieces[2] != "T":
                    continue
                sizes[pieces[3]] = int(pieces[1])

            gcc_instance = Popen(
                gprof_build_str, stderr=PIPE, shell=True
            )  # Compile with gprof to get per-function runtime info

            while not os.path.exists("gcc_plugin.soc"):
                gcc_instance.poll()
                if gcc_instance.returncode != None:
                    print(
                        f"gcc failed: return code {gcc_instance.returncode}\n",
                        file=sys.stderr,
                    )
                    print(
                        gcc_instance.communicate()[1].decode("utf-8"), file=sys.stderr
                    )
                    exit(1)
                pass

            while gcc_instance.poll() == None:
                try:
                    fun_name = gcc_socket.recv(4096, socket.MSG_DONTWAIT).decode(
                        "utf-8"
                    )
                    if fun_name in active_funcs_lists:
                        gcc_socket.sendto(
                            active_funcs_lists[fun_name].encode("utf-8"),
                            "gcc_plugin.soc".encode("utf-8"),
                        )
                        embedding = gcc_socket.recv(1024)
                    else:
                        list_msg = fun_name.ljust(100, "\0") + bytes(3996).decode(
                            "utf-8"
                        )
                        gcc_socket.sendto(
                            list_msg.encode("utf-8"), "gcc_plugin.soc".encode("utf-8")
                        )
                        embedding = gcc_socket.recv(1024)
                except BlockingIOError:
                    pass

            if gcc_instance.wait() != 0:
                print(
                    f"gcc failed: return code {gcc_instance.returncode}\n",
                    file=sys.stderr,
                )
                print(gcc_instance.communicate()[1].decode("utf-8"), file=sys.stderr)
                exit(1)

            for i in range(0, args.bench_repeats):
                run(
                    f"qemu-aarch64 -L /usr/aarch64-linux-gnu ./pg_main.elf {args.run_string}",
                    shell=True,
                )
                run("gprof -s pg_main.elf", shell=True, check=True)

            runtime_data = (
                run(
                    "gprof -bp pg_main.elf", shell=True, capture_output=True, check=True
                )
                .stdout.decode("utf-8")
                .splitlines()[5:]
            )
            os.unlink("gmon.out")
            os.unlink("gmon.sum")

            runtimes = {}
            if " no time accumulated" in runtime_data:
                for fun_name in active_funcs_lists.keys():
                    runtimes[fun_name] = (0.0, 0.0)
            else:
                for line in runtime_data:
                    pieces = line.split()
                    if len(pieces) == 7:
                        runtimes[pieces[6]] = (float(pieces[0]), float(pieces[2]))

            for fun_name in active_funcs_lists.keys():
                func_env_address = f"\0{args.bench_name}:{fun_name}_{args.instance}"
                if fun_name not in sizes:
                    print(
                        f"Symbol [{fun_name}] was not properly profiled, size or runtime data missing",
                        file=sys.stderr,
                    )
                try:
                    env_socket.sendto(
                        bytes(
                            struct.pack(  # runtime_percent runtime_sec size
                                "ddi",
                                runtimes.get(fun_name, (0.0, 0.0))[0],
                                runtimes.get(fun_name, (0.0, 0.0))[1],
                                sizes[fun_name],
                            )
                        ),
                        func_env_address,
                    )
                except (ConnectionError, FileNotFoundError):
                    print(
                        f"Environment [{func_env_address}] unexpectedly died",
                        file=sys.stderr,
                    )

            if len(active_funcs_lists) == 0 and len(afk_env) == 0:
                break

        cwd = os.path.abspath(os.getcwd())
        if cwd.startswith("/tmp") or cwd.startswith("/run"):
            shutil.rmtree(os.getcwd())
        else:
            os.unlink(socket_name)


def test_gcc():
    plugin_path = "test/plugin.so"
    socket_name = "kernel.soc"
    gcc_str = (
        f"$AARCH_GCC -fplugin={plugin_path} -O2 -fplugin-arg-plugin-dyn_replace=learning "
        f"-fplugin-arg-plugin-remote_socket={socket_name} -fplugin-arg-plugin-dump_format=executed "
        "test/main.c -o test/main.elf > test/log.log"
    )

    # setup socket for communication with gcc plugin
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM, 0) as rec_socket:
        rec_socket.bind(socket_name)

        gcc_instance = Popen(gcc_str, stderr=PIPE, shell=True)

        while not os.path.exists("gcc_plugin.soc"):
            gcc_instance.poll()
            if gcc_instance.returncode != None:
                print(f"gcc failed: return code {gcc_instance.returncode}\n")
                print(gcc_instance.communicate()[1].decode("utf-8"))
                exit(1)
            pass

        rec_socket.connect("gcc_plugin.soc")

        while gcc_instance.poll() == None:
            try:
                fun_name = rec_socket.recv(4096, socket.MSG_DONTWAIT)
                print(f"Got new function {fun_name}")
                rec_socket.send(fun_name)
                embedding = rec_socket.recv(47)  # current embedding length
                print(f"Got new embedding {embedding}")
            except BlockingIOError:
                pass

        if gcc_instance.returncode != 0:
            print(f"gcc failed: return code {gcc_instance.returncode}\n")
            print(gcc_instance.communicate()[1].decode("utf-8"))

        os.unlink(socket_name)


if __name__ == "__main__":
    start_kernel()
