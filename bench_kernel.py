#! /usr/bin/env python3

import logging
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


class MultienvBenchKernel:
    def __init__(self, env_socket, gcc_socket):
        logging.debug("KERNEL: init start")
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "-r", "--run", dest="run_string", action="append", default=[]
        )
        self.parser.add_argument(
            "-b", "--build", dest="build_string", action="store", default=""
        )
        self.parser.add_argument(
            "-e",
            "--embedding-length",
            type=int,
            dest="emb_length",
            action="store",
            default=47,
        )
        self.parser.add_argument(
            "-n", "--name", dest="bench_name", action="store", required=True
        )
        self.parser.add_argument(
            "-i",
            "--instance-num",
            type=int,
            dest="instance",
            action="store",
            required=True,
        )
        self.parser.add_argument(
            "--repeats", type=int, dest="bench_repeats", action="store", default=1
        )
        self.parser.add_argument(
            "-p", "--plugin", dest="plugin_path", action="store", default="plugin"
        )
        self.args = self.parser.parse_args()

        if self.args.run_string == []:
            self.args.run_string = [""]

        self.socket_name = "kernel.soc"

        self.build_str = (
            f"$AARCH_GCC -fplugin={self.args.plugin_path} -O2 -fplugin-arg-plugin-dyn_replace=learning "
            f"-fplugin-arg-plugin-remote_socket={self.socket_name} "
            f"{self.args.build_string} -o main.elf"
        )

        self.gprof_build_str = (
            f"$AARCH_GCC -fplugin={self.args.plugin_path} -O2 -fplugin-arg-plugin-dyn_replace=learning "
            f"-fplugin-arg-plugin-remote_socket={self.socket_name} -pg "
            f"{self.args.build_string} -o pg_main.elf"
        )

        symbols_list = Path("benchmark_info.txt")
        lines = [x.strip() for x in symbols_list.read_text().splitlines()]
        index = lines.index("functions:") + 1
        self.bench_symbols = lines[index:]

        self.env_socket = env_socket
        self.gcc_socket = gcc_socket

        self.gcc_socket.bind(self.socket_name)
        self.env_socket.bind(f"\0{self.args.bench_name}:backend_{self.args.instance}")
        logging.debug("KERNEL: init end")

    def final_cleanup(self):
        cwd = os.path.abspath(os.getcwd())
        if cwd.startswith("/tmp") or cwd.startswith("/run"):
            shutil.rmtree(os.getcwd())
        else:
            os.unlink(self.socket_name)
        exit(0)

    def sendout_profiles(self):
        for fun_name in self.active_funcs_lists.keys():
            func_env_address = (
                f"\0{self.args.bench_name}:{fun_name}_{self.args.instance}"
            )
            if fun_name not in self.sizes:
                print(
                    f"Symbol [{fun_name}] was not properly profiled, size or runtime data missing",
                    file=sys.stderr,
                )
                self.sizes[fun_name] = 0
            try:
                self.env_socket.sendto(
                    bytes(
                        struct.pack(  # runtime_percent runtime_sec size
                            "ddi",
                            self.runtimes.get(fun_name, (0.0, 0.0))[0],
                            self.runtimes.get(fun_name, (0.0, 0.0))[1],
                            self.sizes[fun_name],
                        )
                    ),
                    func_env_address,
                )
            except (ConnectionError, FileNotFoundError):
                print(
                    f"Environment [{func_env_address}] unexpectedly died",
                    file=sys.stderr,
                )

    def get_sizes(self):
        size_info = (
            run(
                "${AARCH_PREFIX}nm --print-size --size-sort --radix=d main.elf",
                shell=True,
                capture_output=True,
            )
            .stdout.decode("utf-8")
            .splitlines()
        )
        self.sizes = {}
        for line in size_info:
            pieces = line.split()
            self.sizes[pieces[3]] = int(pieces[1])

    def get_runtimes(self):
        for i in range(0, self.args.bench_repeats):
            for run_str in self.args.run_string:
                run(
                    f"qemu-aarch64 -L /usr/aarch64-linux-gnu ./pg_main.elf {run_str}",
                    shell=True,
                )
            run(
                "${AARCH_PREFIX}nm --extern-only --defined-only -v --print-file-name pg_main.elf > symtab",
                shell=True,
            )
            run("${AARCH_PREFIX}gprof -s -Ssymtab pg_main.elf", shell=True, check=True)

        runtime_data = (
            run(
                "${AARCH_PREFIX}gprof -bp --no-demangle pg_main.elf",
                shell=True,
                capture_output=True,
                check=True,
            )
            .stdout.decode("utf-8")
            .splitlines()[5:]
        )
        os.unlink("gmon.out")
        os.unlink("gmon.sum")

        self.runtimes = {}
        if " no time accumulated" in runtime_data:
            for fun_name in self.active_funcs_lists.keys():
                self.runtimes[fun_name] = (0.0, 0.0)
        else:
            for line in runtime_data:
                pieces = line.split()
                self.runtimes[pieces[-1]] = (float(pieces[0]), float(pieces[2]))

    def compile_instrumented(self):
        gcc_instance = Popen(
            self.gprof_build_str, stderr=PIPE, shell=True
        )  # Compile with gprof to get per-function runtime info

        while not os.path.exists("gcc_plugin.soc"):
            gcc_instance.poll()
            if gcc_instance.returncode != None:
                print(
                    f"gcc failed: return code {gcc_instance.returncode}\n",
                    file=sys.stderr,
                )
                print(gcc_instance.communicate()[1].decode("utf-8"), file=sys.stderr)
                exit(1)
            pass

        while gcc_instance.poll() == None:
            try:
                fun_name = self.gcc_socket.recv(4096, socket.MSG_DONTWAIT).decode(
                    "utf-8"
                )
                if fun_name in self.active_funcs_lists:
                    self.gcc_socket.sendto(
                        self.active_funcs_lists[fun_name],
                        "gcc_plugin.soc".encode("utf-8"),
                    )
                    embedding = self.gcc_socket.recv(1024)
                else:
                    list_msg = fun_name.ljust(100, "\0").encode("utf-8") + bytes(1)
                    self.gcc_socket.sendto(list_msg, "gcc_plugin.soc".encode("utf-8"))
                    embedding = self.gcc_socket.recv(1024)
            except BlockingIOError:
                pass

        if gcc_instance.wait() != 0:
            print(
                f"gcc failed: return code {gcc_instance.returncode}\n",
                file=sys.stderr,
            )
            print(gcc_instance.communicate()[1].decode("utf-8"), file=sys.stderr)
            exit(1)

    def validate_addr(self, parsed_addr):
        if parsed_addr[1] != self.args.bench_name:
            print(
                f"Got message from env with incorrect bench name. "
                f"Expected '{self.args.bench_name}' got '{parsed_addr[1]}'",
                file=sys.stderr,
            )
            exit(1)
        if int(parsed_addr[3]) != self.args.instance:
            print(
                f"Got message from env with incorrect instance number. "
                f"Expected '{self.args.instance}' got '{parsed_addr[3]}'",
                file=sys.stderr,
            )
            exit(1)
        if parsed_addr[2] not in self.bench_symbols:
            print(
                f"Got message from env with incorrect function name. "
                f"Got '{self.parsed_addr[2]}'",
                file=sys.stderr,
            )
            exit(1)

    def add_env_to_list(self, pass_list, addr):
        parsed_addr = re.match("\0(.*):(.*)_(\d*)", addr.decode("utf-8"))
        self.validate_addr(parsed_addr)
        self.active_funcs_lists[parsed_addr[2]] = (
            parsed_addr[2].ljust(100, "\0").encode("utf-8") + pass_list
        )

    def gather_active_envs(self):
        while True:
            self.active_funcs_lists = {}

            self.env_socket.settimeout(60)
            try:
                self.add_env_to_list(*self.env_socket.recvfrom(4096))
                self.env_socket.settimeout(None)
                logging.debug("KERNEL: got first env")
                while True:
                    try:
                        logging.debug("KERNEL: env cycling wewo")
                        self.add_env_to_list(
                            *self.env_socket.recvfrom(4096, socket.MSG_DONTWAIT)
                        )
                    except BlockingIOError:
                        break
                break
            except (TimeoutError, socket.timeout):
                logging.debug("KERNEL: Got in timeout")
                afk_envs_exist = False
                for fun_name in self.bench_symbols:
                    try:
                        self.env_socket.sendto(
                            bytes(0),
                            f"\0{self.args.bench_name}:{fun_name}_{self.args.instance}".encode(
                                "utf-8"
                            ),
                        )
                        afk_envs_exist = True
                        logging.debug(f"KERNEL: saved by afk env '{fun_name}'")
                        break
                    except:
                        continue
                if not afk_envs_exist:
                    logging.debug("KERNEL: I have fallen and will not get up")
                    self.final_cleanup()

    def compile_for_size(self):
        gcc_instance = Popen(
            self.build_str, stderr=PIPE, shell=True
        )  # Compile without gprof do all the compilation stuff

        while not os.path.exists("gcc_plugin.soc"):
            gcc_instance.poll()
            if gcc_instance.returncode != None:
                print(
                    f"gcc failed: return code {gcc_instance.returncode}\n",
                    file=sys.stderr,
                )
                print(gcc_instance.communicate()[1].decode("utf-8"), file=sys.stderr)
                exit(1)
            pass

        while gcc_instance.poll() == None:
            try:
                fun_name = self.gcc_socket.recv(4096, socket.MSG_DONTWAIT).decode(
                    "utf-8"
                )
                if fun_name in self.active_funcs_lists.keys():
                    self.gcc_socket.sendto(
                        self.active_funcs_lists[fun_name],
                        "gcc_plugin.soc".encode("utf-8"),
                    )
                    embedding = self.gcc_socket.recv(1024)
                    self.env_socket.sendto(
                        embedding,
                        f"\0{self.args.bench_name}:{fun_name}_{self.args.instance}".encode(
                            "utf-8"
                        ),
                    )
                else:
                    list_msg = fun_name.ljust(100, "\0").encode("utf-8") + bytes(1)
                    self.gcc_socket.sendto(list_msg, "gcc_plugin.soc".encode("utf-8"))
                    embedding = self.gcc_socket.recv(1024)
            except BlockingIOError:
                pass

        if gcc_instance.wait() != 0:
            print(
                f"gcc failed: return code {gcc_instance.returncode}\n",
                file=sys.stderr,
            )
            print(gcc_instance.communicate()[1].decode("utf-8"), file=sys.stderr)
            exit(1)

    def kernel_loop(self):
        while True:
            logging.debug("KERNEL: compilation cucle")
            self.gather_active_envs()
            logging.debug(f"KERNEL: collected lists {self.active_funcs_lists}")
            self.compile_for_size()
            logging.debug("KERNEL: compiled for size")
            self.get_sizes()
            logging.debug("KERNEL: got sizes")
            self.compile_instrumented()
            logging.debug("KERNEL: gprof compiled")
            self.get_runtimes()
            logging.debug("KERNEL: got runtimes")
            self.sendout_profiles()
            logging.debug("KERNEL: sent profiles")


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
    with socket.socket(
        socket.AF_UNIX, socket.SOCK_DGRAM, 0
    ) as env_socket, socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM, 0) as gcc_socket:
        kernel = MultienvBenchKernel(env_socket, gcc_socket)
        kernel.kernel_loop()
