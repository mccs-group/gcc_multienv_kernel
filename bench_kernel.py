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
import hashlib
import base64
import signal


def sigterm_handler(sig, frame):
    exit(1)


signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, signal.default_int_handler)


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

        self.EMBED_LEN_MULTIPLIER = 200

        symbols_list = Path("benchmark_info.txt")
        lines = [x.strip() for x in symbols_list.read_text().splitlines()]
        index = lines.index("functions:") + 1
        self.bench_symbols = [self.encode_fun_name(x) for x in lines[index:]]

        self.env_socket = env_socket
        self.gcc_socket = gcc_socket

        self.gcc_instance = None

        self.gcc_socket.bind(self.socket_name)
        self.env_socket.bind(f"\0{self.args.bench_name}:backend_{self.args.instance}")
        logging.debug("KERNEL: init end")

    def final_cleanup(self):
        if self.gcc_instance != None:
            try:
                self.gcc_instance.wait(30)
            except TimeoutExpired:
                pass
        cwd = os.path.abspath(os.getcwd())
        if cwd.startswith("/tmp") or cwd.startswith("/run"):
            shutil.rmtree(os.getcwd())
        else:
            os.unlink(self.socket_name)

    def sendout_profiles(self):
        for fun_name in self.active_funcs_lists:
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
            self.sizes[self.encode_fun_name(pieces[3])] = int(pieces[1])

    def get_runtimes(self):
        for i in range(0, self.args.bench_repeats):
            for run_str in self.args.run_string:
                run(
                    f"qemu-aarch64 -L /usr/aarch64-linux-gnu ./pg_main.elf {run_str}",
                    shell=True,
                    stdout=DEVNULL,
                    stderr=DEVNULL,
                )
            run(
                "${AARCH_PREFIX}nm --extern-only --defined-only -v --print-file-name pg_main.elf > symtab",
                shell=True,
                capture_output=True,
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
                self.runtimes[self.encode_fun_name(fun_name)] = (0.0, 0.0)
        else:
            for line in runtime_data:
                pieces = line.split()
                self.runtimes[self.encode_fun_name(pieces[-1])] = (
                    float(pieces[0]),
                    float(pieces[2]),
                )

    def compile_instrumented(self):
        self.gcc_instance = Popen(
            self.gprof_build_str, shell=True
        )  # Compile with gprof to get per-function runtime info

        while not os.path.exists("gcc_plugin.soc"):
            self.gcc_instance.poll()
            if self.gcc_instance.returncode != None:
                print(
                    f"gcc failed: return code {self.gcc_instance.returncode}\n",
                    file=sys.stderr,
                )
                exit(1)
            pass

        while self.gcc_instance.poll() == None:
            try:
                fun_name = self.gcc_socket.recv(4096, socket.MSG_DONTWAIT).decode(
                    "utf-8"
                )
                sock_fun_name = self.encode_fun_name(fun_name)
                if sock_fun_name in self.active_funcs_lists:
                    self.gcc_socket.sendto(
                        self.active_funcs_lists[sock_fun_name],
                        "gcc_plugin.soc".encode("utf-8"),
                    )
                    embedding = self.gcc_socket.recv(1024 * self.EMBED_LEN_MULTIPLIER)
                else:
                    list_msg = bytes(1)
                    self.gcc_socket.sendto(list_msg, "gcc_plugin.soc".encode("utf-8"))
                    embedding = self.gcc_socket.recv(1024 * self.EMBED_LEN_MULTIPLIER)
            except BlockingIOError:
                pass

        if self.gcc_instance.wait() != 0:
            print(
                f"gcc failed: return code {self.gcc_instance.returncode}\n",
                file=sys.stderr,
            )
            exit(1)

    def encode_fun_name(self, fun_name):
        fun_name = fun_name.partition(".")[0]
        avail_length = (
            107 - len(self.args.bench_name) - len(str(self.args.instance)) - 2
        )
        if len(fun_name) > avail_length or len(fun_name) > 100:
            return base64.b64encode(
                hashlib.sha256(fun_name.encode("utf-8")).digest(),
                "-_".encode("utf-8"),
            ).decode("utf-8")
        else:
            return fun_name

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
                f"Got '{parsed_addr[2]}'",
                file=sys.stderr,
            )
            exit(1)

    def add_env_to_list(self, pass_list, addr):
        parsed_addr = re.match("\0(.*):(.*)_(\d*)", addr.decode("utf-8"))
        self.validate_addr(parsed_addr)
        self.active_funcs_lists[parsed_addr[2]] = pass_list

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
                    exit(0)

    def compile_for_size(self):
        self.gcc_instance = Popen(
            self.build_str, shell=True
        )  # Compile without gprof do all the compilation stuff

        while not os.path.exists("gcc_plugin.soc"):
            self.gcc_instance.poll()
            if self.gcc_instance.returncode != None:
                print(
                    f"gcc failed: return code {self.gcc_instance.returncode}\n",
                    file=sys.stderr,
                )
                exit(1)
            pass

        while self.gcc_instance.poll() == None:
            try:
                fun_name = self.gcc_socket.recv(4096, socket.MSG_DONTWAIT).decode(
                    "utf-8"
                )
                sock_fun_name = self.encode_fun_name(fun_name)
                if sock_fun_name in self.active_funcs_lists:
                    logging.debug(f"KERNEL: Sending list for {sock_fun_name}")
                    self.gcc_socket.sendto(
                        self.active_funcs_lists[sock_fun_name],
                        "gcc_plugin.soc".encode("utf-8"),
                    )
                    logging.debug(
                        f"KERNEL: Sent list {self.active_funcs_lists[sock_fun_name]} to gcc"
                    )
                    embedding = self.gcc_socket.recv(1024 * self.EMBED_LEN_MULTIPLIER)
                    logging.debug(f"KERNEL: Got embedding from gcc")
                    try:
                        self.env_socket.sendto(
                            embedding,
                            f"\0{self.args.bench_name}:{sock_fun_name}_{self.args.instance}".encode(
                                "utf-8"
                            ),
                        )
                        logging.debug(f"KERNEL: Sent embedding to env")
                    except (ConnectionError, FileNotFoundError):
                        print(
                            f"Environment [{sock_fun_name}] unexpectedly died",
                            file=sys.stderr,
                        )
                else:
                    logging.debug(f"KERNEL: No list for {sock_fun_name}")
                    list_msg = bytes(1)
                    self.gcc_socket.sendto(list_msg, "gcc_plugin.soc".encode("utf-8"))
                    embedding = self.gcc_socket.recv(1024 * self.EMBED_LEN_MULTIPLIER)
            except BlockingIOError:
                pass

        if self.gcc_instance.wait() != 0:
            print(
                f"gcc failed: return code {self.gcc_instance.returncode}\n",
                file=sys.stderr,
            )
            exit(1)

    def kernel_loop(self):
        try:
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
        except (Exception, SystemExit, KeyboardInterrupt) as e:
            self.final_cleanup()
            raise e


if __name__ == "__main__":
    with socket.socket(
        socket.AF_UNIX, socket.SOCK_DGRAM, 0
    ) as env_socket, socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM, 0) as gcc_socket:
        kernel = MultienvBenchKernel(env_socket, gcc_socket)
        kernel.kernel_loop()
