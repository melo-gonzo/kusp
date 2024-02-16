"""
A minimal server for serving ML objects for KLIFF.

Incoming data specification:
    [Necessary] First 4 bytes: size of integer on the system (int_width), 32 bit integer
    [Necessary] Next int_width bytes: number of atoms (n_atoms), default int_width integer
    [Necessary] Next int_width x n_atoms bytes: atomic numbers
    [Necessary] Next 8 x 3 x n_atoms bytes: positions of atoms (x, y, z), double precision
    [Optional] Next int_width x n_atoms bytes: Which atoms to compute energy for (contributing atoms)

"""

import socket
import struct
import time
from typing import Any, Callable, Union

import numpy as np
from loguru import logger

from .server import Server


def get_all_data(client_socket, size):
    all_data = b""
    # while True:
    data = client_socket.recv(size)
    # if not data or len(data) < BUFFER_SIZE:
    #    break
    all_data += data
    return all_data, len(all_data)


class KUSPServer(Server):
    def __init__(self, exec_func: Callable[..., Any], configuration: Union[dict, str]):
        super().__init__()
        if isinstance(configuration, dict):
            optional_args = configuration.get("optional", {})
            server_args = configuration.get("server", {})
            super().__init__(**server_args, **optional_args)
        elif isinstance(configuration, str):
            super().configuration_from_yaml(configuration)
        else:
            raise ValueError("Configuration should be a dictionary or a string")

        self.exec_func = exec_func
        self.global_information = self.configuration.get("global", {})

        logger.info(f"kim server listening on {self.host}:{self.port}")

    def serve(self):
        self.start()
        while True:
            client_socket, client_address = self.server_socket.accept()
            with client_socket:
                logger.info(f"kim server connected from {client_address}")

                while True:  # Continuous loop to keep connection open
                    try:
                        # Receive the first 4 bytes to determine integer width
                        data = client_socket.recv(4)
                        if not data:
                            break  # Break the loop if client closes the connection or sends empty data

                        int_width = struct.unpack("i", data)[0]
                        int_type_py = np.int32 if int_width == 4 else np.int64

                        # Next int_width bytes: number of atoms (n_atoms), default int_width integer
                        n_atoms = struct.unpack(f"i", client_socket.recv(int_width))[0]

                        # Next int_width x n_atoms bytes: atomic numbers
                        b_atomic_numbers, n_bytes = get_all_data(
                            client_socket, size=int_width * n_atoms
                        )
                        atomic_numbers = np.frombuffer(
                            b_atomic_numbers, dtype=int_type_py
                        )

                        # Next 8 x 3 x n_atoms bytes: positions of atoms (x, y, z), double precision
                        b_positions, n_bytes = get_all_data(
                            client_socket, size=8 * 3 * n_atoms
                        )
                        positions = np.frombuffer(
                            b_positions, dtype=np.float64
                        ).reshape((n_atoms, 3))

                        # Next int_width x n_atoms bytes: Which atoms to compute energy for (contributing atoms)
                        b_contributing_atoms, n_bytes = get_all_data(
                            client_socket, size=int_width * n_atoms
                        )
                        if n_bytes != int_width * n_atoms:
                            if int_width == 4:
                                b_contributing_atoms = b"\x01\x00\x00\x00" * n_atoms
                            else:
                                b_contributing_atoms = (
                                    b"\x01\x00\x00\x00\x00\x00\x00\x00" * n_atoms
                                )

                        contributing_atoms = np.frombuffer(
                            b_contributing_atoms, dtype=int_type_py
                        )
                        start_all_time = time.perf_counter()

                        # execute the model
                        model_inputs = self.prepare_model_inputs(
                            atomic_numbers, positions, contributing_atoms
                        )
                        model_outputs = self.execute_model(**model_inputs)
                        model_outputs = self.prepare_model_outputs(model_outputs)

                        end_time = time.perf_counter()

                        client_socket.sendall(model_outputs["energy"].tobytes())
                        client_socket.sendall(model_outputs["forces"].tobytes())
                        logger.info(
                            f"Evaluated Configuration. Time taken: {(end_time - start_all_time) * 1000}"
                        )
                    except e:
                        logger.error(f"Error: {e}")
                        break
                logger.info("Client closed the connection.")
        self.stop()

    def prepare_model_inputs(
        self, atomic_numbers, positions, contributing_atoms, **kwargs
    ):
        return {
            "atomic_numbers": atomic_numbers,
            "positions": positions,
            "contributing_atoms": contributing_atoms,
        }

    def prepare_model_outputs(self, kwargs):
        dict_out = {}
        dict_out["energy"] = kwargs["energy"].detach().cpu().numpy()
        dict_out["forces"] = kwargs["forces"].detach().cpu().numpy()
        return dict_out

    def execute_model(self, **kwargs):
        return self.exec_func(**kwargs)
