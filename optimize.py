import sys
import argparse
import pennylane as qml
from pennylane import numpy as np
import torch
from timeit import default_timer as timer
from datetime import datetime, timedelta

from utils import clustered_chain_graph, get_qaoa_circuit, grad_reduction_qaoa_circuit

import ray

seed = 1967

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device_name",
        type=str,
        default="lightning.qubit",
        const="lightning.qubit",
        nargs="?",
        choices=(
            "lightning.qubit",
            "lightning.gpu",
            "lightning.kokkos",
            "default.qubit",
        ),
        help="PennyLane device name.",
    )

    # qml.gradients method names
    parser.add_argument(
        "--diff_method",
        type=str,
        default="param_shift",
        const="param_shift",
        nargs="?",
        choices=("param_shift", "finite_diff"),
        help="The method of differentiation to be used by the QNode.",
    )

    # QAOA Hamiltonian configuration:
    parser.add_argument("--layers", type=int, default=2, help="Number of repetition layers.")

    # Graph configuration:
    parser.add_argument("--num_clusters", type=int, default=2, help="Number of graph nodes.")  # r

    parser.add_argument(
        "--nodes_per_cluster",
        type=int,
        default=2,
        help="Number of nodes within each cluster.",
    )  # n

    parser.add_argument(
        "--vertex_separators",
        type=int,
        default=1,
        help="Vertex separators between each cluster.",
    )  # k

    parser.add_argument(
        "--intra_cluster_edge_prob",
        type=float,
        default=0.7,
        help="Probability of having an intra-cluster edge.",
    )  # q1

    parser.add_argument(
        "--num_gpus",
        type=float,
        default=0.0,
        help="Number of gpus per tape execution (can be fractional). Set as zero for cpu only.",
    )  # gpu control

    return parser.parse_args()


def find_depth(tapes):
    # Assuming the same depth for all configurations of largest fragments
    largest_width = 0
    all_depths = []
    for tpe in tapes:
        all_depths.append(tpe.specs["depth"])
        wire_num = len(tpe.wires)
        if wire_num > largest_width:
            largest_width = wire_num
            largest_frag = tpe

    return (largest_frag.specs["depth"], max(all_depths))


########################################################################
# Execute methods
########################################################################
def _execute_tape(tape, device_name, frag_wires):
    dev = qml.device(device_name, wires=frag_wires)
    res = dev.execute(tape)
    return res

def execute_tape(_num_gpus):
    if (_num_gpus == None) or (_num_gpus == 0):
        return ray.remote(_execute_tape)
    else:
        return ray.remote(num_gpus=_num_gpus, max_calls=1)(_execute_tape)

def _execute_tape_jac(tape, device_name, frag_wires):
    dev = qml.device(device_name, wires=frag_wires)
    return dev.adjoint_jacobian(tape)

def execute_tape_jac(_num_gpus):
    if (_num_gpus == None):
        return ray.remote(_execute_tape_jac)
    else:
        return ray.remote(num_gpus=_num_gpus, max_calls=1)(_execute_tape_jac)

########################################################################
# Add samples Ray calls for S/R and for circuit execution
########################################################################
class RayExecutor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, tape, frag_wires, device_name, num_gpus):
        ctx.tape = tape
        ctx.device_name = device_name
        ctx.frag_wires = frag_wires
        ctx.num_gpus = num_gpus
        return execute_tape(ctx.num_gpus).remote(tape, ctx.device_name, ctx.frag_wires)

    @staticmethod
    def backward(ctx, dy):
        jac = torch.tensor(
            ray.get(execute_tape_jac(ctx.num_gpus).remote(ctx.tape, ctx.device_name, ctx.frag_wires)),
            requires_grad=True,
        )
        return dy * jac, None


def QAOA_cost(args, frag_wires, params):
    """
    Executes the QAOA circuit for a given set of parameters and returns a cost value.
    """
    G, cluster_nodes, separator_nodes = clustered_chain_graph(
        args.nodes_per_cluster,
        args.num_clusters,
        args.vertex_separators,
        args.intra_cluster_edge_prob,
        1 - args.intra_cluster_edge_prob,
        seed=seed,
    )
    circuit = get_qaoa_circuit(G, cluster_nodes, separator_nodes, params, args.layers)

    start_frag = timer()

    print(f"Finding fragments ... ", flush=True)
    fragment_configs, processing_fn = qml.cut_circuit(circuit, device_wires=range(frag_wires))
    end_frag = timer()
    elapsed_frag = end_frag - start_frag
    format_frag = str(timedelta(seconds=elapsed_frag))
    print(f"Fragmentation time: {format_frag}")

    print(f"Total number of fragment tapes = {len(fragment_configs)}", flush=True)

    frag_depth, deepest_tape = find_depth(fragment_configs)
    print(f"Depth of largest fragment = {frag_depth}", flush=True)
    print(f"Depth of deepest tape = {deepest_tape}", flush=True)

    start_cut = timer()
    results = ray.get(
        [
            RayExecutor.apply(t.get_parameters(), t, frag_wires, args.device_name, args.num_gpus)
            for t in fragment_configs
        ]
    )

    end_cut = timer()
    elapsed_cut = end_cut - start_cut
    format_cut = str(timedelta(seconds=elapsed_cut))
    print(f"Circuit cutting time: {format_cut}", flush=True)
    return np.sum(processing_fn(results))


########################################################################
# Gradients
########################################################################
def execute_grad(params, args, frag_wires, graph_data):
    """
    Function to find and execute gradient tapes
    """
    start_grad = timer()

    print(f"Creating gradient tapes...", flush=True)

    qaoa_tape = get_qaoa_circuit(
        graph_data["G"],
        graph_data["cluster_nodes"],
        graph_data["separator_nodes"],
        params,
        args.layers,
    )

    gradient_tapes, fn_grad = getattr(qml.gradients, args.diff_method)(qaoa_tape)

    print(f"Total number of gradient tapes = {len(gradient_tapes)}", flush=True)

    print("Finding and executing gradient fragments...", flush=True)

    grad_res = []
    for grad_tape in gradient_tapes:
        fragment_tapes, fn_cut = qml.cut_circuit(grad_tape, device_wires=range(frag_wires))
        results = ray.get(
            [
                RayExecutor.apply(t.get_parameters(), t, frag_wires, args.device_name, args.num_gpus)
                for t in fragment_tapes
            ]
        )
        grad_res.append(sum(fn_cut(results)))

    final_grad = fn_grad(grad_res)

    end_grad = timer()
    formated_time = str(timedelta(seconds=(end_grad - start_grad)))
    print(f"Gradient evaluation time: {formated_time}", flush=True)

    # When building the QAOA circuit, for consistency with analytic results, we multiply the parameters by 2.
    # Because of that, we divide the gradients by 2 here, so we can update the original parameters.
    return 0.5 * grad_reduction_qaoa_circuit(
        graph_data["G"],
        graph_data["cluster_nodes"],
        graph_data["separator_nodes"],
        final_grad,
        params,
    )


def grad_descent(steps, args, frag_wires, graph_data):
    """
    Function to perform gradient descent
    """
    init_params = np.array([[0.15, 0.2]] * args.layers, requires_grad=True)
    print(f"Initial params:")
    print(init_params)

    circuit = get_qaoa_circuit(
        graph_data["G"],
        graph_data["cluster_nodes"],
        graph_data["separator_nodes"],
        init_params,
        args.layers,
    )

    print(f"Total number of qubits = {len(circuit.wires)}", flush=True)
    full_depth = circuit.specs["depth"]
    print(f"Depth of full (uncut) circuit = {full_depth}", flush=True)

    params = init_params
    start_opt = timer()

    for i in range(steps):
        print(f"\n=====================================")
        print(f"Step {i+1}/{steps}:")
        print(f"Number of params = {params.size}", flush=True)
        cost = QAOA_cost(args, frag_wires, params)
        print(f"Cost at step {i} = {cost}", flush=True)
        grad = execute_grad(params, args, frag_wires, graph_data)
        print(f"Grad len = {len(grad)}", flush=True)
        params -= 0.0001 * grad
    print(f"=====================================")

    print(f"\nFinal report:", flush=True)
    end_opt = timer()
    elapsed_opt = end_opt - start_opt
    format_opt = str(timedelta(seconds=elapsed_opt))
    print(f"Optimization time: {format_opt}", flush=True)

    print(f"Final full parameters:", flush=True)
    print(params, flush=True)

    print(f"Final cost = {cost}", flush=True)


if __name__ == "__main__":
    args = parse_args()

    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    filename = f"./data/optimisation/opt_p={args.layers}_r={args.num_clusters}_n={args.nodes_per_cluster}_k={args.vertex_separators}_{time_stamp}.out"
    sys.stdout = open(filename, "w")

    ray.init()  # Should be updated according to system config

    print("\nNodes in the Ray cluster:", flush=True)
    print(ray.nodes(), flush=True)

    print(f"cluster resources: {ray.available_resources()}", flush=True)
    print(f"resources: {ray.available_resources()}", flush=True)

    print(
        "Problem: Graph with,",
        args.num_clusters,
        "clusters of ",
        args.nodes_per_cluster,
        "nodes and",
        args.vertex_separators,
        "vertex separators",
        flush=True,
    )

    frag_wires = (
        args.nodes_per_cluster + (3 * args.layers - 1) * args.vertex_separators
    )  # number of wires on biggest fragment
    print(f"\nSimulating {frag_wires} qubits for largest fragment", flush=True)

    G, cluster_nodes, separator_nodes = clustered_chain_graph(
        args.nodes_per_cluster,  # n
        args.num_clusters,  # r
        args.vertex_separators,  # k
        args.intra_cluster_edge_prob,  # q1
        1 - args.intra_cluster_edge_prob,  # q2
        seed=seed,
    )
    graph_data = dict(G=G, cluster_nodes=cluster_nodes, separator_nodes=separator_nodes)

    grad_descent(steps=1, args=args, frag_wires=frag_wires, graph_data=graph_data)
