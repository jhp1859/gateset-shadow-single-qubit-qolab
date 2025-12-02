
# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM, Transmon
from quam_libs.macros import qua_declaration, active_reset, readout_state
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_decay_exp, decay_exp
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ["q6"]
    use_state_discrimination: bool = True
    use_strict_timing: bool = False
    num_random_sequences: int = 5  # Number of random sequences
    #num_averages: int = 1
    max_circuit_depth: int = 5  # Maximum circuit depth
    delta_clifford: int = 2
    seed: int = 345324
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False

description = """Typical Runtime w/Default Params:
40-50s for all qubits
8-10s per qubit
"""

node = QualibrationNode(name="GSS4", description=description, parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations

config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# %% {QUA_program_parameters}
num_of_sequences = node.parameters.num_random_sequences  # Number of random sequences
# Number of averaging loops for each random sequence
#n_avg = node.parameters.num_averages #######################check
max_circuit_depth = node.parameters.max_circuit_depth  # Maximum circuit depth
if node.parameters.delta_clifford < 1:
    raise NotImplementedError("Delta clifford < 2 is not supported.")
#  Play each sequence with a depth step equals to 'delta_clifford - Must be > 1
delta_clifford = node.parameters.delta_clifford
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_thermal_or_active
assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
num_depths = max_circuit_depth // delta_clifford + 1
seed = node.parameters.seed  # Pseudo-random number generator seed
# Flag to enable state discrimination if the readout has been calibrated (rotated blobs and threshold)
state_discrimination = node.parameters.use_state_discrimination
strict_timing = node.parameters.use_strict_timing
# List of recovery gates from the lookup table
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]


def generate_sequence():
    cayley = declare(int, value=c1_table.flatten().tolist())
    inv_list = declare(int, value=inv_gates)
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=max_circuit_depth + 1)
    inv_gate = declare(int, size=max_circuit_depth + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < max_circuit_depth, i + 1):
        assign(step, rand.rand_int(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)
        assign(inv_gate[i], inv_list[current_state])

    return sequence, inv_gate


def play_sequence(sequence_list, depth, qubit: Transmon):
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                qubit.xy.wait(qubit.xy.operations["x180"].length // 4)
            with case_(1):  # x180
                qubit.xy.play("x180")
            with case_(2):  # y180
                qubit.xy.play("y180")
            with case_(3):  # Z180
                qubit.xy.play("y180")
                qubit.xy.play("x180")
            with case_(4):  # Z90 X180 Z-180
                qubit.xy.play("x90")
                qubit.xy.play("y90")
            with case_(5):  # Z-90 Y-90 Z-90
                qubit.xy.play("x90")
                qubit.xy.play("-y90")
            with case_(6):  # Z-90 X180 Z-180
                qubit.xy.play("-x90")
                qubit.xy.play("y90")
            with case_(7):  # Z-90 Y90 Z-90
                qubit.xy.play("-x90")
                qubit.xy.play("-y90")
            with case_(8):  # X90 Z90
                qubit.xy.play("y90")
                qubit.xy.play("x90")
            with case_(9):  # X-90 Z-90
                qubit.xy.play("y90")
                qubit.xy.play("-x90")
            with case_(10):  # z90 X90 Z90
                qubit.xy.play("-y90")
                qubit.xy.play("x90")
            with case_(11):  # z90 X-90 Z90
                qubit.xy.play("-y90")
                qubit.xy.play("-x90")
            with case_(12):  # x90
                qubit.xy.play("x90")
            with case_(13):  # -x90
                qubit.xy.play("-x90")
            with case_(14):  # y90
                qubit.xy.play("y90")
            with case_(15):  # -y90
                qubit.xy.play("-y90")
            with case_(16):  # Z90
                qubit.xy.play("-x90")
                qubit.xy.play("y90")
                qubit.xy.play("x90")
            with case_(17):  # -Z90
                qubit.xy.play("-x90")
                qubit.xy.play("-y90")
                qubit.xy.play("x90")
            with case_(18):  # Y-90 Z-90
                qubit.xy.play("x180")
                qubit.xy.play("y90")
            with case_(19):  # Y90 Z90
                qubit.xy.play("x180")
                qubit.xy.play("-y90")
            with case_(20):  # Y90 Z-90
                qubit.xy.play("y180")
                qubit.xy.play("x90")
            with case_(21):  # Y-90 Z90
                qubit.xy.play("y180")
                qubit.xy.play("-x90")
            with case_(22):  # x90 Z-90
                qubit.xy.play("x90")
                qubit.xy.play("y90")
                qubit.xy.play("x90")
            with case_(23):  # -x90 Z90
                qubit.xy.play("-x90")
                qubit.xy.play("y90")
                qubit.xy.play("-x90")


# %% {QUA_program}
with program() as randomized_benchmarking_individual:
    depth = declare(int)  # QUA variable for the varying depth
    # QUA variable for the current depth (changes in steps of delta_clifford)
    depth_target = declare(int)
    # QUA variable to store the last Clifford gate of the current sequence which is replaced by the recovery gate
    saved_gate = declare(int)
    m = declare(int)  # QUA variable for the loop over random sequences
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(int) for _ in range(num_qubits)]
    
    # The relevant streams
    m_st = declare_stream()
    # state_st = declare_stream()
    state_st = [declare_stream() for _ in range(num_qubits)]
    # NEW: streams for sequences
    seq_st = [declare_stream() for _ in range(num_qubits)]
    #inv_st = [declare_stream() for _ in range(num_qubits)]

    for i, qubit in enumerate(qubits):

        align()

        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        # QUA for_ loop over the random sequences
        with for_(m, 0, m < num_of_sequences, m + 1):
            # Generate the random sequence of length max_circuit_depth
            sequence_list, inv_gate_list = generate_sequence()
            assign(depth_target, 0)  # Initialize the current depth to 0

            # save the full random sequence for this m
            k = declare(int)
            with for_(k, 0, k < max_circuit_depth, k + 1):
                save(sequence_list[k], seq_st[i])
            
            with for_(depth, 1, depth <= max_circuit_depth, depth + 1):
                # Replacing the last gate in the sequence with the sequence's inverse gate
                # The original gate is saved in 'saved_gate' and is being restored at the end
                ###(rm1)assign(saved_gate, sequence_list[depth])
                ###(rm2)assign(sequence_list[depth], inv_gate_list[depth - 1])
                # Only played the depth corresponding to target_depth
                with if_((depth == 1) | (depth == depth_target)):
                    
                    # Initialize the qubits
                    if reset_type == "active":
                        active_reset(qubit, "readout")
                    else:
                        qubit.resonator.wait(qubit.thermalization_time * u.ns)
                        # Align the two elements to play the sequence after qubit initialization
                    qubit.align()
                        # The strict_timing ensures that the sequence will be played without gaps
                    if strict_timing:
                        with strict_timing_():
                                # Play the random sequence of desired depth
                            play_sequence(sequence_list, depth, qubit)
                    else:
                        play_sequence(sequence_list, depth, qubit)
                        # Align the two elements to measure after playing the circuit.
                    qubit.align()
                    readout_state(qubit, state[i])

                    save(state[i], state_st[i])

                    # Go to the next depth
                    assign(depth_target, depth_target + delta_clifford)
                # Reset the last gate of the sequence back to the original Clifford gate
                # (that was replaced by the recovery gate at the beginning)
                ###(rm3)assign(sequence_list[depth], saved_gate)
            # Save the counter for the progress bar
            save(m, m_st)


    with stream_processing():
        m_st.save("iteration")
        for i in range(num_qubits):
            # 1) RAW measurement outcomes:
            # shape: (num_sequences, num_depths, n_avg)
            state_st[i].buffer(num_depths).buffer(num_of_sequences).save(f"state_raw_q{i+1}")

            # 2) Full gate sequence for each random sequence:
            # shape: (num_sequences, max_circuit_depth)
            seq_st[i].buffer(max_circuit_depth).buffer(num_of_sequences).save(f"sequence_q{i+1}")




# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=100_000)  # in clock cycles
    job = qmm.simulate(config, randomized_benchmarking_individual, simulation_config)
    samples = job.get_simulated_samples()
    ###(rm4)fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    ###(rm5)for i, con in enumerate(samples.keys()):
    ###(rm6)    plt.subplot(len(samples.keys()),1,i+1)
    ###(rm7)    samples[con].plot()
    ###(rm8)    plt.title(con)
    ###(rm9)plt.tight_layout()
    ###(rm10)node.results["figure"] = plt.gcf()
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    # Prepare data for saving
    node.results = {}
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        if not node.parameters.multiplexed:
            job = qm.execute(randomized_benchmarking_individual)
        else:
            job = qm.execute(randomized_benchmarking_multiplexed)
            
        results = fetching_tool(job, ["iteration"], mode="live")
        while results.is_processing():
            # Fetch results
            m = results.fetch_all()[0]
            # Progress bar
            progress_counter(m, num_of_sequences, start_time=results.start_time)


    # %% {Data_fetching_and_dataset_creation}
    if node.parameters.load_data_id is None:
        depths = np.arange(0, max_circuit_depth + 0.1, delta_clifford)
        depths[0] = 1

        # --- NEW: split handles by variable name ---
        handles_all = job.result_handles

        # Only the state streams (from state_st[i])
        handles_state = {
            k: handles_all.get(k)
            for k in handles_all.keys()
            if "state_raw_q" in k
        }

        # Only the sequence streams (from seq_st[i])
        handles_seq = {
            k: handles_all.get(k)
            for k in handles_all.keys()
            if "sequence_q" in k
        }

        # --- Build dataset for measurement outcomes ---
        # Axes: qubit × sequence × depths × n_avg
        ds_stat = fetch_results_as_xarray(
            handles_state,
            qubits,
            {
                "depths":  depths,
                "sequence": np.arange(num_of_sequences),
            },
        )

        # --- Build dataset for gate sequences ---
        # Axes: qubit × sequence × positions (gate index)
        ds_seq = fetch_results_as_xarray(
            handles_seq,
            qubits,
            {
                "positions": np.arange(max_circuit_depth),
                "sequence":  np.arange(num_of_sequences),
            },
        )

    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds_stat = node.results["ds_stat"]
        ds_seq  = node.results["ds_seq"]

    # Add both datasets to the node (don’t overwrite!)
    node.results = {
        "ds_stat": ds_stat,
        "ds_seq":  ds_seq,
    }
    node.save()

