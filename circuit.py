import pennylane as qml


def add_layer_pennylane(params, num_wires, noise_level=0.0):
    idx = 0
    for wire in range(num_wires - 1):
        p1 = params[idx]
        p2 = params[idx + 1]
        p3 = params[idx + 2]
        qml.U3(p1, p2, p3, wires=wire)
        idx += 3

        p_phase = params[idx]
        qml.ControlledPhaseShift(p_phase, wires=[wire + 1, wire])
        idx += 1

    if noise_level > 0.0:
        for dep_wire in range(num_wires):
            qml.DepolarizingChannel(noise_level, wires=dep_wire)


def create_qnode(num_qubits, num_layers, dev, noise_level=0.0, diff_method="adjoint"):
    total_wires = num_qubits + 1
    params_per_layer = num_qubits * 4

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(weights):
        for wire in range(total_wires):
            qml.Hadamard(wires=wire)
        for layer_idx in range(num_layers):
            start = layer_idx * params_per_layer
            end = start + params_per_layer
            add_layer_pennylane(weights[start:end], total_wires, noise_level)
        for wire in range(total_wires):
            qml.Hadamard(wires=wire)
        return qml.probs(wires=range(total_wires))

    return circuit


def get_device(num_wires, backend_preference=None, shots=None):
    """
    Створює квантовий device, пробуючи бекенди в порядку від найшвидшого
    (GPU) до гарантовано доступного (default.qubit).

    Для малої к-сті кубітів (як у вихідному MNIST-коді, ~7 wires) різниці
    майже немає. Для високої роздільності (13+ wires) lightning.qubit на CPU
    стає дуже повільним для навчання — тоді потрібен lightning.gpu або
    lightning.kokkos (вимагають окремої установки, див. README).
    """
    candidates = backend_preference or [
        "lightning.gpu",
        "lightning.kokkos",
        "lightning.qubit",
        "default.qubit",
    ]
    last_err = None
    for name in candidates:
        try:
            return qml.device(name, wires=num_wires, shots=shots), name
        except Exception as e:  # noqa: BLE001 - хочемо спробувати наступний бекенд
            last_err = e
            continue
    raise RuntimeError(f"Не вдалося створити жоден device з {candidates}. Остання помилка: {last_err}")
