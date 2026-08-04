import psi4
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import logging
from typing import Dict, Any, Optional
import os

# login setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# configure system
@dataclass
class QuantumConfig:
    method: str = "B3LYP"
    basis: str = "6-31G(d)"
    memory: str = "4 GB"
    threads: int = 4
    reference: str = "rhf"
    cube_properties: list = field(default_factory=lambda: ["DENSITY"])
    grid_size: int = 60
    grid_range: float = 4.0
    output_dir: str = "results"


# core engine system 
class QuantumSystem:
    def __init__(self, geometry: str, config: QuantumConfig):
        self.geometry = geometry
        self.config = config
        self.molecule = None
        self.wfn = None
        self.energy = None

        self._setup_environment()

    def _setup_environment(self):
        psi4.set_memory(self.config.memory)
        psi4.set_num_threads(self.config.threads)
        psi4.core.set_output_file("psi4.log", False)

        os.makedirs(self.config.output_dir, exist_ok=True)
        logger.info("Environment initialized.")

    def build_molecule(self):
        try:
            self.molecule = psi4.geometry(self.geometry)
            logger.info("Molecule built successfully.")
        except Exception as e:
            logger.error("Failed to build molecule.")
            raise e

    def run_scf(self):
        try:
            logger.info("Starting SCF/DFT calculation...")
            self.energy, self.wfn = psi4.energy(
                f"{self.config.method}/{self.config.basis}",
                return_wfn=True
            )
            logger.info(f"SCF completed. Energy: {self.energy:.8f}")
        except Exception as e:
            logger.error("SCF calculation failed.")
            raise e


# analysis module
class QuantumAnalyzer:
    def __init__(self, system: QuantumSystem):
        self.sys = system
        self.wfn = system.wfn

    def density_matrices(self) -> Dict[str, np.ndarray]:
        Da = self.wfn.Da().np
        Db = self.wfn.Db().np
        return {
            "alpha": Da,
            "beta": Db,
            "total": Da + Db
        }

    def energy_components(self) -> Dict[str, float]:
        D = self.wfn.Da().np + self.wfn.Db().np
        H = self.wfn.H().np
        J = self.wfn.J()[0].np
        K = self.wfn.K()[0].np

        E_one = np.sum(D * H)
        E_coulomb = np.sum(D * J)
        E_exchange = -0.5 * np.sum(self.wfn.Da().np * K)

        return {
            "one_electron": E_one,
            "coulomb": E_coulomb,
            "exchange": E_exchange
        }

    def orbital_analysis(self) -> Dict[str, Any]:
        eps = self.wfn.epsilon_a().np

        homo = np.max(np.where(eps < 0))
        lumo = homo + 1

        return {
            "orbital_energies": eps,
            "HOMO": eps[homo],
            "LUMO": eps[lumo],
            "gap": eps[lumo] - eps[homo]
        }

    def population_analysis(self):
        logger.info("Running Mulliken analysis...")
        psi4.mulliken_population_analysis(self.wfn)

        logger.info("Running Lowdin analysis...")
        psi4.lowdin_population_analysis(self.wfn)

    def real_space_density(self) -> np.ndarray:
        logger.info("Computing real-space electron density...")

        grid_size = self.sys.config.grid_size
        grid_range = self.sys.config.grid_range

        x = np.linspace(-grid_range, grid_range, grid_size)
        y = np.linspace(-grid_range, grid_range, grid_size)
        z = 0.0

        D = self.wfn.Da().np + self.wfn.Db().np
        basis = self.wfn.basisset()
        mints = psi4.core.MintsHelper(basis)

        density = np.zeros((grid_size, grid_size))

        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                point = psi4.core.Vector.from_array([xi, yj, z])
                phi = np.array(mints.basisset().compute_phi(point))
                density[i, j] = phi @ D @ phi.T

        return density

    def generate_cube(self):
        logger.info("Generating cube file...")
        psi4.cubeprop(self.wfn, properties=self.sys.config.cube_properties)


# visualization

class QuantumVisualizer:
    @staticmethod
    def plot_density(density: np.ndarray, config: QuantumConfig):
        plt.figure(figsize=(6,5))
        plt.imshow(
            density,
            extent=[
                -config.grid_range,
                config.grid_range,
                -config.grid_range,
                config.grid_range
            ],
            origin="lower"
        )
        plt.colorbar(label="Electron Density")
        plt.title("Electron Density Map")
        plt.xlabel("X (Bohr)")
        plt.ylabel("Y (Bohr)")
        plt.tight_layout()
        plt.show()


# controller

class QuantumWorkflow:
    def __init__(self, geometry: str, config: QuantumConfig):
        self.system = QuantumSystem(geometry, config)
        self.analyzer = None

    def run(self):
        self.system.build_molecule()
        self.system.run_scf()

        self.analyzer = QuantumAnalyzer(self.system)

        densities = self.analyzer.density_matrices()
        energies = self.analyzer.energy_components()
        orbitals = self.analyzer.orbital_analysis()

        logger.info(f"Energy Components: {energies}")
        logger.info(f"HOMO-LUMO Gap: {orbitals['gap']}")

        self.analyzer.population_analysis()

        density_grid = self.analyzer.real_space_density()

        QuantumVisualizer.plot_density(density_grid, self.system.config)

        self.analyzer.generate_cube()

        return {
            "energy": self.system.energy,
            "densities": densities,
            "energies": energies,
            "orbitals": orbitals
        }


# energy points

if __name__ == "__main__":

    geometry = """
    0 1
    O  0.000000  0.000000  0.000000
    H  0.758602  0.000000  0.504284
    H -0.758602  0.000000  0.504284
    symmetry c1
    """

    config = QuantumConfig(
        method="B3LYP",
        basis="6-31G(d)",
        grid_size=80
    )

    workflow = QuantumWorkflow(geometry, config)
    results = workflow.run()

    print("\nFinal Results Summary:")
    print(results)
