import os
import sys
import logging
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt

from openmm.app import *
from openmm import *
from openmm.unit import *

# logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# configuration

class Config:
    pdb_file = "docked_complex.pdb"
    output_dir = "md_results"
    temperature = 300
    pressure = 1
    timestep = 2  # fs
    equil_steps = 5000
    prod_steps = 20000
    report_interval = 1000
    platform = "CPU"  # change to CUDA if available

cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)

# system validation

if not os.path.exists(cfg.pdb_file):
    logger.error("PDB file not found.")
    sys.exit()

logger.info("Loading structure...")

pdb = PDBFile(cfg.pdb_file)

# system preparation

forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')

modeller = Modeller(pdb.topology, pdb.positions)

logger.info("Adding hydrogens...")
modeller.addHydrogens(forcefield)

logger.info("Adding solvent...")
modeller.addSolvent(forcefield, model='tip3p', padding=1.0*nanometer)

logger.info("System size: %d atoms", modeller.topology.getNumAtoms())

system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=1.0*nanometer,
    constraints=HBonds
)

# Add barostat
system.addForce(MonteCarloBarostat(cfg.pressure*bar, cfg.temperature*kelvin))

# integrator

integrator = LangevinIntegrator(
    cfg.temperature*kelvin,
    1/picosecond,
    cfg.timestep*femtoseconds
)

# simulation modeller

platform = Platform.getPlatformByName(cfg.platform)

simulation = Simulation(modeller.topology, system, integrator, platform)
simulation.context.setPositions(modeller.positions)

# minimization

logger.info("Minimizing energy...")
simulation.minimizeEnergy()

# equilibration

logger.info("Equilibration...")

simulation.context.setVelocitiesToTemperature(cfg.temperature*kelvin)

simulation.reporters.append(StateDataReporter(
    os.path.join(cfg.output_dir, "equil.log"),
    1000,
    step=True,
    temperature=True,
    potentialEnergy=True
))

simulation.step(cfg.equil_steps)

# produciton run

logger.info("Production run...")

dcd_file = os.path.join(cfg.output_dir, "trajectory.dcd")

simulation.reporters.append(DCDReporter(dcd_file, cfg.report_interval))

simulation.reporters.append(StateDataReporter(
    os.path.join(cfg.output_dir, "prod.log"),
    cfg.report_interval,
    step=True,
    temperature=True,
    potentialEnergy=True
))

simulation.step(cfg.prod_steps)

# Save final structure
state = simulation.context.getState(getPositions=True)
PDBFile.writeFile(
    modeller.topology,
    state.getPositions(),
    open(os.path.join(cfg.output_dir, "final.pdb"), "w")
)

logger.info("MD completed.")

# trajectory analysis

logger.info("Loading trajectory for analysis...")

traj = md.load(dcd_file, top=os.path.join(cfg.output_dir, "final.pdb"))

# RMSD
rmsd = md.rmsd(traj, traj, 0)

# RMSF
rmsf = md.rmsf(traj, traj, 0)

# Radius of gyration
rg = md.compute_rg(traj)

# SASA
sasa = md.shrake_rupley(traj)

# Hydrogen bonds
hbonds = md.baker_hubbard(traj)

# plots

def plot(data, title, ylabel, filename):
    plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(cfg.output_dir, filename))
    plt.close()

plot(rmsd, "RMSD", "nm", "rmsd.png")
plot(rg, "Radius of Gyration", "nm", "rg.png")

plt.figure()
plt.plot(rmsf)
plt.title("RMSF")
plt.xlabel("Residue")
plt.ylabel("nm")
plt.savefig(os.path.join(cfg.output_dir, "rmsf.png"))
plt.close()

# SASA plot
plt.figure()
plt.plot(np.sum(sasa, axis=1))
plt.title("SASA")
plt.xlabel("Frame")
plt.ylabel("nm^2")
plt.savefig(os.path.join(cfg.output_dir, "sasa.png"))
plt.close()

np.save(os.path.join(cfg.output_dir, "rmsd.npy"), rmsd)
np.save(os.path.join(cfg.output_dir, "rmsf.npy"), rmsf)
np.save(os.path.join(cfg.output_dir, "rg.npy"), rg)
np.save(os.path.join(cfg.output_dir, "sasa.npy"), sasa)

with open(os.path.join(cfg.output_dir, "hbonds.txt"), "w") as f:
    for h in hbonds:
        f.write(f"{h}\n")

logger.info("Analysis completed.")

# mmpbsa

logger.info("Preparing MM-PBSA (external step)...")

print("""
To perform MM-PBSA:

1. Convert trajectory to GROMACS format:
   mdconvert trajectory.dcd -o traj.xtc

2. Use gmx_MMPBSA:
   gmx_MMPBSA -O -i mmpbsa.in -cs topol.tpr -ct traj.xtc

(This step requires GROMACS + gmx_MMPBSA)
""")

logger.info("Pipeline finished successfully.")
