import sys

sys.path.append("/store/code/open-catalyst/public-repo/matsciml")

from ase import io
from ase.calculators.kim import KIM

# Initialize KIM Model
model = KIM("KUSP__MO_000000000000_000")

# Using this as a 'dummy config' for now.
config = io.read("./Si.xyz")
# Set it as calculator
i = 0

# Compute energy/forces
config.set_calculator(model)
energy = config.get_potential_energy()
# forces returned are not the right shape for some reason. correct in serve_matsciml_models.py
forces = config.get_forces()
print(f"Energy: {energy}")
print(f"Forces: {forces}")
