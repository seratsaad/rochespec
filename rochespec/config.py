"""Data locations. ROCHESPEC_DATA is the data root (trailing slash added); it holds the input tables, the spectra and the
payne-zero intensity grid in the layout described in the README. ROCHESPEC_A4B points to a checkout of the agent4binary
repository, whose DR19 census manifest supplies the APOGEE single-dwarf calibrators."""
import os

DATA = os.path.join(os.path.expanduser(os.environ.get("ROCHESPEC_DATA", "~/rochespec_data")), "")
A4B = os.path.join(os.path.expanduser(os.environ.get("ROCHESPEC_A4B", "~/projects/agent4binary")), "")
