import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np
import h5py


class VQVAEDataset:
    """HDF5 dataloader with SPMD batch sharding.

    Loads a single field from HDF5, adds a channel dim, and yields
    batches sharded across the device mesh.

    Args:
        path: path to HDF5 file
        field: dataset name under /fields/
        batch_size: samples per batch (must be divisible by device count)
        shuffle: whether to shuffle each epoch
        seed: random seed for shuffling
        mesh: jax device mesh for SPMD sharding (None = no sharding)
    """

    def __init__(self, path: str, field: str = "omega", batch_size: int = 16,
                 shuffle: bool = True, seed: int = 42, mesh=None,
                 sample_start: int = 0, sample_stop: int = None):
        with h5py.File(path, "r") as f:
            self.data = f[f"fields/{field}"][sample_start:sample_stop]  # (N, H, W) float32

        # Add channel dim: (N, H, W) -> (N, 1, H, W)
        self.data = self.data[:, None, :, :]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        self.mesh = mesh
        self.sharding = (
            NamedSharding(mesh, P("batch", None, None, None))
            if mesh is not None else None
        )

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def sample_shape(self) -> tuple:
        return self.data.shape[1:]                           # (1, H, W)

    def __len__(self) -> int:
        return self.n_samples // self.batch_size

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            self.rng.shuffle(indices)

        for i in range(len(self)):
            batch_idx = indices[i * self.batch_size:(i + 1) * self.batch_size]
            batch = jnp.array(self.data[batch_idx])
            if self.sharding is not None:
                batch = jax.device_put(batch, self.sharding)
            yield batch
