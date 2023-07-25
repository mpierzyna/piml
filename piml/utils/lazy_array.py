import pathlib
import shutil
import uuid
import warnings
from typing import List, Union

import joblib


class LazyArray:
    """ Array that transparently pickles all data to disk rather than keeping it in RAM. """

    def __init__(self, output_dir: Union[str, pathlib.Path], overwrite: bool = False, compress: int = 0):
        self.output_dir = pathlib.Path(output_dir)

        if self.output_dir.exists() and not overwrite:
            # Reload list of old array elements and UUID
            warnings.warn(f"Directory {output_dir} is not empty and on-disk content "
                          f"will be available in this instance.")

            data_files = sorted(self.output_dir.glob("*.joblib"))
            self._data: List[str] = [f.name for f in data_files]
            self._uuid = self._data[0].replace(".joblib", "")
        else:
            if overwrite:
                # Delete old array
                shutil.rmtree(self.output_dir, ignore_errors=True)

            # Create new directory and empty array instance
            self.output_dir.mkdir(parents=True)  # Create parents if needed
            self._data: List[str] = []
            self._uuid = uuid.uuid4().hex

        self._i_iter = 0  # internal variable for iterator
        self._compress = compress  # joblib compression level

    def __repr__(self):
        return self._data.__repr__()

    def __str__(self):
        return self._data.__str__()

    def __getitem__(self, key):
        """ Read value for position key from disk and return it. """
        if isinstance(key, int):
            # Get single requested element
            item_name = self._data[key]
            return joblib.load(self.output_dir / item_name)
        elif isinstance(key, slice):
            # Get slice of data
            item_names = self._data[key]
            return [joblib.load(self.output_dir / p) for p in item_names]
        else:
            raise IndexError(f"Unknown index {key}. Has to be integer or slice of integers.")

    def __setitem__(self, key, value):
        """ Overwrite existing value at position key on disk. """
        if not isinstance(key, int):
            raise IndexError(f"Cannot set data with key of type {type(key)}. Only integer is allowed.")
        item_name = self._data[key]
        joblib.dump(value, self.output_dir / item_name, compress=self._compress)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        self._i_iter = 0
        return self

    def __next__(self):
        if self._i_iter < len(self):
            item = self[self._i_iter]
            self._i_iter += 1
            return item
        else:
            raise StopIteration

    def append(self, value):
        item_name = f"{self._uuid}_{len(self._data) + 1:d}.joblib"
        self._data.append(item_name)
        joblib.dump(value, self.output_dir / item_name, compress=self._compress)

    def health_check(self):
        """ Raises error if any of the on-disk array elements is missing. """
        for i, p in enumerate(self._data):
            p = self.output_dir / p
            if not p.exists():
                raise FileNotFoundError(f"File {p} for array index {i} could not be found!")

    def gather_to_disk(self, output_path: Union[str, pathlib.Path] = None) -> pathlib.Path:
        """ Load all data into RAM (can be heavy!) and dump them as standard list back to disk. """
        if output_path is None:
            output_path = self.output_dir.parent / f"{self.output_dir.stem}.joblib"

        data = [i for i in self]
        joblib.dump(data, output_path, compress=self._compress)

        return output_path

    def gather_to_mem(self) -> List:
        """ Load all data into RAM and return them. """
        return [i for i in self]


if __name__ == '__main__':
    la = LazyArray(output_dir="/tmp/bar", overwrite=False)
    la.append(5)
    la.append("test")
