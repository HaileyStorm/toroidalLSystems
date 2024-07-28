import numpy as np
from typing import Optional


class Torus:
    def __init__(self, arr, wrap_stop: bool = False, wrap_set_vals: bool = False):
        """
        Initialize a Torus object.

        :param arr: Input array-like object.
        :param wrap_stop: If True, wrap stop indices in slices.
                          If False (default), treat stop indices as non-inclusive - leave them untouched in
                          _wrap_indices, so that in __getitem__ and __setitem__ they wrap to the beginning of the
                          dimension when the end of the dimension is reached (that is, expand the slice to a list of
                          indices and wrap those individual indices).
        :param wrap_set_vals: If True, when setting values and the provided sequence of values is longer than the index
                              slice (in a given dimension), the values are wrapped - the values fill the slice, then
                              continue filling/overwriting from the beginning of the slice, etc.
                              If False (default), __setitem__ will throw an error if the values are longer than the slice.
        """
        self.arr = np.array(arr)
        self.shape = self.arr.shape
        self.dtype = self.arr.dtype
        self.wrap_stop = wrap_stop
        self.wrap_set_vals = wrap_set_vals

    def _wrap_indices(self, idx, wrap_stop: Optional[bool] = None):
        """
        Wrap indices to implement toroidal behavior.

        :param idx: Index or slice, or tuple of indices and/or slices, to wrap.
        :param wrap_stop: Override default wrap_stop behavior if not None.
        :return: Wrapped index or slice, or tuple of wrapped indices and/or slices.
        """
        # Use the provided wrap_stop value or fall back to the instance attribute
        wrap_stop = wrap_stop if wrap_stop is not None else self.wrap_stop
        if isinstance(idx, tuple):
            # Get the slice for each dimension and rebuild it with the wrapped indices
            return tuple(
                slice(i.start % size if i.start is not None else None,
                      i.stop if not wrap_stop else ((i.stop - 1) % size) + 1 if i.stop is not None else None,
                      i.step) if isinstance(i, slice) else i % size
                for i, size in zip(idx, self.shape)
            )
        else:
            return idx % self.shape[0] if not isinstance(idx, slice) else slice(idx.start % self.shape[0] if idx.start is not None else None,
                                                                                idx.stop if not wrap_stop else ((idx.stop - 1) % self.shape[0]) + 1 if idx.stop is not None else None,
                                                                                idx.step)

    def __getitem__(self, idx):
        """
        Get item(s) from the Torus, handling toroidal indexing.
        Behavior is effected by :attr:`Torus.wrap_stop` and :attr:`Torus.wrap_set_vals`.

        :param idx: Index or slice.
        :return: Single item or new Torus object.
        """
        idx = self._wrap_indices(idx)
        if isinstance(idx, tuple):
            if any(isinstance(i, slice) for i in idx):
                indices = []
                # Generate a list of wrapped indices for each slice
                for i, size in zip(idx, self.shape):
                    if isinstance(i, slice):
                        start = i.start if i.start is not None else 0
                        stop = i.stop if i.stop is not None else size
                        step = i.step if i.step is not None else 1
                        indices.append([idx % size for idx in range(start, stop, step)])
                    else:
                        indices.append([i % size])

                # Use NumPy's advanced indexing to create a new Torus from the lists of indices
                grid = np.meshgrid(*indices, indexing='ij')
                return Torus(self.arr[tuple(grid)])
            else:
                return self.arr[idx]
        else:
            if isinstance(idx, slice):
                # Generate a list of wrapped indices
                start = idx.start if idx.start is not None else 0
                stop = idx.stop if idx.stop is not None else self.shape[0]
                step = idx.step if idx.step is not None else 1
                wrapped_indices = [idx % self.shape[0] for idx in range(start, stop, step)]
                # Use NumPy's advanced indexing to create a new Torus from the lists of indices
                return Torus(self.arr[wrapped_indices])
            else:
                return self.arr[idx]

    def __setitem__(self, idx, val):
        """
        Set item(s) in the Torus, handling toroidal indexing.
        Behavior is effected by :attr:`Torus.wrap_stop` and :attr:`Torus.wrap_set_vals`, see :meth:`Torus.__init__`.

        :param idx: Index, slice, or tuple of indices/slices.
        :param val: Value(s) to set. Can be scalar or array-like.
        """
        idx = self._wrap_indices(idx)
        if isinstance(val, Torus):
            val = val.arr
        else:
            val = np.array(val)

        if isinstance(idx, tuple):
            if any(isinstance(i, slice) for i in idx):
                indices = []
                for i, size in zip(idx, self.shape):
                    if isinstance(i, slice):
                        start = i.start if i.start is not None else 0
                        stop = i.stop if i.stop is not None else size
                        step = i.step if i.step is not None else 1
                        indices.append([idx % size for idx in range(start, stop, step)])
                    else:
                        indices.append([i % size])

                grid = np.meshgrid(*indices, indexing='ij')
                target_shape = tuple(len(i) for i in indices)

                if val.shape == ():  # scalar value
                    self.arr[tuple(grid)] = val
                elif val.shape == target_shape:
                    self.arr[tuple(grid)] = val
                elif self.wrap_set_vals:
                    # Wrap values if they exceed target shape
                    wrapped_val = np.tile(val, tuple((s + v - 1) // v for s, v in zip(target_shape, val.shape)))
                    self.arr[tuple(grid)] = wrapped_val[:target_shape[0], :target_shape[1]]
                else:
                    raise ValueError(
                        f"Shape mismatch: cannot assign array of shape {val.shape} to slice of shape {target_shape}.")
            else:
                self.arr[idx] = val
        else:
            if isinstance(idx, slice):
                start = idx.start if idx.start is not None else 0
                stop = idx.stop if idx.stop is not None else self.shape[0]
                step = idx.step if idx.step is not None else 1
                wrapped_indices = [idx % self.shape[0] for idx in range(start, stop, step)]
                target_shape = len(wrapped_indices)

                if val.shape == ():  # scalar value
                    print(wrapped_indices)
                    print(val)
                    self.arr[wrapped_indices] = val
                elif val.shape == (target_shape,):
                    self.arr[wrapped_indices] = val
                elif self.wrap_set_vals:
                    # Wrap values if they exceed target shape
                    wrapped_val = np.tile(val, (target_shape + val.shape[0] - 1) // val.shape[0])
                    self.arr[wrapped_indices] = wrapped_val[:target_shape]
                else:
                    raise ValueError(
                        f"Shape mismatch: cannot assign array of shape {val.shape} to slice of shape ({target_shape},).")
            else:
                self.arr[idx] = val

    def __repr__(self):
        """
        String representation of the Torus.
        """
        return f"Torus({repr(self.arr)})"

    def dot(self, other):
        """
        Compute the dot product with another Torus or array-like object.

        :param other: Another Torus or compatible array-like object.
        :return: A new Torus with the dot product.
        """
        if isinstance(other, Torus):
            return Torus(np.dot(self.arr, other.arr))
        else:
            try:
                other = np.array(other)
            except:
                raise ValueError(f"Type mismatch: `other` must be a Torus or array-like, but is {type(other)}.")
            return Torus(np.dot(self.arr, other))

    def _prepare_indices(self, other):
        """
        Prepare indices for operations between two Torus objects of different sizes.

        :param other: Another Torus or array-like object.
        :return: Tuple of expanded and wrapped indices for self and other.
        """
        if isinstance(other, Torus):
            other_arr = other.arr
        else:
            other_arr = np.array(other)

        target_shape = tuple(np.lcm(s, o) for s, o in zip(self.shape, other_arr.shape))

        self_indices = [np.arange(s).repeat(target_shape[i] // s) for i, s in enumerate(self.shape)]
        other_indices = [np.arange(o).repeat(target_shape[i] // o) for i, o in enumerate(other_arr.shape)]

        self_indices = [np.resize(np.arange(s), target_shape[i]) for i, s in enumerate(self.shape)]
        other_indices = [np.resize(np.arange(o), target_shape[i]) for i, o in enumerate(other_arr.shape)]

        return np.meshgrid(*self_indices, indexing='ij'), np.meshgrid(*other_indices, indexing='ij')

    def __add__(self, other):
        """
        Add a scalar or another Torus or array-like object to this Torus.

        :param other: A scalar or another Torus or compatible array-like object.
        :return: A new Torus with the result.
        """
        if isinstance(other, Torus):
            self_indices, other_indices = self._prepare_indices(other)
        else:
            if np.isscalar(other):
                return Torus(self.arr + other)
            else:
                try:
                    other = Torus(np.array(other))
                    self_indices, other_indices = self._prepare_indices(other)
                except:
                    raise ValueError(f"Type mismatch: `other` must be a scalar, Torus, or array-like, but is {type(other)}.")
        return Torus(self.arr[tuple(self_indices)] + other.arr[tuple(other_indices)])

    def __radd__(self, other):
        """
        Right-hand add a scalar or another Torus or array-like object to this Torus.

        :param other: A scalar or another Torus or compatible array-like object.
        :return: A new Torus with the result.
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtract a scalar or another Torus or array-like object from this Torus.

        :param other: A scalar or another Torus or compatible array-like object.
        :return: A new Torus with the result.
        """
        if isinstance(other, Torus):
            self_indices, other_indices = self._prepare_indices(other)
        else:
            if np.isscalar(other):
                return Torus(self.arr - other)
            else:
                try:
                    other = Torus(np.array(other))
                    self_indices, other_indices = self._prepare_indices(other)
                except:
                    raise ValueError(
                        f"Type mismatch: `other` must be a scalar, Torus, or array-like, but is {type(other)}.")
        return Torus(self.arr[tuple(self_indices)] - other.arr[tuple(other_indices)])

    def __rsub__(self, other):
        """
        Right-hand subtract a scalar or another Torus or array-like object from this Torus.

        :param other: A scalar or another Torus or compatible array-like object.
        :return: A new Torus with the result.
        """
        if isinstance(other, Torus):
            self_indices, other_indices = self._prepare_indices(other)
        else:
            if np.isscalar(other):
                return Torus(other - self.arr)
            else:
                try:
                    other = Torus(np.array(other))
                    self_indices, other_indices = self._prepare_indices(other)
                except:
                    raise ValueError(
                        f"Type mismatch: `other` must be a scalar, Torus, or array-like, but is {type(other)}.")
        return Torus(other.arr[tuple(other_indices)] - self.arr[tuple(self_indices)])

    def __mul__(self, other):
        """
        Multiply a scalar or another Torus or array-like object with this Torus.

        :param other: A scalar or another Torus or compatible array-like object.
        :return: A new Torus with the result.
        """
        if isinstance(other, Torus):
            self_indices, other_indices = self._prepare_indices(other)
        else:
            if np.isscalar(other):
                return Torus(self.arr * other)
            else:
                try:
                    other = Torus(np.array(other))
                    self_indices, other_indices = self._prepare_indices(other)
                except:
                    raise ValueError(
                        f"Type mismatch: `other` must be a scalar, Torus, or array-like, but is {type(other)}.")
        return Torus(self.arr[tuple(self_indices)] * other.arr[tuple(other_indices)])

    def __rmul__(self, other):
        """
        Right-hand multiply a scalar or another Torus or array-like object with this Torus.

        :param other: A scalar or another Torus or compatible array-like object.
        :return: A new Torus with the result.
        """
        return self.__mul__(other)

    def __pow__(self, other):
        """
        Multiply a scalar or another Torus or array-like object with this Torus.

        :param other: A scalar or another Torus or compatible array-like object.
        :return: A new Torus with the result.
        """
        if isinstance(other, Torus):
            self_indices, other_indices = self._prepare_indices(other)
        else:
            if np.isscalar(other):
                return Torus(self.arr * other)
            else:
                try:
                    other = Torus(np.array(other))
                    self_indices, other_indices = self._prepare_indices(other)
                except:
                    raise ValueError(
                        f"Type mismatch: `other` must be a scalar, Torus, or array-like, but is {type(other)}.")
        return Torus(self.arr[tuple(self_indices)] * other.arr[tuple(other_indices)])
