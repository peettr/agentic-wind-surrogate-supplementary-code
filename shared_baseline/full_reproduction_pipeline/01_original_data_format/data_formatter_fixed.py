"""
Fix rot90 restore bug in DataFormatter.
Lu's code does np.rot90(raw_data, k=wd//90) at line 224 during preprocessing,
but restore never reverses it. Fix: apply inverse rot90 at the END of 
_restore_single_raw_output_data_from_slices, AFTER building mask is set.
"""
import numpy as np
from data_formatter import DataFormatter

class DataFormatterFixed(DataFormatter):
    """DataFormatter with rot90 restore bug fixed."""
    
    def __init__(self, raw_data, wind_angles=None, formatted_shape=1280):
        self._original_wind_angles = list(wind_angles) if wind_angles else [0] * len(raw_data)
        super().__init__(raw_data, wind_angles, formatted_shape)
    
    def _restore_single_raw_output_data_from_slices(self, data_idx, slices, slice_start_idx, fill_value):
        # Call original restore (stitch + reverse residual angle + crop + building mask)
        # All in the rot90'd coordinate system
        from scipy import ndimage
        nm, nn = self._rotated_blocks[data_idx]
        fm, fn = self.fmt_shape
        expanded_shape = (nm * fm, nn * fn)
        expanded_output = np.empty(expanded_shape, dtype=self._data_type)
        cur_slice_idx = slice_start_idx
        for i in range(expanded_shape[0] // fm):
            for j in range(expanded_shape[1] // fn):
                mbeg_idx, mend_idx = i * fm, (i + 1) * fm
                nbeg_idx, nend_idx = j * fn, (j + 1) * fn
                expanded_output[mbeg_idx:mend_idx, nbeg_idx:nend_idx] = slices[cur_slice_idx, 0, :, :]
                cur_slice_idx += 1
        
        raw_output = self._extract_single_raw_output(data_idx, expanded_output)
        raw_output[self.raw_data[data_idx] < 0] = fill_value
        
        # FIX: reverse the rot90 that was applied during preprocessing
        wd = self._original_wind_angles[data_idx]
        k = int(wd // 90)
        if k > 0:
            raw_output = np.rot90(raw_output, k=-k)
        
        return raw_output



