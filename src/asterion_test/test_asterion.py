# https://gist.github.com/melix/421c742a946e91c123ee09ce1d2ef86e

import asterion
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def main() -> None:
    with asterion.Solver() as solver:
        data = fits.getdata("./data/IC5070_V1490_Cyg_2026_06_20_stack_10x180sec-V.fits")
        # Vérifie que les données lues sont bien un tableau NumPy.
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"Un tableau NumPy était attendu, reçu : {type(data).__name__}"
            )
        
        pixels = data.astype(np.float32)
        result = solver.solve(pixels)
        print(result)

        print("\nwcs")
        wcs = WCS(result.wcs)
        print(wcs)

        print("\nmatched_stars")
        print(result.matched_stars)
        for match in result.matches:
            print(match)


if __name__ == "__main__":
    main()