# https://gist.github.com/melix/421c742a946e91c123ee09ce1d2ef86e

from pathlib import Path

import asterion
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


SOURCE_FILE = Path("data/IC5070_V1490_Cyg_2026_06_20_stack_10x180sec-V.fits")


def main() -> None:
    with asterion.Solver() as solver:
        data = fits.getdata(SOURCE_FILE)
        # Vérifie que les données lues sont bien un tableau NumPy.
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"Un tableau NumPy était attendu, reçu : {type(data).__name__}"
            )
        
        pixels = data.astype(np.float32)
        result = solver.solve(pixels)
        print(result)

        if result.solved:
            print("\nwcs")
            wcs = WCS(result.wcs)
            print(wcs)

            print("\nmatched_stars")
            print(result.matched_stars)
            for match in result.matches:
                print(match)

            output_dir = SOURCE_FILE.parent / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{SOURCE_FILE.stem}_solved.fits"
            with fits.open(SOURCE_FILE) as hdul:
                hdul[0].header.update(wcs.to_header(relax=True))
                hdul.writeto(output_file, overwrite=True)
            print(f"\nFichier résolu sauvegardé : {output_file}")


if __name__ == "__main__":
    main()
