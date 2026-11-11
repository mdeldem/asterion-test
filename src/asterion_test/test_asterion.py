# https://gist.github.com/melix/421c742a946e91c123ee09ce1d2ef86e

from pathlib import Path
from typing import cast

import asterion
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image, ImageDraw


SOURCE_FILE = Path("data/IC5070_V1490_Cyg_2026_06_20_stack_10x180sec-V.fits")


def create_matches_jpg(
    data: np.ndarray,
    matches: list[asterion.Match],
    scale_arcsec_per_pixel: float,
    output_file: Path,
) -> None:
    """Crée un JPG de l'image en entourant chaque étoile appariée."""
    if data.ndim != 2:
        raise ValueError("L'image FITS doit être un tableau 2D en niveaux de gris.")
    if not np.isfinite(scale_arcsec_per_pixel) or scale_arcsec_per_pixel <= 0:
        raise ValueError("L'échelle astrométrique doit être positive et finie.")

    finite_pixels = data[np.isfinite(data)]
    if finite_pixels.size == 0:
        raise ValueError("L'image ne contient aucun pixel fini.")

    low, high = np.percentile(finite_pixels, (1, 99.5))
    if high <= low:
        high = low + 1
    normalized = np.clip((data - low) / (high - low), 0, 1)
    grayscale = (np.nan_to_num(normalized) * 255).astype(np.uint8)

    image = Image.fromarray(grayscale).convert("RGB")
    draw = ImageDraw.Draw(image)
    radius_pixels = 10.0 / scale_arcsec_per_pixel
    for match in matches:
        draw.ellipse(
            (
                match.x - radius_pixels,
                match.y - radius_pixels,
                match.x + radius_pixels,
                match.y + radius_pixels,
            ),
            outline=(255, 60, 60),
            width=2,
        )

    image.save(output_file, format="JPEG", quality=95)


def main() -> None:
    with asterion.Solver() as solver:
        with fits.open(SOURCE_FILE) as hdul:
            primary_hdu = cast(fits.PrimaryHDU, hdul[0])
            data = primary_hdu.data
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
                primary_hdu.header.update(wcs.to_header(relax=True))
                hdul.writeto(output_file, overwrite=True, output_verify="ignore")
                print(f"\nFichier résolu sauvegardé : {output_file}")

                if result.scale is None:
                    raise RuntimeError("La résolution a réussi sans fournir d'échelle.")
                jpg_file = output_dir / f"{SOURCE_FILE.stem}_solved.jpg"
                create_matches_jpg(data, result.matches, result.scale, jpg_file)
                print(f"Image avec étoiles appariées sauvegardée : {jpg_file}")


if __name__ == "__main__":
    main()
