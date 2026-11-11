# https://gist.github.com/melix/421c742a946e91c123ee09ce1d2ef86e

from pathlib import Path
from typing import Callable, cast

import asterion
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image, ImageDraw, ImageFont


SOURCE_FILE = Path("data/IC5070_V1490_Cyg_2026_06_20_stack_10x180sec-V.fits")


def create_matches_jpg(
    data: np.ndarray,
    matches: list[asterion.Match],
    scale_arcsec_per_pixel: float,
    wcs: WCS,
    output_file: Path,
) -> None:
    """Crée un JPG annoté avec la grille céleste, le repère N/E et les étoiles appariées."""
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
    original_width, original_height = image.size
    image, transform_pixel, north_direction, east_direction = _apply_sky_orientation_symmetries(
        image, wcs.celestial
    )
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    _draw_celestial_grid(
        draw,
        wcs.celestial,
        original_width,
        original_height,
        image.width,
        image.height,
        transform_pixel,
        font,
    )
    _draw_north_east_axes(
        draw, image.width, image.height, north_direction, east_direction, font
    )

    radius_pixels = 10.0 / scale_arcsec_per_pixel
    for match in matches:
        x, y = transform_pixel(match.x, match.y)
        draw.ellipse(
            (
                x - radius_pixels,
                y - radius_pixels,
                x + radius_pixels,
                y + radius_pixels,
            ),
            outline=(255, 60, 60),
            width=2,
        )

    image.save(output_file, format="JPEG", quality=95)


def _apply_sky_orientation_symmetries(
    image: Image.Image,
    wcs: WCS,
) -> tuple[
    Image.Image,
    Callable[[float, float], tuple[float, float]],
    tuple[float, float],
    tuple[float, float],
]:
    """Applique seulement les symétries nécessaires pour orienter le nord en haut et l'est à gauche."""
    width, height = image.size
    center_x, center_y = (width - 1) / 2, (height - 1) / 2
    ra, dec = wcs.pixel_to_world_values(center_x, center_y)
    center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    north = center.directional_offset_by(0 * u.deg, 1 * u.arcmin)
    east = center.directional_offset_by(90 * u.deg, 1 * u.arcmin)
    north_x, north_y = wcs.world_to_pixel_values(north.ra.deg, north.dec.deg)
    east_x, east_y = wcs.world_to_pixel_values(east.ra.deg, east.dec.deg)

    flip_horizontally = east_x > center_x
    flip_vertically = north_y > center_y
    if flip_horizontally:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_vertically:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    def transform_point(x: float, y: float) -> tuple[float, float]:
        transformed_x, transformed_y = x, y
        if flip_horizontally:
            transformed_x = width - 1 - transformed_x
        if flip_vertically:
            transformed_y = height - 1 - transformed_y
        return transformed_x, transformed_y

    north_vector = np.array((north_x - center_x, north_y - center_y), dtype=float)
    east_vector = np.array((east_x - center_x, east_y - center_y), dtype=float)
    if flip_horizontally:
        north_vector[0] *= -1
        east_vector[0] *= -1
    if flip_vertically:
        north_vector[1] *= -1
        east_vector[1] *= -1

    north_direction = tuple(north_vector / np.linalg.norm(north_vector))
    east_direction = tuple(east_vector / np.linalg.norm(east_vector))
    return image, transform_point, north_direction, east_direction


def _nice_grid_step(span_degrees: float) -> float:
    """Choisit un pas lisible pour la grille, avec environ cinq intervalles."""
    target = span_degrees / 5
    if target <= 0 or not np.isfinite(target):
        return 1 / 60
    magnitude = 10 ** np.floor(np.log10(target))
    fraction = target / magnitude
    factor = 1 if fraction <= 1 else 2 if fraction <= 2 else 5 if fraction <= 5 else 10
    return float(factor * magnitude)


def _nice_dec_grid_step(dec_min: float, dec_max: float) -> float:
    """Choisit un pas de déclinaison lisible en minutes d'arc, avec au moins trois repères si possible."""
    span_arcminutes = (dec_max - dec_min) * 60
    target_step = span_arcminutes / 5
    if target_step <= 0 or not np.isfinite(target_step):
        return 1 / 60

    nice_steps_arcminutes = sorted(
        {base * 10**power for power in range(4) for base in (1, 2, 5, 10, 15, 20, 30)}
    )

    def line_count(step_arcminutes: float) -> int:
        step_degrees = step_arcminutes / 60
        return int(np.floor(dec_max / step_degrees) - np.ceil(dec_min / step_degrees) + 1)

    candidates = [step for step in nice_steps_arcminutes if 3 <= line_count(step) <= 7]
    if not candidates:
        candidates = [step for step in nice_steps_arcminutes if line_count(step) >= 2]
    if not candidates:
        candidates = nice_steps_arcminutes

    selected_step = min(candidates, key=lambda step: abs(np.log(step / target_step)))
    return selected_step / 60


def _draw_projected_curve(
    draw: ImageDraw.ImageDraw,
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
    width: int,
    height: int,
    color: tuple[int, int, int],
    transform_pixel: Callable[[float, float], tuple[float, float]],
) -> list[tuple[float, float]]:
    """Trace une courbe WCS en coupant les portions éloignées ou non finies."""
    visible_points: list[tuple[float, float]] = []
    segments: list[list[tuple[float, float]]] = []
    segment: list[tuple[float, float]] = []
    margin = max(width, height)

    for x, y in zip(x_pixels, y_pixels):
        if np.isfinite(x) and np.isfinite(y) and -margin <= x <= width + margin and -margin <= y <= height + margin:
            point = transform_pixel(float(x), float(y))
            x, y = point
            if not (-margin <= x <= width + margin and -margin <= y <= height + margin):
                if len(segment) > 1:
                    segments.append(segment)
                segment = []
                continue
            visible_points.append(point)
            segment.append(point)
        else:
            if len(segment) > 1:
                segments.append(segment)
            segment = []
    if len(segment) > 1:
        segments.append(segment)

    for points in segments:
        draw.line(points, fill=color, width=1)
    return visible_points


def _format_ra(degrees: float) -> str:
    total_minutes = round((degrees % 360) * 4)
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:02d}h {minutes:02d}m"


def _format_dec(degrees: float) -> str:
    total_minutes = round(abs(degrees) * 60)
    whole_degrees, minutes = divmod(total_minutes, 60)
    sign = "+" if degrees >= 0 else "−"
    return f"{sign}{whole_degrees:02d}° {minutes:02d}'"


def _draw_celestial_grid(
    draw: ImageDraw.ImageDraw,
    wcs: WCS,
    original_width: int,
    original_height: int,
    width: int,
    height: int,
    transform_pixel: Callable[[float, float], tuple[float, float]],
    font: ImageFont.ImageFont,
) -> None:
    """Projette les méridiens d'ascension droite et parallèles de déclinaison."""
    edge = np.linspace(0, 1, 120)
    border_x = np.concatenate((edge * (original_width - 1), edge * (original_width - 1), np.zeros_like(edge), np.full_like(edge, original_width - 1)))
    border_y = np.concatenate((np.zeros_like(edge), np.full_like(edge, original_height - 1), edge * (original_height - 1), edge * (original_height - 1)))
    border_ra, border_dec = wcs.pixel_to_world_values(border_x, border_y)
    center_ra, center_dec = wcs.pixel_to_world_values((original_width - 1) / 2, (original_height - 1) / 2)
    unwrapped_ra = center_ra + ((border_ra - center_ra + 180) % 360) - 180
    ra_min, ra_max = float(np.nanmin(unwrapped_ra)), float(np.nanmax(unwrapped_ra))
    dec_min, dec_max = float(np.nanmin(border_dec)), float(np.nanmax(border_dec))
    ra_step = _nice_grid_step(ra_max - ra_min)
    dec_step = _nice_dec_grid_step(dec_min, dec_max)
    grid_color = (75, 125, 185)
    label_color = (190, 215, 245)
    line_samples = np.linspace(0, 1, 300)
    ra_lines = np.arange(np.ceil(ra_min / ra_step), np.floor(ra_max / ra_step) + 1) * ra_step
    dec_lines = np.arange(np.ceil(dec_min / dec_step), np.floor(dec_max / dec_step) + 1) * dec_step

    for ra in ra_lines:
        ras = np.full_like(line_samples, ra % 360)
        decs = dec_min + line_samples * (dec_max - dec_min)
        x_pixels, y_pixels = wcs.world_to_pixel_values(ras, decs)
        points = _draw_projected_curve(draw, x_pixels, y_pixels, width, height, grid_color, transform_pixel)
        bottom_points = [(x, y) for x, y in points if 0 <= x < width and 0 <= y < height]
        if bottom_points:
            x, y = min(bottom_points, key=lambda point: abs(point[1] - (height - 1)))
            draw.text((x + 3, min(y - 20, height - 20)), _format_ra(ra), font=font, fill=label_color, stroke_width=2, stroke_fill=(0, 0, 0))

    for dec in dec_lines:
        ras_unwrapped = ra_min + line_samples * (ra_max - ra_min)
        ras = np.mod(ras_unwrapped, 360)
        decs = np.full_like(line_samples, dec)
        x_pixels, y_pixels = wcs.world_to_pixel_values(ras, decs)
        points = _draw_projected_curve(draw, x_pixels, y_pixels, width, height, grid_color, transform_pixel)
        left_points = [(x, y) for x, y in points if 0 <= x < width and 0 <= y < height]
        if left_points:
            x, y = min(left_points, key=lambda point: point[0])
            draw.text((x + 4, y + 3), _format_dec(dec), font=font, fill=label_color, stroke_width=2, stroke_fill=(0, 0, 0))


def _draw_north_east_axes(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    north_direction: tuple[float, float],
    east_direction: tuple[float, float],
    font: ImageFont.ImageFont,
) -> None:
    """Dessine le repère cardinal en respectant l'angle réel des directions célestes."""
    origin = (width - 72, 78)
    arrow_length = 48
    color = (245, 245, 245)
    for label, direction_tuple in (("N", north_direction), ("E", east_direction)):
        direction = np.asarray(direction_tuple)
        start = np.asarray(origin, dtype=float)
        end = start + direction * arrow_length
        draw.line((tuple(start), tuple(end)), fill=color, width=3)
        text_bbox = draw.textbbox((0, 0), label, font=font, stroke_width=2)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_gap = 14 if label == "N" else 6
        label_center = end + direction * (max(text_width, text_height) / 2 + label_gap)
        text_x = np.clip(label_center[0] - text_width / 2, 2, width - text_width - 2)
        text_y = np.clip(label_center[1] - text_height / 2, 2, height - text_height - 2)
        draw.text(
            (text_x - text_bbox[0], text_y - text_bbox[1]),
            label,
            font=font,
            fill=color,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )


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
                create_matches_jpg(data, result.matches, result.scale, wcs, jpg_file)
                print(f"Image avec étoiles appariées sauvegardée : {jpg_file}")


if __name__ == "__main__":
    main()
