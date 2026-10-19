"""Physiological Modeling Challenge 9: Parts 1-3.

Run every implemented part:
    python main.py --part all

Run one part independently:
    python main.py --part 1
    python main.py --part 2
    python main.py --part 3

The model is based on papers and mathematical equations rather than patient-specific. It uses hours, liters,
and millimoles, and it requires only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs"


@dataclass(frozen=True)
class Parameters:
    """Literature-based parameters; citations are listed in appers.txt."""

    fasting_ph: float = 1.7
    fasting_fluid_volume_l: float = 0.025
    basal_secretion_mmol_h: float = 3.3

    heavy_meal_volume_l: float = 0.750
    heavy_meal_buffer_mmol: float = 119.0
    meal_buffered_ph: float = 5.0
    buffer_exhaustion_ph: float = 2.0
    buffer_titration_shape: float = 8.0
    meal_emptying_half_time_h: float = 2.5

    stimulated_peak_mmol_h: float = 40.0
    secretion_rise_time_h: float = 0.25
    secretion_wane_time_h: float = 2.5

    omeprazole_max_inhibition: float = 0.66
    omeprazole_onset_time_h: float = 24.0
    omeprazole_treatment_days: float = 7.0
    pump_recovery_half_time_h: float = 24.0

    simulation_start_h: float = -0.5
    simulation_end_h: float = 6.0
    time_step_h: float = 0.005
    clinical_ph_threshold: float = 4.0

    @property
    def fasting_hydrogen_mmol_l(self) -> float:
        return 1000.0 * 10.0 ** (-self.fasting_ph)

    @property
    def fasting_acid_mmol(self) -> float:
        return self.fasting_hydrogen_mmol_l * self.fasting_fluid_volume_l

    @property
    def meal_emptying_rate_h(self) -> float:
        return math.log(2.0) / self.meal_emptying_half_time_h

    @property
    def basal_liquid_outflow_l_h(self) -> float:
        return self.basal_secretion_mmol_h / self.fasting_hydrogen_mmol_l


@dataclass(frozen=True)
class MealPoint:
    time_h: float
    ph: float
    hydrogen_mmol_l: float
    secretion_mmol_h: float
    stomach_volume_l: float
    buffer_mmol: float


def meal_stimulus(time_h: float, parameters: Parameters) -> float:
    """Normalized secretion signal with a gradual rise and slower decay."""
    if time_h < 0.0:
        return 0.0

    rise = parameters.secretion_rise_time_h
    wane = parameters.secretion_wane_time_h
    signal = (1.0 - math.exp(-time_h / rise)) * math.exp(-time_h / wane)

    peak_time = rise * math.log((rise + wane) / rise)
    peak_signal = (1.0 - math.exp(-peak_time / rise)) * math.exp(
        -peak_time / wane
    )
    return signal / peak_signal


def secretion_rate(time_h: float, parameters: Parameters) -> float:
    stimulation = meal_stimulus(time_h, parameters)
    return parameters.basal_secretion_mmol_h + (
        parameters.stimulated_peak_mmol_h
        - parameters.basal_secretion_mmol_h
    ) * stimulation


def calculate_ph(
    acid_mmol: float,
    buffer_remaining_mmol: float,
    buffer_reference_mmol: float,
    stomach_volume_l: float,
    parameters: Parameters,
) -> tuple[float, float]:
    """Convert total acid and food-buffer capacity to free [H+] and pH.

    Before the food buffer is saturated, an empirical titration curve connects
    the measured meal-buffered pH to the pH 2 titration endpoint. After that
    endpoint, excess acid determines free hydrogen-ion concentration.
    """
    if buffer_remaining_mmol > 0.0 and buffer_reference_mmol > 0.0:
        fraction_neutralized = 1.0 - (
            buffer_remaining_mmol / buffer_reference_mmol
        )
        buffer_saturation = max(0.0, min(fraction_neutralized, 1.0)) ** (
            parameters.buffer_titration_shape
        )
        ph = parameters.meal_buffered_ph - (
            parameters.meal_buffered_ph - parameters.buffer_exhaustion_ph
        ) * buffer_saturation
        hydrogen_mmol_l = 1000.0 * 10.0 ** (-ph)
        return ph, hydrogen_mmol_l

    endpoint_hydrogen_mmol_l = 1000.0 * 10.0 ** (
        -parameters.buffer_exhaustion_ph
    )
    hydrogen_mmol_l = (
        endpoint_hydrogen_mmol_l + acid_mmol / stomach_volume_l
    )
    ph = -math.log10(hydrogen_mmol_l / 1000.0)
    return ph, hydrogen_mmol_l


def simulate_untreated_meal(parameters: Parameters) -> list[MealPoint]:
    """Simulate fasting baseline followed by one untreated heavy meal."""
    acid_mmol = parameters.fasting_acid_mmol
    buffer_mmol = 0.0
    buffer_reference_mmol = 0.0
    meal_has_started = False
    points: list[MealPoint] = []

    number_of_steps = round(
        (parameters.simulation_end_h - parameters.simulation_start_h)
        / parameters.time_step_h
    )

    for step in range(number_of_steps + 1):
        time_h = parameters.simulation_start_h + step * parameters.time_step_h

        if time_h >= 0.0 and not meal_has_started:
            buffer_mmol = parameters.heavy_meal_buffer_mmol
            buffer_reference_mmol = parameters.heavy_meal_buffer_mmol
            immediately_neutralized = min(acid_mmol, buffer_mmol)
            acid_mmol -= immediately_neutralized
            buffer_mmol -= immediately_neutralized
            meal_has_started = True

        if time_h < 0.0:
            meal_volume_l = 0.0
        else:
            meal_volume_l = parameters.heavy_meal_volume_l * math.exp(
                -parameters.meal_emptying_rate_h * time_h
            )

        stomach_volume_l = parameters.fasting_fluid_volume_l + meal_volume_l
        current_secretion = secretion_rate(time_h, parameters)

        if meal_has_started:
            ph, hydrogen_mmol_l = calculate_ph(
                acid_mmol,
                buffer_mmol,
                buffer_reference_mmol,
                stomach_volume_l,
                parameters,
            )
        else:
            hydrogen_mmol_l = acid_mmol / stomach_volume_l
            ph = -math.log10(hydrogen_mmol_l / 1000.0)

        points.append(
            MealPoint(
                time_h=time_h,
                ph=ph,
                hydrogen_mmol_l=hydrogen_mmol_l,
                secretion_mmol_h=current_secretion,
                stomach_volume_l=stomach_volume_l,
                buffer_mmol=buffer_mmol,
            )
        )

        if step == number_of_steps:
            break

        meal_outflow_l_h = parameters.meal_emptying_rate_h * meal_volume_l
        total_outflow_l_h = parameters.basal_liquid_outflow_l_h + meal_outflow_l_h
        outflow_fraction_h = total_outflow_l_h / stomach_volume_l

        acid_mmol = max(
            0.0,
            acid_mmol
            - parameters.time_step_h * outflow_fraction_h * acid_mmol,
        )
        buffer_mmol = max(
            0.0,
            buffer_mmol
            - parameters.time_step_h
            * parameters.meal_emptying_rate_h
            * buffer_mmol,
        )
        buffer_reference_mmol = max(
            0.0,
            buffer_reference_mmol
            - parameters.time_step_h
            * parameters.meal_emptying_rate_h
            * buffer_reference_mmol,
        )

        secreted_acid_mmol = parameters.time_step_h * current_secretion
        neutralized_acid_mmol = min(secreted_acid_mmol, buffer_mmol)
        buffer_mmol -= neutralized_acid_mmol
        acid_mmol += secreted_acid_mmol - neutralized_acid_mmol

    return points


def omeprazole_course(parameters: Parameters) -> list[tuple[float, float]]:
    """Return time in days and modeled proton-pump inhibition in percent."""
    treatment_stop_h = 24.0 * parameters.omeprazole_treatment_days
    simulation_end_h = treatment_stop_h + 4.0 * parameters.pump_recovery_half_time_h
    inhibition_at_stop = parameters.omeprazole_max_inhibition * (
        1.0 - math.exp(-treatment_stop_h / parameters.omeprazole_onset_time_h)
    )

    course: list[tuple[float, float]] = []
    for time_h in range(round(simulation_end_h) + 1):
        if time_h <= treatment_stop_h:
            inhibition = parameters.omeprazole_max_inhibition * (
                1.0 - math.exp(-time_h / parameters.omeprazole_onset_time_h)
            )
        else:
            inhibition = inhibition_at_stop * 2.0 ** (
                -(time_h - treatment_stop_h)
                / parameters.pump_recovery_half_time_h
            )
        course.append((time_h / 24.0, 100.0 * inhibition))

    return course


def run_part_1(parameters: Parameters) -> None:
    result = (
        "PART 1: FASTING BASELINE\n"
        "========================\n"
        f"Fasting gastric pH: {parameters.fasting_ph:.2f}\n"
        f"Hydrogen-ion concentration: {parameters.fasting_hydrogen_mmol_l:.2f} mmol/L\n"
        f"Fasting gastric fluid volume: {1000 * parameters.fasting_fluid_volume_l:.0f} mL\n"
        f"Initial free acid amount: {parameters.fasting_acid_mmol:.3f} mmol\n"
        f"Basal acid secretion: {parameters.basal_secretion_mmol_h:.1f} mmol/h\n"
    )
    print(result)
    (OUTPUT_DIRECTORY / "part1_baseline.txt").write_text(result, encoding="utf-8")


def run_part_2(parameters: Parameters) -> None:
    points = simulate_untreated_meal(parameters)
    post_meal = [point for point in points if point.time_h >= 0.0]
    peak_ph = max(point.ph for point in post_meal)
    minimum_ph = min(point.ph for point in post_meal)
    minutes_below_four = (
        60.0
        * parameters.time_step_h
        * sum(point.ph < parameters.clinical_ph_threshold for point in post_meal)
    )

    write_meal_csv(OUTPUT_DIRECTORY / "part2_untreated.csv", points)
    write_svg_chart(
        OUTPUT_DIRECTORY / "part2_untreated.svg",
        title="Untreated gastric pH after a heavy meal",
        x_label="Time relative to meal (hours)",
        y_label="Gastric pH",
        x_limits=(-0.5, 6.0),
        y_limits=(0.0, 7.0),
        series=[
            (
                "Untreated heavy meal",
                "#b33a3a",
                [(point.time_h, point.ph) for point in points],
            )
        ],
        threshold=parameters.clinical_ph_threshold,
    )

    print("PART 2: UNTREATED HEAVY MEAL")
    print("==============================")
    print(f"Maximum post-meal pH: {peak_ph:.2f}")
    print(f"Minimum pH over six hours: {minimum_ph:.2f}")
    print(f"Time below pH 4: {minutes_below_four:.1f} minutes")
    print("Created outputs/part2_untreated.csv and outputs/part2_untreated.svg\n")


def run_part_3(parameters: Parameters) -> None:
    course = omeprazole_course(parameters)
    write_course_csv(OUTPUT_DIRECTORY / "part3_omeprazole_course.csv", course)
    write_svg_chart(
        OUTPUT_DIRECTORY / "part3_omeprazole_course.svg",
        title="Modeled onset and recovery of omeprazole inhibition",
        x_label="Time (days; treatment stops after day 7)",
        y_label="Proton-pump inhibition (%)",
        x_limits=(0.0, max(day for day, _ in course)),
        y_limits=(0.0, 70.0),
        series=[("Omeprazole", "#277da1", course)],
    )

    inhibition_at_stop = course[round(24 * parameters.omeprazole_treatment_days)][1]
    explanation = (
        "PART 3: OMEPRAZOLE\n"
        "==================\n"
        "Omeprazole is a proton-pump inhibitor. It is activated in the acidic "
        "secretory canaliculus of active parietal cells and covalently inhibits "
        "the H+/K+-ATPase, the final step in gastric acid secretion. Its plasma "
        "half-life is short, but pump inhibition lasts much longer. Repeated "
        "doses are needed because not all pumps are active at the same time.\n\n"
        f"Modeled maximum inhibition: {100 * parameters.omeprazole_max_inhibition:.0f}%\n"
        f"Modeled inhibition after seven treatment days: {inhibition_at_stop:.1f}%\n"
        f"Modeled recovery half-time after stopping: "
        f"{parameters.pump_recovery_half_time_h:.0f} hours\n"
    )
    print(explanation)
    (OUTPUT_DIRECTORY / "part3_omeprazole.txt").write_text(
        explanation, encoding="utf-8"
    )
    print(
        "Created outputs/part3_omeprazole_course.csv, "
        "outputs/part3_omeprazole_course.svg, and outputs/part3_omeprazole.txt\n"
    )


def write_meal_csv(path: Path, points: list[MealPoint]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            (
                "time_h",
                "pH",
                "hydrogen_mmol_L",
                "secretion_mmol_h",
                "stomach_volume_L",
                "buffer_mmol",
            )
        )
        for point in points:
            writer.writerow(
                (
                    f"{point.time_h:.4f}",
                    f"{point.ph:.5f}",
                    f"{point.hydrogen_mmol_l:.5f}",
                    f"{point.secretion_mmol_h:.5f}",
                    f"{point.stomach_volume_l:.5f}",
                    f"{point.buffer_mmol:.5f}",
                )
            )


def write_course_csv(path: Path, course: list[tuple[float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("day", "pump_inhibition_percent"))
        writer.writerows(
            (f"{day:.4f}", f"{inhibition:.4f}")
            for day, inhibition in course
        )


def write_svg_chart(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    series: list[tuple[str, str, list[tuple[float, float]]]],
    threshold: float | None = None,
) -> None:
    width, height = 900, 520
    left, right, top, bottom = 80, 30, 55, 70
    x_min, x_max = x_limits
    y_min, y_max = y_limits

    def coordinates(x_value: float, y_value: float) -> tuple[float, float]:
        x_pixel = left + (width - left - right) * (x_value - x_min) / (
            x_max - x_min
        )
        y_pixel = top + (height - top - bottom) * (y_max - y_value) / (
            y_max - y_min
        )
        return x_pixel, y_pixel

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{escape(title)}</text>',
    ]

    for tick_number in range(8):
        y_value = y_min + tick_number * (y_max - y_min) / 7
        _, y_pixel = coordinates(x_min, y_value)
        elements.append(
            f'<line x1="{left}" y1="{y_pixel:.1f}" x2="{width - right}" y2="{y_pixel:.1f}" stroke="#e5e5e5"/>'
        )
        elements.append(
            f'<text x="{left - 12}" y="{y_pixel + 4:.1f}" text-anchor="end" font-family="Arial" font-size="13">{y_value:g}</text>'
        )

    for tick_number in range(7):
        x_value = x_min + tick_number * (x_max - x_min) / 6
        x_pixel, _ = coordinates(x_value, y_min)
        elements.append(
            f'<line x1="{x_pixel:.1f}" y1="{top}" x2="{x_pixel:.1f}" y2="{height - bottom}" stroke="#e5e5e5"/>'
        )
        elements.append(
            f'<text x="{x_pixel:.1f}" y="{height - bottom + 24}" text-anchor="middle" font-family="Arial" font-size="13">{x_value:.1f}</text>'
        )

    if threshold is not None:
        _, threshold_y = coordinates(x_min, threshold)
        elements.append(
            f'<line x1="{left}" y1="{threshold_y:.1f}" x2="{width - right}" y2="{threshold_y:.1f}" stroke="#555" stroke-width="2" stroke-dasharray="7 6"/>'
        )
        elements.append(
            f'<text x="{width - right - 8}" y="{threshold_y - 7:.1f}" text-anchor="end" font-family="Arial" font-size="12">pH {threshold:g}</text>'
        )

    elements.extend(
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111" stroke-width="2"/>',
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111" stroke-width="2"/>',
            f'<text x="{width / 2}" y="{height - 16}" text-anchor="middle" font-family="Arial" font-size="15">{escape(x_label)}</text>',
            f'<text x="20" y="{height / 2}" text-anchor="middle" transform="rotate(-90 20 {height / 2})" font-family="Arial" font-size="15">{escape(y_label)}</text>',
        )
    )

    for index, (label, color, values) in enumerate(series):
        polyline_points = " ".join(
            f"{x_pixel:.2f},{y_pixel:.2f}"
            for x_pixel, y_pixel in (
                coordinates(x_value, y_value) for x_value, y_value in values
            )
        )
        elements.append(
            f'<polyline points="{polyline_points}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        legend_y = top + 20 + 23 * index
        elements.append(
            f'<line x1="{left + 15}" y1="{legend_y}" x2="{left + 43}" y2="{legend_y}" stroke="{color}" stroke-width="4"/>'
        )
        elements.append(
            f'<text x="{left + 51}" y="{legend_y + 5}" font-family="Arial" font-size="13">{escape(label)}</text>'
        )

    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Parts 1-3 of the gastric acid modeling challenge."
    )
    parser.add_argument(
        "--part",
        choices=("1", "2", "3", "all"),
        default="all",
        help="Part to run; the default is all implemented parts.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    parameters = Parameters()
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    part_functions = {
        "1": run_part_1,
        "2": run_part_2,
        "3": run_part_3,
    }
    selected_parts = (
        part_functions.keys() if arguments.part == "all" else (arguments.part,)
    )
    for part in selected_parts:
        part_functions[part](parameters)

    print(f"Output directory: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
