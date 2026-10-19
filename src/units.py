"""emission-unit conversion via fair's unit tables; in: value + units; out: converted value."""
from fair.structure.units import compound_convert, prefix_convert, time_convert


def emissions_unit_factor(from_unit, to_unit):
    # factor for "<prefix> <compound>/<time>", e.g. "Mt N2O/yr"; NO2==NOx via fair
    if from_unit == to_unit:
        return 1.0
    try:
        from_prefix, from_rest = from_unit.split()
        to_prefix, to_rest = to_unit.split()
        from_compound, from_time = from_rest.split("/")
        to_compound, to_time = to_rest.split("/")
    except ValueError as exc:
        raise ValueError(
            f"cannot parse emission unit {from_unit!r} or {to_unit!r} "
            "(expected '<prefix> <compound>/<time>')"
        ) from exc
    try:
        return (
            prefix_convert[from_prefix][to_prefix]
            * compound_convert[from_compound][to_compound]
            * time_convert[from_time][to_time]
        )
    except KeyError as exc:
        raise ValueError(
            f"no fair emission-unit conversion {from_unit!r} -> {to_unit!r} "
            f"(unknown token {exc})"
        ) from exc


def convert_emissions(values, from_unit, to_unit):
    return values * emissions_unit_factor(from_unit, to_unit)
