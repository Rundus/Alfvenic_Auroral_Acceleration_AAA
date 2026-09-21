

class DetectorFluxToggles:

    use_esa_specs_bool = False
    count_threshold = 1 # count level required for flux to output non-zero value
    esa_geometric_factor = 1.74E-4
    esa_deadtime = 674E-9 # [seconds]
    esa_acqusition_time = 0.9E-3 # [seconds]