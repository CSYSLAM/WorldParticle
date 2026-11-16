"""Select residual differentiation by physical contract, without changing values."""


def freeze(configuration,contract):
    frozen=configuration.get('frozen_residual_backward',False)
    composed=configuration.get('elastic_rod_differentiated_residual_backward',False)
    complete=configuration.get('protected_complete_response',False)
    if type(frozen) is not bool or type(composed) is not bool or type(complete) is not bool:
        raise ValueError('Explicit Boolean residual backward policies required')
    if contract=='elastic_rod_surface_joint_self_contact_v222' and composed:return False
    if complete:return False
    return frozen
