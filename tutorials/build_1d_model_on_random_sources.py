from kcsd.sources import get_dipoles

real, fakes = get_dipoles([1, -1, -1, 1, 1, -1], .2, .2, -1, 1, real=.5, randomness=1e-2)

print(real)


print(fakes)