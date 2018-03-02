from kcsd.sources import generate_sources_along_the_curve
from kcsd.csd_profile import add_2d_gaussians
import numpy as np
import matplotlib.pyplot as plt
import time

x = np.linspace(-2, 2, num=7)
y = np.exp(-x*2)

real_states, fake_states = generate_sources_along_the_curve(np.vstack((x, y)).T, 5,
                                                            seed=int(time.time()),
                                                            n_divisions=5, with_fakes=True)

# real_states = list(zip(*real))[0]
# real_states = np.array(real_states)

# fake_states = list(zip(*fakes))[0]
# fake_states = np.array(fake_states)

xx, yy = np.meshgrid(np.linspace(min(x), max(x), num=100), np.linspace(min(y), max(y), num=100))

plt.subplot(311)
plt.imshow(add_2d_gaussians(xx, yy, real_states))
plt.subplot(312)
plt.imshow(add_2d_gaussians(xx, yy, fake_states))
plt.subplot(313)
plt.imshow(add_2d_gaussians(xx, yy, np.vstack((real_states, fake_states))))
# plt.colorbar()
plt.show()