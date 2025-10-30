## Actuator Torque Model Parameters
The positional gain and force parameters of the actuator model in Mujoco were
set according to the following derived model of a DC motor. This modelled has
been rearranged to match the [force generation model in
Mujoco](https://mujoco.readthedocs.io/en/stable/computation/index.html#force-generation). More
specifically, this derivation shows how to calculate the values $a$, $b_1$, and
$b_2$ in

$$\tau_m = a u_i + b_1 l_i + b_2 \dot{l}_i$$

Then, by the [actuator/position xml element](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-position), 

$k_p = a$, $k_p = -b_1$, $k_v = -b_2$ (1).

These parameters are calculated for the servo using the [motor model calculation
script](./calc_actuator_model.py) and servo datasheet information. **_This
script assumes that the servo proportional gain is set to a specific value (see
script) - if this gain is changed then these model values need to be
recalculated._**


The motor output torque ($\tau_m$) is related to the motor armature current
($i_a$) by the following expression, where $K_m$ is the motor torque constant:

$$\tau_m = K_m i_a \quad (2)$$

Using Kirchoff's voltage law on the circuit of a DC motor gives:

$$L \frac{di_a}{dt} + R i_a = V - V_b$$

where $L$ is the inductance, $R$ the armature resistance, $V$ the voltage
applied to the motor terminals, and $V_b$ the motor back EMF
voltage. This can then be rearranged as:

$$i_a = \frac{V - V_b - L di_a/dt}{R} \quad (3)$$

Substituting (3) into (2) gives:

$$\tau_m = K_m \frac{V - V_b - L \frac{di_a}{dt}}{R}$$

Assuming $L << R$, then $L/R = 0$, which gives

$$\tau_m = K_m \frac{V - V_b}{R} \quad (4)$$

Next, the back EMF voltage is related to the rotational speed of the motor
($\dot{\theta}$) by:

$$ V_b = K_b \dot{\theta} \quad (5)$$

where $K_b$ is the motor speed constant.

It is assumed that the PID controller of the servo is set to only use a
proportional gain term ($G_p$), and that the position error is in degrees
instead of radians. This gives the voltage applied to the motor by the servo PID
controller as:

$$ V = G_p e = G_p \frac{180}{\pi} (\theta_d - \theta) \quad (6)$$

where $\theta_d$ is the desired angular position, and $\theta$ is the current
angular velocity.

Substituting (5) and (6) into (4) gives:

$$
\tau_m = K_m (G_p \frac{180}{\pi} (\theta_d - \theta) - K_b \dot{\theta})/R
$$

Rearranging gives the desired form where $u_i = \theta_d$, $l_i = \theta$, and
$\dot{l}_i = \dot{\theta}$:

$$
\tau_m = (K_m G_p \frac{180}{\pi} / R) \theta_d - (K_m G_p \frac{180}{\pi} / R)\theta - (K_m K_b / R) \dot{\theta} \quad \quad (7)
$$

Finally, using (1) and (7) gives

$$k_p = a = -b_1 = (K_m G_p \frac{180}{\pi} / R)$$

$$k_v = -b_2 = -K_m K_b / R$$
