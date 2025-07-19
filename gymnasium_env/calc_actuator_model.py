import numpy as np

# Calculations are based on squid notes under the "Directed Research"
# folder. These are derived from DC motor model equations. It assumes that the
# gain used in servo uses units of gain/deg instead of gain/raw_pos. The
# ST-3215-C047 model data is used. All the parameters are calculated based on
# the output of the servo (not the motor), so the gear ratio does not need to be
# used.


def calcKm():
    # use stall conditions
    tau_stall = 30 * 9.8 / 100 # 30 kg.cm x 9.8 N/kg x 1 m / 100 cm => [N.m]
    print("stall torque: {}".format(tau_stall))
    i_stall = 2.7 # A
    Km = tau_stall / i_stall
    return Km


def calcKb(R):
    # use no load conditions
    V_rated = 12 # V
    i_nl = 180e-3 # A
    omega_nl = 45 * 2*np.pi / 60 # 45 rev/min x 2*pi rad/rev x 1 min / 60 s
    Kb = (V_rated - R*i_nl)/omega_nl
    return Kb


def main():
    R = 1 # Ohm
    Km = calcKm()
    Kb = calcKb(R)
    print("Km = {}, Kb = {}".format(Km, Kb))

    # proportional gain
    kp_p_deg = 12
    kp_p_rad = kp_p_deg * 180 / np.pi
    
    # calculate mujoco actuator model values
    a = Km * kp_p_rad / R
    b1 = -Km * kp_p_rad / R
    b2 = -Km * Kb / R
    b0 = 0

    print("Mujoco actuator parameters:\na={}\nb1={}\nb2={}\nb0=0".format(a,b1,b2))
    

if __name__ == "__main__":
    main()
