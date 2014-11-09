"""
Tests for hybrid
"""

import hybrid

import numpy as np
import numpy.testing as npt

from scipy.misc import logsumexp


def tau_sum_test():
    init = 0.5
    trans = 0.5

    prob_0 = np.exp(hybrid.tau_logp(0, init, trans))
    prob_greater_0 = np.exp(hybrid.tau_logp(0, init, trans))

    npt.assert_almost_equal(prob_0 + prob_greater_0, 1)


def run_trial_logp(value, guess, slip, init, trans):


    tau_logps = np.array([hybrid.tau_logp(i, init, trans)
                          for i in xrange(value.shape[0]+1)])

    logps_per_tau = []
    for tau in xrange(value.shape[0]):

        if tau == 0:
            logp_tau = np.log(init)
        else:
            logp_tau = (np.log(1 - init) + np.log(trans) +
                        (np.log(1 - trans) * (tau - 1)))

        npt.assert_equal(tau_logps[tau], logp_tau)

        logp_given_tau = 0.0
        for i, obs in enumerate(value):
            if i < tau:
                p_correct = guess[i]
            else:
                p_correct = 1 - slip[i]

            if obs == 1:
                p_obs = p_correct
            else:
                p_obs = 1 - p_correct

            logp_given_tau += np.log(p_obs)

        logps_per_tau.append(logp_tau + logp_given_tau)

    logps_given_not_mastered = 0.0
    for i, obs in enumerate(value):
        if obs == 1:
            logps_given_not_mastered += np.log(guess[i])
        else:
            logps_given_not_mastered += np.log(1 - guess[i])

    logp_trials_not_mastered = (hybrid.tau_greater_logp(value.shape[0] - 1,
                                                        init, trans) +
                                logps_given_not_mastered)

    logp_trials = np.logaddexp(logsumexp(logps_per_tau),
                               logp_trials_not_mastered)

    actual_logp_trials = hybrid.trial_logp(value, guess, slip, init, trans,
                                           tau_logps)
    npt.assert_almost_equal(actual_logp_trials, logp_trials)

def test_trial_logp():

    init = 0.2
    trans = 0.5
    value = np.array([1])
    guess = np.array([0.2])
    slip = np.array([0.2])

    yield run_trial_logp, value, guess, slip, init, trans

    yield run_trial_logp, np.array([0]), guess, slip, init, trans

    yield (run_trial_logp, np.array([0, 0]), np.array([0.2, 0.2]),
           np.array([0.4, 0.5]), init, trans)
    yield (run_trial_logp, np.array([0, 1]), np.array([0.2, 0.2]),
           np.array([0.4, 0.5]), init, trans)
    yield (run_trial_logp, np.array([1, 0]), np.array([0.2, 0.2]),
           np.array([0.4, 0.5]), init, trans)
    yield (run_trial_logp, np.array([1, 1]), np.array([0.2, 0.2]),
           np.array([0.4, 0.5]), init, trans)

    yield (run_trial_logp,
           np.array([0, 1, 1]),
           np.array([0.3, 0.1, 0.2]),
           np.array([0.4, 0.2, 0.1]),
           init,
           trans)

