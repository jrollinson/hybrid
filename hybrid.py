"""
This module provides the model found in Integrating Latent-Factor and Knowledge
Tracing to Predict Individual Differences in Learning.
"""

import pymc as pm
import numpy as np
import pandas as pd

from scipy.misc import logsumexp

import os.path
import itertools

DATAPATH = 'data/'

DF = pd.read_pickle(os.path.join(DATAPATH, 'glops.pkl'))
# first_skill = df['skill'][0]
# df = df[df['skill'] == first_skill]

def skill_to_problem(skills, seq_lengths):
    """
    Creates a list of problem sequences from skills.
    """
    problems_list = []
    for skill, length in zip(skills, seq_lengths):
        problems_list.append(np.repeat(skill, length))
    return problems_list


def stack_ragged(array_list, axis=0):
    """
    Concatenates a list of ragged arrays and also returns the index positions of
    the array
    """
    lengths = [np.shape(a)[axis] for a in array_list]
    idx = np.cumsum(lengths[:-1])
    stacked = np.concatenate(array_list, axis=axis)
    return stacked, idx

def index_mapper(vals):
    """
    Takes an iterable of vals and creates a dictionary mapping vals to an index
    """
    d = {}
    i = 0
    for v in vals:
        if v not in d:
            d[v] = i
            i += 1
    return d

def tau_greater_logp(i, init, trans):
    """
    Calculates the probability that tau > i.
    In other words that the student does not achieve mastery in the first i
    questions.
    """
    return np.log(1 - init) + (i * np.log(1 - trans))

def tau_logp(tau, init, trans):
    """
    Calculates the probability of a tau value.

    In other words, the probability that tau is the first observation in which
    the student has mastered the material.
    """

    log_init = np.log(init)
    log_init_fail = np.log(1 - init)
    log_trans = np.log(trans)
    log_trans_fail = np.log(1 - trans)

    if tau < 0:
        return - np.inf
    elif tau == 0:
        return log_init
    else: # tau > 0
        return log_init_fail + log_trans + (log_trans_fail * (tau - 1))

def trial_logp(value, guess, slip, init, trans, tau_logps):

    # TODO: Check trans and init

    length, = value.shape
    index_array = np.arange(length)
    tau_array = np.arange(length+1)

    is_mastered = (np.expand_dims(index_array, axis=1) >=
                   np.expand_dims(tau_array, axis=0))
    p_correct = np.where(is_mastered,
                         np.expand_dims(1-slip, axis=1),
                         np.expand_dims(guess, axis=1))

    ps = np.where(np.expand_dims(value, axis=1) == 1,
                  p_correct, 1 - p_correct)

    logps = np.log(ps)

    logps_given_tau = np.sum(logps, axis=0)
    logps_mastered = logps_given_tau[:length] + tau_logps[:length]

    logp_mastered = logsumexp(logps_mastered)
    logp_not_mastered = (tau_greater_logp(length-1, init, trans) +
                         logps_given_tau[length])

    return np.logaddexp(logp_mastered, logp_not_mastered)

def trial_logp_given_prev_seq(trials, guesses, slips, init, trans):
    """Log probability of the final trial given the previous trials.

    :trials: The sequences of trials (0s and 1s)
    :guesses: The guess probabilities for each trial
    :slips: The slip probabilities for each trial
    :init: The initial probability of mastery
    :trans: The probability of transition to mastery
    :returns: log probability of the final trial given previous trials.

    """

    seq_len, = trials.shape
    tau_logps = [tau_logp(tau, init, trans) for tau in xrange(seq_len)]

    seq_logp = trial_logp(trials, guesses, slips, init, trans, tau_logps)

    if seq_len == 1:
        result = seq_logp

    else:
        prev_seq_logp = trial_logp(trials[:-1], guesses[:-1], slips[:-1], init,
                                   trans, tau_logps)
        result = seq_logp - prev_seq_logp

    return result


def hybrid_vars(sequence_list, student_list, problems_list):
    """
    Creates all hybrid variables

    Arguments:
        sequence_list: A list of observation sequences.
        student_list: A list of students for each observation sequence.
        problems_list: A list of problem sequences for each observation sequnce.

    Returns: All variables in a dict

    """

    student_dict = index_mapper(student_list)
    problem_dict = index_mapper(itertools.chain.from_iterable(problems_list))

    n_students = len(student_dict)
    n_problems = len(problem_dict)

    # Create a stacked ragged array of student indices per trial.
    student_per_seq = [student_dict[student] for student in student_list]

    # Create a stacked ragged array of problem indices per trail
    problems_per_trial = []
    for problem_seq in problems_list:
        problem_ids = np.array([problem_dict[x] for x in problem_seq])
        problems_per_trial.append(problem_ids)

    tau_len = max([seq.shape[0] for seq in sequence_list]) + 1

    # Make the guess and slip offset vars
    guess_offset = pm.Uniform('Guess Offset', -3.0, 3.0)
    slip_offset = pm.Uniform('Slip Offset', -3.0, 3.0)
    print 'made guess and slip offset'

    # The probability that a student begins with mastery.
    initial_mastered = pm.Uniform('Initial Mastered Probability', 0.0, 1.0)

    # The probability that a student transitions to mastery.
    transition_mastered = pm.Uniform('Transition Mastered Probability',
                                     0.0, 1.0)
    print 'made init and trans'

    # The variance in student aptitude and problem difficulty.
    alpha_variance = pm.InverseGamma('Student Aptitude Variance', 1, 2)
    delta_variance = pm.InverseGamma('Problem Difficulty Variance', 1, 2)

    alphas = np.empty(n_students, dtype=object)
    for i in xrange(n_students):
        alphas[i] = pm.Normal('Student Aptitudes %d' % i, 0, alpha_variance)
    print 'made alphas'

    alphas_seq = np.empty(len(sequence_list), dtype=object)
    for i, student in enumerate(student_per_seq):
        alphas_seq[i] = alphas[student]
    print 'made alphas_seq'

    # The difficulty of each problem.
    deltas = np.empty(n_problems, dtype=object)
    for i in xrange(n_problems):
        deltas[i] = pm.Normal('Problem Difficulties %d' % i,
                              0, delta_variance)
    print 'made deltas'

    deltas_seq = np.empty(len(sequence_list), dtype=object)
    for i, problems in enumerate(problems_per_trial):
        ds = np.array([deltas[problem] for problem in problems])
        deltas_seq[i] = ds
    print 'made deltas_seq'

    guesses = np.empty(len(sequence_list), dtype=object)
    for i in xrange(len(sequence_list)):
        guesses[i] = pm.InvLogit('guess %d' % i,
                                 alphas_seq[i] - deltas_seq[i] + guess_offset)
    print 'made guesses'

    slips = np.empty(len(sequence_list), dtype=object)
    for i in xrange(len(sequence_list)):
        slips[i] = pm.InvLogit('slip %d' % i,
                               deltas_seq[i] - alphas_seq[i] + slip_offset)
    print 'made seqs'

    @pm.deterministic
    def tau_logps(init=initial_mastered, trans=transition_mastered,
                  tau_len=tau_len):
        logps = np.empty(tau_len)
        for i in xrange(tau_len):
            logps[i] = tau_logp(i, init, trans)
        return logps

    print 'made tau_logps'

    trials = np.empty(len(sequence_list), dtype=object)
    for i in xrange(len(sequence_list)):

        guess = guesses[i]
        slip = slips[i]
        seq = sequence_list[i]

        trials[i] = pm.Stochastic(logp=trial_logp,
                                  doc='The observed trial',
                                  name='trial %d' % i,
                                  parents={'guess' : guess,
                                           'slip' : slip,
                                           'init' : initial_mastered,
                                           'trans' : transition_mastered,
                                           'tau_logps' : tau_logps},
                                  trace=False,
                                  value=seq,
                                  observed=True,
                                  plot=False)
    print 'made trials'

    return {"guess_offset" : guess_offset,
            "slip_offset" : slip_offset,
            "initial_mastered" : initial_mastered,
            "transition_mastered" : transition_mastered,
            "alpha_variance" : alpha_variance,
            "delta_variance" : delta_variance,
            "alphas" : alphas,
            "deltas" : deltas,
            "guesses" : guesses,
            "slips" : slips,
            "trials" : trials}



def hybrid_model(sequence_list, student_list, problems_list, db='ram',
        dbname='MCMC.pkl'):
    """Returns a hybrid model

    :sequence_list: A list of observation sequences.
    :student_list: A list of students for each observation sequence.
    :problems_list: A list of problem sequences for each observation sequnce.
    :single_alpha: Whether a single alpha array variable should be used or
                   one variable per student.
   : single_delta: Whether a single delta array variable should be used or
                   one variable per problem type.

    :returns: A MCMC model with variables set to step method.

    """
    variables = hybrid_vars(sequence_list, student_list, problems_list)

    model = pm.MCMC(variables, db, dbname=dbname)

    model.use_step_method(pm.Slicer, model.guess_offset, w=0.1, m=6.0)
    model.use_step_method(pm.Slicer, model.slip_offset, w=0.1, m=6.0)
    model.use_step_method(pm.Slicer, model.initial_mastered, w=0.01, m=1.0)
    model.use_step_method(pm.Slicer, model.transition_mastered, w=0.01, m=1.0)
    model.use_step_method(pm.Slicer, model.alpha_variance, w=0.1, m=1.0)
    model.use_step_method(pm.Slicer, model.delta_variance, w=0.1, m=1.0)

    for alpha in model.alphas:
        model.use_step_method(pm.Slicer, alpha, w=0.1, m=3.0)

    for delta in model.deltas:
        model.use_step_method(pm.Slicer, delta, w=0.1, m=3.0)

    return model

